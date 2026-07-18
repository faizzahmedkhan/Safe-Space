import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
import torchvision.transforms as T
import numpy as np
import cv2
import os
import time
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from collections import deque
from camera import CameraStream, CLIP_LENGTH # Importing the Producer we just built!

""
INFERENCE_STRIDE = 8
ALERT_THRESHOLD = 0.60
ALERT_COOLDOWN_SECONDS = 5
SCORE_HISTORY_LEN = 3

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

alert_log: list[dict] = []

score_history: deque = deque(maxlen=SCORE_HISTORY_LEN)

# ==========================================
# 1. LOAD THE BRAIN (System Startup)
# ==========================================
# If your laptop doesn't have a GPU, PyTorch gracefully falls back to the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Booting AI Core on: {device}")

# Rebuild the empty architecture
model = video_models.r3d_18()
model.fc = nn.Linear(model.fc.in_features, 2)

# Inject your trained weights into the architecture
# map_location=device ensures it loads safely on a Laptop CPU
model.load_state_dict(torch.load('safespace_bullying_model.pth', map_location=device))
model = model.to(device)
model.eval() # STRICTLY set to evaluation mode (no learning)

normalize = T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD)

# Turn on the camera (Thread 1 - The Producer). Set show_preview=False to avoid cv2.imshow on headless setups.
cam = CameraStream(src=0, target_fps=15, show_preview=False).start()

os.makedirs("evidence", exist_ok=True)

def _save_evidence_sync(buffer_snapshot: list, filename: str) -> None:
  """
  Synchronous disk-write function - intentionally NOT async.
  Called via run_in_executor so it runs in a thread pool and never blocks the FastAPI event loop while writing frames to disk.
  """
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(filename, fourcc, 15.0, (112, 112))
  for f_rgb in buffer_snapshot:
    out.write(cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR))
  out.release()


# ==========================================
# 2. THE CONSUMER LOOP (Background Worker)
# ==========================================
async def ai_watcher():
  """
  This function runs continuously in the background.
  It checks the camera buffer, runs the math, and triggers alerts.
  """
  print(f"[AI WATCHER] Online. Waiting for the camera buffer to fill...")

  last_processed_frame_count = 0
  last_alert_time = 0.0

  while True:
    buffer = cam.get_buffer()
    current_frame_count = cam.get_frame_count()

    new_frames_since_last = current_frame_count - last_processed_frame_count
    buffer_ready = len(buffer) == CLIP_LENGTH
    stride_reached = new_frames_since_last >= INFERENCE_STRIDE

    # Only run the heavy math if we have complete 16-frame video clip
    if buffer_ready and stride_reached:
      last_processed_frame_count = current_frame_count

      # Step A: Convert the list of images into a single NumPy array
      # Current shape: (Time=16, Height=112, Width=112, Channels=3)
      frames_array = np.stack(buffer)

      # Step B: Convert to a PyTorch Tensor and normalize (0.0 to 1.0)
      # First move channels to dim=1 so torchvision.Normalize can broadcast correctly.
      tensor = torch.from_numpy(frames_array.copy()).permute(0, 3, 1, 2).float()  # (T, C, H, W)
      tensor.div_(255.0)
      tensor = normalize(tensor)

      # Step C: Reorganize to match R3D_18 expectations -> (C, T, H, W)
      tensor = tensor.permute(1, 0, 2, 3)

      # Add the "Batch" dimension -> [1, 3, 16, 112, 112]
      inputs = tensor.unsqueeze(0).to(device)

      # Step D: The Prediction (The actual AI thinking)
      with torch.no_grad():
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        current_prob = probabilities[0][1].item()

      # Step E: The Output Logic
      score_history.append(current_prob)
      if len(score_history) < score_history.maxlen:
        print(
            f"[WARMUP] {len(score_history)}/{score_history.maxlen} "
            f"predictions collected. Raw: {current_prob*100:.1f}%"
        )
      else:
        avg_prob = sum(score_history) / len(score_history)
        

        print(f"Stats -> [Raw Spike: {current_prob*100:.1f}%] | [SMOOTHED AVG: {avg_prob*100:.1f}%]")

        if avg_prob >= ALERT_THRESHOLD:
          current_time = time.time()

          if current_time - last_alert_time > ALERT_COOLDOWN_SECONDS:
            print(f"🚨🚨 ALERT: PHYSICAL ALTERCATION DETECTED! 🚨🚨")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"evidence/incident_{timestamp}.mp4"

            loop = asyncio.get_event_loop()
            buffer_snapshot = list(buffer)

            await loop.run_in_executor(None, _save_evidence_sync, buffer_snapshot, filename)

            print(f"📁 EVIDENCE SECURED: Saved to {filename}")

            last_alert_time = current_time

            alert_log.append({
              "timestamp": timestamp,
              "confidence": round(avg_prob * 100, 1),
              "evidence_file": filename,
            })
          # In the future, this is where you write code to send an email or SMS!
        else:
          print(f"✅ Safe.")
    
    # CRITICAL: Pause this thread for 1 second
    # This prevents the AI from hogging 100% of your laptop's CPU and allows the camera to keep recording smoothly.
    await asyncio.sleep(1)

# ==========================================
# 3. FASTAPI ROUTING (The Web Interface)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
  # The moment the web server starts, launch our AI watcher in the background
  asyncio.create_task(ai_watcher())
  yield
  print("[SYSTEM] Shutting down SafeSpace...")
  cam.stop()

# Initialize the web server
app = FastAPI(title="SafeSpace AI Backend", lifespan=lifespan)

@app.get("/")
def read_root():
  # A simple endpoint so we can verify the server is running from the web browser
  return {"status": "SafeSpace Core is Active", "camera_active": not cam.stopped}

@app.get("/status")
def get_status():
  """
  Returns real-time system diagnostics:
   - Whether the camera is running
   - How full the frame buffer is
   - The latest raw and smoothed detection scores
  """
  scores = list(score_history)
  latest_raw = round(scores[-1] * 100, 1) if scores else None
  latest_avg = round((sum(scores) / len(scores)) * 100, 1) if len(scores) == SCORE_HISTORY_LEN else None

  return {
    "camera_active": not cam.stopped,
    "buffer_fill": f"{len(cam.get_buffer())}/{CLIP_LENGTH}",
    "total_frames": cam.get_frame_count(),
    "latest_raw_pct": latest_raw,
    "smoothed_avg_pct": latest_avg,
    "alert_threshold": ALERT_THRESHOLD * 100,
  }

@app.get("/alerts")
def get_alerts():
  """
  Returns the 20 most recent detected incidents with timestamps, confidence scores, and paths to saved evidence clips.
  """
  return {
    "total_alerts": len(alert_log),
    "recent": alert_log[-20:],
  }