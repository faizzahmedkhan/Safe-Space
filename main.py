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
import json
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import tempfile
from collections import deque
from camera import CameraStream, CLIP_LENGTH # Importing the Producer we just built!

# --- HARDWARE SETTINGS ---
# Set CAMERA_SOURCE to 0 for your laptop webcam.
# To use your mobile phone as a CCTV camera:
# 1. Download "IP Webcam" on Android or "IP Camera Lite" on iPhone.
# 2. Connect your phone and laptop to the same Wi-Fi.
# 3. Start the server in the app. It will give you a URL like: http://192.168.1.5:8080/video
# 4. Paste that exact URL below (in quotes) instead of 0.
CAMERA_SOURCE = "http://192.168.100.249:8080/video"  # e.g., "http://192.168.1.5:8080/video"

# --- THRESHOLDS & TIMINGS ---
INFERENCE_STRIDE = 8
ALERT_THRESHOLD = 0.60
ALERT_COOLDOWN_SECONDS = 5
SCORE_HISTORY_LEN = 6
MAX_ALERTS = 500

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

ALERTS_FILE = "alerts.json"
alert_log: list[dict] = []

def _save_alerts_atomic(alerts: list, path: str) -> None:
    dir_name = os.path.dirname(path) or "."
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp") as tmp:
            json.dump(alerts, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[SYSTEM] Could not persist alerts: {e}")

if os.path.exists(ALERTS_FILE):
    try:
        with open(ALERTS_FILE, "r") as f:
            alert_log = json.load(f)
    except Exception as e:
        print(f"[SYSTEM] Could not load past alerts: {e}")

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
model_path = 'safespace_bullying_model.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please run 'git lfs pull' to download the weights.")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval() # STRICTLY set to evaluation mode (no learning)

normalize = T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD)

# Turn on the camera (Thread 1 - The Producer). Set show_preview=False to avoid cv2.imshow on headless setups.
cam = CameraStream(src=CAMERA_SOURCE, target_fps=15, show_preview=False).start()

os.makedirs("evidence", exist_ok=True)

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
  alert_active = False

  # Sleep just long enough for INFERENCE_STRIDE new frames to arrive
  # e.g. STRIDE=8 at 15 FPS -> sleep 0.53s
  INFERENCE_SLEEP = INFERENCE_STRIDE / cam.target_fps

  while True:
    inference_frames, evidence_frames, current_frame_count = cam.get_snapshot()

    new_frames_since_last = current_frame_count - last_processed_frame_count
    buffer_ready = len(inference_frames) == CLIP_LENGTH
    stride_reached = new_frames_since_last >= INFERENCE_STRIDE

    if not buffer_ready:
      if score_history:
        print("[AI WATCHER] Camera not ready — clearing stale score history.")
        score_history.clear()
        alert_active = False
      await asyncio.sleep(INFERENCE_SLEEP)
      continue

    # Only run the heavy math if we have complete 16-frame video clip and hit our stride
    if stride_reached:
      last_processed_frame_count = current_frame_count

      # Step A: Convert the list of images into a single NumPy array
      # Current shape: (Time=16, Height=112, Width=112, Channels=3)
      frames_array = np.stack(inference_frames)

      # Step B: Convert to a PyTorch Tensor and normalize (0.0 to 1.0)
      # First move channels to dim=1 so torchvision.Normalize can broadcast correctly.
      tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
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
          if not alert_active:
            alert_active = True
            current_time = time.time()
            print(f"🚨🚨 ALERT: PHYSICAL ALTERCATION DETECTED! 🚨🚨")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"evidence/incident_{timestamp}.mp4"

            cam.start_recording(filename)

            print(f"📁 EVIDENCE SECURED: Started recording to {filename}")

            last_alert_time = current_time
            global alert_log
            alert_log.append({
              "timestamp": timestamp,
              "confidence": round(avg_prob * 100, 1),
              "evidence_file": filename,
            })
            
            if len(alert_log) > MAX_ALERTS:
              alert_log = alert_log[-MAX_ALERTS:]
            
            # Save to disk atomically so we don't corrupt history on crash
            _save_alerts_atomic(alert_log, ALERTS_FILE)
                
          # In the future, this is where you write code to send an email or SMS!
        else:
          if alert_active:
            cam.stop_recording()
            print(f"📁 EVIDENCE RECORDING STOPPED.")
            alert_active = False
          print(f"✅ Safe.")
    
    # Sleep to pace the loop
    await asyncio.sleep(INFERENCE_SLEEP)

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

def _generate_mjpeg():
  """
  Generator that yields JPEG frames as an MJPEG stream.
  Open http://127.0.0.1:8000/preview in your browser to see the live feed.
  """
  while True:
    jpeg_bytes = cam.get_latest_frame_jpeg()
    if jpeg_bytes is not None:
      yield (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
      )
    time.sleep(1.0 / 10)  # ~10 FPS for the browser preview

@app.get("/feed")
def video_feed():
  """Raw MJPEG stream of the camera feed."""
  return StreamingResponse(
    _generate_mjpeg(),
    media_type="multipart/x-mixed-replace; boundary=frame"
  )

@app.get("/preview")
def preview_page():
  """
  Opens a simple HTML page in your browser that shows the live camera feed
  so you can verify the phone camera is connected and the model is seeing real frames.
  """
  scores = list(score_history)
  latest_raw = round(scores[-1] * 100, 1) if scores else 0
  latest_avg = round((sum(scores) / len(scores)) * 100, 1) if len(scores) == SCORE_HISTORY_LEN else 0

  html = f"""
  <!DOCTYPE html>
  <html>
  <head>
    <title>SafeSpace Live Preview</title>
    <style>
      body {{ background: #0f172a; color: #e2e8f0; font-family: sans-serif; text-align: center; padding: 20px; }}
      h1 {{ color: #38bdf8; }}
      img {{ border: 2px solid #334155; border-radius: 12px; max-width: 90vw; }}
      .stats {{ margin-top: 16px; font-size: 18px; color: #94a3b8; }}
    </style>
  </head>
  <body>
    <h1>🛡️ SafeSpace — Live Camera Preview</h1>
    <p>If you see your phone's camera feed below, the connection is working.</p>
    <img src="/feed" alt="Live Camera Feed" />
    <div class="stats">
      <p>Latest Raw Score: <strong>{latest_raw}%</strong> | Smoothed Avg: <strong>{latest_avg}%</strong></p>
      <p>Camera Active: <strong>{not cam.stopped}</strong> | Total Frames: <strong>{cam.get_frame_count()}</strong></p>
    </div>
  </body>
  </html>
  """
  return HTMLResponse(content=html)