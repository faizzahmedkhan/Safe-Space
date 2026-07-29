import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
import torchvision.transforms as T
import numpy as np
import os
import time
import json
import re
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import tempfile
import shutil
from collections import deque
from camera import CameraStream, CLIP_LENGTH, diagnose_http_camera
from bridge_ctl import ensure_lan_bridge
from evidence_video import ensure_browser_compatible
from video_analyze import analyze_video_file

# --- HARDWARE SETTINGS ---
# Do NOT auto-open the laptop webcam — connect from the dashboard so the UI
# matches the real source (phone IP Webcam vs laptop).
CAMERA_SOURCE = None

# --- THRESHOLDS & TIMINGS ---
INFERENCE_STRIDE = 8
ALERT_THRESHOLD = 0.75
ALERT_COOLDOWN_SECONDS = 5
SCORE_HISTORY_LEN = 6
MAX_ALERTS = 500
MAX_LOGS = 200
FRAME_STALE_MS = 2000

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

ALERTS_FILE = "alerts.json"
DEVICE_CONFIG_FILE = "device_config.json"
EVIDENCE_DIR = "evidence"
UPLOADS_DIR = "uploads"
DATA_SAFE = os.path.join("data", "safe")
DATA_UNSAFE = os.path.join("data", "unsafe")

alert_log: list[dict] = []
system_logs: deque = deque(maxlen=MAX_LOGS)
device_config: dict = {
    "ip": "",
    "port": 8080,
    "path": "/video",
    "wifi_name": "",
    "stream_url": None,
}


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_log(message: str, level: str = "info", kind: str = "system") -> None:
    entry = {
        "timestamp": _now_iso(),
        "level": level,
        "message": message,
        "kind": kind,
    }
    system_logs.append(entry)
    print(f"[{kind.upper()}] {message}")


def _save_alerts_atomic(alerts: list, path: str) -> None:
    dir_name = os.path.dirname(path) or "."
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp") as tmp:
            json.dump(alerts, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, path)
    except Exception as e:
        add_log(f"Could not persist alerts: {e}", level="error", kind="system")


def _save_device_config(config: dict) -> None:
    try:
        with tempfile.NamedTemporaryFile("w", dir=".", delete=False, suffix=".tmp") as tmp:
            json.dump(config, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, DEVICE_CONFIG_FILE)
    except Exception as e:
        add_log(f"Could not persist device config: {e}", level="error", kind="system")


def _load_device_config() -> dict:
    if not os.path.exists(DEVICE_CONFIG_FILE):
        return dict(device_config)
    try:
        with open(DEVICE_CONFIG_FILE, "r") as f:
            loaded = json.load(f)
        merged = dict(device_config)
        merged.update(loaded)
        return merged
    except Exception as e:
        add_log(f"Could not load device config: {e}", level="error", kind="system")
        return dict(device_config)


if os.path.exists(ALERTS_FILE):
    try:
        with open(ALERTS_FILE, "r") as f:
            alert_log = json.load(f)
    except Exception as e:
        print(f"[SYSTEM] Could not load past alerts: {e}")

device_config = _load_device_config()
score_history: deque = deque(maxlen=SCORE_HISTORY_LEN)

# ==========================================
# 1. LOAD THE BRAIN (System Startup)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Booting AI Core on: {device}")

model = video_models.r3d_18()
model.fc = nn.Linear(model.fc.in_features, 2)

model_path = "safespace_bullying_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please run 'git lfs pull' to download the weights."
    )
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

normalize = T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD)

os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_SAFE, exist_ok=True)
os.makedirs(DATA_UNSAFE, exist_ok=True)

# Soft-start: only auto-reconnect a previously saved phone URL
_initial_src = device_config.get("stream_url") or CAMERA_SOURCE
cam = CameraStream(src=_initial_src, target_fps=15, show_preview=False, soft_fail=True)
if _initial_src is None:
    add_log(
        "Camera idle at startup. Connect a phone or laptop webcam from the dashboard.",
        kind="camera",
    )
elif cam.stream is not None:
    cam.start()
    add_log(f"Camera started with src={_initial_src}", kind="camera")
else:
    add_log(
        f"Camera not connected at startup ({cam.open_error}). Use the dashboard to connect.",
        level="warning",
        kind="camera",
    )


def build_stream_url(ip: str, port: int = 8080, path: str = "/video") -> str:
    ip = ip.strip()
    path = path.strip() or "/video"
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{ip}:{port}{path}"


# ==========================================
# 2. THE CONSUMER LOOP (Background Worker)
# ==========================================
async def ai_watcher():
    print("[AI WATCHER] Online. Waiting for the camera buffer to fill...")

    last_processed_frame_count = 0
    alert_active = False
    INFERENCE_SLEEP = INFERENCE_STRIDE / cam.target_fps

    while True:
        inference_frames, evidence_frames, current_frame_count = cam.get_snapshot()

        # Camera restart resets frame_count to 0 — reset stride tracker too,
        # otherwise inference never runs again and the dashboard shows "—".
        if current_frame_count < last_processed_frame_count:
            last_processed_frame_count = 0
            score_history.clear()
            if alert_active:
                cam.stop_recording()
                alert_active = False
            print("[AI WATCHER] Camera frame counter reset — restarting inference.")

        new_frames_since_last = current_frame_count - last_processed_frame_count
        buffer_ready = len(inference_frames) == CLIP_LENGTH
        stride_reached = new_frames_since_last >= INFERENCE_STRIDE

        if not buffer_ready:
            if score_history:
                print("[AI WATCHER] Camera not ready — clearing stale score history.")
                score_history.clear()
                if alert_active:
                    cam.stop_recording()
                    add_log("Evidence recording stopped (camera not ready).", kind="recording")
                    alert_active = False
            await asyncio.sleep(INFERENCE_SLEEP)
            continue

        if stride_reached:
            last_processed_frame_count = current_frame_count

            frames_array = np.stack(inference_frames)
            tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()
            tensor.div_(255.0)
            tensor = normalize(tensor)
            tensor = tensor.permute(1, 0, 2, 3)
            inputs = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                current_prob = probabilities[0][1].item()

            score_history.append(current_prob)
            if len(score_history) < score_history.maxlen:
                print(
                    f"[WARMUP] {len(score_history)}/{score_history.maxlen} "
                    f"predictions collected. Raw: {current_prob*100:.1f}%"
                )
            else:
                avg_prob = sum(score_history) / len(score_history)
                scene = cam.get_scene_meta()
                suitable = scene.get("suitable_for_model", True)
                print(
                    f"Stats -> [Raw Spike: {current_prob*100:.1f}%] | "
                    f"[SMOOTHED AVG: {avg_prob*100:.1f}%] | "
                    f"[scene: {scene.get('method')} "
                    f"subject={scene.get('subject_area_frac')} "
                    f"suitable={suitable}]"
                )

                if avg_prob >= ALERT_THRESHOLD and not suitable:
                    # High score on overview CCTV is usually unreliable at 112x112
                    if alert_active:
                        cam.stop_recording()
                        add_log(
                            "Recording stopped — scene unsuitable for model framing.",
                            kind="recording",
                        )
                        alert_active = False
                    print(
                        "⚠️ Score high but subjects too small/far for reliable "
                        "112x112 analysis — alert suppressed."
                    )
                elif avg_prob >= ALERT_THRESHOLD:
                    if not alert_active:
                        alert_active = True
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"{EVIDENCE_DIR}/incident_{timestamp}.mp4"

                        cam.start_recording(filename)
                        add_log(
                            f"ALERT: Physical altercation detected ({avg_prob*100:.1f}%). "
                            f"Recording {filename}",
                            level="alert",
                            kind="alert",
                        )

                        global alert_log
                        alert_log.append(
                            {
                                "timestamp": timestamp,
                                "confidence": round(avg_prob * 100, 1),
                                "evidence_file": filename,
                                "scene": scene,
                            }
                        )

                        if len(alert_log) > MAX_ALERTS:
                            alert_log = alert_log[-MAX_ALERTS:]

                        _save_alerts_atomic(alert_log, ALERTS_FILE)
                else:
                    if alert_active:
                        cam.stop_recording()
                        add_log("Evidence recording stopped.", kind="recording")
                        alert_active = False
                    print("✅ Safe.")

        await asyncio.sleep(INFERENCE_SLEEP)


# ==========================================
# 3. FASTAPI ROUTING
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(ai_watcher())
    yield
    print("[SYSTEM] Shutting down SafeSpace...")
    cam.stop()


app = FastAPI(title="SafeSpace AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CameraConnectRequest(BaseModel):
    ip: str = Field(..., min_length=1, description="Phone IP address on local Wi-Fi")
    port: int = Field(8080, ge=1, le=65535)
    path: str = Field("/video", description="Stream path, e.g. /video for IP Webcam")
    wifi_name: str = Field("", description="Operator label for the Wi-Fi network")


def _evidence_basename(path: str) -> str:
    return os.path.basename(path)


def _alert_with_url(alert: dict) -> dict:
    out = dict(alert)
    evidence = alert.get("evidence_file") or ""
    name = _evidence_basename(evidence)
    out["evidence_url"] = f"/evidence/{name}" if name else None
    return out


def _camera_receiving_frames() -> bool:
    if not cam.is_active():
        return False
    age = cam.get_last_frame_age_ms()
    if age is None:
        return False
    return age <= FRAME_STALE_MS


@app.get("/")
def read_root():
    return {
        "status": "SafeSpace Core is Active",
        "camera_active": cam.is_active(),
        "camera_receiving": _camera_receiving_frames(),
    }


@app.get("/status")
def get_status():
    scores = list(score_history)
    latest_raw = round(scores[-1] * 100, 1) if scores else None
    # Show running average even during warmup so the dashboard isn't blank
    latest_avg = (
        round((sum(scores) / len(scores)) * 100, 1) if scores else None
    )
    age_ms = cam.get_last_frame_age_ms()
    camera_active = cam.is_active()
    receiving = _camera_receiving_frames()

    return {
        "backend_online": True,
        "camera_active": camera_active,
        "camera_receiving": receiving,
        "camera_kind": (
            "phone"
            if device_config.get("stream_url")
            else ("webcam" if camera_active or receiving else "none")
        ),
        "last_frame_age_ms": age_ms,
        "stream_url": device_config.get("stream_url"),
        "capture_url": device_config.get("capture_url"),
        "via_bridge": bool(device_config.get("via_bridge")),
        "wifi_name": device_config.get("wifi_name") or "",
        "ip": device_config.get("ip") or "",
        "port": device_config.get("port", 8080),
        "path": device_config.get("path", "/video"),
        "buffer_fill": f"{len(cam.get_buffer())}/{CLIP_LENGTH}",
        "total_frames": cam.get_frame_count(),
        "latest_raw_pct": latest_raw,
        "smoothed_avg_pct": latest_avg,
        "score_warmup": f"{len(scores)}/{SCORE_HISTORY_LEN}",
        "alert_threshold": ALERT_THRESHOLD * 100,
        "open_error": cam.open_error,
        "scene": cam.get_scene_meta(),
    }


@app.post("/camera/connect")
def camera_connect(body: CameraConnectRequest):
    global device_config

    ip = body.ip.strip()
    if not re.match(r"^[\w.\-]+$", ip):
        raise HTTPException(status_code=400, detail="Invalid IP address or hostname.")

    stream_url = build_stream_url(ip, body.port, body.path)
    add_log(
        f"Connecting to camera at {stream_url}"
        + (f" (Wi-Fi: {body.wifi_name})" if body.wifi_name.strip() else ""),
        kind="camera",
    )

    # Prefer direct phone URL; if Cursor lacks Local Network access, use LAN bridge.
    capture_url = stream_url
    used_bridge = False
    diag = diagnose_http_camera(stream_url)
    if not diag["ok"]:
        add_log(
            "Direct phone HTTP failed (common inside Cursor). Starting LAN bridge via Terminal.app...",
            level="warning",
            kind="camera",
        )
        try:
            capture_url, note = ensure_lan_bridge(stream_url)
            used_bridge = True
            add_log(note, kind="camera")
        except Exception as e:
            add_log(f"LAN bridge failed: {e}", level="error", kind="camera")
            device_config = {
                "ip": ip,
                "port": body.port,
                "path": body.path if body.path.startswith("/") else f"/{body.path}",
                "wifi_name": body.wifi_name.strip(),
                "stream_url": None,
            }
            _save_device_config(device_config)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{e} "
                    "Alternatively: System Settings → Privacy & Security → Local Network → enable Cursor, "
                    "or run start_backend.command in Terminal.app."
                ),
            )

    success, error = cam.restart(capture_url)
    score_history.clear()

    device_config = {
        "ip": ip,
        "port": body.port,
        "path": body.path if body.path.startswith("/") else f"/{body.path}",
        "wifi_name": body.wifi_name.strip(),
        "stream_url": stream_url if success else None,
        "capture_url": capture_url if success else None,
        "via_bridge": used_bridge,
        "source": "phone",
    }
    _save_device_config(device_config)

    if not success:
        add_log(f"Camera connect failed: {error}", level="error", kind="camera")
        raise HTTPException(
            status_code=400,
            detail=error
            or "Could not open camera stream. Check IP Webcam is running and both devices are on the same Wi-Fi.",
        )

    # Give the producer a moment to grab first frames
    time.sleep(0.5)
    add_log(
        "Camera connected successfully"
        + (" via LAN bridge." if used_bridge else "."),
        kind="camera",
    )
    return {
        "ok": True,
        "stream_url": stream_url,
        "capture_url": capture_url,
        "via_bridge": used_bridge,
        "wifi_name": device_config["wifi_name"],
        "camera_active": cam.is_active(),
        "camera_receiving": _camera_receiving_frames(),
    }


@app.post("/camera/connect-webcam")
def camera_connect_webcam():
    """Connect the laptop's built-in webcam (for testing without a phone)."""
    global device_config
    add_log("Connecting laptop webcam (src=0)...", kind="camera")
    success, error = cam.restart(0)
    score_history.clear()
    device_config = {
        "ip": "",
        "port": 8080,
        "path": "/video",
        "wifi_name": "Laptop webcam",
        "stream_url": None,
        "source": "webcam",
    }
    _save_device_config(device_config)
    if not success:
        add_log(f"Webcam connect failed: {error}", level="error", kind="camera")
        raise HTTPException(status_code=400, detail=error or "Could not open laptop webcam.")
    time.sleep(0.3)
    add_log("Laptop webcam connected.", kind="camera")
    return {
        "ok": True,
        "camera_kind": "webcam",
        "camera_active": cam.is_active(),
        "camera_receiving": _camera_receiving_frames(),
    }


@app.post("/camera/disconnect")
def camera_disconnect():
    global device_config
    cam.stop()
    score_history.clear()
    device_config = {
        **device_config,
        "stream_url": None,
    }
    _save_device_config(device_config)
    add_log("Camera disconnected by operator.", kind="camera")
    return {"ok": True, "camera_active": False}


class LabelEvidenceRequest(BaseModel):
    label: str = Field(..., description="'safe' (false alarm) or 'unsafe' (true incident)")


@app.post("/analyze/video")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    """
    Upload a recorded clip and run it through the same model as live mode.
    Use this to check whether an alert was wrong (false positive) or missed.
    """
    name = os.path.basename(file.filename or "clip.mp4")
    if not re.search(r"\.(mp4|avi|mov|mkv|webm)$", name, re.I):
        raise HTTPException(status_code=400, detail="Upload an mp4/avi/mov/mkv/webm video.")

    safe = re.sub(r"[^\w.\-]", "_", name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(UPLOADS_DIR, f"review_{stamp}_{safe}")

    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save upload: {e}") from e

    add_log(f"Analyzing uploaded video {dest}", kind="review")
    try:
        result = analyze_video_file(
            dest,
            model=model,
            device=device,
            alert_threshold=ALERT_THRESHOLD,
        )
    except Exception as e:
        add_log(f"Video analysis failed: {e}", level="error", kind="review")
        raise HTTPException(status_code=400, detail=str(e)) from e

    result["saved_as"] = dest
    add_log(
        f"Review result: {result['verdict']} "
        f"(peak smoothed {result['peak_smoothed_pct']}%) for {name}",
        kind="review",
    )
    return result


@app.post("/analyze/evidence/{filename}")
def analyze_evidence_file(filename: str):
    """Re-run the model on a saved evidence clip from alerts."""
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid evidence filename.")
    if not re.match(r"^[\w.\-]+\.mp4$", safe_name):
        raise HTTPException(status_code=400, detail="Invalid evidence filename.")

    path = os.path.join(EVIDENCE_DIR, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Evidence file not found.")

    add_log(f"Re-analyzing evidence {safe_name}", kind="review")
    try:
        result = analyze_video_file(
            path,
            model=model,
            device=device,
            alert_threshold=ALERT_THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    result["evidence_file"] = f"{EVIDENCE_DIR}/{safe_name}"
    return result


@app.post("/review/label-evidence/{filename}")
def label_evidence(filename: str, body: LabelEvidenceRequest):
    """
    Copy an evidence clip into data/safe (false alarm) or data/unsafe (true fight)
    so you can fine-tune later with train.py.
    """
    label = body.label.strip().lower()
    if label not in ("safe", "unsafe"):
        raise HTTPException(status_code=400, detail="label must be 'safe' or 'unsafe'.")

    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid evidence filename.")
    if not re.match(r"^[\w.\-]+\.mp4$", safe_name):
        raise HTTPException(status_code=400, detail="Invalid evidence filename.")

    src = os.path.join(EVIDENCE_DIR, safe_name)
    if not os.path.isfile(src):
        raise HTTPException(status_code=404, detail="Evidence file not found.")

    dest_dir = DATA_SAFE if label == "safe" else DATA_UNSAFE
    dest = os.path.join(dest_dir, safe_name)
    shutil.copy2(src, dest)
    add_log(
        f"Labeled {safe_name} as {label} → {dest}",
        kind="review",
    )
    return {
        "ok": True,
        "label": label,
        "copied_to": dest,
        "hint": "When you have both safe and unsafe examples, run: python train.py",
    }


@app.get("/alerts")
def get_alerts():
    recent = [_alert_with_url(a) for a in alert_log[-20:]]
    return {
        "total_alerts": len(alert_log),
        "recent": list(reversed(recent)),
    }


@app.get("/logs")
def get_logs():
    return {
        "total": len(system_logs),
        "recent": list(reversed(list(system_logs)))[:100],
    }


@app.api_route("/evidence/{filename}", methods=["GET", "HEAD"])
def get_evidence(filename: str):
    # Path-safe: only basename under evidence/
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid evidence filename.")
    if not re.match(r"^[\w.\-]+\.mp4$", safe_name):
        raise HTTPException(status_code=400, detail="Invalid evidence filename.")

    path = os.path.join(EVIDENCE_DIR, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Evidence file not found.")

    try:
        playable = ensure_browser_compatible(path)
    except Exception as e:
        add_log(f"Evidence transcode error for {safe_name}: {e}", level="warning", kind="system")
        playable = path

    return FileResponse(
        playable,
        media_type="video/mp4",
        filename=safe_name,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


def _generate_mjpeg():
    while True:
        jpeg_bytes = cam.get_latest_frame_jpeg()
        if jpeg_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(1.0 / 10)


@app.get("/feed")
def video_feed():
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/preview")
def preview_page():
    scores = list(score_history)
    latest_raw = round(scores[-1] * 100, 1) if scores else 0
    latest_avg = (
        round((sum(scores) / len(scores)) * 100, 1)
        if len(scores) == SCORE_HISTORY_LEN
        else 0
    )

    html = f"""
  <!DOCTYPE html>
  <html>
  <head>
    <title>SafeSpace Live Preview</title>
    <style>
      body {{ background: #0f172a; color: #e2e8f0; font-family: sans-serif; text-align: center; padding: 20px; }}
      h1 {{ color: #2dd4bf; }}
      img {{ border: 2px solid #334155; border-radius: 12px; max-width: 90vw; }}
      .stats {{ margin-top: 16px; font-size: 18px; color: #94a3b8; }}
    </style>
  </head>
  <body>
    <h1>SafeSpace — Live Camera Preview</h1>
    <p>Prefer the Next.js dashboard at <code>http://localhost:3000</code>.</p>
    <img src="/feed" alt="Live Camera Feed" />
    <div class="stats">
      <p>Latest Raw Score: <strong>{latest_raw}%</strong> | Smoothed Avg: <strong>{latest_avg}%</strong></p>
      <p>Camera Active: <strong>{cam.is_active()}</strong> | Total Frames: <strong>{cam.get_frame_count()}</strong></p>
    </div>
  </body>
  </html>
  """
    return HTMLResponse(content=html)
