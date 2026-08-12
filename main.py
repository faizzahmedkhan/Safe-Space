import asyncio
import threading
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
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import tempfile
import shutil
from collections import deque
from camera import CameraStream, CLIP_LENGTH, diagnose_http_camera
from bridge_ctl import ensure_lan_bridge
from video_analyze import analyze_video_file
from video_calibration import process_video, cut_clip, get_ffmpeg_binary
import yt_dlp

def ensure_web_mp4(src_path: str) -> str:
    """Remuxes video file to standard ISO MP4 if it is in mpegts or non-web container format."""
    if not src_path or not os.path.exists(src_path):
        return src_path
    ffmpeg_exe = get_ffmpeg_binary()
    try:
        res = subprocess.run([ffmpeg_exe, "-i", src_path], capture_output=True, text=True)
        if "mpegts" in res.stderr.lower():
            out_path = os.path.splitext(src_path)[0] + "_web.mp4"
            subprocess.run([ffmpeg_exe, "-y", "-i", src_path, "-c", "copy", out_path], capture_output=True, check=True)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                os.replace(out_path, src_path)
    except Exception:
        pass
    return src_path

# --- THRESHOLDS & TIMINGS ---
INFERENCE_STRIDE = 8
ALERT_THRESHOLD = 0.75
# A single raw clip score at or above this fires an alert immediately,
# bypassing the rolling average. A brief, sharp act (a slap, a sudden shove)
# only shows up strongly in one or two ~1s clips — averaging it in with
# SCORE_HISTORY_LEN older, calmer clips dilutes it below ALERT_THRESHOLD
# and it never fires. This spike path exists so sudden violence isn't
# lost to smoothing, while ALERT_THRESHOLD + the rolling average still
# catches sustained-but-lower-confidence altercations.
SPIKE_THRESHOLD = 0.90
ALERT_COOLDOWN_SECONDS = 5
# Shrunk from 10 -> 4. At INFERENCE_STRIDE=8 / CAMERA_TARGET_FPS=15, 10
# samples averaged ~6 seconds of real time — a single violent clip could
# only move that average by 1/10th, so sustained violence for 5-6s was
# effectively required to cross ALERT_THRESHOLD. 4 samples span ~1.6-2s,
# which still smooths out single-frame noise/false detections without
# burying a real short incident.
SCORE_HISTORY_LEN = 4
MAX_ALERTS = 500
MAX_LOGS = 200
FRAME_STALE_MS = 2000
CAMERA_TARGET_FPS = 15
# Total evidence-buffer RAM budget shared across ALL cameras, not per camera.
# Each camera's individual cap shrinks as more cameras join, so total memory
# stays roughly constant (e.g. ~300MB total) instead of growing linearly
# with camera count (which is what a flat per-camera cap would do).
TOTAL_EVIDENCE_BUDGET_MB = 300

# NOTE: the training notebook (SchoolBullyingDataset.__getitem__) only ever
# does `vframes / 255.0` — it never applies Kinetics mean/std normalization.
# The model's weights were fit to that [0,1]-scaled distribution, so we must
# match it exactly here. Do NOT reintroduce T.Normalize(KINETICS_MEAN, ...)
# unless you also add it to the training pipeline and retrain — otherwise
# every live prediction is silently out-of-distribution.

ALERTS_FILE = "alerts.json"
CAMERAS_CONFIG_FILE = "cameras_config.json"
EVIDENCE_DIR = "evidence"
UPLOADS_DIR = "uploads"
DATA_SAFE = os.path.join("data", "safe")
DATA_UNSAFE = os.path.join("data", "unsafe")

alert_log: list[dict] = []
system_logs: deque = deque(maxlen=MAX_LOGS)


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


# ==========================================
# 0. CAMERA MANAGER
# ==========================================
# Each entry in `cameras` is one independently-running CameraStream plus the
# per-camera bookkeeping that used to be a handful of module-level globals
# (device_config, score_history, alert_active, ...). Every camera captures on
# its own background thread exactly as before — nothing about CameraStream
# itself changed. The only new piece is that we now keep a dict of them
# instead of a single `cam` variable, and ai_watcher() below sweeps all of
# them once per cycle.
_cameras_lock = threading.Lock()
cameras: dict[str, dict] = {}


def _default_camera_config() -> dict:
    return {
        "ip": "",
        "port": 8080,
        "path": "/video",
        "wifi_name": "",
        "stream_url": None,
        "capture_url": None,
        "via_bridge": False,
        "source": "none",
        "label": "",
    }


def _new_camera_slot(camera_id: str, label: str = "") -> dict:
    return {
        "camera_id": camera_id,
        "stream": CameraStream(src=None, target_fps=CAMERA_TARGET_FPS, show_preview=False, soft_fail=True),
        "config": {**_default_camera_config(), "label": label or camera_id},
        "score_history": deque(maxlen=SCORE_HISTORY_LEN),
        "alert_active": False,
        "last_processed_frame_count": 0,
        "warmup_remaining": 12,
        "consecutive_above": 0,
    }


def get_camera_or_404(camera_id: str) -> dict:
    slot = cameras.get(camera_id)
    if slot is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera_id '{camera_id}'.")
    return slot


def _save_cameras_config() -> None:
    payload = {cid: slot["config"] for cid, slot in cameras.items()}
    try:
        with tempfile.NamedTemporaryFile("w", dir=".", delete=False, suffix=".tmp") as tmp:
            json.dump(payload, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, CAMERAS_CONFIG_FILE)
    except Exception as e:
        add_log(f"Could not persist camera config: {e}", level="error", kind="system")


def _load_cameras_config() -> dict:
    if not os.path.exists(CAMERAS_CONFIG_FILE):
        return {}
    try:
        with open(CAMERAS_CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        add_log(f"Could not load camera config: {e}", level="error", kind="system")
        return {}


def _rebalance_evidence_budgets() -> None:
    """Re-split TOTAL_EVIDENCE_BUDGET_MB evenly across whatever cameras exist
    right now. Call this any time a camera is added or removed."""
    n = max(1, len(cameras))
    per_camera_mb = TOTAL_EVIDENCE_BUDGET_MB / n
    for slot in cameras.values():
        slot["stream"].set_evidence_budget_mb(per_camera_mb)


def build_stream_url(ip: str, port: int = 8080, path: str = "/video") -> str:
    ip = ip.strip()
    path = path.strip() or "/video"
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{ip}:{port}{path}"


if os.path.exists(ALERTS_FILE):
    try:
        with open(ALERTS_FILE, "r") as f:
            alert_log = json.load(f)
    except Exception as e:
        print(f"[SYSTEM] Could not load past alerts: {e}")

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

os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_SAFE, exist_ok=True)
os.makedirs(DATA_UNSAFE, exist_ok=True)

# Soft-start: re-create any cameras that were connected last time we ran,
# and try to auto-reconnect ones that had a saved phone stream_url.
_persisted_cameras = _load_cameras_config()
if not _persisted_cameras:
    add_log("No cameras configured yet. Add one from the dashboard.", kind="camera")
else:
    for _camera_id, _cfg in _persisted_cameras.items():
        slot = _new_camera_slot(_camera_id, label=_cfg.get("label", _camera_id))
        slot["config"] = {**_default_camera_config(), **_cfg}
        cameras[_camera_id] = slot

        _src = slot["config"].get("capture_url") or slot["config"].get("stream_url")
        if slot["config"].get("source") == "webcam":
            _src = 0
        if _src is None:
            add_log(f"[{_camera_id}] Idle at startup. Connect from the dashboard.", kind="camera")
            continue

        ok, err = slot["stream"].restart(_src)
        if ok:
            add_log(f"[{_camera_id}] Camera started with src={_src}", kind="camera")
        else:
            add_log(f"[{_camera_id}] Not connected at startup ({err}). Use the dashboard.", level="warning", kind="camera")

_rebalance_evidence_budgets()


# ==========================================
# 2. THE CONSUMER LOOP (Background Worker)
# ==========================================
def _fire_alert(camera_id: str, slot: dict, confidence_pct: float, scene: dict, reason: str) -> None:
    """Start recording + log an alert. Shared by the spike path and the
    sustained/averaged path so both record identically."""
    stream = slot["stream"]
    slot["alert_active"] = True
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{EVIDENCE_DIR}/incident_{camera_id}_{timestamp}.mp4"

    stream.start_recording(filename)
    add_log(
        f"[{camera_id}] ALERT ({reason}): Physical altercation detected ({confidence_pct:.1f}%). Recording {filename}",
        level="alert",
        kind="alert",
    )

    global alert_log
    alert_log.append(
        {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "confidence": round(confidence_pct, 1),
            "trigger": reason,
            "evidence_file": filename,
            "scene": scene,
        }
    )
    if len(alert_log) > MAX_ALERTS:
        alert_log = alert_log[-MAX_ALERTS:]
    _save_alerts_atomic(alert_log, ALERTS_FILE)


def _handle_camera_score(camera_id: str, slot: dict, current_prob: float, scene: dict) -> None:
    """Alert/recording state machine, scoped to one camera_id.

    Two independent triggers feed the same recording path:
      1. SPIKE path: a single raw clip score >= SPIKE_THRESHOLD fires
         immediately. Sudden, short violence (a slap, a shove) only lights
         up one or two ~1s clips, so it must not be forced to wait on the
         rolling average below.
      2. SUSTAINED path: the SCORE_HISTORY_LEN-sample rolling average
         crossing ALERT_THRESHOLD for 2 consecutive cycles, same as before
         — this still catches lower-confidence-but-sustained altercations
         and filters one-off noisy frames.
    """
    stream = slot["stream"]
    score_history = slot["score_history"]
    suitable = scene.get("suitable_for_model", True)

    if len(score_history) < score_history.maxlen:
        print(f"[WARMUP][{camera_id}] {len(score_history)}/{score_history.maxlen} predictions. Raw: {current_prob*100:.1f}%")
        return

    avg_prob = sum(score_history) / len(score_history)
    print(
        f"[{camera_id}] Stats -> [Raw Spike: {current_prob*100:.1f}%] | "
        f"[SMOOTHED AVG: {avg_prob*100:.1f}%] | "
        f"[scene: {scene.get('method')} subject={scene.get('subject_area_frac')} suitable={suitable}]"
    )

    if not suitable:
        if slot["alert_active"]:
            stream.stop_recording()
            add_log(f"[{camera_id}] Recording stopped — scene unsuitable for model framing.", kind="recording")
            slot["alert_active"] = False
        slot["consecutive_above"] = 0
        if avg_prob >= ALERT_THRESHOLD or current_prob >= SPIKE_THRESHOLD:
            print(f"⚠️ [{camera_id}] Score high but subjects too small/far — alert suppressed.")
        return

    # --- 1. Spike path: don't let smoothing eat a short, sharp incident ---
    if current_prob >= SPIKE_THRESHOLD and not slot["alert_active"]:
        _fire_alert(camera_id, slot, current_prob * 100, scene, reason="spike")
        slot["consecutive_above"] = 0
        return

    # --- 2. Sustained path: rolling average over consecutive cycles ---
    if avg_prob >= ALERT_THRESHOLD:
        slot["consecutive_above"] = slot.get("consecutive_above", 0) + 1
        # Require 2 consecutive above-threshold cycles to trigger an alert.
        # This filters transient false-positive spikes (e.g. sitting with
        # minor movements that the motion crop amplifies).
        if slot["consecutive_above"] >= 2 and not slot["alert_active"]:
            _fire_alert(camera_id, slot, avg_prob * 100, scene, reason="sustained")
        return

    # Score dropped below threshold on both paths
    slot["consecutive_above"] = 0
    if slot["alert_active"] and current_prob < SPIKE_THRESHOLD:
        stream.stop_recording()
        add_log(f"[{camera_id}] Evidence recording stopped.", kind="recording")
        slot["alert_active"] = False
    print(f"✅ [{camera_id}] Safe.")


def _run_inference_batch(batch: torch.Tensor) -> torch.Tensor:
    """Blocking model call, meant to be run via asyncio.to_thread so it
    doesn't stall the event loop that also serves your async endpoints
    (e.g. /analyze/video). As camera count grows, batches get bigger and
    this call takes longer — without to_thread, every concurrent request
    would queue behind it on the same thread."""
    with torch.no_grad():
        outputs = model(batch)
        return F.softmax(outputs, dim=1)


async def ai_watcher():
    """
    One consumer loop for every camera. Each cycle:
      1. Snapshot every camera's buffer.
      2. Collect the ones that are full AND due for inference (stride reached)
         into a single batch — however many that is (0..N).
      3. Run ONE model() forward pass on that batch (still works fine on CPU;
         batching doesn't require a GPU, it just amortizes per-call overhead
         and keeps the code simple as camera count grows).
      4. Fan results back out to each camera's own score history / alert logic.
    Cameras that aren't ready this cycle are simply skipped — nobody blocks
    anybody else, there is no round-robin turn-taking.
    """
    print("[AI WATCHER] Online. Waiting for camera buffers to fill...")
    inference_sleep = INFERENCE_STRIDE / CAMERA_TARGET_FPS

    while True:
        ready_ids: list[str] = []
        ready_tensors: list[torch.Tensor] = []
        ready_scene: list[dict] = []

        for camera_id, slot in list(cameras.items()):
            stream = slot["stream"]
            inference_frames, _evidence_frames, current_frame_count = stream.get_snapshot()

            # Camera restart resets frame_count to 0 — reset stride tracker too.
            if current_frame_count < slot["last_processed_frame_count"]:
                slot["last_processed_frame_count"] = 0
                slot["score_history"].clear()
                if slot["alert_active"]:
                    stream.stop_recording()
                    slot["alert_active"] = False
                slot["warmup_remaining"] = 12
                print(f"[AI WATCHER][{camera_id}] Frame counter reset — restarting inference.")

            new_frames_since_last = current_frame_count - slot["last_processed_frame_count"]
            buffer_ready = len(inference_frames) == CLIP_LENGTH
            stride_reached = new_frames_since_last >= INFERENCE_STRIDE

            if not buffer_ready:
                if slot["score_history"]:
                    slot["score_history"].clear()
                    if slot["alert_active"]:
                        stream.stop_recording()
                        add_log(f"[{camera_id}] Evidence recording stopped (camera not ready).", kind="recording")
                        slot["alert_active"] = False
                continue

            if not stride_reached:
                continue

            slot["last_processed_frame_count"] = current_frame_count

            frames_array = np.stack(inference_frames)
            tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()
            tensor.div_(255.0)  # matches training: SchoolBullyingDataset only does /255.0
            tensor = tensor.permute(1, 0, 2, 3)  # (C, T, H, W)

            ready_ids.append(camera_id)
            ready_tensors.append(tensor)
            ready_scene.append(stream.get_scene_meta())

        if ready_tensors:
            batch = torch.stack(ready_tensors).to(device)  # (B, C, T, H, W)
            probabilities = await asyncio.to_thread(_run_inference_batch, batch)

            for i, camera_id in enumerate(ready_ids):
                slot = cameras.get(camera_id)
                if slot is None:
                    continue  # camera was removed mid-cycle
                current_prob = probabilities[i][1].item()
                slot["score_history"].append(current_prob)
                if slot.get("warmup_remaining", 0) > 0:
                    slot["warmup_remaining"] -= 1
                    print(f"[WARMUP][{camera_id}] Remaining warmup steps: {slot['warmup_remaining']}. Raw score: {current_prob*100:.1f}% (Suppressed)")
                    continue
                _handle_camera_score(camera_id, slot, current_prob, ready_scene[i])

        await asyncio.sleep(inference_sleep)


# ==========================================
# 3. FASTAPI ROUTING
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(ai_watcher())
    yield
    print("[SYSTEM] Shutting down SafeSpace...")
    for slot in cameras.values():
        slot["stream"].stop()


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


class CameraCreateRequest(BaseModel):
    camera_id: str = Field(..., min_length=1, description="Unique id, e.g. 'entrance', 'hallway-2'")
    label: str = Field("", description="Human-friendly display name")


class CameraConnectRequest(BaseModel):
    ip: str = Field(..., min_length=1, description="Phone IP address on local Wi-Fi")
    port: int = Field(8080, ge=1, le=65535)
    path: str = Field("/video", description="Stream path, e.g. /video for IP Webcam")
    wifi_name: str = Field("", description="Operator label for the Wi-Fi network")


class WebcamConnectRequest(BaseModel):
    device_index: int = Field(0, ge=0, description="OpenCV device index, e.g. 0 or 1 for a second USB webcam")


def _evidence_basename(path: str) -> str:
    return os.path.basename(path)


def _alert_with_url(alert: dict) -> dict:
    out = dict(alert)
    evidence = alert.get("evidence_file") or ""
    name = _evidence_basename(evidence)
    out["evidence_url"] = f"/evidence/{name}" if name else None
    return out


def _camera_receiving_frames(slot: dict) -> bool:
    stream = slot["stream"]
    if not stream.is_active():
        return False
    age = stream.get_last_frame_age_ms()
    if age is None:
        return False
    return age <= FRAME_STALE_MS


def _camera_status_payload(camera_id: str, slot: dict) -> dict:
    stream = slot["stream"]
    config = slot["config"]
    scores = list(slot["score_history"])
    latest_raw = round(scores[-1] * 100, 1) if scores else None
    latest_avg = round((sum(scores) / len(scores)) * 100, 1) if scores else None
    camera_active = stream.is_active()
    receiving = _camera_receiving_frames(slot)

    return {
        "camera_id": camera_id,
        "label": config.get("label") or camera_id,
        "backend_online": True,
        "camera_active": camera_active,
        "camera_receiving": receiving,
        "camera_kind": (
            "phone" if config.get("stream_url") else ("webcam" if camera_active or receiving else "none")
        ),
        "last_frame_age_ms": stream.get_last_frame_age_ms(),
        "target_fps": stream.target_fps,
        "actual_fps": stream.get_actual_fps(),
        "stream_url": config.get("stream_url"),
        "capture_url": config.get("capture_url"),
        "via_bridge": bool(config.get("via_bridge")),
        "wifi_name": config.get("wifi_name") or "",
        "ip": config.get("ip") or "",
        "port": config.get("port", 8080),
        "path": config.get("path", "/video"),
        "buffer_fill": f"{len(stream.get_buffer())}/{CLIP_LENGTH}",
        "total_frames": stream.get_frame_count(),
        "latest_raw_pct": latest_raw,
        "smoothed_avg_pct": latest_avg,
        "score_warmup": f"{len(scores)}/{SCORE_HISTORY_LEN}",
        "alert_threshold": ALERT_THRESHOLD * 100,
        "open_error": stream.open_error,
        "scene": stream.get_scene_meta(),
    }


@app.get("/")
def read_root():
    return {
        "status": "SafeSpace Core is Active",
        "camera_count": len(cameras),
        "cameras_active": sum(1 for s in cameras.values() if s["stream"].is_active()),
    }


@app.get("/cameras")
def list_cameras():
    """One entry per registered camera — this is what a multi-camera dashboard polls."""
    return {
        "total": len(cameras),
        "cameras": [_camera_status_payload(cid, slot) for cid, slot in cameras.items()],
    }


@app.post("/cameras")
def create_camera(body: CameraCreateRequest):
    camera_id = body.camera_id.strip()
    if not re.match(r"^[\w\-]+$", camera_id):
        raise HTTPException(status_code=400, detail="camera_id may only contain letters, numbers, - and _.")
    with _cameras_lock:
        if camera_id in cameras:
            raise HTTPException(status_code=409, detail=f"camera_id '{camera_id}' already exists.")
        cameras[camera_id] = _new_camera_slot(camera_id, label=body.label)
    _rebalance_evidence_budgets()
    _save_cameras_config()
    add_log(f"[{camera_id}] Camera slot created.", kind="camera")
    return {"ok": True, "camera_id": camera_id}


@app.delete("/cameras/{camera_id}")
def delete_camera(camera_id: str):
    with _cameras_lock:
        slot = cameras.pop(camera_id, None)
    if slot is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera_id '{camera_id}'.")
    slot["stream"].stop()
    _rebalance_evidence_budgets()
    _save_cameras_config()
    add_log(f"[{camera_id}] Camera removed.", kind="camera")
    return {"ok": True}


@app.get("/cameras/{camera_id}/status")
def camera_status(camera_id: str):
    slot = get_camera_or_404(camera_id)
    return _camera_status_payload(camera_id, slot)


@app.post("/cameras/{camera_id}/connect")
def camera_connect(camera_id: str, body: CameraConnectRequest):
    with _cameras_lock:
        slot = cameras.get(camera_id)
        if slot is None:
            slot = _new_camera_slot(camera_id)
            cameras[camera_id] = slot
            _rebalance_evidence_budgets()

    ip = body.ip.strip()
    if not re.match(r"^[\w.\-]+$", ip):
        raise HTTPException(status_code=400, detail="Invalid IP address or hostname.")

    stream_url = build_stream_url(ip, body.port, body.path)
    add_log(
        f"[{camera_id}] Connecting to camera at {stream_url}"
        + (f" (Wi-Fi: {body.wifi_name})" if body.wifi_name.strip() else ""),
        kind="camera",
    )

    capture_url = stream_url
    used_bridge = False
    diag = diagnose_http_camera(stream_url)
    if not diag["ok"]:
        add_log(
            f"[{camera_id}] Direct phone HTTP failed. Starting LAN bridge via Terminal.app...",
            level="warning",
            kind="camera",
        )
        try:
            capture_url, note = ensure_lan_bridge(stream_url)
            used_bridge = True
            add_log(f"[{camera_id}] {note}", kind="camera")
        except Exception as e:
            add_log(f"[{camera_id}] LAN bridge failed: {e}", level="error", kind="camera")
            slot["config"] = {
                **_default_camera_config(),
                "ip": ip,
                "port": body.port,
                "path": body.path if body.path.startswith("/") else f"/{body.path}",
                "wifi_name": body.wifi_name.strip(),
                "label": slot["config"].get("label") or camera_id,
            }
            _save_cameras_config()
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{e} "
                    "Alternatively: System Settings → Privacy & Security → Local Network → enable Cursor, "
                    "or run start_backend.command in Terminal.app."
                ),
            )

    success, error = slot["stream"].restart(capture_url)
    slot["score_history"].clear()
    slot["warmup_remaining"] = 12

    slot["config"] = {
        **_default_camera_config(),
        "ip": ip,
        "port": body.port,
        "path": body.path if body.path.startswith("/") else f"/{body.path}",
        "wifi_name": body.wifi_name.strip(),
        "stream_url": stream_url if success else None,
        "capture_url": capture_url if success else None,
        "via_bridge": used_bridge,
        "source": "phone",
        "label": slot["config"].get("label") or camera_id,
    }
    _save_cameras_config()

    if not success:
        add_log(f"[{camera_id}] Camera connect failed: {error}", level="error", kind="camera")
        raise HTTPException(
            status_code=400,
            detail=error or "Could not open camera stream. Check IP Webcam is running and both devices are on the same Wi-Fi.",
        )

    time.sleep(0.5)
    add_log(f"[{camera_id}] Camera connected successfully" + (" via LAN bridge." if used_bridge else "."), kind="camera")
    return {
        "ok": True,
        "camera_id": camera_id,
        "stream_url": stream_url,
        "capture_url": capture_url,
        "via_bridge": used_bridge,
        "wifi_name": slot["config"]["wifi_name"],
        "camera_active": slot["stream"].is_active(),
        "camera_receiving": _camera_receiving_frames(slot),
    }


@app.post("/cameras/{camera_id}/connect-webcam")
def camera_connect_webcam(camera_id: str, body: WebcamConnectRequest = WebcamConnectRequest()):
    """Connect a local USB/built-in webcam (device_index lets you attach a 2nd USB camera)."""
    with _cameras_lock:
        slot = cameras.get(camera_id)
        if slot is None:
            slot = _new_camera_slot(camera_id)
            cameras[camera_id] = slot
            _rebalance_evidence_budgets()

    add_log(f"[{camera_id}] Connecting webcam (device_index={body.device_index})...", kind="camera")
    success, error = slot["stream"].restart(body.device_index)
    slot["score_history"].clear()
    slot["warmup_remaining"] = 12
    slot["config"] = {
        **_default_camera_config(),
        "wifi_name": "Local webcam",
        "source": "webcam",
        "label": slot["config"].get("label") or camera_id,
    }
    _save_cameras_config()
    if not success:
        add_log(f"[{camera_id}] Webcam connect failed: {error}", level="error", kind="camera")
        raise HTTPException(status_code=400, detail=error or "Could not open webcam.")

    time.sleep(0.3)
    add_log(f"[{camera_id}] Webcam connected.", kind="camera")
    return {
        "ok": True,
        "camera_id": camera_id,
        "camera_kind": "webcam",
        "camera_active": slot["stream"].is_active(),
        "camera_receiving": _camera_receiving_frames(slot),
    }


@app.post("/cameras/{camera_id}/disconnect")
def camera_disconnect(camera_id: str):
    slot = get_camera_or_404(camera_id)
    slot["stream"].stop()
    slot["score_history"].clear()
    slot["config"] = {**slot["config"], "stream_url": None, "capture_url": None}
    _save_cameras_config()
    add_log(f"[{camera_id}] Camera disconnected by operator.", kind="camera")
    return {"ok": True, "camera_id": camera_id, "camera_active": False}


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
    add_log(f"Labeled {safe_name} as {label} → {dest}", kind="review")
    return {
        "ok": True,
        "label": label,
        "copied_to": dest,
        "hint": "When you have both safe and unsafe examples, run: python train.py",
    }


@app.get("/alerts")
def get_alerts(camera_id: Optional[str] = None):
    entries = alert_log if camera_id is None else [a for a in alert_log if a.get("camera_id") == camera_id]
    recent = [_alert_with_url(a) for a in entries[-20:]]
    return {
        "total_alerts": len(entries),
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

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=safe_name,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


def _generate_mjpeg(camera_id: str):
    slot = get_camera_or_404(camera_id)
    stream = slot["stream"]
    while True:
        jpeg_bytes = stream.get_latest_frame_jpeg()
        if jpeg_bytes is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
        time.sleep(1.0 / CAMERA_TARGET_FPS)


@app.get("/cameras/{camera_id}/feed")
def video_feed(camera_id: str):
    get_camera_or_404(camera_id)  # 404 early, before the generator starts streaming
    return StreamingResponse(
        _generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/cameras/{camera_id}/preview")
def preview_page(camera_id: str):
    slot = get_camera_or_404(camera_id)
    scores = list(slot["score_history"])
    latest_raw = round(scores[-1] * 100, 1) if scores else 0
    latest_avg = (
        round((sum(scores) / len(scores)) * 100, 1)
        if len(scores) == SCORE_HISTORY_LEN
        else 0
    )
    stream = slot["stream"]

    html = f"""
  <!DOCTYPE html>
  <html>
  <head>
    <title>SafeSpace Live Preview — {camera_id}</title>
    <style>
      body {{ background: #0f172a; color: #e2e8f0; font-family: sans-serif; text-align: center; padding: 20px; }}
      h1 {{ color: #2dd4bf; }}
      img {{ border: 2px solid #334155; border-radius: 12px; max-width: 90vw; }}
      .stats {{ margin-top: 16px; font-size: 18px; color: #94a3b8; }}
    </style>
  </head>
  <body>
    <h1>SafeSpace — Live Camera Preview ({camera_id})</h1>
    <p>Prefer the Next.js dashboard at <code>http://localhost:3000</code>.</p>
    <img src="/cameras/{camera_id}/feed" alt="Live Camera Feed" />
    <div class="stats">
      <p>Latest Raw Score: <strong>{latest_raw}%</strong> | Smoothed Avg: <strong>{latest_avg}%</strong></p>
      <p>Camera Active: <strong>{stream.is_active()}</strong> | Total Frames: <strong>{stream.get_frame_count()}</strong></p>
    </div>
    </body>
  </html>
  """
    return HTMLResponse(content=html)


CALIBRATION_DIR = "uploads/calibration"
os.makedirs(CALIBRATION_DIR, exist_ok=True)

@app.post("/calibration/run")
async def run_calibration(
    video_file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    target_fps: float = Form(CAMERA_TARGET_FPS),
    spike_threshold: float = Form(SPIKE_THRESHOLD),
    alert_threshold: float = Form(ALERT_THRESHOLD),
    history_len: int = Form(SCORE_HISTORY_LEN),
    pad_seconds: float = Form(2.0),
    merge_gap_seconds: float = Form(1.5),
):
    if not video_file and not youtube_url:
        raise HTTPException(status_code=400, detail="Must provide video_file or youtube_url.")
    
    video_path = None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if youtube_url:
        # Download with yt-dlp
        add_log(f"Downloading YouTube video: {youtube_url}", kind="calibration")
        ffmpeg_exe = get_ffmpeg_binary()
        ydl_opts = {
            'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            'outtmpl': os.path.join(CALIBRATION_DIR, f'yt_{stamp}_%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'nocheckcertificate': True,
            'remote_components': ['ejs:github'],
            'js_runtimes': {'node': {}},
            'quiet': True,
        }
        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            ydl_opts['ffmpeg_location'] = ffmpeg_exe
        if os.path.exists("cookies.txt"):
            ydl_opts["cookiefile"] = "cookies.txt"
        try:
            # Need to run yt-dlp in a thread to avoid blocking event loop
            def download_yt():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=True)
                    target = ydl.prepare_filename(info)
                    if os.path.exists(target):
                        return target
                    mp4_target = os.path.splitext(target)[0] + '.mp4'
                    if os.path.exists(mp4_target):
                        return mp4_target
                    base = os.path.basename(target).split('.')[0]
                    for f in os.listdir(CALIBRATION_DIR):
                        if base in f:
                            return os.path.join(CALIBRATION_DIR, f)
                    return target
            video_path = await asyncio.to_thread(download_yt)
            video_path = ensure_web_mp4(video_path)
        except Exception as e:
            add_log(f"YouTube download failed: {e}", level="error", kind="calibration")
            raise HTTPException(status_code=400, detail=f"YouTube download failed: {e}")
    else:
        name = os.path.basename(video_file.filename or "clip.mp4")
        if not re.search(r"\.(mp4|avi|mov|mkv|webm)$", name, re.I):
            raise HTTPException(status_code=400, detail="Upload an mp4/avi/mov/mkv/webm video.")
        safe = re.sub(r"[^\w.\-]", "_", name)
        video_path = os.path.join(CALIBRATION_DIR, f"upload_{stamp}_{safe}")
        try:
            with open(video_path, "wb") as out:
                shutil.copyfileobj(video_file.file, out)
            video_path = ensure_web_mp4(video_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save upload: {e}")
            
    add_log(f"Starting calibration analysis on {video_path}", kind="calibration")
    try:
        events, per_sample_log, native_fps = await asyncio.to_thread(
            process_video,
            video_path,
            model,
            device,
            target_fps=target_fps,
            spike_threshold=spike_threshold,
            alert_threshold=alert_threshold,
            history_len=history_len,
            pad_seconds=pad_seconds,
            merge_gap_seconds=merge_gap_seconds
        )
    except Exception as e:
        add_log(f"Calibration processing failed: {e}", level="error", kind="calibration")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
        
    clips = []
    base = os.path.splitext(os.path.basename(video_path))[0]
    for i, (s, e, peak, trig) in enumerate(events, 1):
        out_name = f"{base}_event{i:02d}_{trig}_{s:.1f}s-{e:.1f}s_peak{peak:.2f}.mp4"
        out_path = os.path.join(CALIBRATION_DIR, out_name)
        try:
            await asyncio.to_thread(cut_clip, video_path, out_path, s, e)
            clips.append({
                "start_sec": round(s, 2),
                "end_sec": round(e, 2),
                "peak_score": round(peak, 4),
                "trigger": trig,
                "url": f"/calibration/clip/{out_name}"
            })
        except Exception as exc:
            add_log(f"ffmpeg failed for {s:.1f}s-{e:.1f}s: {exc}", level="error", kind="calibration")

    add_log(f"Calibration complete for {video_path}. {len(clips)} events.", kind="calibration")
    return {
        "ok": True,
        "video_path": video_path,
        "native_fps": native_fps,
        "events": clips,
        "per_sample_log": per_sample_log
    }

@app.api_route("/calibration/clip/{filename}", methods=["GET", "HEAD"])
def get_calibration_clip(filename: str):
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid clip filename.")
    
    path = os.path.join(CALIBRATION_DIR, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Clip file not found.")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=safe_name,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )