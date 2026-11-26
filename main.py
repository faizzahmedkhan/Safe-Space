import asyncio
import html as html_lib
import json
import logging
import logging.handlers
import os
import queue
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from urllib.parse import parse_qs

import torch
import yt_dlp
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from bridge_ctl import ensure_lan_bridge
from calibration_runs import CalibrationStore, build_record as build_calibration_record, is_run_id
from camera import CameraStream, diagnose_http_camera
from clip_sampler import window_coverage
from evidence_video import make_browser_compatible
from inference_core import (
    CAMERA_TARGET_FPS,
    CLIP_LENGTH,
    INFERENCE_STRIDE,
    POST_ROLL_SECONDS,
    load_model,
    make_tracker,
    preprocess_clip,
    predict_violence,
)
from retention import prune_dir, retention_budgets
from validators import (
    is_clip_filename,
    is_evidence_filename,
    is_video_filename,
    keys_match,
    parse_origins,
    validate_camera_id,
    validate_host,
    validate_path,
)
from video_analyze import analyze_video_file
from video_calibration import cut_clip, process_video

# ==========================================
# LOGGING (file + console).  add_log() below still feeds the dashboard feed.
# ==========================================
LOG_DIR = os.environ.get("SAFESPACE_LOG_DIR", "logs")


def _setup_logging() -> logging.Logger:
    root = logging.getLogger("safespace")
    if root.handlers:
        return root
    root.setLevel(logging.INFO)
    root.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(LOG_DIR, "safespace.log"), maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as e:  # read-only checkout etc. - console logging still works
        root.warning("File logging disabled: %s", e)
    return root


_setup_logging()
logger = logging.getLogger("safespace.app")

# --- THRESHOLDS & TIMINGS ---
# The canonical values live in inference_core / thresholds.py and may be
# overridden per model by its JSON sidecar (see load_model).  Everything below
# is read from THRESHOLDS after the model loads.
MAX_ALERTS = 500
MAX_LOGS = 200
FRAME_STALE_MS = 2000
# Total evidence-buffer RAM budget shared across ALL cameras, not per camera.
TOTAL_EVIDENCE_BUDGET_MB = 300
WARMUP_RESTART = 12   # scored windows ignored after a connect/restart
WARMUP_GAP = 6        # ... after a frame gap / grab failure
COVERAGE_WARN_PCT = 90.0
WATCHER_STALL_S = 20.0
RECONNECT_INTERVAL_S = 30.0
RETENTION_INTERVAL_S = 3600.0
MAX_UPLOAD_BYTES = int(float(os.environ.get("SAFESPACE_MAX_UPLOAD_MB", "500")) * 1024 * 1024)
MAX_YT_DURATION_S = 15 * 60

# NOTE: cropping and normalization follow the loaded model's preprocessing
# contract (MODEL_INFO["preprocessing"], from its sidecar; legacy models =
# per-frame crop, /255 only).  See preprocessing.py.

ALERTS_FILE = "alerts.json"
CAMERAS_CONFIG_FILE = "cameras_config.json"
EVIDENCE_DIR = "evidence"
UPLOADS_DIR = "uploads"
DATA_SAFE = os.path.join("data", "safe")
DATA_UNSAFE = os.path.join("data", "unsafe")

# ---- auth / CORS (F1) ----
API_KEY = os.environ.get("SAFESPACE_API_KEY", "").strip()
AUTH_ENABLED = bool(API_KEY)
CORS_ORIGINS = parse_origins(
    os.environ.get("SAFESPACE_CORS_ORIGINS"),
    ["http://localhost:3000", "http://127.0.0.1:3000"],
)

alert_log: list = []
# F3: alert_log is mutated from the watcher, transcode worker, retention job
# and request threads; every mutation + persist happens under this lock.
alert_lock = threading.RLock()
system_logs: deque = deque(maxlen=MAX_LOGS)
_stop_evt = threading.Event()


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR, "alert": logging.WARNING}


def add_log(message: str, level: str = "info", kind: str = "system") -> None:
    entry = {
        "timestamp": _now_iso(),
        "level": level,
        "message": message,
        "kind": kind,
    }
    system_logs.append(entry)
    logging.getLogger(f"safespace.{kind}").log(_LEVELS.get(level, logging.INFO), "%s", message)


def _save_alerts_atomic(alerts: list, path: str) -> None:
    """Callers must hold alert_lock."""
    dir_name = os.path.dirname(path) or "."
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp") as tmp:
            json.dump(alerts, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, path)
    except Exception as e:
        add_log(f"Could not persist alerts: {e}", level="error", kind="system")


def _persist_alerts() -> None:
    with alert_lock:
        _save_alerts_atomic(alert_log, ALERTS_FILE)


# ==========================================
# 1. LOAD THE BRAIN (System Startup)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Booting AI Core on: %s", device)

# Model path: env SAFESPACE_MODEL, else first existing of v4_baseline / v3 / smart_crop / base.
# Thresholds: from <model>.json sidecar if present, else inference_core defaults.
model, MODEL_INFO = load_model(device=device)
MODEL_VERSION = MODEL_INFO["version"]
THRESHOLDS = MODEL_INFO["thresholds"]
ALERT_THRESHOLD = THRESHOLDS["alert_threshold"]
SPIKE_THRESHOLD = THRESHOLDS["spike_threshold"]
SCORE_HISTORY_LEN = int(THRESHOLDS["score_history_len"])
PREPROC = MODEL_INFO["preprocessing"]
add_log(
    f"Model {MODEL_VERSION} loaded on {device} (alert>={ALERT_THRESHOLD}, spike>={SPIKE_THRESHOLD}"
    + (", thresholds from sidecar)" if MODEL_INFO["thresholds_from_sidecar"] else ", default thresholds)")
    + f"; preprocessing {PREPROC.name} (crop={PREPROC.crop}, normalize={PREPROC.normalize}, "
    f"from {MODEL_INFO['preprocessing_source']})",
    kind="system",
)
if not AUTH_ENABLED:
    add_log(
        "SAFESPACE_API_KEY is not set: API authentication is DISABLED. Anyone who can reach "
        "this server can view feeds and evidence. Set SAFESPACE_API_KEY before exposing it.",
        level="warning",
        kind="system",
    )

if os.path.exists(ALERTS_FILE):
    try:
        with open(ALERTS_FILE, "r") as f:
            alert_log = json.load(f)
    except Exception as e:
        logger.error("Could not load past alerts: %s", e)

os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_SAFE, exist_ok=True)
os.makedirs(DATA_UNSAFE, exist_ok=True)


# ==========================================
# 2. EVIDENCE TRANSCODE WORKER (E1)
# ==========================================
# ffmpeg used to run synchronously inside the asyncio watcher, freezing every
# camera while it worked.  Now one background thread drains a queue.
_transcode_q: "queue.Queue[Optional[str]]" = queue.Queue()


def _set_alert_status(path: str, status: str) -> None:
    with alert_lock:
        for a in alert_log:
            if a.get("evidence_file") == path:
                a["status"] = status
        _save_alerts_atomic(alert_log, ALERTS_FILE)


def _transcode_worker() -> None:
    while True:
        path = _transcode_q.get()
        if path is None:
            return
        try:
            ok = make_browser_compatible(path, timeout=300)
        except Exception as e:  # never let the worker die
            logger.error("Transcode crashed for %s: %s", path, e)
            ok = False
        _set_alert_status(path, "ready" if ok else "failed")
        add_log(
            f"Evidence {os.path.basename(path)} " + ("ready." if ok else "could not be transcoded (status failed)."),
            level="info" if ok else "error",
            kind="recording",
        )


def _on_recording_finished(camera_id: str, path: str, ok: bool) -> None:
    """Called from a camera's evidence-writer thread once the file is closed."""
    if ok:
        _set_alert_status(path, "processing")
        _transcode_q.put(path)
    else:
        _set_alert_status(path, "failed")
        add_log(f"[{camera_id}] Evidence file {os.path.basename(path)} could not be written.", level="error", kind="recording")


def _recover_alert_states() -> None:
    """Old alerts have no status (they are playable -> 'ready').  Anything left
    'recording'/'processing' by a crash gets re-queued, or failed if the file
    is gone."""
    changed = False
    with alert_lock:
        for a in alert_log:
            st = a.get("status")
            if st is None:
                a["status"] = "ready"
                changed = True
            elif st in ("recording", "processing"):
                f = a.get("evidence_file") or ""
                if f and os.path.isfile(f):
                    a["status"] = "processing"
                    _transcode_q.put(f)
                else:
                    a["status"] = "failed"
                changed = True
        if changed:
            _save_alerts_atomic(alert_log, ALERTS_FILE)


_recover_alert_states()


# ==========================================
# 3. CAMERA MANAGER
# ==========================================
# Each entry in `cameras` is one independently-running CameraStream plus its
# per-camera bookkeeping (IncidentTracker, warmup, generation, ...).
_cameras_lock = threading.Lock()
cameras: dict = {}


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
        "device_index": 0,
        "label": "",
    }


def _new_camera_slot(camera_id: str, label: str = "") -> dict:
    stream = CameraStream(src=None, target_fps=CAMERA_TARGET_FPS, show_preview=False, soft_fail=True,
                          crop_mode=PREPROC.crop)
    tracker = make_tracker(THRESHOLDS)
    stream.on_recording_finished = lambda path, ok, _cid=camera_id: _on_recording_finished(_cid, path, ok)
    return {
        "camera_id": camera_id,
        "stream": stream,
        "config": {**_default_camera_config(), "label": label or camera_id},
        "tracker": tracker,
        # alias kept for status/preview code: IncidentTracker.history is the rolling score deque
        "score_history": tracker.history,
        "incident_alert": None,
        "generation": stream.get_generation(),
        "warmup_remaining": WARMUP_RESTART,
        "last_scored_t": None,
        "scored_ts": deque(maxlen=40),
        "coverage_pct": None,
        "effective_stride_s": None,
        "last_coverage_warn": 0.0,
        "last_persist": 0.0,
        "manual_stop": False,
        "lock": threading.Lock(),   # serialises connect/restart vs. the reconnect supervisor
        "health": {"offline": False, "frozen": False, "degraded": False, "stale": False},
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


def _saved_source(slot: dict):
    """The source this camera should reconnect to, or None."""
    cfg = slot["config"]
    if cfg.get("source") == "webcam":
        return int(cfg.get("device_index") or 0)
    return cfg.get("capture_url") or cfg.get("stream_url")


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

        _src = _saved_source(slot)
        if _src is None:
            add_log(f"[{_camera_id}] Idle at startup. Connect from the dashboard.", kind="camera")
            continue

        ok, err = slot["stream"].restart(_src)
        slot["generation"] = slot["stream"].get_generation()
        if ok:
            add_log(f"[{_camera_id}] Camera started with src={_src}", kind="camera")
        else:
            add_log(f"[{_camera_id}] Not connected at startup ({err}). Will retry automatically.", level="warning", kind="camera")

_rebalance_evidence_budgets()


# ==========================================
# 4. INCIDENTS (C1-C4): tracker events <-> alerts + evidence
# ==========================================
def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _incident_open(camera_id: str, slot: dict, ev: dict, scene: dict) -> None:
    stream = slot["stream"]
    timestamp = _stamp()
    filename = f"{EVIDENCE_DIR}/incident_{camera_id}_{timestamp}.mp4"
    recording = stream.start_recording(filename)
    alert = {
        "camera_id": camera_id,
        "timestamp": timestamp,
        "confidence": round(ev["confidence"] * 100, 1),
        "trigger": ev["trigger"],
        "evidence_file": filename,
        "scene": scene,
        "status": "recording" if recording else "failed",
        "peak": round(ev["peak"] * 100, 1),
        "n_triggers": ev["n_triggers"],
        "end_timestamp": None,
        "duration_sec": None,
        "model_version": MODEL_VERSION,
    }
    with alert_lock:
        alert_log.append(alert)
        if len(alert_log) > MAX_ALERTS:
            del alert_log[: len(alert_log) - MAX_ALERTS]
        _save_alerts_atomic(alert_log, ALERTS_FILE)
    slot["incident_alert"] = alert
    slot["last_persist"] = time.monotonic()
    add_log(
        f"[{camera_id}] INCIDENT OPEN ({ev['trigger']}): physical altercation detected "
        f"({alert['confidence']:.1f}%). "
        + (f"Recording {filename}" if recording else "Evidence recording could not start."),
        level="alert",
        kind="alert",
    )


def _incident_extend(camera_id: str, slot: dict, ev: dict) -> None:
    alert = slot.get("incident_alert")
    if alert is None:
        return
    with alert_lock:
        alert["peak"] = round(ev["peak"] * 100, 1)
        alert["n_triggers"] = ev["n_triggers"]
        if time.monotonic() - slot["last_persist"] > 5.0:  # throttle disk writes
            slot["last_persist"] = time.monotonic()
            _save_alerts_atomic(alert_log, ALERTS_FILE)


def _incident_close(camera_id: str, slot: dict, ev: dict) -> None:
    slot["stream"].stop_recording()   # writer thread finishes -> _on_recording_finished
    alert = slot.get("incident_alert")
    slot["incident_alert"] = None
    if alert is not None:
        with alert_lock:
            alert["end_timestamp"] = _stamp()
            alert["duration_sec"] = round(ev.get("duration", 0.0), 1)
            alert["peak"] = round(ev["peak"] * 100, 1)
            alert["n_triggers"] = ev["n_triggers"]
            alert["close_reason"] = ev["reason"]
            _save_alerts_atomic(alert_log, ALERTS_FILE)
    add_log(
        f"[{camera_id}] Incident closed ({ev['reason']}), {ev.get('duration', 0.0):.0f}s, "
        f"peak {ev['peak']*100:.0f}%, {ev['n_triggers']} trigger(s).",
        kind="recording",
    )


def _apply_events(camera_id: str, slot: dict, events: list, scene: dict) -> None:
    for ev in events:
        kind = ev["type"]
        if kind == "open":
            _incident_open(camera_id, slot, ev, scene)
        elif kind == "extend":
            _incident_extend(camera_id, slot, ev)
        elif kind == "close":
            _incident_close(camera_id, slot, ev)
        elif kind == "spike_unconfirmed":
            logger.info("[%s] Spike at t=%.1f not confirmed by the next window - ignored.", camera_id, ev["spike_t"])
        elif kind == "spike_pending":
            logger.info("[%s] Spike %.0f%% - waiting for confirmation window.", camera_id, ev["score"] * 100)


def _reset_slot_state(camera_id: str, slot: dict, reason: str, warmup: int) -> None:
    """Camera generation changed (restart / frame gap / failure) or the camera
    was lost: forget score history, re-warm. An open incident is closed only on
    a restart or a lost camera; a brief gap keeps it open (see IncidentTracker.reset)."""
    keep = reason not in ("restart", "camera_lost")
    events = slot["tracker"].reset(time.monotonic(), reason=reason, keep_incident=keep)
    _apply_events(camera_id, slot, events, {})
    slot["warmup_remaining"] = warmup
    slot["last_scored_t"] = None
    slot["scored_ts"].clear()
    slot["coverage_pct"] = None
    slot["effective_stride_s"] = None
    slot["generation"] = slot["stream"].get_generation()


def _update_coverage(camera_id: str, slot: dict) -> None:
    ts = list(slot["scored_ts"])
    window_s = (CLIP_LENGTH - 1) / CAMERA_TARGET_FPS
    slot["coverage_pct"] = window_coverage(ts, window_s=window_s, horizon_s=20.0)
    if len(ts) >= 2:
        recent = ts[-10:]
        slot["effective_stride_s"] = round((recent[-1] - recent[0]) / (len(recent) - 1), 3)
    cov = slot["coverage_pct"]
    if cov is not None and cov < COVERAGE_WARN_PCT and len(ts) >= 10:
        if time.monotonic() - slot["last_coverage_warn"] > 60.0:
            slot["last_coverage_warn"] = time.monotonic()
            add_log(
                f"[{camera_id}] Only {cov:.0f}% of recent video is being scored (effective stride "
                f"{slot['effective_stride_s']}s vs target {INFERENCE_STRIDE / CAMERA_TARGET_FPS:.2f}s). "
                "Short events may be missed - inference is too slow for this many cameras.",
                level="warning",
                kind="system",
            )


# ==========================================
# 5. THE CONSUMER LOOP (Background Worker)
# ==========================================
_watcher_state = {"last_cycle": None}


def _run_inference_batch(batch: torch.Tensor) -> torch.Tensor:
    """Blocking model call (run via asyncio.to_thread).  Serialised with every
    other forward pass in the process by inference_core.FORWARD_LOCK."""
    return predict_violence(model, batch)


async def _watcher_cycle(stride_s: float) -> None:
    ready_ids: list = []
    ready_snaps: list = []
    ready_tensors: list = []
    tol = 0.5 / CAMERA_TARGET_FPS  # half a frame of arrival jitter

    for camera_id, slot in list(cameras.items()):
        stream = slot["stream"]
        snap = stream.get_inference_snapshot()

        # A3/C7: any discontinuity (restart, grab failure, frame gap) bumps the
        # camera generation - drop all per-camera state and re-warm.
        if snap["generation"] != slot["generation"]:
            reason = snap["reset_reason"]
            _reset_slot_state(camera_id, slot, reason, WARMUP_RESTART if reason == "restart" else WARMUP_GAP)
            logger.info("[%s] Camera discontinuity (%s) - inference state reset.", camera_id, reason)

        age = snap["age_s"]
        if (not snap["ready"]) or age is None or age > 1.0:
            # A camera that can no longer be observed must not hold an incident open.
            tracker = slot["tracker"]
            if tracker.active and (age is None or age > POST_ROLL_SECONDS):
                _reset_slot_state(camera_id, slot, "camera_lost", WARMUP_GAP)
                add_log(f"[{camera_id}] Incident closed - camera stopped delivering frames.", level="warning", kind="recording")
            continue

        last = slot["last_scored_t"]
        if last is not None and (snap["t_end"] - last) < (stride_s - tol):
            continue

        ready_ids.append(camera_id)
        ready_snaps.append(snap)
        ready_tensors.append(preprocess_clip(snap["frames"], normalize=PREPROC.normalize))

    if not ready_tensors:
        return

    batch = torch.stack(ready_tensors).to(device)  # (B, C, T, H, W)
    probabilities = await asyncio.to_thread(_run_inference_batch, batch)

    for i, camera_id in enumerate(ready_ids):
        slot = cameras.get(camera_id)
        snap = ready_snaps[i]
        if slot is None:
            continue  # camera was removed mid-cycle
        if snap["generation"] != slot["stream"].get_generation():
            continue  # discontinuity happened while scoring; drop this stale result
        current_prob = float(probabilities[i].item())
        t_end = snap["t_end"]
        slot["last_scored_t"] = t_end
        slot["scored_ts"].append(t_end)
        _update_coverage(camera_id, slot)

        if slot["warmup_remaining"] > 0:
            slot["warmup_remaining"] -= 1
            logger.info(
                "[WARMUP][%s] %d steps left. Raw %.1f%% (suppressed)",
                camera_id, slot["warmup_remaining"], current_prob * 100,
            )
            continue

        tracker = slot["tracker"]
        events = tracker.update(
            t_end,
            current_prob,
            suitable=snap["suitable"],
            degraded=snap["degraded"],
            global_motion=snap["global_motion"],
        )
        avg = tracker.average() or 0.0
        logger.info(
            "[%s] raw=%.1f%% avg=%.1f%% scene=%s suitable=%s global=%s degraded=%s incident=%s",
            camera_id, current_prob * 100, avg * 100, snap["scene"].get("method"),
            snap["suitable"], snap["global_motion"], snap["degraded"], tracker.active,
        )
        _apply_events(camera_id, slot, events, snap["scene"])


async def ai_watcher():
    """
    One consumer loop for every camera.  Each cycle (deadline-scheduled, so
    inference time does not stretch the stride):
      1. Take a timestamped snapshot of every camera and sample a 16-frame clip
         on a 15 fps timeline.
      2. Batch every camera that is due into ONE forward pass.
      3. Feed each camera's IncidentTracker, which opens/extends/closes incidents.
    """
    logger.info("[AI WATCHER] Online. Waiting for camera buffers to fill...")
    stride_s = INFERENCE_STRIDE / CAMERA_TARGET_FPS
    # Poll twice per stride so a window is scored as soon as it is due (a
    # camera clock that is a few ms off never costs a whole extra cycle).
    period_s = stride_s / 2.0
    next_deadline = time.monotonic()
    last_error_log = 0.0

    while True:
        _watcher_state["last_cycle"] = time.monotonic()
        try:
            await _watcher_cycle(stride_s)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # keep the watcher alive; surface the failure
            logger.exception("AI watcher cycle failed")
            if time.monotonic() - last_error_log > 30:
                last_error_log = time.monotonic()
                add_log(f"AI watcher error (continuing): {type(e).__name__}: {e}", level="error", kind="system")
        _watcher_state["last_cycle"] = time.monotonic()

        next_deadline += period_s
        delay = next_deadline - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            next_deadline = time.monotonic()  # overran: do not try to catch up


def _watcher_age_ms() -> Optional[float]:
    t = _watcher_state["last_cycle"]
    return None if t is None else round((time.monotonic() - t) * 1000, 1)


# ==========================================
# 6. WATCHDOG (E3), RECONNECT SUPERVISOR (E2), RETENTION (F3)
# ==========================================
def _watchdog_tick(state: dict) -> None:
    age = _watcher_age_ms()
    if age is not None:
        stalled = age / 1000.0 > WATCHER_STALL_S
        if stalled and not state["watcher_stalled"]:
            add_log(f"AI watcher has not completed a cycle for {age/1000:.0f}s. Detection may be down.", level="error", kind="system")
        elif not stalled and state["watcher_stalled"]:
            add_log("AI watcher recovered.", kind="system")
        state["watcher_stalled"] = stalled

    for camera_id, slot in list(cameras.items()):
        stream = slot["stream"]
        h = slot["health"]
        offline = stream.offline_since is not None
        if offline and not h["offline"]:
            add_log(f"[{camera_id}] Camera OFFLINE since {stream.get_offline_since_iso()}.", level="error", kind="camera")
        elif not offline and h["offline"]:
            add_log(f"[{camera_id}] Camera back online.", kind="camera")
        h["offline"] = offline

        frozen = bool(stream.frozen) and stream.is_active()
        if frozen and not h["frozen"]:
            add_log(f"[{camera_id}] Camera FROZEN: the same image has repeated for several seconds.", level="error", kind="camera")
        elif not frozen and h["frozen"]:
            add_log(f"[{camera_id}] Camera no longer frozen.", kind="camera")
        h["frozen"] = frozen

        degraded = bool(stream.degraded) and stream.is_active()
        if degraded and not h["degraded"]:
            add_log(
                f"[{camera_id}] Camera DEGRADED ({stream.get_actual_fps()} fps). Alerts are suppressed until it recovers.",
                level="warning", kind="camera",
            )
        elif not degraded and h["degraded"]:
            add_log(f"[{camera_id}] Camera frame rate recovered.", kind="camera")
        h["degraded"] = degraded

        fage = stream.get_last_frame_age_ms()
        stale = stream.is_active() and fage is not None and fage > 5000
        if stale and not h["stale"]:
            add_log(f"[{camera_id}] No new frames for {fage/1000:.0f}s.", level="warning", kind="camera")
        elif not stale and h["stale"]:
            add_log(f"[{camera_id}] Frames flowing again.", kind="camera")
        h["stale"] = stale


def _watchdog_loop() -> None:
    state = {"watcher_stalled": False}
    while not _stop_evt.wait(2.0):
        try:
            _watchdog_tick(state)
        except Exception:
            logger.exception("watchdog tick failed")


def _reconnect_tick() -> None:
    for camera_id, slot in list(cameras.items()):
        if slot["manual_stop"] or slot["stream"].is_active():
            continue
        src = _saved_source(slot)
        if src is None:
            continue
        if not slot["lock"].acquire(blocking=False):
            continue  # an operator connect is in progress
        try:
            if slot["manual_stop"] or slot["stream"].is_active():
                continue
            add_log(f"[{camera_id}] Auto-reconnect attempt (src={src}).", kind="camera")
            ok, err = slot["stream"].restart(src)
            slot["generation"] = slot["stream"].get_generation()
            _reset_slot_state(camera_id, slot, "restart", WARMUP_RESTART)
            if ok:
                add_log(f"[{camera_id}] Auto-reconnect succeeded.", kind="camera")
            else:
                add_log(f"[{camera_id}] Auto-reconnect failed ({err}). Next attempt in {int(RECONNECT_INTERVAL_S)}s.", level="warning", kind="camera")
        finally:
            slot["lock"].release()


def _reconnect_loop() -> None:
    while not _stop_evt.wait(RECONNECT_INTERVAL_S):
        try:
            _reconnect_tick()
        except Exception:
            logger.exception("reconnect supervisor failed")


def run_retention() -> dict:
    """F3: prune evidence/uploads/calibration by size and age.  Never touches
    evidence of open incidents or clips copied into data/."""
    budgets = retention_budgets()
    with alert_lock:
        protected = {
            os.path.basename(a.get("evidence_file") or "")
            for a in alert_log
            if a.get("status") in ("recording", "processing")
        }
    for d in (DATA_SAFE, DATA_UNSAFE):
        try:
            protected.update(os.listdir(d))
        except OSError:
            pass
    protected.discard("")

    removed = {
        "evidence": prune_dir(EVIDENCE_DIR, *budgets["evidence"], protected_names=protected),
        "uploads": prune_dir(UPLOADS_DIR, *budgets["uploads"], protected_names=protected),
        "calibration": prune_dir(
            [CALIBRATION_DIR, calibration_store.inputs, calibration_store.outputs],
            *budgets["calibration"],
            protected_names=protected | calibration_store.run_json_names(),
        ),
    }
    if removed["evidence"]:
        gone = set(removed["evidence"])
        with alert_lock:
            for a in alert_log:
                if os.path.basename(a.get("evidence_file") or "") in gone:
                    a["evidence_pruned"] = True
            _save_alerts_atomic(alert_log, ALERTS_FILE)
    n = sum(len(v) for v in removed.values())
    if n:
        add_log(
            f"Retention removed {len(removed['evidence'])} evidence, {len(removed['uploads'])} upload, "
            f"{len(removed['calibration'])} calibration file(s).",
            kind="system",
        )
    return removed


def _retention_loop() -> None:
    while True:
        try:
            run_retention()
        except Exception:
            logger.exception("retention job failed")
        if _stop_evt.wait(RETENTION_INTERVAL_S):
            return


# ==========================================
# 7. FASTAPI ROUTING
# ==========================================
_bg_threads: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(ai_watcher())
    for target, name in (
        (_transcode_worker, "transcode"),
        (_watchdog_loop, "watchdog"),
        (_reconnect_loop, "reconnect"),
        (_retention_loop, "retention"),
    ):
        t = threading.Thread(target=target, name=f"safespace-{name}", daemon=True)
        t.start()
        _bg_threads.append(t)
    yield
    logger.info("Shutting down SafeSpace...")
    _stop_evt.set()
    _transcode_q.put(None)
    for slot in cameras.values():
        slot["stream"].stop()


app = FastAPI(title="SafeSpace AI Backend", lifespan=lifespan)


class ApiKeyMiddleware:
    """Pure-ASGI (safe for the endless MJPEG feed).  If SAFESPACE_API_KEY is set,
    every request except GET / (and CORS preflight) needs header X-SafeSpace-Key
    or ?key= (for <img>/<video> tags).  Also rejects oversize uploads early."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope["method"]
        path = scope["path"]
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}

        if AUTH_ENABLED and method != "OPTIONS" and not (path == "/" and method in ("GET", "HEAD")):
            provided = headers.get("x-safespace-key")
            if not provided:
                q = parse_qs(scope.get("query_string", b"").decode("latin-1"))
                provided = (q.get("key") or [None])[0]
            if not keys_match(provided, API_KEY):
                resp = JSONResponse({"detail": "Missing or invalid API key."}, status_code=401)
                await resp(scope, receive, send)
                return

        if method in ("POST", "PUT"):
            try:
                length = int(headers.get("content-length", "0"))
            except ValueError:
                length = 0
            if length > MAX_UPLOAD_BYTES + 1024 * 1024:
                resp = JSONResponse(
                    {"detail": f"Upload too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."}, status_code=413
                )
                await resp(scope, receive, send)
                return

        await self.app(scope, receive, send)


# Order matters: the LAST middleware added is the outermost.  CORS must wrap the
# auth middleware so 401/413 responses still carry CORS headers.
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-SafeSpace-Key", "Content-Type"],
)


async def _save_upload(upload: UploadFile, dest: str) -> int:
    """Stream an upload to disk with a hard byte cap (413 when exceeded)."""
    total = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
                    )
                out.write(chunk)
    except HTTPException:
        _silent_remove(dest)
        raise
    except Exception as e:
        _silent_remove(dest)
        raise HTTPException(status_code=500, detail=f"Could not save upload: {e}") from e
    return total


def _silent_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


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
    out.setdefault("status", "ready")
    evidence = alert.get("evidence_file") or ""
    name = _evidence_basename(evidence)
    out["evidence_url"] = (
        f"/evidence/{name}" if name and not alert.get("evidence_pruned") else None
    )
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
    tracker = slot["tracker"]
    latest_raw = round(scores[-1] * 100, 1) if scores else None
    latest_avg = round((sum(scores) / len(scores)) * 100, 1) if scores else None
    camera_active = stream.is_active()
    receiving = _camera_receiving_frames(slot)
    fill, _ready = stream.get_buffer_fill()

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
        "buffer_fill": f"{fill}/{CLIP_LENGTH}",
        "total_frames": stream.get_frame_count(),
        "latest_raw_pct": latest_raw,
        "smoothed_avg_pct": latest_avg,
        "score_warmup": f"{len(scores)}/{tracker.history.maxlen}",
        "alert_threshold": ALERT_THRESHOLD * 100,
        "open_error": stream.open_error,
        "scene": stream.get_scene_meta(),
        # --- reliability fields (A1/A3/C5/C7/E3) ---
        "degraded": bool(stream.degraded) if camera_active else False,
        "frozen": bool(stream.frozen) if camera_active else False,
        "offline_since": stream.get_offline_since_iso(),
        "coverage_pct": slot.get("coverage_pct"),
        "effective_stride_s": slot.get("effective_stride_s"),
        "generation": stream.get_generation(),
        "incident_active": tracker.active,
    }


@app.get("/")
def read_root():
    return {
        "status": "SafeSpace Core is Active",
        "camera_count": len(cameras),
        "cameras_active": sum(1 for s in cameras.values() if s["stream"].is_active()),
        "watcher_last_cycle_age_ms": _watcher_age_ms(),
        "model_version": MODEL_VERSION,
        "model_name": MODEL_INFO["name"],
        "preprocessing": PREPROC.name,
        "thresholds": {
            "spike_threshold": SPIKE_THRESHOLD,
            "alert_threshold": ALERT_THRESHOLD,
            "score_history_len": SCORE_HISTORY_LEN,
        },
        "thresholds_from_sidecar": bool(MODEL_INFO["thresholds_from_sidecar"]),
        "auth_enabled": AUTH_ENABLED,
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
    if not validate_camera_id(camera_id):
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
    _reset_slot_state(camera_id, slot, "camera_removed", WARMUP_RESTART)
    slot["stream"].stop()
    _rebalance_evidence_budgets()
    _save_cameras_config()
    add_log(f"[{camera_id}] Camera removed.", kind="camera")
    return {"ok": True}


@app.get("/cameras/{camera_id}/status")
def camera_status(camera_id: str):
    slot = get_camera_or_404(camera_id)
    return _camera_status_payload(camera_id, slot)


def _get_or_create_slot(camera_id: str) -> dict:
    with _cameras_lock:
        slot = cameras.get(camera_id)
        if slot is None:
            if not validate_camera_id(camera_id):
                raise HTTPException(status_code=400, detail="camera_id may only contain letters, numbers, - and _.")
            slot = _new_camera_slot(camera_id)
            cameras[camera_id] = slot
            _rebalance_evidence_budgets()
    return slot


@app.post("/cameras/{camera_id}/connect")
def camera_connect(camera_id: str, body: CameraConnectRequest):
    # F2: validate before touching anything (no slot is created for bad input).
    ip = body.ip.strip()
    if not validate_host(ip):
        raise HTTPException(
            status_code=400,
            detail="Invalid IP address or hostname (use e.g. 192.168.1.5 - no http://, port or path).",
        )
    path = body.path.strip() or "/video"
    if not path.startswith("/"):
        path = "/" + path
    if not validate_path(path):
        raise HTTPException(status_code=400, detail="Invalid stream path (letters, digits, / . _ - only).")

    slot = _get_or_create_slot(camera_id)
    with slot["lock"]:
        slot["manual_stop"] = False
        stream_url = build_stream_url(ip, body.port, path)
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
                capture_url, note = ensure_lan_bridge(stream_url, camera_id=camera_id)
                used_bridge = True
                add_log(f"[{camera_id}] {note}", kind="camera")
            except Exception as e:
                add_log(f"[{camera_id}] LAN bridge failed: {e}", level="error", kind="camera")
                slot["config"] = {
                    **_default_camera_config(),
                    "ip": ip,
                    "port": body.port,
                    "path": path,
                    "wifi_name": body.wifi_name.strip(),
                    "label": slot["config"].get("label") or camera_id,
                }
                _save_cameras_config()
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{e} "
                        "Alternatively: System Settings -> Privacy & Security -> Local Network -> enable Cursor, "
                        "or run start_backend.command in Terminal.app."
                    ),
                )

        success, error = slot["stream"].restart(capture_url)
        _reset_slot_state(camera_id, slot, "restart", WARMUP_RESTART)

        slot["config"] = {
            **_default_camera_config(),
            "ip": ip,
            "port": body.port,
            "path": path,
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
    slot = _get_or_create_slot(camera_id)
    with slot["lock"]:
        slot["manual_stop"] = False
        add_log(f"[{camera_id}] Connecting webcam (device_index={body.device_index})...", kind="camera")
        success, error = slot["stream"].restart(body.device_index)
        _reset_slot_state(camera_id, slot, "restart", WARMUP_RESTART)
        slot["config"] = {
            **_default_camera_config(),
            "wifi_name": "Local webcam",
            "source": "webcam",
            "device_index": body.device_index,
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
    slot["manual_stop"] = True   # the reconnect supervisor must not undo an operator disconnect
    _reset_slot_state(camera_id, slot, "disconnect", WARMUP_RESTART)
    slot["stream"].stop()
    slot["stream"].offline_since = None
    slot["config"] = {**slot["config"], "stream_url": None, "capture_url": None, "source": "none"}
    _save_cameras_config()
    add_log(f"[{camera_id}] Camera disconnected by operator.", kind="camera")
    return {"ok": True, "camera_id": camera_id, "camera_active": False}


class LabelEvidenceRequest(BaseModel):
    label: str = Field(..., description="'safe' (false alarm) or 'unsafe' (true incident)")


@app.post("/analyze/video")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    """
    Upload a recorded clip and run it through the same pipeline as live mode
    (same preprocessing, 15 fps resampling, thresholds and incident rules).
    Use this to check whether an alert was wrong (false positive) or missed.
    """
    name = os.path.basename(file.filename or "clip.mp4")
    if not is_video_filename(name):
        raise HTTPException(status_code=400, detail="Upload an mp4/avi/mov/mkv/webm video.")

    safe = re.sub(r"[^\w.\-]", "_", name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(UPLOADS_DIR, f"review_{stamp}_{safe}")
    await _save_upload(file, dest)

    add_log(f"Analyzing uploaded video {dest}", kind="review")
    try:
        # E1: heavy CPU work off the event loop so every camera keeps being scored.
        result = await asyncio.to_thread(
            analyze_video_file,
            dest,
            model,
            device,
            thresholds=THRESHOLDS,
            model_version=MODEL_VERSION,
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
    if not is_evidence_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid evidence filename.")
    safe_name = filename

    path = os.path.join(EVIDENCE_DIR, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Evidence file not found.")

    add_log(f"Re-analyzing evidence {safe_name}", kind="review")
    try:
        result = analyze_video_file(
            path,
            model=model,
            device=device,
            thresholds=THRESHOLDS,
            model_version=MODEL_VERSION,
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

    if not is_evidence_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid evidence filename.")
    safe_name = filename

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
        "hint": (
            "When you have both safe and unsafe examples, run: python train.py "
            "(it saves a NEW timestamped model + .json sidecar and never overwrites the current one). "
            "Then restart the backend with SAFESPACE_MODEL=<the new .pth path> to use it."
        ),
    }


@app.get("/alerts")
def get_alerts(camera_id: Optional[str] = None):
    with alert_lock:  # snapshot: the watcher mutates open incidents in place
        entries = [dict(a) for a in alert_log if camera_id is None or a.get("camera_id") == camera_id]
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
    if not is_evidence_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid evidence filename.")
    safe_name = filename

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
def preview_page(camera_id: str, request: Request):
    slot = get_camera_or_404(camera_id)
    key = request.query_params.get("key")
    feed_src = f"/cameras/{camera_id}/feed" + (f"?key={html_lib.escape(key, quote=True)}" if key else "")
    scores = list(slot["score_history"])
    latest_raw = round(scores[-1] * 100, 1) if scores else 0
    latest_avg = (
        round((sum(scores) / len(scores)) * 100, 1)
        if len(scores) == slot["tracker"].history.maxlen
        else 0
    )
    stream = slot["stream"]

    page = f"""
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
    <img src="{feed_src}" alt="Live Camera Feed" />
    <div class="stats">
      <p>Latest Raw Score: <strong>{latest_raw}%</strong> | Smoothed Avg: <strong>{latest_avg}%</strong></p>
      <p>Camera Active: <strong>{stream.is_active()}</strong> | Total Frames: <strong>{stream.get_frame_count()}</strong></p>
    </div>
    </body>
  </html>
  """
    return HTMLResponse(content=page)


CALIBRATION_DIR = "uploads/calibration"
# Run history: inputs/ (source videos) and outputs/ (one <run_id>_run.json per
# run + the clips it cut).  See calibration_runs.py.
calibration_store = CalibrationStore(CALIBRATION_DIR)


def _yt_match_filter(info_dict, *, incomplete=False):
    """Reject anything longer than MAX_YT_DURATION_S (and live/unknown length)."""
    duration = info_dict.get("duration")
    if duration is None:
        return "Skipping: unknown duration (live stream?)"
    if duration > MAX_YT_DURATION_S:
        return f"Skipping: video is {duration}s, limit is {MAX_YT_DURATION_S}s"
    return None


def _calibration_settings(
    target_fps: float,
    spike_threshold: Optional[float],
    alert_threshold: Optional[float],
    history_len: Optional[int],
    pad_seconds: float,
    merge_gap_seconds: float,
) -> dict:
    """Effective settings: anything not sent falls back to the LOADED MODEL's
    thresholds (its sidecar), so a run always records what it actually used."""
    s = {
        "target_fps": float(target_fps),
        "spike_threshold": float(SPIKE_THRESHOLD if spike_threshold is None else spike_threshold),
        "alert_threshold": float(ALERT_THRESHOLD if alert_threshold is None else alert_threshold),
        "history_len": int(SCORE_HISTORY_LEN if history_len is None else history_len),
        "pad_seconds": float(pad_seconds),
        "merge_gap_seconds": float(merge_gap_seconds),
    }
    if not (1.0 <= s["target_fps"] <= 60.0):
        raise HTTPException(status_code=400, detail="target_fps must be between 1 and 60.")
    for k in ("spike_threshold", "alert_threshold"):
        if not (0.0 < s[k] <= 1.0):
            raise HTTPException(status_code=400, detail=f"{k} must be in (0, 1].")
    if not (1 <= s["history_len"] <= 20):
        raise HTTPException(status_code=400, detail="history_len must be between 1 and 20.")
    if not (0.0 <= s["pad_seconds"] <= 30.0) or not (0.0 <= s["merge_gap_seconds"] <= 30.0):
        raise HTTPException(status_code=400, detail="pad_seconds and merge_gap_seconds must be 0-30 s.")
    return s


async def _calibrate_and_record(video_path: str, source: dict, settings: dict) -> dict:
    """Run the live pipeline over a video with the loaded model, cut a clip per
    incident into outputs/, save the run record, and return it."""
    add_log(f"Starting calibration analysis on {video_path} with {MODEL_VERSION}", kind="calibration")
    try:
        events, per_sample_log, native_fps = await asyncio.to_thread(
            process_video,
            video_path,
            model,
            device,
            target_fps=settings["target_fps"],
            spike_threshold=settings["spike_threshold"],
            alert_threshold=settings["alert_threshold"],
            history_len=settings["history_len"],
            pad_seconds=settings["pad_seconds"],
            merge_gap_seconds=settings["merge_gap_seconds"],
            thresholds=THRESHOLDS,
        )
    except Exception as e:
        add_log(f"Calibration processing failed: {e}", level="error", kind="calibration")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    run_id = CalibrationStore.new_run_id(video_path, MODEL_INFO["name"], MODEL_INFO["sha256_12"])
    clips = []
    for i, (s, e, peak, trig) in enumerate(events, 1):
        out_name = CalibrationStore.clip_name(run_id, i, trig, s, e, peak)
        out_path = os.path.join(calibration_store.outputs, out_name)
        try:
            await asyncio.to_thread(cut_clip, video_path, out_path, s, e)
            clips.append({
                "start_sec": round(s, 2),
                "end_sec": round(e, 2),
                "peak_score": round(peak, 4),
                "trigger": trig,
                "url": f"/calibration/clip/{out_name}",
            })
        except Exception as exc:
            add_log(f"ffmpeg failed for {s:.1f}s-{e:.1f}s: {exc}", level="error", kind="calibration")

    record = build_calibration_record(
        run_id=run_id,
        video_path=video_path,
        source=source,
        model_info=MODEL_INFO,
        settings=settings,
        native_fps=native_fps,
        events=clips,
        per_sample_log=per_sample_log,
    )
    try:
        calibration_store.save(record)
    except Exception as exc:  # the result is still returned; only history misses it
        add_log(f"Could not save calibration run {run_id}: {exc}", level="error", kind="calibration")
    add_log(f"Calibration complete for {video_path}. {len(clips)} events (run {run_id}).", kind="calibration")
    loaded = calibration_store.load(run_id)
    return loaded if loaded is not None else record


@app.post("/calibration/run")
async def run_calibration(
    video_file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    target_fps: float = Form(CAMERA_TARGET_FPS),
    spike_threshold: Optional[float] = Form(None),
    alert_threshold: Optional[float] = Form(None),
    history_len: Optional[int] = Form(None),
    pad_seconds: float = Form(2.0),
    merge_gap_seconds: float = Form(1.5),
):
    """Calibrate a new video (upload or YouTube) with the loaded model.
    Thresholds not sent default to the model's own (sidecar) thresholds."""
    if not video_file and not youtube_url:
        raise HTTPException(status_code=400, detail="Must provide video_file or youtube_url.")
    settings = _calibration_settings(target_fps, spike_threshold, alert_threshold, history_len,
                                     pad_seconds, merge_gap_seconds)

    video_path = None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if youtube_url:
        if not re.match(r"^https?://[^\s]+$", youtube_url.strip(), re.I):
            raise HTTPException(status_code=400, detail="youtube_url must be an http(s) URL.")
        youtube_url = youtube_url.strip()
        add_log(f"Downloading YouTube video: {youtube_url}", kind="calibration")
        ydl_opts = {
            'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            'outtmpl': os.path.join(calibration_store.inputs, f'yt_{stamp}_%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
            'noplaylist': True,
            # F2: TLS verification stays ON; bound size and duration.
            'max_filesize': 500 * 1024 * 1024,
            'match_filter': _yt_match_filter,
        }
        try:
            # Need to run yt-dlp in a thread to avoid blocking event loop
            def download_yt():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=True)
                    if info is None:
                        raise RuntimeError("Download rejected (video too long, too large or unavailable).")
                    target = ydl.prepare_filename(info)
                    if os.path.exists(target):
                        return target
                    mp4_target = os.path.splitext(target)[0] + '.mp4'
                    if os.path.exists(mp4_target):
                        return mp4_target
                    base = os.path.basename(target).split('.')[0]
                    for f in os.listdir(calibration_store.inputs):
                        if base in f:
                            return os.path.join(calibration_store.inputs, f)
                    raise RuntimeError("Download rejected (video too long, too large or unavailable).")
            video_path = await asyncio.to_thread(download_yt)
        except Exception as e:
            add_log(f"YouTube download failed: {e}", level="error", kind="calibration")
            raise HTTPException(status_code=400, detail=f"YouTube download failed: {e}")
        source = {"kind": "youtube", "url": youtube_url}
    else:
        name = os.path.basename(video_file.filename or "clip.mp4")
        if not is_video_filename(name):
            raise HTTPException(status_code=400, detail="Upload an mp4/avi/mov/mkv/webm video.")
        video_path = calibration_store.new_input_path(name, stamp)
        await _save_upload(video_file, video_path)
        source = {"kind": "upload", "original_name": name}

    return await _calibrate_and_record(video_path, source, settings)


@app.post("/calibration/rerun")
async def rerun_calibration(
    video_name: str = Form(...),
    rerun_of: Optional[str] = Form(None),
    target_fps: float = Form(CAMERA_TARGET_FPS),
    spike_threshold: Optional[float] = Form(None),
    alert_threshold: Optional[float] = Form(None),
    history_len: Optional[int] = Form(None),
    pad_seconds: float = Form(2.0),
    merge_gap_seconds: float = Form(1.5),
):
    """Re-calibrate a video that is already on the server (by file name) with
    the loaded model and new settings: compare thresholds, or new weights after
    a restart, without uploading again.  A new run is recorded; the old one stays."""
    video_path = calibration_store.find_input(video_name)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Source video not found (it may have been removed by retention).")
    if rerun_of is not None and not is_run_id(rerun_of):
        raise HTTPException(status_code=400, detail="Invalid rerun_of run id.")
    settings = _calibration_settings(target_fps, spike_threshold, alert_threshold, history_len,
                                     pad_seconds, merge_gap_seconds)
    return await _calibrate_and_record(video_path, {"kind": "rerun", "rerun_of": rerun_of}, settings)


@app.get("/calibration/runs")
def list_calibration_runs():
    """Every recorded run, newest first (summaries; open one for its details)."""
    return {"runs": calibration_store.list()}


@app.get("/calibration/runs/{run_id}")
def get_calibration_run(run_id: str):
    if not is_run_id(run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    try:
        rec = calibration_store.load(run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Run record is unreadable: {e}")
    if rec is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return rec


@app.api_route("/calibration/clip/{filename}", methods=["GET", "HEAD"])
def get_calibration_clip(filename: str):
    """An incident clip (outputs/) or a source video (inputs/, or the old flat folder)."""
    if not is_clip_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid clip filename.")
    path = calibration_store.find_clip(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Clip file not found (it may have been removed by retention).")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )
