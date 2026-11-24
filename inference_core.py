"""Single source of truth for model loading, preprocessing and thresholds.

Everything that scores video (live watcher in main.py, /analyze/*,
video_calibration.py, train.py) goes through this module so the model always
sees the same input the training pipeline produced:

  * frames from frame_prep.prepare_for_model (HOG off, no person-box cache)
  * sampled on a 15 fps timeline (clip_sampler)
  * scaled by /255 ONLY (no Kinetics mean/std), layout (C, T, H, W)
  * decisions made by incident.IncidentTracker with the thresholds below
    (or the ones stored in the model's JSON sidecar)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models

from clip_sampler import ClipWindow, FpsResampler, select_clip_indices  # noqa: F401
from incident import IncidentTracker, default_thresholds, make_tracker  # noqa: F401  (re-exported)
from thresholds import (  # noqa: F401  (re-exported)
    ALERT_COOLDOWN_SECONDS,
    ALERT_THRESHOLD,
    CAMERA_TARGET_FPS,
    CLIP_FRACTION,
    CLIP_LENGTH,
    DEGRADED_FPS,
    INFERENCE_STRIDE,
    MAX_FRAME_GAP_S,
    MAX_INCIDENT_SECONDS,
    MIN_INCIDENT_SECONDS,
    POST_ROLL_SECONDS,
    SCORE_HISTORY_LEN,
    SPIKE_REQUIRES_CONFIRMATION,
    SPIKE_THRESHOLD,
    SUSTAINED_CONSECUTIVE,
)

log = logging.getLogger("safespace.core")

MODEL_ENV = "SAFESPACE_MODEL"
MODEL_CANDIDATES = [
    "safespace_bullying_model_v3_weaponized.pth",
    "safespace_bullying_model_smart_crop.pth",
    "safespace_bullying_model.pth",
]

# One lock for every forward pass in the process (live watcher, uploads,
# calibration, re-analysis all share one model object; PyTorch modules are not
# guaranteed re-entrant while eval/no_grad state is shared).
FORWARD_LOCK = threading.Lock()


# ---------------------------------------------------------------- preprocessing
def preprocess_clip(frames_rgb_112: List[np.ndarray]) -> torch.Tensor:
    """List of T uint8 RGB frames (H, W, 3) -> float tensor (C, T, H, W) in
    [0, 1].  /255 only: the training notebook never applies Kinetics
    mean/std, so doing so here would silently put every prediction
    out-of-distribution."""
    return preprocess_batch(np.stack(frames_rgb_112)[None])[0]


def preprocess_batch(clips_uint8: np.ndarray) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 RGB clips -> float tensor (B, C, T, H, W) in
    [0, 1].  The one implementation behind preprocess_clip, the training
    datasets and batched evaluation (clip_eval), so they cannot drift."""
    arr = np.ascontiguousarray(clips_uint8)  # flipped / memmap views are fine
    if not arr.flags.writeable:  # torch.from_numpy warns on read-only memory
        arr = arr.copy()
    if arr.ndim != 5 or arr.shape[-1] != 3:
        raise ValueError(f"expected (B, T, H, W, 3) clips, got shape {arr.shape}")
    tensor = torch.from_numpy(arr).float()  # (B, T, H, W, C)
    tensor.div_(255.0)
    return tensor.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, H, W)


def predict_violence(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) on the model's device -> (B,) probability of class 1
    (violent).  Serialised by FORWARD_LOCK."""
    with FORWARD_LOCK:
        with torch.no_grad():
            probs = F.softmax(model(batch), dim=1)
    return probs[:, 1].detach().cpu()


# ---------------------------------------------------------------- model loading
def build_model() -> nn.Module:
    model = video_models.r3d_18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def resolve_model_path(path: Optional[str] = None) -> str:
    if path:
        return path
    env = os.environ.get(MODEL_ENV, "").strip()
    if env:
        return env
    for cand in MODEL_CANDIDATES:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "No model weights found. Set SAFESPACE_MODEL=<path to .pth> or place one of "
        f"{', '.join(MODEL_CANDIDATES)} in the working directory (git lfs pull if the "
        "weights are stored in LFS)."
    )


def file_sha256_12(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _load_sidecar(path: str) -> Tuple[Dict[str, Any], Optional[str]]:
    sidecar = os.path.splitext(path)[0] + ".json"
    if not os.path.isfile(sidecar):
        return {}, None
    try:
        with open(sidecar, "r") as f:
            data = json.load(f)
        raw = data.get("thresholds", data)
        if not isinstance(raw, dict):
            return {}, sidecar
        allowed = default_thresholds()
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            if k == "history_len":
                k = "score_history_len"
            if k in allowed and isinstance(v, (int, float, bool)):
                out[k] = v
        return out, sidecar
    except Exception as e:
        log.warning("Could not read model sidecar %s: %s", sidecar, e)
        return {}, sidecar


def load_model(
    path: Optional[str] = None, device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load weights (env SAFESPACE_MODEL, else first existing candidate).
    Returns (model, info); info has path/name/sha256_12/version/thresholds."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = resolve_model_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Model file '{resolved}' not found. Run 'git lfs pull' or set SAFESPACE_MODEL."
        )
    sha = file_sha256_12(resolved)
    model = build_model()
    state = torch.load(resolved, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    sidecar_vals, sidecar_path = _load_sidecar(resolved)
    thresholds = {**default_thresholds(), **sidecar_vals}
    stem = os.path.splitext(os.path.basename(resolved))[0]
    info = {
        "path": resolved,
        "name": stem,
        "sha256_12": sha,
        "version": f"{stem}@{sha}",
        "sidecar": sidecar_path,
        "thresholds": thresholds,
        "thresholds_from_sidecar": bool(sidecar_vals),
        "device": str(device),
    }
    log.info(
        "Loaded model %s sha256=%s on %s (thresholds %s)",
        resolved, sha, device,
        "from sidecar " + str(sidecar_path) if sidecar_vals else "default",
    )
    return model, info
