"""Calibration run history: where calibration files live and how runs are recorded.

Layout (under uploads/calibration):
    inputs/   the source videos (uploads, YouTube downloads)
    outputs/  per run: <run_id>_run.json + the incident clips it cut,
              named <run_id>_eventNN_<trigger>_<start>s-<end>s_peak<p>.mp4

A run record says exactly what produced it, so runs stay comparable later:
the video, the model weights (name + SHA-256 prefix), the preprocessing
contract, the thresholds and the other settings, plus the events and the
per-window score log.  Runs made before records carried that information
(schema 1, the August "model_type" runs) are still listed and opened; their
model is shown by its old model_type label and their settings as unknown.

Pure Python (no torch / FastAPI) so it can be unit-tested anywhere.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set

SCHEMA = 2
RUN_SUFFIX = "_run.json"
_RUN_ID_RE = re.compile(r"^[\w.\-]{1,200}$")
_VIDEO_RE = re.compile(r"\.(mp4|avi|mov|mkv|webm)$", re.I)


def safe_component(text: str, max_len: int = 80) -> str:
    """Filesystem- and URL-safe piece of a name: letters, digits, _ . -"""
    out = re.sub(r"[^\w.\-]+", "_", text or "").strip("._")
    out = out.replace("..", "_")
    return (out or "x")[:max_len]


def is_run_id(run_id: str) -> bool:
    return bool(run_id) and bool(_RUN_ID_RE.fullmatch(run_id)) and ".." not in run_id


def is_safe_name(name: str) -> bool:
    return bool(name) and os.path.basename(name) == name and ".." not in name and "/" not in name and "\\" not in name


def basename_any(path: str) -> str:
    """Basename of a path written on Windows or POSIX (old records store
    Windows paths with backslashes)."""
    return re.split(r"[\\/]", path or "")[-1]


class CalibrationStore:
    def __init__(self, root: str):
        self.root = root
        self.inputs = os.path.join(root, "inputs")
        self.outputs = os.path.join(root, "outputs")
        os.makedirs(self.inputs, exist_ok=True)
        os.makedirs(self.outputs, exist_ok=True)

    # ------------------------------------------------------------ files
    def new_input_path(self, original_name: str, stamp: str) -> str:
        stem, ext = os.path.splitext(os.path.basename(original_name or "clip.mp4"))
        ext = ext.lower() if _VIDEO_RE.search(ext or "") else ".mp4"
        return os.path.join(self.inputs, f"upload_{stamp}_{safe_component(stem)}{ext}")

    def find_input(self, name: str) -> Optional[str]:
        """An existing source video by file name (inputs/, or the old flat layout)."""
        if not is_safe_name(name) or not _VIDEO_RE.search(name):
            return None
        for d in (self.inputs, self.root):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    def find_clip(self, name: str) -> Optional[str]:
        """A servable video: an event clip (outputs/) or a source video."""
        if not is_safe_name(name) or not _VIDEO_RE.search(name):
            return None
        for d in (self.outputs, self.inputs, self.root):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    # ------------------------------------------------------------ naming
    @staticmethod
    def new_run_id(video_name: str, model_name: str, model_sha: str, now: Optional[datetime] = None) -> str:
        """<video stem>__<model name>-<sha8>__<YYYYmmdd_HHMMSS>: unique per run,
        and readable in a directory listing."""
        now = now or datetime.now()
        stem = safe_component(os.path.splitext(basename_any(video_name))[0], 90)
        model = safe_component(f"{model_name}-{(model_sha or '')[:8]}".rstrip("-"), 60)
        return f"{stem}__{model}__{now:%Y%m%d_%H%M%S}"

    @staticmethod
    def clip_name(run_id: str, index: int, trigger: str, start: float, end: float, peak: float) -> str:
        return (f"{run_id}_event{index:02d}_{safe_component(trigger, 20)}_"
                f"{start:.1f}s-{end:.1f}s_peak{peak:.2f}.mp4")

    def _run_path(self, run_id: str) -> str:
        return os.path.join(self.outputs, f"{run_id}{RUN_SUFFIX}")

    def run_json_names(self) -> Set[str]:
        try:
            return {n for n in os.listdir(self.outputs) if n.endswith(RUN_SUFFIX)}
        except OSError:
            return set()

    # ------------------------------------------------------------ records
    def save(self, record: Dict[str, Any]) -> str:
        run_id = record["run_id"]
        if not is_run_id(run_id):
            raise ValueError(f"invalid run id {run_id!r}")
        path = self._run_path(run_id)
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f)
        os.replace(tmp, path)
        return path

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not is_run_id(run_id):
            return None
        path = self._run_path(run_id)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.normalize(data, run_id, os.path.getmtime(path))

    def list(self) -> List[Dict[str, Any]]:
        """Summaries of every run, newest first.  Unreadable files are skipped."""
        out = []
        for name in self.run_json_names():
            run_id = name[: -len(RUN_SUFFIX)]
            if not is_run_id(run_id):
                continue
            try:
                rec = self.load(run_id)
            except Exception:
                continue
            if rec is not None:
                out.append(self.summary(rec))
        out.sort(key=lambda s: s["created_ts"], reverse=True)
        return out

    def normalize(self, data: Dict[str, Any], run_id: str, mtime: float) -> Dict[str, Any]:
        """Give old and new records the same shape (fields are only added)."""
        rec = dict(data)
        rec["run_id"] = run_id
        rec.setdefault("schema", 1)
        video_name = rec.get("video_name") or basename_any(rec.get("video_path", ""))
        rec["video_name"] = video_name
        rec.setdefault("video_url", f"/calibration/clip/{video_name}" if video_name else None)
        rec["video_available"] = self.find_input(video_name) is not None if video_name else False
        if not rec.get("created_ts"):
            rec["created_ts"] = mtime
        rec.setdefault("created_at", datetime.fromtimestamp(rec["created_ts"]).isoformat(timespec="seconds"))
        if not isinstance(rec.get("model"), dict):
            legacy = rec.get("model_type") or "original"
            rec["model"] = {"version": None, "name": f"legacy: {legacy}", "sha256_12": None,
                            "preprocessing": None, "legacy_model_type": legacy}
        rec.setdefault("settings", None)
        rec.setdefault("events", [])
        rec.setdefault("per_sample_log", [])
        for ev in rec["events"]:
            clip = basename_any(ev.get("url", ""))
            ev["clip_available"] = self.find_clip(clip) is not None if clip else False
        if not isinstance(rec.get("summary"), dict):
            rec["summary"] = summarize_samples(rec["per_sample_log"], rec["events"])
        return rec

    @staticmethod
    def summary(rec: Dict[str, Any]) -> Dict[str, Any]:
        model = rec.get("model") or {}
        settings = rec.get("settings") or {}
        return {
            "run_id": rec["run_id"],
            "schema": rec.get("schema", 1),
            "created_at": rec.get("created_at"),
            "created_ts": rec.get("created_ts", 0),
            "video_name": rec.get("video_name"),
            "video_available": rec.get("video_available", False),
            "source": (rec.get("source") or {}).get("kind"),
            "model_name": model.get("name"),
            "model_version": model.get("version"),
            "preprocessing": model.get("preprocessing"),
            "spike_threshold": settings.get("spike_threshold"),
            "alert_threshold": settings.get("alert_threshold"),
            "history_len": settings.get("history_len"),
            "n_events": len(rec.get("events") or []),
            **{k: (rec.get("summary") or {}).get(k) for k in ("duration_sec", "n_windows", "peak_raw")},
        }


def summarize_samples(per_sample_log: Iterable[Dict[str, Any]], events: Iterable[Any]) -> Dict[str, Any]:
    samples = list(per_sample_log or [])
    raws = [float(s.get("raw_score", 0.0)) for s in samples]
    suit = [bool(s.get("suitable_for_model")) for s in samples if "suitable_for_model" in s]
    return {
        "duration_sec": max((float(s.get("time_sec", 0.0)) for s in samples), default=0.0),
        "n_windows": len(samples),
        "n_events": len(list(events or [])),
        "peak_raw": max(raws, default=0.0),
        "framing_suitable_ratio": (sum(suit) / len(suit)) if suit else None,
    }


def build_record(
    *,
    run_id: str,
    video_path: str,
    source: Dict[str, Any],
    model_info: Dict[str, Any],
    settings: Dict[str, Any],
    native_fps: float,
    events: List[Dict[str, Any]],
    per_sample_log: List[Dict[str, Any]],
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """The schema-2 record saved for every new run."""
    now = time.time() if now is None else now
    spec = model_info.get("preprocessing")
    video_name = basename_any(video_path)
    return {
        "schema": SCHEMA,
        "ok": True,
        "run_id": run_id,
        "created_ts": now,
        "created_at": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        "video_path": video_path.replace("\\", "/"),
        "video_name": video_name,
        "video_url": f"/calibration/clip/{video_name}",
        "source": source,
        "model": {
            "version": model_info.get("version"),
            "name": model_info.get("name"),
            "sha256_12": model_info.get("sha256_12"),
            "preprocessing": getattr(spec, "name", spec),
            "thresholds_from_sidecar": model_info.get("thresholds_from_sidecar"),
        },
        "settings": settings,
        "summary": summarize_samples(per_sample_log, events),
        "native_fps": native_fps,
        "events": events,
        "per_sample_log": per_sample_log,
        "model_version": model_info.get("version"),  # kept for older dashboard code
    }
