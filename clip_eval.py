"""Clip-level evaluation that matches how SafeSpace is actually deployed.

The old validation scored only the FIRST 16-frame window of each clip at a
0.5 threshold.  Live mode instead scores a window every INFERENCE_STRIDE
frames, and raises an alert through IncidentTracker (a spike >= 0.90
confirmed by the next window >= 0.67, or a 4-window rolling average >= 0.67
twice in a row).  A checkpoint chosen by first-window accuracy is not
necessarily the checkpoint that alerts best.  This module evaluates each
clip the way live mode would see it:

  1. windows = clip_sampler.scoring_schedule over the cached 15 fps timeline
     (the same windows video_calibration.analyze_windows scores);
  2. every window is scored in batches (score_videos, the only torch part);
  3. the window scores are replayed through the real IncidentTracker
     (simulate_live) -> would_alert, exactly /analyze/video's verdict;
  4. clip scores for threshold-free metrics: first / max / mean window and
     the peak rolling average (peak_smoothed).

summarize() reports PR-AUC and ROC-AUC per clip score, Dangerous recall at a
fixed Normal recall (i.e. at a fixed false-alarm rate), and the live
decision's confusion matrix.  SELECTION_METRICS names the numbers a training
run may select its best epoch by (higher is better for all of them).

Default selection is "pr_auc_max": threshold-free (thresholds are recalibrated
per model afterwards with video_calibration.py, so selecting on today's
0.67/0.90 would conflate model quality with threshold fit) and shaped like
the spike path, which is what opens most incidents.

Metric code is numpy-only (no sklearn) so it runs in the backend venv and in
unit tests without extra dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from incident import make_tracker

CLIP_SCORES = ("first_p", "max_p", "mean_p", "peak_smoothed")
DEFAULT_TARGET_NORMAL_RECALL = 0.90


# ---------------------------------------------------------------- data
@dataclass
class VideoScores:
    """Every scored window of one clip, in time order."""

    path: str
    label: int  # 0 = Normal, 1 = Dangerous
    cls: str  # raw class name (e.g. Normal / Violence / Weaponized)
    t_end: np.ndarray  # (W,) seconds
    raw: np.ndarray  # (W,) P(Dangerous) per window
    suitable: np.ndarray  # (W,) bool, clip-level flag per window
    global_motion: np.ndarray  # (W,) bool
    segment: np.ndarray  # (W,) int, bumps after a frame gap
    short: bool = False  # clip shorter than one window: fallback window scored

    @property
    def n_windows(self) -> int:
        return int(len(self.raw))


@dataclass
class EvalResult:
    rows: List[Dict[str, Any]]  # one dict per clip (see clip_row)
    summary: Dict[str, Any]  # see summarize
    scores: List[VideoScores]


# ---------------------------------------------------------------- live replay
def simulate_live(vs: VideoScores, thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Replay window scores through IncidentTracker like analyze_windows does
    (degraded=False: a file never has a slow camera)."""
    tracker = make_tracker(thresholds)
    n_incidents = 0
    first_trigger: Optional[str] = None
    first_alert_t: Optional[float] = None
    peak_smoothed = 0.0
    prev_segment: Optional[int] = None
    t = 0.0
    for t, raw, suit, glob, seg in zip(
        vs.t_end.tolist(), vs.raw.tolist(), vs.suitable.tolist(),
        vs.global_motion.tolist(), vs.segment.tolist(),
    ):
        if prev_segment is not None and seg != prev_segment:
            tracker.reset(t, reason="gap")
        prev_segment = seg
        for ev in tracker.update(t, raw, suitable=suit, degraded=False, global_motion=glob):
            if ev["type"] == "open":
                n_incidents += 1
                if first_trigger is None:
                    first_trigger, first_alert_t = ev["trigger"], t
        peak_smoothed = max(peak_smoothed, tracker.average() or 0.0)
    tracker.finish(t)
    return {
        "would_alert": n_incidents > 0,
        "n_incidents": n_incidents,
        "trigger": first_trigger or "",
        "first_alert_t": first_alert_t,
        "peak_smoothed": peak_smoothed,
    }


def clip_row(vs: VideoScores, thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    live = simulate_live(vs, thresholds)
    raw = vs.raw
    return {
        "path": vs.path,
        "class": vs.cls,
        "y_true": int(vs.label),
        "n_windows": vs.n_windows,
        "short": bool(vs.short),
        "first_p": float(raw[0]),
        "max_p": float(raw.max()),
        "mean_p": float(raw.mean()),
        "peak_smoothed": float(live["peak_smoothed"]),
        "would_alert": bool(live["would_alert"]),
        "n_incidents": int(live["n_incidents"]),
        "trigger": live["trigger"],
        "suitable_frac": float(np.mean(vs.suitable)),
    }


# ---------------------------------------------------------------- metrics
def _as_arrays(y: Sequence[int], s: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same length")
    return y, s


def average_precision(y: Sequence[int], s: Sequence[float]) -> float:
    """PR-AUC as sklearn.metrics.average_precision_score (step-wise, ties
    grouped).  nan when there are no positives."""
    y, s = _as_arrays(y, s)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    last_of_group = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[last_of_group]
    predicted = last_of_group + 1
    precision = tps / predicted
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y: Sequence[int], s: Sequence[float]) -> float:
    """ROC-AUC via the rank-sum statistic (ties get average ranks).  nan when
    one class is missing."""
    y, s = _as_arrays(y, s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    _, inverse, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    avg_rank = starts + (counts + 1) / 2.0  # 1-based average rank per value
    ranks = np.empty(len(s), np.float64)
    ranks[order] = avg_rank[inverse]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def recall_at_normal_recall(
    y: Sequence[int], s: Sequence[float], target: float = DEFAULT_TARGET_NORMAL_RECALL
) -> Dict[str, float]:
    """Best Dangerous recall while keeping Normal recall >= target (i.e. at
    most 1 - target of normal clips flagged).  A clip is flagged when its
    score is strictly greater than `threshold`."""
    y, s = _as_arrays(y, s)
    neg = np.sort(s[y == 0])
    pos = s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return {"dangerous_recall": float("nan"), "normal_recall": float("nan"),
                "threshold": float("nan")}
    k = int(math.ceil(target * len(neg) - 1e-9))
    thr = -math.inf if k <= 0 else float(neg[min(k, len(neg)) - 1])
    return {
        "dangerous_recall": float(np.mean(pos > thr)),
        "normal_recall": float(np.mean(neg <= thr)),
        "threshold": thr,
    }


def decision_metrics(y: Sequence[int], pred: Sequence[int]) -> Dict[str, float]:
    """Confusion matrix + recalls/precision/macro-F1 for hard decisions."""
    y, p = _as_arrays(y, pred)
    p = p.astype(np.int64)
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    def f1(prec: float, rec: float) -> float:
        if math.isnan(prec) or math.isnan(rec) or prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    d_prec, d_rec = ratio(tp, tp + fp), ratio(tp, tp + fn)
    n_prec, n_rec = ratio(tn, tn + fn), ratio(tn, tn + fp)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": ratio(tp + tn, len(y)),
        "normal_recall": n_rec,
        "dangerous_recall": d_rec,
        "dangerous_precision": d_prec,
        "false_alarm_rate": ratio(fp, tn + fp),
        "macro_f1": (f1(d_prec, d_rec) + f1(n_prec, n_rec)) / 2.0,
    }


def summarize(
    rows: Sequence[Dict[str, Any]], target_normal_recall: float = DEFAULT_TARGET_NORMAL_RECALL
) -> Dict[str, Any]:
    y = np.array([r["y_true"] for r in rows], np.int64)
    out: Dict[str, Any] = {
        "n_clips": int(len(rows)),
        "n_dangerous": int(y.sum()),
        "n_short_clips": int(sum(1 for r in rows if r.get("short"))),
        "target_normal_recall": target_normal_recall,
        "scores": {},
    }
    for name in CLIP_SCORES:
        s = np.array([r[name] for r in rows], np.float64)
        at = recall_at_normal_recall(y, s, target_normal_recall)
        out["scores"][name] = {
            "pr_auc": average_precision(y, s),
            "roc_auc": roc_auc(y, s),
            "dangerous_recall_at_target": at["dangerous_recall"],
            "threshold_at_target": at["threshold"],
        }
    out["live"] = decision_metrics(y, [int(r["would_alert"]) for r in rows])
    by_cls: Dict[str, List[int]] = {}
    for r in rows:
        by_cls.setdefault(r["class"], []).append(int(r["would_alert"]))
    out["live_alert_rate_by_class"] = {c: float(np.mean(v)) for c, v in sorted(by_cls.items())}
    return out


# Higher is better for every entry.
SELECTION_METRICS: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "pr_auc_max": lambda m: m["scores"]["max_p"]["pr_auc"],
    "pr_auc_peak_smoothed": lambda m: m["scores"]["peak_smoothed"]["pr_auc"],
    "roc_auc_max": lambda m: m["scores"]["max_p"]["roc_auc"],
    "recall_at_target_max": lambda m: m["scores"]["max_p"]["dangerous_recall_at_target"],
    "live_macro_f1": lambda m: m["live"]["macro_f1"],
}
DEFAULT_SELECTION_METRIC = "pr_auc_max"


def selection_value(summary: Dict[str, Any], metric: str = DEFAULT_SELECTION_METRIC) -> float:
    """The number to maximise; nan (e.g. a class missing) becomes -inf so it
    never wins."""
    if metric not in SELECTION_METRICS:
        raise KeyError(f"unknown selection metric {metric!r}; choose from {sorted(SELECTION_METRICS)}")
    v = float(SELECTION_METRICS[metric](summary))
    return -math.inf if math.isnan(v) else v


def format_summary(summary: Dict[str, Any]) -> str:
    def f(v: float) -> str:
        return "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"

    tgt = summary["target_normal_recall"]
    lines = [
        f"{summary['n_clips']} clips ({summary['n_dangerous']} dangerous, "
        f"{summary['n_short_clips']} shorter than one window)",
        f"{'clip score':<15}{'PR-AUC':>8}{'ROC-AUC':>9}  Dangerous recall @ Normal recall>={tgt:.2f}",
    ]
    for name, m in summary["scores"].items():
        lines.append(
            f"{name:<15}{f(m['pr_auc']):>8}{f(m['roc_auc']):>9}  {f(m['dangerous_recall_at_target'])}"
            f" (score > {f(m['threshold_at_target'])})"
        )
    lv = summary["live"]
    lines.append(
        f"live incident rule: TN={lv['tn']} FP={lv['fp']} FN={lv['fn']} TP={lv['tp']} | "
        f"Normal recall {f(lv['normal_recall'])}  Dangerous recall {f(lv['dangerous_recall'])}  "
        f"macro-F1 {f(lv['macro_f1'])}"
    )
    rates = ", ".join(f"{c}={f(v)}" for c, v in summary["live_alert_rate_by_class"].items())
    lines.append(f"live alert rate by class: {rates}")
    return "\n".join(lines)


# ---------------------------------------------------------------- scoring (torch)
def score_videos(
    model,
    videos: Iterable[Tuple[str, int, str, Any]],
    device,
    batch_size: int = 32,
) -> List[VideoScores]:
    """Score every live-schedule window of every video.

    videos: iterable of (path, label, class_name, CachedVideo); a generator
    that loads each CachedVideo lazily keeps memory flat.  Windows from
    several videos share forward passes.  Cropping and normalization follow
    the model's preprocessing contract (preprocessing.of_model); a video
    cached with another crop mode is an error, not a silent mismatch.
    Leaves the model in eval() mode:
    the caller restores its own train mode (train.py keeps frozen BatchNorm
    in eval, so a blanket model.train() here would be wrong)."""
    import torch  # noqa: F401  (lazy: metric code above stays torch-free)

    import preprocessing
    from inference_core import predict_violence, preprocess_batch

    spec = preprocessing.of_model(model)
    model.eval()
    results: List[VideoScores] = []
    pending: List[Tuple[int, int, np.ndarray]] = []  # (video_idx, window_idx, frames)
    raw_store: List[np.ndarray] = []

    def flush() -> None:
        if not pending:
            return
        batch = preprocess_batch(np.stack([p[2] for p in pending]), normalize=spec.normalize).to(device)
        probs = predict_violence(model, batch).numpy()
        for (vi, wi, _), pr in zip(pending, probs):
            raw_store[vi][wi] = pr
        pending.clear()

    for path, label, cls, video in videos:
        if video.crop_mode != spec.crop:
            raise ValueError(
                f"{path} was cached with crop_mode={video.crop_mode!r} but the model's "
                f"preprocessing is {spec.name} (crop={spec.crop!r}); use a FrameCache with "
                f"crop_mode={spec.crop!r}"
            )
        windows = video.schedule()
        short = not windows
        if short:
            windows = [video.fallback_window()]
        clips = [video.window(w) for w in windows]
        vi = len(results)
        raw_store.append(np.full(len(clips), np.nan, np.float64))
        results.append(
            VideoScores(
                path=path,
                label=int(label),
                cls=str(cls),
                t_end=np.array([c.t_end for c in clips], np.float64),
                raw=raw_store[vi],
                suitable=np.array([c.suitable for c in clips], bool),
                global_motion=np.array([c.global_motion for c in clips], bool),
                segment=np.array([w.segment for w in windows], np.int64),
                short=short,
            )
        )
        for wi, c in enumerate(clips):
            pending.append((vi, wi, c.frames))
            if len(pending) >= batch_size:
                flush()
    flush()
    return results


def evaluate_model(
    model,
    videos: Iterable[Tuple[str, int, str, Any]],
    device,
    thresholds: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
    target_normal_recall: float = DEFAULT_TARGET_NORMAL_RECALL,
) -> EvalResult:
    """score_videos + clip rows + summary in one call."""
    scores = score_videos(model, videos, device, batch_size=batch_size)
    rows = [clip_row(vs, thresholds) for vs in scores]
    return EvalResult(rows=rows, summary=summarize(rows, target_normal_recall), scores=scores)
