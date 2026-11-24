"""Incident state machine shared by live mode, /analyze/* and calibration.

Pure Python (no torch/cv2/numpy) so it can be unit-tested anywhere.

Call `update()` once per scored window.  The tracker returns a list of event
dicts: {"type": "open"|"extend"|"close"|"spike_pending"|"spike_unconfirmed", ...}.

Semantics
- Open on a SPIKE (raw >= spike_threshold, clip not degraded/global-motion/
  unsuitable, confirmed by the next window >= alert_threshold when
  spike_requires_confirmation) or on SUSTAINED (rolling average >= alert
  threshold for `sustained_consecutive` consecutive windows).
- While open, any "hot" window (raw >= spike or average >= alert) extends the
  incident.  It closes after `post_roll_s` of calm, but never before
  `min_s`, and always by `max_s`.
- Degraded / global-motion / unsuitable windows only block OPENING.  They never
  close an active incident: close is evaluated on clean windows only.
- `cooldown_s` separates one incident from the next.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from thresholds import (
    ALERT_COOLDOWN_SECONDS,
    ALERT_THRESHOLD,
    MAX_INCIDENT_SECONDS,
    MIN_INCIDENT_SECONDS,
    POST_ROLL_SECONDS,
    SCORE_HISTORY_LEN,
    SPIKE_REQUIRES_CONFIRMATION,
    SPIKE_THRESHOLD,
    SUSTAINED_CONSECUTIVE,
)


class IncidentTracker:
    def __init__(
        self,
        alert_threshold: float = ALERT_THRESHOLD,
        spike_threshold: float = SPIKE_THRESHOLD,
        history_len: int = SCORE_HISTORY_LEN,
        cooldown_s: float = ALERT_COOLDOWN_SECONDS,
        post_roll_s: float = POST_ROLL_SECONDS,
        min_s: float = MIN_INCIDENT_SECONDS,
        max_s: float = MAX_INCIDENT_SECONDS,
        spike_requires_confirmation: bool = SPIKE_REQUIRES_CONFIRMATION,
        sustained_consecutive: int = SUSTAINED_CONSECUTIVE,
    ):
        self.alert_threshold = alert_threshold
        self.spike_threshold = spike_threshold
        self.cooldown_s = cooldown_s
        self.post_roll_s = post_roll_s
        self.min_s = min_s
        self.max_s = max_s
        self.spike_requires_confirmation = spike_requires_confirmation
        self.sustained_consecutive = sustained_consecutive
        self.history: Deque[float] = deque(maxlen=max(1, int(history_len)))
        self.incident: Optional[Dict[str, Any]] = None
        self._consecutive = 0
        self._pending_spike: Optional[Dict[str, float]] = None
        self._last_close_t: Optional[float] = None

    # ---- introspection -------------------------------------------------
    @property
    def active(self) -> bool:
        return self.incident is not None

    def average(self) -> Optional[float]:
        return sum(self.history) / len(self.history) if self.history else None

    # ---- lifecycle -----------------------------------------------------
    def reset(self, t: Optional[float] = None, reason: str = "reset", keep_incident: bool = False) -> List[dict]:
        """Drop score history/pending state (camera restart, frame gap).  An
        active incident is closed with reason=`reason` (no cooldown), unless
        keep_incident=True: a brief frame gap/grab failure mid-fight must not
        split one incident into several short clips -- the incident then ends
        normally via post-roll, or via the caller's camera-lost check."""
        events: List[dict] = []
        if self.incident is not None and not keep_incident:
            events.append(self._close(t if t is not None else self.incident["last_t"], reason, set_cooldown=False))
        self.history.clear()
        self._consecutive = 0
        self._pending_spike = None
        return events

    def finish(self, t: float) -> List[dict]:
        """End of an offline video: close any open incident."""
        if self.incident is None:
            return []
        return [self._close(t, "end_of_video", set_cooldown=False)]

    # ---- main entry ----------------------------------------------------
    def update(
        self,
        t: float,
        score: float,
        suitable: bool = True,
        degraded: bool = False,
        global_motion: bool = False,
    ) -> List[dict]:
        events: List[dict] = []
        self.history.append(score)
        full = len(self.history) >= self.history.maxlen
        avg = sum(self.history) / len(self.history)
        above = full and avg >= self.alert_threshold
        blocked = degraded or global_motion or not suitable
        spike = score >= self.spike_threshold
        hot = spike or above or (self.incident is not None and avg >= self.alert_threshold)

        # sustained counter (only clean windows count towards opening)
        if above and not blocked:
            self._consecutive += 1
        else:
            self._consecutive = 0

        if self.incident is not None:
            events.extend(self._while_active(t, score, avg, hot, blocked))
            return events

        # ---- no active incident ----
        in_cooldown = (
            self._last_close_t is not None and (t - self._last_close_t) < self.cooldown_s
        )

        # confirmation of a pending spike by THIS window
        if self._pending_spike is not None:
            pend = self._pending_spike
            self._pending_spike = None
            if not in_cooldown and not blocked and score >= self.alert_threshold:
                events.append(
                    self._open(pend["t"], "spike", pend["score"], max(pend["score"], score), t)
                )
                return events
            events.append({"type": "spike_unconfirmed", "t": t, "spike_t": pend["t"], "score": score})

        if in_cooldown or blocked:
            return events

        if spike:
            if self.spike_requires_confirmation:
                self._pending_spike = {"t": t, "score": score}
                events.append({"type": "spike_pending", "t": t, "score": score})
            else:
                events.append(self._open(t, "spike", score, score, t))
            return events

        if self._consecutive >= self.sustained_consecutive:
            events.append(self._open(t, "sustained", avg, score, t))
        return events

    # ---- internals -----------------------------------------------------
    def _open(self, start_t: float, trigger: str, confidence: float, peak: float, now: float) -> dict:
        self.incident = {
            "start_t": start_t,
            "opened_t": now,
            "last_t": now,
            "last_hot_t": now,
            "trigger": trigger,
            "confidence": confidence,
            "peak": peak,
            "n_triggers": 1,
        }
        self._consecutive = 0
        return {"type": "open", "t": now, **self._public()}

    def _public(self) -> dict:
        inc = self.incident or {}
        return {
            "start_t": inc.get("start_t"),
            "trigger": inc.get("trigger"),
            "confidence": inc.get("confidence"),
            "peak": inc.get("peak"),
            "n_triggers": inc.get("n_triggers"),
        }

    def _while_active(self, t: float, score: float, avg: float, hot: bool, blocked: bool) -> List[dict]:
        inc = self.incident
        assert inc is not None
        events: List[dict] = []
        inc["last_t"] = t
        if score > inc["peak"]:
            inc["peak"] = score
        if hot:
            inc["last_hot_t"] = t
            inc["n_triggers"] += 1
            events.append({"type": "extend", "t": t, **self._public()})

        if (t - inc["start_t"]) >= self.max_s:
            events.append(self._close(t, "max_length"))
            return events
        # Blocked windows (unsuitable / degraded / global motion) hold the
        # recording open: never close on evidence we don't trust.
        if blocked:
            return events
        calm_for = t - inc["last_hot_t"]
        if calm_for >= self.post_roll_s and (t - inc["start_t"]) >= self.min_s:
            events.append(self._close(t, "calm"))
        return events

    def _close(self, t: float, reason: str, set_cooldown: bool = True) -> dict:
        inc = self.incident
        assert inc is not None
        self.incident = None
        self._consecutive = 0
        if set_cooldown:
            self._last_close_t = t
        return {
            "type": "close",
            "t": t,
            "reason": reason,
            "end_t": t,
            "duration": t - inc["start_t"],
            **{k: inc[k] for k in ("start_t", "trigger", "confidence", "peak", "n_triggers")},
        }
