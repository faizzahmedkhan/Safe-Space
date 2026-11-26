"""Disk retention for evidence/uploads/calibration (pure stdlib)."""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable, List, Optional, Sequence, Set, Union

log = logging.getLogger("safespace.retention")


def _env_num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def retention_budgets() -> dict:
    """Env-configurable (MB, days).  A value <= 0 disables that limit."""
    return {
        "evidence": (_env_num("SAFESPACE_EVIDENCE_MAX_MB", 5000), _env_num("SAFESPACE_EVIDENCE_MAX_DAYS", 30)),
        "uploads": (_env_num("SAFESPACE_UPLOADS_MAX_MB", 2000), _env_num("SAFESPACE_UPLOADS_MAX_DAYS", 7)),
        # Calibration runs are kept as history (compare models and thresholds
        # over time), so by default only a size cap applies, oldest first, and
        # run records (*_run.json) are never deleted (see main.run_retention).
        "calibration": (_env_num("SAFESPACE_CALIBRATION_MAX_MB", 2000), _env_num("SAFESPACE_CALIBRATION_MAX_DAYS", 0)),
    }


def prune_dir(
    directory: Union[str, Sequence[str]],
    max_mb: float,
    max_days: float,
    protected_names: Optional[Iterable[str]] = None,
    min_age_s: float = 600.0,
    now: Optional[float] = None,
) -> List[str]:
    """Delete files (top level only) older than max_days, then oldest-first
    until the total is within max_mb.  Files named in `protected_names` and
    files modified within `min_age_s` are never deleted.  Returns removed
    basenames.  `directory` may be a list: its folders then share ONE budget
    (oldest file first across all of them)."""
    dirs = [directory] if isinstance(directory, str) else list(directory)
    now = time.time() if now is None else now
    protected: Set[str] = set(protected_names or ())
    entries = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            entries.append((st.st_mtime, name, st.st_size, p))
    if not entries:
        return []
    entries.sort()  # oldest first
    removed: List[str] = []

    def deletable(mtime: float, name: str) -> bool:
        return name not in protected and (now - mtime) >= min_age_s

    def remove(entry) -> bool:
        try:
            os.remove(entry[3])
            removed.append(entry[1])
            return True
        except OSError as e:
            log.warning("retention: could not delete %s: %s", entry[3], e)
            return False

    remaining = []
    for e in entries:
        if max_days > 0 and (now - e[0]) > max_days * 86400 and deletable(e[0], e[1]):
            if remove(e):
                continue
        remaining.append(e)

    if max_mb > 0:
        total = sum(e[2] for e in remaining)
        limit = max_mb * 1024 * 1024
        for e in remaining:
            if total <= limit:
                break
            if deletable(e[0], e[1]) and remove(e):
                total -= e[2]
    return removed
