"""Ensure the LAN bridge (Terminal.app) is running for phone camera access from Cursor."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 8765
BRIDGE_BASE = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}"
ROOT = os.path.dirname(os.path.abspath(__file__))
BRIDGE_COMMAND = os.path.join(ROOT, "start_lan_bridge.command")


def bridge_shot_url() -> str:
    return f"{BRIDGE_BASE}/shot.jpg"


def bridge_health() -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{BRIDGE_BASE}/health", timeout=1.5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _configure_upstream(upstream: str) -> Tuple[bool, str]:
    payload = json.dumps({"upstream": upstream.rstrip("/")}).encode()
    req = urllib.request.Request(
        f"{BRIDGE_BASE}/config",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = json.loads(resp.read().decode())
            if body.get("ok"):
                return True, "ok"
            return False, json.dumps(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        return False, detail or str(e)
    except Exception as e:
        return False, str(e)


def _launch_bridge_via_terminal() -> None:
    """Start bridge in Terminal.app so it inherits Local Network permission."""
    if not os.path.exists(BRIDGE_COMMAND):
        raise RuntimeError(f"Missing {BRIDGE_COMMAND}")
    # Ensure executable
    try:
        os.chmod(BRIDGE_COMMAND, 0o755)
    except OSError:
        pass
    subprocess.run(
        ["open", BRIDGE_COMMAND],
        check=False,
        cwd=ROOT,
    )


def ensure_lan_bridge(upstream: str, wait_seconds: float = 20.0) -> Tuple[str, str]:
    """
    Make sure the LAN bridge is up and pointed at the phone.
    Returns (local_shot_url, note).
    """
    upstream = upstream.rstrip("/")
    health = bridge_health()
    if health is None:
        _launch_bridge_via_terminal()
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            health = bridge_health()
            if health is not None:
                break
            time.sleep(0.4)
        if health is None:
            raise RuntimeError(
                "LAN bridge did not start. Allow Terminal if macOS asks, "
                "keep the 'lan_bridge' Terminal window open, then try Connect again."
            )

    ok, detail = _configure_upstream(upstream)
    if not ok:
        raise RuntimeError(
            f"LAN bridge could not reach the phone ({detail}). "
            "Confirm IP Webcam is running and close other browser tabs to the phone."
        )

    # Verify Cursor can pull a frame via localhost
    try:
        with urllib.request.urlopen(bridge_shot_url(), timeout=8) as resp:
            data = resp.read(1024)
            if not data:
                raise RuntimeError("Bridge returned an empty frame")
    except Exception as e:
        raise RuntimeError(f"Bridge is up but localhost fetch failed: {e}") from e

    note = (
        "Using LAN bridge (Terminal.app) so Cursor can access the phone camera. "
        "Keep the bridge Terminal window open."
    )
    return bridge_shot_url(), note
