#!/usr/bin/env python3
"""
Local Network bridge for SafeSpace.

Runs outside Cursor (via Terminal.app) so macOS Local Network access works.
The SafeSpace backend then pulls frames from this bridge on localhost instead
of talking to the phone directly.

Per-camera upstreams (E6): each camera has its own upstream phone, served at
    GET  /shot/<camera_id>.jpg
    POST /config   {"upstream": "http://PHONE:8080", "camera_id": "<id>"}
so a second phone can never silently replace camera 1's feed.  The legacy single
endpoint (GET /shot.jpg, POST /config without camera_id) still works.

Hardening (F2): CORS limited to the dashboard origins (SAFESPACE_CORS_ORIGINS),
POST bodies must be application/json, and upstreams must be private LAN hosts
(no loopback/link-local/public addresses), so the bridge cannot be used as an
open proxy.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional
from urllib.parse import urlparse

from validators import is_private_upstream, parse_origins, validate_camera_id

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 8765
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lan_bridge_config.json")
USER_AGENT = "SafeSpace-LAN-Bridge/1.0"
MAX_BODY = 8192
ALLOWED_ORIGINS = parse_origins(
    os.environ.get("SAFESPACE_CORS_ORIGINS"),
    ["http://localhost:3000", "http://127.0.0.1:3000"],
)

_lock = threading.Lock()
_upstream: Optional[str] = None            # legacy single upstream
_cameras: Dict[str, str] = {}              # camera_id -> upstream


def _load_config() -> None:
    global _upstream
    if not os.path.exists(CONFIG_PATH):
        return
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        _upstream = (data.get("upstream") or "").rstrip("/") or None
        cams = data.get("cameras") or {}
        if isinstance(cams, dict):
            for cid, up in cams.items():
                if validate_camera_id(cid) and isinstance(up, str) and up:
                    _cameras[cid] = up.rstrip("/")
    except Exception as e:
        print(f"[BRIDGE] Could not load config: {e}", flush=True)


def _write_config_locked() -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump({"upstream": _upstream, "cameras": _cameras}, f)


def _save_config(upstream: str, camera_id: Optional[str] = None) -> None:
    global _upstream
    upstream = upstream.rstrip("/")
    with _lock:
        if camera_id:
            _cameras[camera_id] = upstream
        else:
            _upstream = upstream
        _write_config_locked()


def _upstream_allowed(upstream: str) -> bool:
    p = urlparse(upstream)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    return is_private_upstream(p.hostname)


def _fetch_shot(upstream: str, timeout: float = 6.0) -> bytes:
    url = upstream.rstrip("/") + "/shot.jpg"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Connection": "close"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("[BRIDGE] " + (fmt % args) + "\n")
        sys.stdout.flush()

    def _cors_origin(self) -> Optional[str]:
        origin = (self.headers.get("Origin") or "").rstrip("/")
        return origin if origin and origin in ALLOWED_ORIGINS else None

    def _cors_headers(self) -> None:
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        if not self._cors_origin():
            self._send(403, b"origin not allowed", "text/plain")
            return
        self.send_response(204)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_shot(self, up: Optional[str]) -> None:
        if not up:
            self._send(503, b"upstream not configured", "text/plain")
            return
        try:
            data = _fetch_shot(up)
            if not data:
                self._send(502, b"empty frame", "text/plain")
                return
            self._send(200, data, "image/jpeg")
        except Exception as e:
            msg = f"upstream fetch failed: {e}".encode()
            self._send(502, msg, "text/plain")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/health", "/"):
            with _lock:
                payload = json.dumps(
                    {"ok": True, "upstream": _upstream, "cameras": sorted(_cameras), "port": BRIDGE_PORT}
                ).encode()
            self._send(200, payload, "application/json")
            return

        if path == "/shot.jpg":
            with _lock:
                up = _upstream
            self._serve_shot(up)
            return

        if path.startswith("/shot/") and path.endswith(".jpg"):
            camera_id = path[len("/shot/"):-len(".jpg")]
            if not validate_camera_id(camera_id):
                self._send(400, b"invalid camera id", "text/plain")
                return
            with _lock:
                up = _cameras.get(camera_id)
            self._serve_shot(up)
            return

        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._send(415, b"Content-Type must be application/json", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self._send(413, b"body too large", "text/plain")
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode() or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, b"invalid json", "text/plain")
            return
        if not isinstance(data, dict):
            self._send(400, b"json object expected", "text/plain")
            return

        if path == "/config":
            upstream = (data.get("upstream") or "").strip()
            camera_id = data.get("camera_id")
            if camera_id is not None and not validate_camera_id(str(camera_id)):
                self._send(400, b"invalid camera_id", "text/plain")
                return
            if not upstream.startswith(("http://", "https://")):
                self._send(400, b"upstream must be http(s) URL", "text/plain")
                return
            if not _upstream_allowed(upstream):
                self._send(403, b"upstream must be a private LAN host (e.g. 192.168.x.x)", "text/plain")
                return
            # Validate we can reach the phone from THIS process (Terminal LAN access)
            try:
                probe = _fetch_shot(upstream, timeout=8.0)
                if not probe:
                    self._send(400, b"upstream returned empty shot.jpg", "text/plain")
                    return
            except Exception as e:
                self._send(
                    400,
                    f"cannot reach upstream from bridge: {e}".encode(),
                    "text/plain",
                )
                return
            _save_config(upstream, str(camera_id) if camera_id else None)
            body = json.dumps({"ok": True, "upstream": upstream, "camera_id": camera_id}).encode()
            self._send(200, body, "application/json")
            return

        self._send(404, b"not found", "text/plain")


def main() -> None:
    _load_config()
    server = ThreadingHTTPServer((BRIDGE_HOST, BRIDGE_PORT), BridgeHandler)
    print(
        f"[BRIDGE] SafeSpace LAN bridge on http://{BRIDGE_HOST}:{BRIDGE_PORT} "
        f"(upstream={_upstream}, cameras={sorted(_cameras)})",
        flush=True,
    )
    print("[BRIDGE] Keep this Terminal window open while using the phone camera.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[BRIDGE] Shutting down.", flush=True)


if __name__ == "__main__":
    main()
