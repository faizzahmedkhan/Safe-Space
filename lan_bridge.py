#!/usr/bin/env python3
"""
Local Network bridge for SafeSpace.

Runs outside Cursor (via Terminal.app) so macOS Local Network access works.
Cursor's uvicorn then pulls frames from http://127.0.0.1:8765/shot.jpg
instead of talking to the phone directly.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 8765
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lan_bridge_config.json")
USER_AGENT = "SafeSpace-LAN-Bridge/1.0"

_lock = threading.Lock()
_upstream: Optional[str] = None


def _load_config() -> None:
    global _upstream
    if not os.path.exists(CONFIG_PATH):
        return
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        _upstream = (data.get("upstream") or "").rstrip("/") or None
    except Exception as e:
        print(f"[BRIDGE] Could not load config: {e}", flush=True)


def _save_config(upstream: str) -> None:
    global _upstream
    upstream = upstream.rstrip("/")
    with _lock:
        _upstream = upstream
        with open(CONFIG_PATH, "w") as f:
            json.dump({"upstream": upstream}, f)


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

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/health", "/"):
            with _lock:
                up = _upstream
            payload = json.dumps(
                {"ok": True, "upstream": up, "port": BRIDGE_PORT}
            ).encode()
            self._send(200, payload, "application/json")
            return

        if path != "/shot.jpg":
            self._send(404, b"not found", "text/plain")
            return

        with _lock:
            up = _upstream
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

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._send(400, b"invalid json", "text/plain")
            return

        if path == "/config":
            upstream = (data.get("upstream") or "").strip()
            if not upstream.startswith("http"):
                self._send(400, b"upstream must be http(s) URL", "text/plain")
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
            _save_config(upstream)
            body = json.dumps({"ok": True, "upstream": upstream}).encode()
            self._send(200, body, "application/json")
            return

        self._send(404, b"not found", "text/plain")


def main() -> None:
    _load_config()
    server = ThreadingHTTPServer((BRIDGE_HOST, BRIDGE_PORT), BridgeHandler)
    print(
        f"[BRIDGE] SafeSpace LAN bridge on http://{BRIDGE_HOST}:{BRIDGE_PORT} "
        f"(upstream={_upstream})",
        flush=True,
    )
    print("[BRIDGE] Keep this Terminal window open while using the phone camera.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[BRIDGE] Shutting down.", flush=True)


if __name__ == "__main__":
    main()
