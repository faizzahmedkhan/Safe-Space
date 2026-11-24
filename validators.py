"""Pure-Python input validators (no third-party imports)."""

from __future__ import annotations

import hmac
import ipaddress
import re
import socket
from typing import Callable, Iterable, Optional

_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_DIGITS_DOTS_RE = re.compile(r"^[\d.]+$")
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
_HOST_RE = re.compile(rf"^{_LABEL}(?:\.{_LABEL})*$")
_PATH_RE = re.compile(r"^/[\w./\-]*$")
_CAMERA_ID_RE = re.compile(r"^[\w\-]{1,64}$")
VIDEO_EXT_RE = re.compile(r"\.(mp4|avi|mov|mkv|webm)$", re.I)
_EVIDENCE_RE = re.compile(r"^[\w.\-]+\.mp4$")
_CLIP_RE = re.compile(r"^[\w.\-]+\.(mp4|avi|mov|mkv|webm)$", re.I)


def validate_host(host: str) -> bool:
    """IPv4 address or DNS hostname.  No scheme, port, path or whitespace."""
    if not isinstance(host, str):
        return False
    if not host or len(host) > 253:
        return False
    if _DIGITS_DOTS_RE.fullmatch(host):
        if not _IPV4_RE.fullmatch(host):
            return False
        return all(0 <= int(p) <= 255 for p in host.split("."))
    return bool(_HOST_RE.fullmatch(host))


def validate_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    return bool(_PATH_RE.fullmatch(path)) and ".." not in path


def validate_camera_id(camera_id: str) -> bool:
    return isinstance(camera_id, str) and bool(_CAMERA_ID_RE.fullmatch(camera_id))


def is_video_filename(name: str) -> bool:
    return bool(VIDEO_EXT_RE.search(name or ""))


def is_safe_basename(name: str) -> bool:
    import os

    return bool(name) and os.path.basename(name) == name and ".." not in name


def is_evidence_filename(name: str) -> bool:
    return is_safe_basename(name) and bool(_EVIDENCE_RE.fullmatch(name))


def is_clip_filename(name: str) -> bool:
    return is_safe_basename(name) and bool(_CLIP_RE.fullmatch(name))


def is_http_url(url: str) -> bool:
    return isinstance(url, str) and bool(re.match(r"^https?://[^\s]+$", url, re.I))


def is_private_address(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return bool(ip.is_private and not ip.is_loopback and not ip.is_link_local and not ip.is_unspecified)


def is_private_upstream(
    host: str,
    resolver: Optional[Callable[[str], Iterable[str]]] = None,
) -> bool:
    """True only if `host` is (or resolves exclusively to) private LAN addresses.
    Loopback/link-local are rejected so the bridge cannot be aimed at local
    services (including the SafeSpace backend itself)."""
    if not validate_host(host):
        return False
    if _IPV4_RE.fullmatch(host):
        return is_private_address(host)
    if resolver is None:
        def resolver(h: str):  # type: ignore[misc]
            return [ai[4][0] for ai in socket.getaddrinfo(h, None)]
    try:
        addrs = list(resolver(host))
    except Exception:
        return False
    return bool(addrs) and all(is_private_address(a) for a in addrs)


def keys_match(provided: Optional[str], expected: str) -> bool:
    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def parse_origins(raw: Optional[str], default: Iterable[str]) -> list:
    if raw is None or not raw.strip():
        return list(default)
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
