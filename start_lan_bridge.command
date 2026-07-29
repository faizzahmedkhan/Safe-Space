#!/bin/bash
# Keep this Terminal window open — it gives SafeSpace Local Network access.
cd "$(dirname "$0")"
source .venv/bin/activate
exec python lan_bridge.py
