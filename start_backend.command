#!/bin/bash
# Double-click this file, or run: open start_backend.command
# Must run in Terminal.app so macOS Local Network access reaches your phone.

cd "$(dirname "$0")"
source .venv/bin/activate

echo "=============================================="
echo " SafeSpace backend"
echo " Open the dashboard at http://localhost:3000"
echo " Connect phone AFTER this server is running."
echo "=============================================="
echo ""

# Quick phone reachability check (optional IP from device_config)
if [[ -f device_config.json ]]; then
  IP=$(python -c "import json; print(json.load(open('device_config.json')).get('ip') or '')" 2>/dev/null || true)
  PORT=$(python -c "import json; print(json.load(open('device_config.json')).get('port') or 8080)" 2>/dev/null || true)
  if [[ -n "$IP" ]]; then
    echo "Testing phone http://$IP:$PORT/shot.jpg ..."
    if curl -sS -o /tmp/safespace_probe.jpg --connect-timeout 4 --max-time 6 -A "Mozilla/5.0" "http://$IP:$PORT/shot.jpg"; then
      echo "Phone HTTP OK ($(wc -c </tmp/safespace_probe.jpg) bytes)."
    else
      echo "WARNING: phone not reachable from Terminal yet."
      echo "Start IP Webcam, close other browser tabs to the phone, retry Connect."
    fi
    echo ""
  fi
fi

exec uvicorn main:app --reload --host 0.0.0.0 --port 8000
