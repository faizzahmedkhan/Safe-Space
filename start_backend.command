#!/bin/bash
# Double-click this file, or run: open start_backend.command
# Must run in Terminal.app so macOS Local Network access reaches your phone.
#
# Environment (optional):
#   SAFESPACE_API_KEY       require this key (header X-SafeSpace-Key or ?key=) on every endpoint but GET /
#   SAFESPACE_LAN=1         listen on all interfaces (default: 127.0.0.1 only)
#   SAFESPACE_CORS_ORIGINS  comma-separated dashboard origins (default: localhost:3000)
#   SAFESPACE_MODEL         path to the .pth weights to load

cd "$(dirname "$0")"
source .venv/bin/activate

HOST=127.0.0.1
if [[ "${SAFESPACE_LAN:-0}" == "1" ]]; then
  HOST=0.0.0.0
fi

echo "=============================================="
echo " SafeSpace backend on $HOST:8000"
echo " Open the dashboard at http://localhost:3000"
echo " Connect phone AFTER this server is running."
if [[ "$HOST" == "0.0.0.0" && -z "${SAFESPACE_API_KEY:-}" ]]; then
  echo ""
  echo " WARNING: listening on the LAN with NO API key."
  echo " Anyone on this network can view cameras and evidence."
  echo " Set SAFESPACE_API_KEY=<secret> first."
fi
echo "=============================================="
echo ""

# Camera config check: probe the first saved phone camera (cameras_config.json)
if [[ -f cameras_config.json ]]; then
  read -r IP PORT < <(python - <<'PY' 2>/dev/null
import json
cfg = json.load(open("cameras_config.json"))
for c in cfg.values():
    if c.get("ip") and c.get("source") == "phone":
        print(c["ip"], c.get("port") or 8080)
        break
PY
)
  if [[ -n "${IP:-}" ]]; then
    echo "Testing phone http://$IP:$PORT/shot.jpg ..."
    if curl -sS -o /tmp/safespace_probe.jpg --connect-timeout 4 --max-time 6 -A "Mozilla/5.0" "http://$IP:$PORT/shot.jpg"; then
      echo "Phone HTTP OK ($(wc -c </tmp/safespace_probe.jpg) bytes)."
    else
      echo "WARNING: phone not reachable from Terminal yet."
      echo "Start IP Webcam, close other browser tabs to the phone, retry Connect."
    fi
    echo ""
  else
    echo "cameras_config.json has no saved phone camera. Add one from the dashboard."
    echo ""
  fi
else
  echo "No cameras_config.json yet - add a camera from the dashboard (it is created automatically)."
  echo ""
fi

# No --reload: a code reload would restart the AI watcher and drop live cameras mid-incident.
exec uvicorn main:app --host "$HOST" --port 8000
