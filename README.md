# SafeSpace - AI Safety Monitor (Windows)

SafeSpace is a real-time AI safety monitor for Windows that watches a camera feed (laptop webcam or phone IP Webcam), looks for suspicious physical activity like bullying or fights, and saves evidence clips when an incident is detected.

Stack:

- **Backend:** FastAPI + OpenCV + PyTorch (R3D-18)
- **Dashboard:** Next.js operator UI at `http://localhost:3000`

## Windows Setup

### 1. Clone the repository

```powershell
git clone <repo_url>
cd "Safe-Space"
```

### 2. Get the model weights

The `.pth` weights are large and are **not** stored in normal git history
(`*.pth` is git-ignored; `.gitattributes` marks `*.pth` for Git LFS if you choose to
track them with `git add -f`). Put the weights file in the project root, or use
`git lfs pull` if your remote stores them in LFS.

The backend picks the model like this:

1. `SAFESPACE_MODEL=<path to .pth>` if set, otherwise
2. the first existing file of `safespace_bullying_model_v3_weaponized.pth`,
   `safespace_bullying_model_smart_crop.pth`, `safespace_bullying_model.pth`.

At boot it logs the file's SHA-256 (first 12 chars) and every alert is stamped with
`model_version` (`<file stem>@<sha12>`).

**Thresholds sidecar:** if `<model_stem>.json` exists next to the weights it can carry the
thresholds tuned for that model (any of `alert_threshold`, `spike_threshold`,
`score_history_len`, `alert_cooldown_seconds`, `post_roll_seconds`,
`min_incident_seconds`, `max_incident_seconds`, `spike_requires_confirmation`, either at
the top level or under a `"thresholds"` key). Otherwise the defaults in `thresholds.py` /
`inference_core.py` apply (alert 0.67, spike 0.90). Re-run `video_calibration.py` for each
new model and update its sidecar.

### 3. Create and activate the Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you prefer Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 5. Install dashboard dependencies

```powershell
cd frontend
npm install
cd ..
```

## Running the App

SafeSpace runs as two separate processes: backend and frontend.

### Backend (FastAPI)

From the project root in PowerShell:

```powershell
.\start_backend.ps1          # Windows (binds 127.0.0.1 only)
./start_backend.command      # macOS (run in Terminal.app)
```

or manually: `uvicorn main:app --host 127.0.0.1 --port 8000` (no `--reload`: a reload
restarts the AI watcher and drops live cameras).

The server binds **127.0.0.1 by default**. Set `SAFESPACE_LAN=1` to listen on all
interfaces, and always set `SAFESPACE_API_KEY` when you do.

If `uvicorn` is not found, ensure the virtual environment is activated and retry.

### Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAFESPACE_API_KEY` | unset | If set, every endpoint except `GET /` needs header `X-SafeSpace-Key: <key>` or query `?key=<key>` (the query form is for `<img>`/`<video>` tags). Unset = auth disabled (a warning is logged at startup). |
| `SAFESPACE_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated dashboard origins (also used by the LAN bridge). |
| `SAFESPACE_LAN` | unset | `1` makes the start scripts bind `0.0.0.0` instead of `127.0.0.1`. |
| `SAFESPACE_MODEL` | auto | Path to the `.pth` weights (see above). |
| `SAFESPACE_MAX_UPLOAD_MB` | `500` | Upload cap for `/analyze/video` and `/calibration/run` (HTTP 413 above it). |
| `SAFESPACE_LOG_DIR` | `logs` | Rotating log file `safespace.log` (5 MB x 5). |
| `SAFESPACE_EVIDENCE_MAX_MB` / `_MAX_DAYS` | `5000` / `30` | Retention for `evidence/`. |
| `SAFESPACE_UPLOADS_MAX_MB` / `_MAX_DAYS` | `2000` / `7` | Retention for `uploads/`. |
| `SAFESPACE_CALIBRATION_MAX_MB` / `_MAX_DAYS` | `2000` / `7` | Retention for `uploads/calibration/`. |

### Authentication

With `SAFESPACE_API_KEY` set, the dashboard must send `X-SafeSpace-Key` on API calls and
`?key=` on `<img>`/`<video>` URLs (feed, evidence, calibration clips). Requests without a
valid key get `401`. `GET /` (health: includes `model_version`, `auth_enabled`,
`watcher_last_cycle_age_ms`) stays open.

### Retention

At startup and then hourly, a retention job prunes `evidence/`, `uploads/` and
`uploads/calibration/` by age and total size (budgets above; set a value to `0` to disable
that limit). It never deletes evidence of incidents that are still recording/processing,
files that were copied into `data/safe` or `data/unsafe`, or files younger than 10 minutes.
Alerts whose clip was pruned keep their record with `evidence_pruned: true` and no
`evidence_url`.

### How incidents work

One fight = one alert. An incident **opens** on a confirmed spike (a window >= 0.90 that
is confirmed by the next window >= 0.67) or on a sustained average (2 consecutive windows
with the 4-window average >= 0.67). It stays open while scores re-trigger, and **closes**
after 5 s of calm (min 10 s, max 120 s), then a 5 s cooldown applies. Alerts carry
`status` (`recording` -> `processing` -> `ready` / `failed`), `trigger`, `peak`,
`n_triggers`, `end_timestamp` and `model_version`. Degraded cameras (< 8 real fps),
global-motion clips (shake / lighting changes) and unsuitable framing only **block opening**
new incidents; they never stop an active recording.

### Dashboard (Next.js)

Open a second terminal, then:

```powershell
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

### Useful backend endpoints

- `http://127.0.0.1:8000/` — health (open even when an API key is set)
- `http://127.0.0.1:8000/cameras` — per-camera diagnostics JSON
- `http://127.0.0.1:8000/alerts` — recent alerts
- `http://127.0.0.1:8000/logs` — system logs
- `http://127.0.0.1:8000/cameras/<id>/preview` — legacy HTML preview

## Phone Camera (IP Webcam) Setup

1. Put the **phone and PC on the same Wi-Fi**.
2. Install **IP Webcam** (Android) or an equivalent IP camera app on the phone.
3. Start the stream server in the app and note the phone's IP address (for example, `192.168.1.5`).
4. Close any browser tabs already viewing `http://PHONE_IP:8080`, since many phone camera apps only allow one client.
5. In the SafeSpace dashboard:
   - Enter the phone **IP address**
   - Use port `8080` and path `/video` for IP Webcam by default
   - Optionally enter a **Wi-Fi name** label for display
   - Click **Connect phone**

If the phone connection fails, Windows firewall or phone app access can be the issue. Allow Python/PowerShell or the browser through the firewall, and try again.

### Quick test without a phone

Use the laptop webcam option in the dashboard if no phone stream is available.

## Training & Fine-Tuning

To fine-tune the model using custom videos:

1. Prepare your training folders:
   - `data/safe/` → short `.mp4` videos of normal behavior
   - `data/unsafe/` → short `.mp4` videos of fights/incidents
     _(Both folders are required. Do not train with only one class.)_

2. Run the training script:

```powershell
python train.py
```

`train.py` fine-tunes the currently selected model (layer4 at 1e-5, fc at 1e-3, earlier
layers frozen, sqrt class weights, 15% validation hold-out, fixed seed) using the same
frame preparation as live mode. It **never overwrites** weights: it writes
`safespace_bullying_model_ft_<timestamp>.pth` plus a `.json` sidecar.

3. Restart the backend with the new weights, then re-run `video_calibration.py` on some
   footage and adjust the sidecar thresholds if needed:

```powershell
$env:SAFESPACE_MODEL = "safespace_bullying_model_ft_<timestamp>.pth"
.\start_backend.ps1
```

## API Overview

| Method | Path                                | Purpose                                                |
| ------ | ----------------------------------- | ------------------------------------------------------ |
| GET    | `/`                                 | Health (open): model version, watcher heartbeat, auth |
| GET    | `/cameras`                          | All cameras: fps, degraded/frozen/offline, coverage    |
| POST   | `/cameras/{id}/connect`             | `{ ip, port?, path?, wifi_name? }` (ip validated)      |
| POST   | `/cameras/{id}/disconnect`          | Stop camera                                            |
| GET    | `/alerts`                           | Recent alerts (one per incident) + evidence URLs       |
| GET    | `/logs`                             | System event log                                       |
| GET    | `/cameras/{id}/feed`                | Live MJPEG stream                                      |
| GET    | `/evidence/{filename}`              | Play/download an evidence MP4                          |
| POST   | `/analyze/video`                    | Upload a recording; get would_alert / safe verdict     |
| POST   | `/analyze/evidence/{filename}`      | Re-analyze a saved evidence clip                       |
| POST   | `/review/label-evidence/{filename}` | Copy clip to `data/safe` or `data/unsafe` for training |

## Notes for Windows Users

- Use PowerShell or Command Prompt as administrator if you hit permission errors.
- If camera access is blocked, allow Python, PowerShell, and your browser through Windows Firewall.
- There is no built-in Windows version of `start_lan_bridge.command`; the app is intended to run directly on the local network. On non-macOS systems the LAN bridge is never auto-started (the helper prints a clear message instead).
- If the phone app offers a browser preview, close that preview before connecting from the dashboard.

## Reliability notes

- **Timing:** every frame is timestamped and clips are built by sampling at 1/15 s steps
  back from the newest frame, so a slow camera never makes motion look faster. Below 8
  real fps a camera is `degraded` and cannot open incidents.
- **Discontinuities:** a grab failure, a frame gap > 0.3 s or a reconnect clears the clip
  buffer and bumps the camera `generation`; the watcher resets score history and warms up.
- **Watchdog:** the dashboard log reports a stalled AI watcher, frozen streams (same image
  for ~3 s), offline cameras and low fps. Stopped cameras with a saved source are retried
  every 30 s (an operator Disconnect is respected).
- **LAN bridge:** each camera has its own upstream (`/shot/<camera_id>.jpg`); only private
  LAN hosts can be configured and CORS is limited to the dashboard origins.
- **Logs:** `logs/safespace.log` (rotating) in addition to the dashboard log feed.
