# SafeSpace - AI Safety Monitor

SafeSpace is a real-time AI safety monitor that watches a camera feed (laptop webcam or phone IP Webcam), looks for suspicious physical activity (like bullying or fights), and saves evidence clips when it thinks an incident occurred.

Stack:
- **Backend:** FastAPI + OpenCV + PyTorch (R3D-18)
- **Dashboard:** Next.js operator UI at `http://localhost:3000`

## Features
- Connect a phone camera by IP (and optional Wi‑Fi label) from the dashboard
- Live MJPEG preview and live violence scores
- Clear dual status: laptop backend + phone camera
- Alerts with evidence play/download, plus system logs
- Continuous incident recording while the smoothed score stays above threshold

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd FYP
   ```

2. **Pull the model weights (Git LFS required):**
   ```bash
   git lfs pull
   ```

3. **Install Python dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Install dashboard dependencies:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

## Running the App (two processes)

### 1. Backend (FastAPI)

You **can** run this from Cursor:

```bash
cd /Users/faizahmedkhan/Downloads/FYP
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

macOS often blocks Cursor from Local Network access to your phone. SafeSpace handles that
automatically: when Connect fails directly, it opens a **LAN bridge** in Terminal.app
(`start_lan_bridge.command`) and pulls frames via `http://127.0.0.1:8765`. Keep that
bridge Terminal window open while using the phone camera.

Optional: System Settings → Privacy & Security → **Local Network** → enable **Cursor**
so the bridge is not needed.

### 2. Dashboard (Next.js)
```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** for the operator dashboard.

Useful backend URLs:
- `http://127.0.0.1:8000/status` — diagnostics JSON
- `http://127.0.0.1:8000/alerts` — recent alerts
- `http://127.0.0.1:8000/logs` — system logs
- `http://127.0.0.1:8000/preview` — legacy HTML preview

## Connect a phone as CCTV

1. Put the **phone and laptop on the same Wi‑Fi**.
2. Install **IP Webcam** (Android) or a similar IP camera app (iPhone).
3. Start the server in the app and note the IP (e.g. `192.168.1.5`).
4. **Close** any Safari/Chrome tab viewing the phone stream (one-client limit).
5. In the SafeSpace dashboard:
   - Enter the **IP address**
   - Port defaults to `8080`, path defaults to `/video` (IP Webcam)
   - Optionally enter a **Wi‑Fi name** label (for display only)
   - Click **Connect phone**
6. If a Terminal “LAN bridge” window opens, leave it running.

### If Connect keeps failing
1. Confirm IP Webcam shows **Start server** and the IP matches the dashboard.
2. Close browser tabs to `http://PHONE_IP:8080`.
3. Allow **Terminal** (and optionally **Cursor**) under Local Network privacy settings.
4. Manually run `open start_lan_bridge.command`, then Connect again.
5. For a quick UI test without the phone, click **Use laptop webcam**.

## Training & Fine-Tuning the AI

If you want to teach the AI on your own custom videos (e.g., to reduce false alarms):

1. **Prepare your data:**
   - Put short `.mp4` videos of normal activities in `data/safe/`
   - Put short `.mp4` videos of fights/incidents in `data/unsafe/`
   *(Both classes are required — never train with only one folder.)*

2. **Run the training script:**
   ```bash
   python train.py
   ```

3. The script fine-tunes and overwrites `safespace_bullying_model.pth`. Restart the backend to load the new weights.

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/status` | Backend/camera status, scores, Wi‑Fi label |
| POST | `/camera/connect` | `{ ip, port?, path?, wifi_name? }` |
| POST | `/camera/disconnect` | Stop camera |
| GET | `/alerts` | Recent alerts + evidence URLs |
| GET | `/logs` | System event log |
| GET | `/feed` | Live MJPEG stream |
| GET | `/evidence/{filename}` | Play/download an evidence MP4 |
| POST | `/analyze/video` | Upload a recording; get would_alert / safe verdict |
| POST | `/analyze/evidence/{filename}` | Re-analyze a saved evidence clip |
| POST | `/review/label-evidence/{filename}` | Copy clip to `data/safe` or `data/unsafe` for training |

