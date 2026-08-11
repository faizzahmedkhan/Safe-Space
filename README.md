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

### 2. Download model weights

If this repo uses Git LFS, run:

```powershell
git lfs pull
```

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
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If `uvicorn` is not found, ensure the virtual environment is activated and retry.

### Dashboard (Next.js)

Open a second terminal, then:

```powershell
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

### Useful backend endpoints

- `http://127.0.0.1:8000/status` — diagnostics JSON
- `http://127.0.0.1:8000/alerts` — recent alerts
- `http://127.0.0.1:8000/logs` — system logs
- `http://127.0.0.1:8000/preview` — legacy HTML preview

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

3. Restart the backend after training to load the updated weights.

## API Overview

| Method | Path                                | Purpose                                                |
| ------ | ----------------------------------- | ------------------------------------------------------ |
| GET    | `/status`                           | Backend/camera status, scores, Wi-Fi label             |
| POST   | `/camera/connect`                   | `{ ip, port?, path?, wifi_name? }`                     |
| POST   | `/camera/disconnect`                | Stop camera                                            |
| GET    | `/alerts`                           | Recent alerts + evidence URLs                          |
| GET    | `/logs`                             | System event log                                       |
| GET    | `/feed`                             | Live MJPEG stream                                      |
| GET    | `/evidence/{filename}`              | Play/download an evidence MP4                          |
| POST   | `/analyze/video`                    | Upload a recording; get would_alert / safe verdict     |
| POST   | `/analyze/evidence/{filename}`      | Re-analyze a saved evidence clip                       |
| POST   | `/review/label-evidence/{filename}` | Copy clip to `data/safe` or `data/unsafe` for training |

## Notes for Windows Users

- Use PowerShell or Command Prompt as administrator if you hit permission errors.
- If camera access is blocked, allow Python, PowerShell, and your browser through Windows Firewall.
- There is no built-in Windows version of `start_lan_bridge.command`; the app is intended to run directly on the local network.
- If the phone app offers a browser preview, close that preview before connecting from the dashboard.
