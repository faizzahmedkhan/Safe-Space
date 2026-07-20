# SafeSpace - AI Safety Monitor

SafeSpace is a real-time AI safety monitor that watches a webcam feed, looks for suspicious physical activity (like bullying or fights), and saves evidence clips when it thinks an incident occurred. It is implemented as a FastAPI backend plus a background video-processing pipeline.

## Features
- Captures frames continuously from the camera
- Runs a pretrained 3D CNN model (`R3D-18`) on clips of 16 frames
- Produces a confidence score for unsafe/physical altercation behavior
- Triggers an alert if the smoothed score crosses a threshold
- Saves MP4 evidence clips locally

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

3. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Running the App

Start the FastAPI application:
```bash
uvicorn main:app --reload
```

Then open your browser to:
- `http://127.0.0.1:8000/` : Health check
- `http://127.0.0.1:8000/status` : System diagnostics
- `http://127.0.0.1:8000/alerts` : Recent alerts and evidence

## Training & Fine-Tuning the AI

If you want to teach the AI on your own custom videos (e.g., to reduce false alarms from accidents):

1. **Prepare your data:**
   - Put short `.mp4` videos of normal activities in `data/safe/`
   - Put short `.mp4` videos of fights/incidents in `data/unsafe/`
   *(Note: The folders will be created automatically when you first run the script).*

2. **Run the training script:**
   ```bash
   python train.py
   ```

3. **Enjoy the smarter model:**
   The script will automatically fine-tune the model and overwrite `safespace_bullying_model.pth`. Next time you start `main.py`, it will use your upgraded model!
