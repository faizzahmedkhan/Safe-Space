import cv2
import time
import threading
from collections import deque

CLIP_LENGTH = 16
EVIDENCE_BUFFER_LEN = 60

class CameraStream:
  def __init__(self, src=0, target_fps=15, show_preview=True):
    """
    src=0 opens the default laptop webcam.
    src="http://..." or "rtsp://..." opens a network camera stream.
    """
    self.src = src
    self.is_network_stream = isinstance(src, str) and ("://" in src)

    self.stream = cv2.VideoCapture(src)
    if not self.stream.isOpened():
        raise RuntimeError(f"Failed to open camera with src={src}. Please check if the camera is connected or the index is correct.")
    
    # For network streams: minimize OpenCV's internal buffer so we always
    # get the LATEST frame, not a stale frame from seconds ago.
    if self.is_network_stream:
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[PRODUCER] Network stream detected. Buffer minimized for real-time processing.")

    self.target_fps = target_fps
    self.show_preview = show_preview
    self.preview_enabled = show_preview

    # THE MAGIC BUFFER: Holds exactly 16 items.
    # When item 17 is added, item 1 is automatically pushed out.
    self.frame_buffer = deque(maxlen=CLIP_LENGTH)
    
    # Secondary buffer for high-res evidence (keeps the last ~4 seconds)
    # This will be dynamically capped based on actual frame size to avoid OOM
    self.evidence_buffer = deque(maxlen=EVIDENCE_BUFFER_LEN)
    self.max_evidence_mb = 150
    self.evidence_buffer_initialized = False

    self._lock = threading.Lock()

    self.frame_count = 0
    self.video_writer = None

    # A flag to safely shut down the background thread
    self.stopped = False
  
  def start(self):
    """Starts the background thread to continuously read frames."""
    print("[PRODUCER] Starting camera thread...")
    #daemon=True means this thread will automatically die when the main program closes
    threading.Thread(target=self.update, args=(), daemon=True).start()
    return self
  
  def update(self):
    """This function runs on an infinite loop in the background."""
    sleep_time = 1.0 / self.target_fps
    consecutive_failures = 0
    max_consecutive_failures = 30

    while not self.stopped:
      loop_start = time.perf_counter()

      # For network streams: drain OpenCV's internal buffer by grabbing
      # frames rapidly until we get the very latest one.
      # Without this, .read() returns old buffered frames and the AI
      # processes footage that is several seconds behind real-time.
      if self.is_network_stream:
          for _ in range(5):  # drain up to 5 stale frames
              self.stream.grab()
          grabbed, frame = self.stream.read()
      else:
          grabbed, frame = self.stream.read()

      if not grabbed:
        consecutive_failures += 1
        wait_time = min(2 ** (consecutive_failures - 1), 30)
        print(f"[PRODUCER] Frame grab failed ({consecutive_failures}/{max_consecutive_failures}). Retrying in {wait_time}s...")
        if consecutive_failures >= max_consecutive_failures:
            print("[PRODUCER] Camera disconnected. Too many failures. Giving up.")
            self.stop()
            break
        time.sleep(wait_time)
        continue
      
      consecutive_failures = 0

      # Initialize evidence buffer size dynamically based on frame size
      if not self.evidence_buffer_initialized:
          frame_bytes = frame.nbytes
          max_frames = int((self.max_evidence_mb * 1024 * 1024) / frame_bytes)
          max_frames = max(max_frames, CLIP_LENGTH) # Ensure we always keep at least one clip
          self.evidence_buffer = deque(maxlen=max_frames)
          self.evidence_buffer_initialized = True
          print(f"[PRODUCER] Evidence buffer capped at {max_frames} frames ({self.max_evidence_mb} MB)")

      # 1. PRE-PROCESS IMMEDIATELY: Resize to 112x112 to save laptop RAM
      frame_resized = cv2.resize(frame, (112, 112))

      frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

      with self._lock:
         self.frame_buffer.append(frame_rgb)
         self.evidence_buffer.append(frame) # Keep raw BGR frame for high-res evidence
         self.frame_count += 1
         
         if self.video_writer is not None:
             self.video_writer.write(frame)
      
      # For local webcams, pace the loop. For network streams, the network
      # itself is the natural rate limiter — no need to add artificial delay.
      if not self.is_network_stream:
          elapsed = time.perf_counter() - loop_start
          remaining = sleep_time - elapsed
          if remaining > 0:
              time.sleep(remaining)
  
  def get_buffer(self):
    """The AI will call this function whenever it is ready for a new batch of 16 frames"""
    # We return a standard list copy of the buffer to prevent thread-collision errors
    with self._lock:
      return list(self.frame_buffer)
      
  def get_evidence_buffer(self):
    """Returns the recent high-res frames for saving as evidence"""
    with self._lock:
      return list(self.evidence_buffer)

  def get_snapshot(self):
    """
    Atomically returns the inference buffer (small, model-ready), the
    evidence buffer (full-res), and the frame count at the exact same moment in time.
    Prevents the race condition where new frames arrive between two separate calls.
    """
    with self._lock:
        return list(self.frame_buffer), list(self.evidence_buffer), self.frame_count

  def get_latest_frame_jpeg(self):
    """
    Returns the latest high-res frame as JPEG bytes for browser preview.
    Returns None if no frames have been captured yet.
    """
    with self._lock:
        if len(self.evidence_buffer) == 0:
            return None
        frame = self.evidence_buffer[-1]  # latest raw BGR frame
    # Encode outside the lock so we don't block the camera thread
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes()

  def start_recording(self, filename):
      """Starts writing the evidence buffer and subsequent live frames to a file."""
      with self._lock:
          if self.video_writer is not None:
              return
          
          if not self.evidence_buffer:
              return

          height, width = self.evidence_buffer[0].shape[:2]
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          self.video_writer = cv2.VideoWriter(filename, fourcc, self.target_fps, (width, height))
          
          # Write the pre-incident buffer so we have the context leading up to the event
          for f in self.evidence_buffer:
              self.video_writer.write(f)

  def stop_recording(self):
      """Stops the current recording."""
      with self._lock:
          if self.video_writer is not None:
              self.video_writer.release()
              self.video_writer = None

  def get_frame_count(self):
     """Returns the total number of frames captured so far"""
     with self._lock:
        return self.frame_count

  def stop(self):
    """Releases the camera hardware."""
    self.stopped = True
    self.stop_recording()
    if getattr(self, 'stream', None) and self.stream.isOpened():
        self.stream.release()
    cv2.destroyAllWindows()

# ==========================================
# TESTING BLOCK (Runs only if you execute this file directly)
# ==========================================
if __name__ == "__main__":
    # 1. Initialize and start the camera in the background
    cam = CameraStream(src=0, target_fps=6, show_preview=True).start()
    
    # Give the camera a second to warm up
    time.sleep(1) 
    
    try:
        # 2. The Main Loop (Simulating the AI waiting for data)
        while True:
            current_buffer = cam.get_buffer()
            
            # Only print if the buffer is completely full (ready for the AI)
            if len(current_buffer) == CLIP_LENGTH:
                print(f"✅ Buffer Full: {len(current_buffer)} frames ready. Shape of one frame: {current_buffer[0].shape} | Total captured: {cam.get_frame_count()}")
            else:
                print(f"⏳ Buffering... {len(current_buffer)}/{CLIP_LENGTH} frames")
                
            time.sleep(1) # Check the buffer every 1 second
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...")
        cam.stop()