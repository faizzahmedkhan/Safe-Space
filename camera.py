import cv2
import time
import threading
from collections import deque

CLIP_LENGTH = 16

class CameraStream:
  def __init__(self, src=0, target_fps=15, show_preview=True):
    """
    src=0 opens the default laptop webcam.
    Later, you will replace 0 with your phone's 'rstp://...' URL
    """
    self.stream = cv2.VideoCapture(src)
    self.target_fps = target_fps
    self.show_preview = show_preview
    self.preview_enabled = show_preview

    # THE MAGIC BUFFER: Holds exactly 16 items.
    # When item 17 is added, item 1 is automatically pushed out.
    self.frame_buffer = deque(maxlen=CLIP_LENGTH)

    self._lock = threading.Lock()

    self.frame_count = 0

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
    # Calculate how long to sleep between frames to hit ~6 FPS
    sleep_time = 1.0 / self.target_fps

    while not self.stopped:
      # Grab the raw, high-res frame from the camera
      loop_start = time.perf_counter()

      grabbed, frame = self.stream.read()

      if not grabbed:
        print("[PRODUCER] Camera disconnected")
        self.stop()
        break

      # 1. PRE-PROCESS IMMEDIATELY: Resize to 112x112 to save laptop RAM
      frame_resized = cv2.resize(frame, (112, 112))

      frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

      with self._lock:
         self.frame_buffer.append(frame_rgb)
         self.frame_count += 1
      
      elapsed = time.perf_counter() - loop_start
      remaining = sleep_time - elapsed
      if remaining > 0:
        time.sleep(remaining)

      # Show preview if enabled and still working. Disable permanently on first failure (headless environments)
      if self.preview_enabled:
        try:
          cv2.imshow("SafeSpace: AI Vision", frame)
          cv2.waitKey(1) # Required for OpenCV to update the window
        except Exception as exc:  # OpenCV raises C++ exceptions in headless mode
          print(f"[PRODUCER] Preview disabled (cv2.imshow failed: {exc})")
          self.preview_enabled = False
  
  def get_buffer(self):
    """The AI will call this function whenever it is ready for a new batch of 16 frames"""
    # We return a standard list copy of the buffer to prevent thread-collision errors
    with self._lock:
      return list(self.frame_buffer)

  def get_frame_count(self):
     """Returns the total number of frames captured so far"""
     with self._lock:
        return self.frame_count

  def stop(self):
    """Releases the camera hardware."""
    self.stopped = True
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