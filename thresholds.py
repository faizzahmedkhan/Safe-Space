"""Pure-Python constants shared by live inference, offline tools and tests.

Kept free of torch/cv2/numpy imports so unit tests can import it anywhere.
`inference_core` re-exports every name here, so the rest of the code base
should treat inference_core as the single source of truth.
"""

CLIP_LENGTH = 16
CAMERA_TARGET_FPS = 15
INFERENCE_STRIDE = 8
SCORE_HISTORY_LEN = 4
ALERT_THRESHOLD = 0.67
# A single raw clip score at/above this can open an incident (after the
# optional confirmation window) without waiting for the rolling average.
SPIKE_THRESHOLD = 0.90
ALERT_COOLDOWN_SECONDS = 5
POST_ROLL_SECONDS = 5
MIN_INCIDENT_SECONDS = 10
MAX_INCIDENT_SECONDS = 120

# A spike must be confirmed by the next window scoring >= ALERT_THRESHOLD.
SPIKE_REQUIRES_CONFIRMATION = True
# Consecutive above-threshold smoothed windows needed for the sustained path.
SUSTAINED_CONSECUTIVE = 2

# Below this real capture rate a camera is "degraded" and cannot open incidents.
DEGRADED_FPS = 8.0
# Frame gap (seconds) that is treated as a discontinuity.
MAX_FRAME_GAP_S = 0.3
# Clip-level fraction of frames that must be suitable / global-motion.
CLIP_FRACTION = 0.5
