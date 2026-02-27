import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions


def create_landmarker(model_path):
    """Creates and returns a MediaPipe PoseLandmarker in IMAGE running mode."""
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def to_mp_image(rgb_frame):
    """Wraps a numpy RGB frame in a MediaPipe Image."""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def extract_wrists(results, frame_shape):
    """
    Returns (right_wrist, left_wrist) as (x, y) pixel tuples.
    Returns (None, None) if no pose is detected.
    MediaPipe landmark indices: 16 = right wrist, 15 = left wrist.
    """
    if not results.pose_landmarks:
        return None, None

    h, w, _ = frame_shape
    landmarks = results.pose_landmarks[0]

    rw = landmarks[16]
    lw = landmarks[15]

    return (int(rw.x * w), int(rw.y * h)), (int(lw.x * w), int(lw.y * h))
