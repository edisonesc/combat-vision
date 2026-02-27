import cv2
from domain.punch_classifier import VELOCITY_THRESHOLD


def open_video(path):
    """Opens a video file and returns a VideoCapture object."""
    return cv2.VideoCapture(str(path))


def create_writer(path, cap):
    """Creates a VideoWriter matching the resolution and FPS of the given capture."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def to_rgb(frame):
    """Converts a BGR OpenCV frame to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def velocity_color(velocity):
    """Returns cyber yellow above threshold, neon green otherwise."""
    if velocity > VELOCITY_THRESHOLD:
        return (0, 255, 255)
    return (0, 255, 120)


def draw_glow_circle(frame, center, color):
    """Draws a neon glow marker using layered semi-transparent circles."""
    for radius in range(12, 0, -3):
        alpha = radius / 12
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, color, -1)
        cv2.addWeighted(overlay, alpha * 0.4, frame, 1 - alpha * 0.4, 0, frame)

    cv2.circle(frame, center, 4, (255, 255, 255), -1)


def draw_hud_panel(frame, right_velocity, left_velocity):
    """Renders a semi-transparent HUD panel with velocity readouts."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (300, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame, f"RIGHT VEL: {right_velocity:.1f}",
        (40, 65), cv2.FONT_HERSHEY_DUPLEX, 0.7,
        velocity_color(right_velocity), 2,
    )
    cv2.putText(
        frame, f"LEFT  VEL: {left_velocity:.1f}",
        (40, 95), cv2.FONT_HERSHEY_DUPLEX, 0.7,
        velocity_color(left_velocity), 2,
    )


def draw_punch_alert(frame, text):
    """Renders a centered semi-transparent alert box with the punch label."""
    h, w, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 2 - 220, 60), (w // 2 + 220, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame, text,
        (w // 2 - 180, 120), cv2.FONT_HERSHEY_DUPLEX, 1.2,
        (0, 255, 255), 3,
    )
