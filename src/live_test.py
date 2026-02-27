import argparse
import cv2
import json
import mediapipe as mp
from abc import ABC, abstractmethod
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from pathlib import Path
from collections import deque
import numpy as np

# ── Detection constants ────────────────────────────────────────────────────────
VELOCITY_THRESHOLD = 40
COOLDOWN_FRAMES    = 10
HISTORY_MAXLEN     = 5
PUNCH_LOG_MAXLEN   = 8
FULL_RANGE_PX      = 150
RANGE_THRESHOLD    = 0.80
MOTION_START_VEL   = 8

# ── Sporty color palette (BGR) ─────────────────────────────────────────────────
C_ORANGE  = (  0, 110, 255)   # primary energy   · RGB(255, 110,   0)
C_RED     = ( 30,  30, 210)   # right wrist      · RGB(210,  30,  30)
C_GOLD    = (  0, 195, 255)   # left wrist       · RGB(255, 195,   0)
C_LIME    = ( 30, 210,  80)   # normal / OK      · RGB( 80, 210,  30)
C_GRAY    = ( 55,  55,  55)   # dim border
C_BG      = ( 22,  22,  26)   # dark charcoal bg
C_TORSO   = ( 65,  65,  70)   # neutral gray torso
C_WHITE   = (255, 255, 255)
C_BLACK   = (  0,   0,   0)


# ── Domain: Velocity ──────────────────────────────────────────────────────────

def compute_velocity(history):
    if len(history) < 2:
        return 0.0
    (x1, y1) = history[-2]
    (x2, y2) = history[-1]
    return float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))


def update_motion_start(motion_start, pos, velocity):
    if motion_start is None and velocity > MOTION_START_VEL:
        return pos
    if motion_start is not None and velocity < MOTION_START_VEL * 0.5:
        return None
    return motion_start


def compute_displacement(start_pos, current_pos):
    dx = current_pos[0] - start_pos[0]
    dy = current_pos[1] - start_pos[1]
    return float(np.sqrt(dx**2 + dy**2))


def detect_punch(velocity, cooldown_counter, motion_start, current_pos):
    if cooldown_counter > 0:
        return False, cooldown_counter - 1
    if motion_start is None:
        return False, 0
    displacement = compute_displacement(motion_start, current_pos)
    if velocity > VELOCITY_THRESHOLD and displacement >= FULL_RANGE_PX * RANGE_THRESHOLD:
        return True, COOLDOWN_FRAMES
    return False, 0


# ── Domain: Direction & Classification ───────────────────────────────────────

def compute_direction(history):
    if len(history) < 2:
        return 0.0, 0.0
    (x1, y1) = history[0]
    (x2, y2) = history[-1]
    return float(x2 - x1), float(y2 - y1)


def classify_punch(dx, dy):
    if abs(dx) >= abs(dy):
        return "STRAIGHT"
    elif dy < 0:
        return "UPPERCUT"
    else:
        return "DOWNWARD"


class PunchClassifier(ABC):
    @abstractmethod
    def classify(self, history, velocity: float) -> str: ...


class HeuristicPunchClassifier(PunchClassifier):
    def classify(self, history, velocity: float) -> str:
        dx, dy = compute_direction(history)
        return classify_punch(dx, dy)


# ── Infrastructure: Event Logger ──────────────────────────────────────────────

class PunchEventLogger:
    def __init__(self, video_path):
        video_path = Path(video_path)
        self._output_path = video_path.with_stem(video_path.stem + "_events").with_suffix(".json")
        self._source_video = video_path.name
        self._events = []

    def log(self, timestamp_ms, wrist, velocity, dx, dy, punch_type):
        self._events.append({
            "timestamp_ms":    round(timestamp_ms, 2),
            "wrist":           wrist,
            "velocity":        round(velocity, 2),
            "direction":       {"dx": round(dx, 2), "dy": round(dy, 2)},
            "classified_type": punch_type,
        })

    def save(self):
        payload = {
            "source_video": self._source_video,
            "total_events": len(self._events),
            "events":       self._events,
        }
        with open(self._output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {len(self._events)} punch events to {self._output_path}")


# ── Rendering: thermal heatmap ────────────────────────────────────────────────

def thermal_color(velocity, max_vel=120):
    """
    Warm sporty thermal gradient (BGR):
      still → dark maroon → red → orange → bright yellow
    """
    t = min(velocity / max_vel, 1.0)
    stops = [
        ( 20,  15,  80),   # dark maroon   (t = 0.00)
        ( 30,  30, 200),   # red           (t = 0.33)
        (  0, 120, 255),   # orange        (t = 0.66)
        ( 60, 230, 255),   # bright yellow (t = 1.00)
    ]
    n   = len(stops) - 1
    pos = t * n
    lo  = int(pos)
    hi  = min(lo + 1, n)
    f   = pos - lo
    return tuple(int(stops[lo][c] + f * (stops[hi][c] - stops[lo][c])) for c in range(3))


def draw_body_heatmap(frame, landmarks, h, w, right_velocity, left_velocity, frame_index=0):
    """
    Renders torso + arms as a blurred warm thermal heatmap:
      1. Draw thick segments on a black canvas using thermal velocity colors.
      2. Gaussian-blur to simulate heat diffusion.
      3. Additively blend onto frame.
      4. Draw sharp wrist markers on top.
    """
    heat = np.zeros_like(frame, dtype=np.uint8)

    def lm(idx):
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    # Torso — neutral gray, thin
    for a, b in [(11, 12), (11, 23), (12, 24), (23, 24)]:
        cv2.line(heat, lm(a), lm(b), C_TORSO, 5)
    for idx in (11, 12, 23, 24):
        cv2.circle(heat, lm(idx), 5, C_TORSO, -1)

    # Right arm — warm thermal by velocity
    r_col = thermal_color(right_velocity)
    cv2.line(heat, lm(12), lm(14), r_col, 12)
    cv2.line(heat, lm(14), lm(16), r_col, 12)
    for idx, r in [(12, 8), (14, 11), (16, 14)]:
        cv2.circle(heat, lm(idx), r, r_col, -1)

    # Left arm — warm thermal by velocity
    l_col = thermal_color(left_velocity)
    cv2.line(heat, lm(11), lm(13), l_col, 12)
    cv2.line(heat, lm(13), lm(15), l_col, 12)
    for idx, r in [(11, 8), (13, 11), (15, 14)]:
        cv2.circle(heat, lm(idx), r, l_col, -1)

    heat_blur = cv2.GaussianBlur(heat, (55, 55), 0)
    frame[:] = np.clip(
        frame.astype(np.float32) + heat_blur.astype(np.float32) * 0.88,
        0, 255
    ).astype(np.uint8)

    # Sharp wrist markers on top
    _draw_wrist_marker(frame, lm(16), C_RED)
    _draw_wrist_marker(frame, lm(15), C_GOLD)


def _draw_wrist_marker(frame, center, color):
    """Bold clean wrist marker: filled circle + white ring + white core."""
    cv2.circle(frame, center, 10, color,   -1)
    cv2.circle(frame, center, 11, C_WHITE,  1)
    cv2.circle(frame, center,  3, C_WHITE, -1)


# ── Rendering: panel system ───────────────────────────────────────────────────

def draw_panel(frame, x1, y1, x2, y2, accent=None):
    """
    Sporty panel:
      · semi-transparent dark charcoal background
      · thin gray border
      · bold colored left-side accent bar
    """
    accent = accent or C_ORANGE
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), C_BG, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_GRAY, 1)
    # Bold left accent bar
    cv2.rectangle(frame, (x1, y1), (x1 + 4, y2), accent, -1)


def _panel_title(frame, x1, y1, x2, title, accent=None):
    """Title row with top accent line and clean white label."""
    accent = accent or C_ORANGE
    cv2.line(frame, (x1, y1), (x2, y1), accent, 2)
    cv2.putText(frame, title, (x1 + 12, y1 + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, C_WHITE, 1)
    cv2.line(frame, (x1 + 6, y1 + 30), (x2 - 6, y1 + 30), C_GRAY, 1)


# ── Rendering: velocity bar ───────────────────────────────────────────────────

def velocity_color(velocity):
    return C_ORANGE if velocity > VELOCITY_THRESHOLD else C_LIME


def _draw_vel_bar(frame, x, y, velocity, color, width=130):
    """Thick solid sport stat bar with threshold tick."""
    fill = int(min(velocity / (VELOCITY_THRESHOLD * 2), 1.0) * width)
    # Track
    cv2.rectangle(frame, (x, y), (x + width, y + 8), (40, 40, 44), -1)
    # Fill
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + 8), color, -1)
    # Threshold tick at 50 %
    tick_x = x + width // 2
    cv2.line(frame, (tick_x, y - 2), (tick_x, y + 10), C_WHITE, 1)


# ── Rendering: HUD panel ──────────────────────────────────────────────────────

def draw_hud_panel(frame, right_velocity, left_velocity):
    x1, y1, x2, y2 = 16, 16, 275, 155
    draw_panel(frame, x1, y1, x2, y2)
    _panel_title(frame, x1, y1, x2, "MOTION ANALYSIS")

    r_col = velocity_color(right_velocity)
    cv2.putText(frame, f"R  {right_velocity:>6.1f}",
                (x1 + 14, y1 + 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, r_col, 1)
    cv2.putText(frame, "px/f", (x1 + 155, y1 + 60),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, C_GRAY, 1)
    _draw_vel_bar(frame, x1 + 14, y1 + 68, right_velocity, r_col)

    l_col = velocity_color(left_velocity)
    cv2.putText(frame, f"L  {left_velocity:>6.1f}",
                (x1 + 14, y1 + 108), cv2.FONT_HERSHEY_DUPLEX, 0.6, l_col, 1)
    cv2.putText(frame, "px/f", (x1 + 155, y1 + 108),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, C_GRAY, 1)
    _draw_vel_bar(frame, x1 + 14, y1 + 116, left_velocity, l_col)


# ── Rendering: punch alert ────────────────────────────────────────────────────

def draw_punch_alert(frame, text, side):
    h, w, _ = frame.shape
    bx1, by1 = w // 2 - 240, 50
    bx2, by2 = w // 2 + 240, 145

    accent = C_RED if side == "right" else C_GOLD

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), C_BG, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Top + bottom bold accent bars
    cv2.rectangle(frame, (bx1, by1),      (bx2, by1 + 4), accent, -1)
    cv2.rectangle(frame, (bx1, by2 - 4),  (bx2, by2),     accent, -1)

    # Side accent pillars
    cv2.rectangle(frame, (bx1,      by1), (bx1 + 6, by2), accent, -1)
    cv2.rectangle(frame, (bx2 - 6,  by1), (bx2,     by2), accent, -1)

    # Text
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
    tx, ty = w // 2 - tw // 2, by1 + 68
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.3, C_WHITE, 2)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.3, accent,  1)


# ── Rendering: combat log ─────────────────────────────────────────────────────

def draw_punch_history(frame, history, current_frame):
    if not history:
        return

    h, w, _ = frame.shape
    ROW_H   = 26
    PANEL_W = 275
    x1      = w - PANEL_W - 16
    y1      = 16
    x2      = x1 + PANEL_W
    y2      = y1 + 44 + len(history) * ROW_H

    draw_panel(frame, x1, y1, x2, y2)
    _panel_title(frame, x1, y1, x2, "COMBAT LOG")

    for i, entry in enumerate(history):
        age   = current_frame - entry["frame"]
        fade  = max(0.25, 1.0 - age / 100)
        base  = C_RED if entry["side"] == "right" else C_GOLD
        color = tuple(int(c * fade) for c in base)

        ts       = entry["timestamp_ms"] / 1000
        m, s     = divmod(ts, 60)
        time_str = f"{int(m):02d}:{s:05.2f}"

        row_y = y1 + 48 + i * ROW_H

        # Side color pip
        cv2.rectangle(frame,
                      (x1 + 8,  row_y - 10),
                      (x1 + 12, row_y + 2),
                      tuple(int(c * fade) for c in base), -1)

        cv2.putText(frame, time_str,
                    (x1 + 20, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1)
        cv2.putText(frame, entry["label"],
                    (x1 + 112, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1)


# ── Setup ─────────────────────────────────────────────────────────────────────

base_options = python.BaseOptions(model_asset_path="models/pose_landmarker_lite.task")
options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)
landmarker = vision.PoseLandmarker.create_from_options(options)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_stem(f"{path.stem}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input video file")
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
video_path   = Path(args.input)

output_dir = PROJECT_ROOT / "samples" / \
    f"{datetime.now().strftime('%Y-%m-%d_%H%M%S_%f')[:-3]} ({video_path.stem})"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = unique_path(output_dir / "output_annotated.mp4")

cap    = cv2.VideoCapture(str(video_path))
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
print("Saving to:", output_path)

classifier   = HeuristicPunchClassifier()
event_logger = PunchEventLogger(output_path)

# ── Main loop ─────────────────────────────────────────────────────────────────
right_wrist_history = deque(maxlen=HISTORY_MAXLEN)
left_wrist_history  = deque(maxlen=HISTORY_MAXLEN)
punch_history       = deque(maxlen=PUNCH_LOG_MAXLEN)
right_cooldown      = 0
left_cooldown       = 0
right_motion_start  = None
left_motion_start   = None
frame_index         = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp_ms = (frame_index / fps) * 1000 if fps > 0 else 0.0

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results   = landmarker.detect(mp_image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]
        h, w, _   = frame.shape

        rw = landmarks[16]
        rw_x, rw_y = int(rw.x * w), int(rw.y * h)
        right_wrist_history.append((rw_x, rw_y))

        lw = landmarks[15]
        lw_x, lw_y = int(lw.x * w), int(lw.y * h)
        left_wrist_history.append((lw_x, lw_y))

        right_velocity = compute_velocity(right_wrist_history)
        left_velocity  = compute_velocity(left_wrist_history)

        draw_body_heatmap(frame, landmarks, h, w, right_velocity, left_velocity, frame_index)

        right_motion_start = update_motion_start(right_motion_start, (rw_x, rw_y), right_velocity)
        left_motion_start  = update_motion_start(left_motion_start,  (lw_x, lw_y), left_velocity)

        right_punch, right_cooldown = detect_punch(right_velocity, right_cooldown,
                                                   right_motion_start, (rw_x, rw_y))
        left_punch,  left_cooldown  = detect_punch(left_velocity,  left_cooldown,
                                                   left_motion_start,  (lw_x, lw_y))

        if right_punch:
            punch_type         = classifier.classify(right_wrist_history, right_velocity)
            dx, dy             = compute_direction(right_wrist_history)
            label              = f"RIGHT_{punch_type}"
            right_motion_start = None
            draw_punch_alert(frame, label, "right")
            punch_history.appendleft({"label": label, "side": "right",
                                      "timestamp_ms": timestamp_ms, "frame": frame_index})
            event_logger.log(timestamp_ms, "right", right_velocity, dx, dy, punch_type)
        elif left_punch:
            punch_type        = classifier.classify(left_wrist_history, left_velocity)
            dx, dy            = compute_direction(left_wrist_history)
            label             = f"LEFT_{punch_type}"
            left_motion_start = None
            draw_punch_alert(frame, label, "left")
            punch_history.appendleft({"label": label, "side": "left",
                                      "timestamp_ms": timestamp_ms, "frame": frame_index})
            event_logger.log(timestamp_ms, "left", left_velocity, dx, dy, punch_type)

        draw_hud_panel(frame, right_velocity, left_velocity)
        draw_punch_history(frame, punch_history, frame_index)

    out.write(frame)
    frame_index += 1

cap.release()
out.release()
landmarker.close()
event_logger.save()
