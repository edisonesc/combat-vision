import sys
from pathlib import Path
from collections import deque

# Ensure src/ is on the path when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domain.velocity import compute_velocity, HISTORY_MAXLEN
from domain.punch_classifier import detect_punch, compute_direction, HeuristicPunchClassifier
from infrastructure.pose_adapter import create_landmarker, to_mp_image, extract_wrists
from infrastructure.video_processor import (
    open_video,
    create_writer,
    to_rgb,
    draw_glow_circle,
    draw_hud_panel,
    draw_punch_alert,
)
from infrastructure.event_logger import PunchEventLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
VIDEO_PATH = PROJECT_ROOT / "samples" / "shadowbox_1.mp4"
OUTPUT_PATH = PROJECT_ROOT / "samples" / "output_annotated.mp4"


def main():
    classifier = HeuristicPunchClassifier()
    landmarker = create_landmarker(MODEL_PATH)
    cap = open_video(VIDEO_PATH)
    writer = create_writer(OUTPUT_PATH, cap)
    event_logger = PunchEventLogger(OUTPUT_PATH)
    print("Saving to:", OUTPUT_PATH)

    fps = cap.get(3)  # CAP_PROP_FPS
    right_wrist_history = deque(maxlen=HISTORY_MAXLEN)
    left_wrist_history = deque(maxlen=HISTORY_MAXLEN)
    right_cooldown = 0
    left_cooldown = 0
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = (frame_index / fps) * 1000 if fps > 0 else 0.0

        mp_image = to_mp_image(to_rgb(frame))
        results = landmarker.detect(mp_image)

        right_wrist, left_wrist = extract_wrists(results, frame.shape)

        if right_wrist and left_wrist:
            right_wrist_history.append(right_wrist)
            left_wrist_history.append(left_wrist)

            draw_glow_circle(frame, right_wrist, (0, 60, 255))
            draw_glow_circle(frame, left_wrist, (255, 255, 0))

            right_velocity = compute_velocity(right_wrist_history)
            left_velocity = compute_velocity(left_wrist_history)

            right_punch, right_cooldown = detect_punch(right_velocity, right_cooldown)
            left_punch, left_cooldown = detect_punch(left_velocity, left_cooldown)

            if right_punch:
                punch_type = classifier.classify(right_wrist_history, right_velocity)
                dx, dy = compute_direction(right_wrist_history)
                draw_punch_alert(frame, f"RIGHT_{punch_type}")
                event_logger.log(timestamp_ms, "right", right_velocity, dx, dy, punch_type)
            elif left_punch:
                punch_type = classifier.classify(left_wrist_history, left_velocity)
                dx, dy = compute_direction(left_wrist_history)
                draw_punch_alert(frame, f"LEFT_{punch_type}")
                event_logger.log(timestamp_ms, "left", left_velocity, dx, dy, punch_type)

            draw_hud_panel(frame, right_velocity, left_velocity)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    landmarker.close()
    event_logger.save()


if __name__ == "__main__":
    main()
