# combat-analysis for boxing

A computer vision pipeline for automated boxing technique analysis. It processes training footage to detect and classify punch events using skeletal pose estimation and motion kinematics — producing annotated video output and structured event data suitable for downstream performance review or ML dataset construction.

Developed as an applied exploration of real-time pose analysis in a combat sports context, with a focus on accurate event detection, clean architecture, and extensibility toward ML-based classification.

> Note: Currently WIP and only support basic movements



https://github.com/user-attachments/assets/b9e92a83-d5b8-414f-9827-20ad098d0577


---

## What it does

- Detects **punch events** from wrist velocity spikes — but only counts a punch if the wrist has traveled at least 80% of a full extension (no false triggers from micro-adjustments)
- **Classifies each punch** as `STRAIGHT`, `UPPERCUT`, or `DOWNWARD` based on the dominant direction vector
- Overlays a **thermal heatmap** on the arms and torso — color shifts from dark maroon at rest to orange/yellow at peak speed
- Renders a **live combat log** of recent strikes with timestamps
- Saves an **annotated MP4** and a **structured JSON events file** (ready for ML labeling) into a timestamped output directory

---

## Technologies

| | |
|---|---|
| **MediaPipe Pose** | Landmark detection (33 keypoints, `IMAGE` mode) |
| **OpenCV** | Video I/O, frame rendering, Gaussian blur heatmap |
| **NumPy** | Velocity and displacement calculations |
| **Python 3.12** | Core runtime |

---

## Algorithm

1. **Pose extraction** — MediaPipe extracts wrist, elbow, shoulder, and hip landmarks per frame
2. **Velocity** — Euclidean distance between the last two wrist positions (px/frame)
3. **Motion phase tracking** — records where a wrist motion began; resets on inactivity
4. **Punch gate** — fires only when velocity > threshold *and* displacement from motion start ≥ 80% of `FULL_RANGE_PX`
5. **Cooldown** — per-wrist debounce prevents double-counting a single strike
6. **Classification** — direction vector from motion start to current position; horizontal dominant = STRAIGHT, vertical up = UPPERCUT, vertical down = DOWNWARD

---

## Usage

```bash
python src/live_test.py samples/shadowbox_1.mp4
```

Output goes to `samples/<datetime> (<input name>)/`.
