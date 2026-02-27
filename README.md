# combat-analysis for boxing

A computer vision pipeline for automated boxing technique analysis. It processes training footage to detect and classify punch events using skeletal pose estimation and motion kinematics — producing annotated video output and structured event data suitable for downstream performance review or ML dataset construction.

Developed as an applied exploration of real-time pose analysis in a combat sports context, with a focus on accurate event detection, clean architecture, and extensibility toward ML-based classification.


![output_annotated-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/d8ba7c21-bd76-4ffb-96dc-e001dbaecda3)

```json
{
  "source_video": "output_annotated.mp4",
  "total_events": 6,
  "events": [
    {
      "timestamp_ms": 552.93,
      "wrist": "left",
      "velocity": 84.48,
      "direction": {
        "dx": 224.0,
        "dy": -100.0
      },
      "classified_type": "STRAIGHT"
    },
    {
      "timestamp_ms": 715.56,
      "wrist": "right",
      "velocity": 151.46,
      "direction": {
        "dx": 68.0,
        "dy": -64.0
      },
      "classified_type": "STRAIGHT"
    },
    {
      "timestamp_ms": 910.71,
      "wrist": "left",
      "velocity": 71.2,
      "direction": {
        "dx": -4.0,
        "dy": -25.0
      },
      "classified_type": "UPPERCUT"
    },
    {
      "timestamp_ms": 1105.86,
      "wrist": "right",
      "velocity": 43.83,
      "direction": {
        "dx": 12.0,
        "dy": 32.0
      },
      "classified_type": "DOWNWARD"
    },
    {
      "timestamp_ms": 1463.64,
      "wrist": "left",
      "velocity": 83.93,
      "direction": {
        "dx": 213.0,
        "dy": -38.0
      },
      "classified_type": "STRAIGHT"
    },
    {
      "timestamp_ms": 1821.42,
      "wrist": "left",
      "velocity": 61.72,
      "direction": {
        "dx": -9.0,
        "dy": 61.0
      },
      "classified_type": "DOWNWARD"
    }
  ]
}
```

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
