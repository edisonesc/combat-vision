Phase 3 — Punch Detection via Velocity Spike

Prompt 3: Add Punch Event Detection with Cooldown

Extend the script with punch event detection logic.

Requirements:

Add configurable constants:

VELOCITY_THRESHOLD

COOLDOWN_FRAMES

When wrist velocity exceeds threshold and cooldown == 0:

Trigger punch event

Reset cooldown

Decrement cooldown each frame

Display punch event label overlay on frame

Futuristic HUD Styling

Prompt 4: Add Futuristic HUD Design

Refactor overlay rendering for aesthetic design.

Requirements:

Neon glow wrist markers (multi-radius blend effect)

Semi-transparent HUD panel for velocity display

Dynamic velocity color:

Normal = neon green

Above threshold = cyber yellow

Centered alert box for punch detection

Keep clean function separation:

draw_glow_circle

draw_hud_panel

draw_punch_alert

velocity_color

Phase 5 — Directional Punch Classification

Prompt 5: Add Direction-Based Punch Classification

Extend punch detection.

Requirements:

Compute direction vector (dx, dy) per wrist

Add function:

def compute_direction(history):
    ...

Add heuristic classification:

def classify_punch(dx, dy):
    ...

If horizontal dominant → STRAIGHT

If vertical upward dominant → UPPERCUT

If vertical downward → DOWNWARD

Combine with LEFT / RIGHT

Maintain cooldown logic.

Architecture Refactor (Enterprise Preparation)

Prompt 6: Refactor Into Modular Architecture

Refactor script into structured modules:

domain/

punch_classifier.py

velocity.py

infrastructure/

pose_adapter.py

video_processor.py

app/

main.py

Requirements:

Domain layer must not import OpenCV or MediaPipe

Separation of detection logic from rendering logic

Maintain same functionality