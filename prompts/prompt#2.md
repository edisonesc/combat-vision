Phase 7 — Prepare for ML Upgrade

Prompt 7: Prepare for ML-Based Classification

Refactor system so that rule-based classifier can later be replaced by ML model.

Requirements:

Define PunchClassifier interface

Current implementation = heuristic classifier

Design so future LSTM / temporal model can plug in

Keep velocity + history logic reusable


Phase 8 — Data Logging for Training

Prompt 8: Add Structured Event Logging

Extend system to:

Log punch events as structured JSON:

timestamp

wrist

velocity

direction

classified type

Save JSON file per video

Prepare format suitable for future ML dataset labeling