import json
from pathlib import Path


class PunchEventLogger:
    """
    Logs structured punch events to a JSON file for later ML dataset labeling.

    Each event records: timestamp (ms), wrist side, velocity, direction vector,
    and the classified punch type.
    """

    def __init__(self, video_path):
        video_path = Path(video_path)
        self._output_path = video_path.with_stem(video_path.stem + "_events").with_suffix(".json")
        self._source_video = video_path.name
        self._events = []

    def log(self, timestamp_ms: float, wrist: str, velocity: float, dx: float, dy: float, punch_type: str):
        self._events.append({
            "timestamp_ms": round(timestamp_ms, 2),
            "wrist": wrist,
            "velocity": round(velocity, 2),
            "direction": {
                "dx": round(dx, 2),
                "dy": round(dy, 2),
            },
            "classified_type": punch_type,
        })

    def save(self):
        payload = {
            "source_video": self._source_video,
            "total_events": len(self._events),
            "events": self._events,
        }
        with open(self._output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {len(self._events)} punch events to {self._output_path}")
