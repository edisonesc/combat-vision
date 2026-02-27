from domain.base_classifier import PunchClassifier

VELOCITY_THRESHOLD = 40
COOLDOWN_FRAMES = 10


def detect_punch(velocity, cooldown_counter):
    """
    Returns (punch_detected: bool, new_cooldown: int).
    Triggers when velocity exceeds threshold and cooldown is zero.
    Decrements cooldown each frame otherwise.
    """
    if velocity > VELOCITY_THRESHOLD and cooldown_counter == 0:
        return True, COOLDOWN_FRAMES
    return False, max(0, cooldown_counter - 1)


def compute_direction(history):
    """
    Returns (dx, dy) direction vector from the oldest to newest position in history.
    """
    if len(history) < 2:
        return 0.0, 0.0

    (x1, y1) = history[0]
    (x2, y2) = history[-1]

    return float(x2 - x1), float(y2 - y1)


def classify_punch(dx, dy):
    """
    Classifies punch type based on dominant axis of direction vector.
    Note: in image coordinates y increases downward, so negative dy = upward movement.

    Returns: "STRAIGHT" | "UPPERCUT" | "DOWNWARD"
    """
    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx >= abs_dy:
        return "STRAIGHT"
    elif dy < 0:
        return "UPPERCUT"
    else:
        return "DOWNWARD"


class HeuristicPunchClassifier(PunchClassifier):
    """
    Rule-based punch classifier using direction vector heuristics.
    Satisfies the PunchClassifier interface so it can be swapped for
    an ML model (e.g. LSTM) without changing the calling code.
    """

    def classify(self, history, velocity: float) -> str:
        dx, dy = compute_direction(history)
        return classify_punch(dx, dy)
