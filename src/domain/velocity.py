import numpy as np

HISTORY_MAXLEN = 5


def compute_velocity(history):
    """Returns scalar velocity (pixel distance) between the last two positions."""
    if len(history) < 2:
        return 0.0

    (x1, y1) = history[-2]
    (x2, y2) = history[-1]

    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
