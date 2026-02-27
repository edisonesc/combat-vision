from abc import ABC, abstractmethod


class PunchClassifier(ABC):
    """
    Interface for punch type classifiers.

    Implementations receive the full wrist position history and current velocity,
    allowing both rule-based heuristics and temporal ML models (e.g. LSTM) to
    satisfy the same contract.
    """

    @abstractmethod
    def classify(self, history, velocity: float) -> str:
        """
        Classify the type of punch from wrist position history and velocity.

        Args:
            history: deque of (x, y) pixel positions, newest last.
            velocity: current frame velocity (pixel distance).

        Returns:
            Punch type string: "STRAIGHT" | "UPPERCUT" | "DOWNWARD"
        """
        ...
