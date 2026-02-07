from dataclasses import dataclass
from .utils import calculate_angle

@dataclass
class RepState:
    counter: int = 0
    stage: str | None = None   # "up" or "down"

class BicepCurlCounter:
    """
    Counts reps using elbow angle.
    """
    def __init__(self, up_angle=30.0, down_angle=160.0, visibility_threshold=0.6):
        self.up_angle = up_angle
        self.down_angle = down_angle
        self.visibility_threshold = visibility_threshold
        self.state = RepState()

    def update(self, shoulder, elbow, wrist, visibility: float):
        """
        shoulder, elbow, wrist: (x, y) normalized
        visibility: float for elbow landmark visibility
        Returns dict with angle, counter, stage
        """
        angle = calculate_angle(shoulder, elbow, wrist)

        # If landmark is not reliable, don't change state
        if visibility < self.visibility_threshold:
            return {"angle": angle, "counter": self.state.counter, "stage": self.state.stage, "reliable": False}

        # Stage logic
        if angle > self.down_angle:
            self.state.stage = "down"
        if angle < self.up_angle and self.state.stage == "down":
            self.state.stage = "up"
            self.state.counter += 1

        return {"angle": angle, "counter": self.state.counter, "stage": self.state.stage, "reliable": True}

class SquatCounter:
    """
    Counts squats using knee angle (hip-knee-ankle).
    """
    def __init__(self, up_angle=165.0, down_angle=95.0, visibility_threshold=0.6):
        self.up_angle = up_angle
        self.down_angle = down_angle
        self.visibility_threshold = visibility_threshold
        self.state = RepState()

    def update(self, hip, knee, ankle, visibility: float):
        angle = calculate_angle(hip, knee, ankle)

        if visibility < self.visibility_threshold:
            return {"angle": angle, "counter": self.state.counter, "stage": self.state.stage, "reliable": False}

        # Standing
        if angle > self.up_angle:
            self.state.stage = "up"

        # Completed rep when going down after being up
        if angle < self.down_angle and self.state.stage == "up":
            self.state.stage = "down"
            self.state.counter += 1

        return {"angle": angle, "counter": self.state.counter, "stage": self.state.stage, "reliable": True}

class ShoulderPressCounter:
    """
    Counts shoulder presses using shoulder–elbow–wrist angle.
    """
    def __init__(self, up_angle=170.0, down_angle=80.0, visibility_threshold=0.6):
        self.up_angle = up_angle
        self.down_angle = down_angle
        self.visibility_threshold = visibility_threshold
        self.state = RepState()

    def update(self, shoulder, elbow, wrist, visibility: float):
        angle = calculate_angle(shoulder, elbow, wrist)

        if visibility < self.visibility_threshold:
            return {"angle": angle, "counter": self.state.counter,
                    "stage": self.state.stage, "reliable": False}

        # Arms fully up
        if angle > self.up_angle:
            self.state.stage = "up"

        # Completed rep when coming down after being up
        if angle < self.down_angle and self.state.stage == "up":
            self.state.stage = "down"
            self.state.counter += 1

        return {"angle": angle, "counter": self.state.counter,
                "stage": self.state.stage, "reliable": True}
