from dataclasses import dataclass

@dataclass
class PoseConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    camera_index: int = 0

@dataclass
class CurlConfig:
    up_angle: float = 30.0      # elbow angle when "up"
    down_angle: float = 160.0   # elbow angle when "down"
    visibility_threshold: float = 0.6
    
@dataclass
class SquatConfig:
    up_angle: float = 165.0      # knee angle when standing
    down_angle: float = 95.0     # knee angle when squatting
    visibility_threshold: float = 0.6

@dataclass
class ShoulderPressConfig:
    up_angle: float = 170.0      # arms fully overhead
    down_angle: float = 80.0     # arms at shoulder level
    visibility_threshold: float = 0.6
