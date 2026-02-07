import numpy as np

def calculate_angle(a, b, c) -> float:
    """
    Returns angle at point b formed by points a-b-c in degrees.
    a, b, c are (x, y) in normalized or pixel coords.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)

def to_pixel(xy_norm, width: int, height: int):
    x, y = xy_norm
    return int(x * width), int(y * height)
