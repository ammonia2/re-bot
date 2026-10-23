import math


def focalLengthPixels(config):
    width, fov = float(config["cameraWidth"]), float(config["cameraFov"])
    if not math.isfinite(width) or width <= 0 or not math.isfinite(fov) or not 0 < fov < math.pi:
        raise ValueError("cameraWidth must be positive and cameraFov must be in (0, pi)")
    return (width / 2) / math.tan(fov / 2)


def estimateTarget(box, config):
    """Return left-positive bearing and horizontal camera range for a full upright body."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("box must contain finite coordinates with positive width and height")
    personHeight = float(config["personHeightMeters"])
    if not math.isfinite(personHeight) or personHeight <= 0:
        raise ValueError("personHeightMeters must be finite and positive")
    focal = focalLengthPixels(config)
    centerX = (x1 + x2) / 2
    # right of image center means clockwise, so the bearing is negative
    bearingError = -math.atan((centerX - config["cameraWidth"] / 2) / focal)
    depth = focal * personHeight / (y2 - y1)
    distance = depth / math.cos(bearingError)
    return bearingError, distance
