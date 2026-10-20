import math


def focalLengthPixels(config):
    return (config["cameraWidth"] / 2) / math.tan(config["cameraFov"] / 2)


def estimateTarget(box, config):
    x1, y1, x2, y2 = box
    focal = focalLengthPixels(config)
    centerX = (x1 + x2) / 2
    # right of image center means clockwise, so the bearing is negative
    bearingError = -math.atan((centerX - config["cameraWidth"] / 2) / focal)
    distance = focal * config["personHeightMeters"] / max(y2 - y1, 1.0)
    return bearingError, distance