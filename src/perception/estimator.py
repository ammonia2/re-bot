import math

# a box edge this close to the frame border is treated as cut off
cropMarginPixels = 2.0
# rays closer to the horizon than this give too little depth resolution to trust
minRayAngle = 0.03
heightRange = (0.5, 2.5)


def focalLengthPixels(config):
    width, fov = float(config["cameraWidth"]), float(config["cameraFov"])
    if not math.isfinite(width) or width <= 0 or not math.isfinite(fov) or not 0 < fov < math.pi:
        raise ValueError("cameraWidth must be positive and cameraFov must be in (0, pi)")
    return (width / 2) / math.tan(fov / 2)


def validBox(box):
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("box must contain finite coordinates with positive width and height")
    return x1, y1, x2, y2


def validHeight(personHeight):
    personHeight = float(personHeight)
    if not math.isfinite(personHeight) or personHeight <= 0:
        raise ValueError("personHeightMeters must be finite and positive")
    return personHeight


def bearingOf(box, config, focal):
    # right of image center means clockwise, so the bearing is negative
    centerX = (box[0] + box[2]) / 2
    return -math.atan((centerX - config["cameraWidth"] / 2) / focal)


def estimateTarget(box, config, personHeight=None):
    """Return left-positive bearing and horizontal camera range for a full upright body."""
    x1, y1, x2, y2 = validBox(box)
    personHeight = validHeight(config["personHeightMeters"] if personHeight is None else personHeight)
    focal = focalLengthPixels(config)
    bearingError = bearingOf((x1, y1, x2, y2), config, focal)
    depth = focal * personHeight / (y2 - y1)
    return bearingError, depth / math.cos(bearingError)


def rayAngles(box, config, focal):
    # angles of the feet below and the head above the horizon, and whether each edge is cut by the frame
    _, y1, _, y2 = box
    imageHeight = float(config["cameraHeight"])
    centerY = imageHeight / 2
    pitch = float(config.get("cameraPitch", 0.0))
    feetAngle = math.atan((y2 - centerY) / focal) + pitch
    headAngle = math.atan((centerY - y1) / focal) - pitch
    return feetAngle, headAngle, y1 <= cropMarginPixels, y2 >= imageHeight - cropMarginPixels


def estimateTargetWithCrop(box, config, personHeight=None):
    """Like estimateTarget, but uses whichever box edges are inside the frame.

    Returns (bearingError, distance), with distance None when the visible edges cannot give a range.
    Without cameraMountHeight only a full body gives a range.
    """
    x1, y1, x2, y2 = validBox(box)
    personHeight = validHeight(config["personHeightMeters"] if personHeight is None else personHeight)
    focal = focalLengthPixels(config)
    bearingError = bearingOf((x1, y1, x2, y2), config, focal)
    feetAngle, headAngle, topCut, bottomCut = rayAngles((x1, y1, x2, y2), config, focal)
    mountHeight = config.get("cameraMountHeight")

    depth = None
    if not topCut and not bottomCut:
        spread = math.tan(feetAngle) + math.tan(headAngle)
        depth = personHeight / spread if spread > 0 else None
    elif mountHeight is not None and not topCut:
        # feet out of frame: range from the head ray and the known height above the camera
        aboveCamera = personHeight - float(mountHeight)
        if abs(headAngle) >= minRayAngle and aboveCamera * headAngle > 0:
            depth = aboveCamera / math.tan(headAngle)
    elif mountHeight is not None and not bottomCut:
        # head out of frame: range from the feet ray on flat ground, independent of the person's height
        if feetAngle >= minRayAngle:
            depth = float(mountHeight) / math.tan(feetAngle)
    if depth is None or not math.isfinite(depth) or depth <= 0:
        return bearingError, None
    return bearingError, depth / math.cos(bearingError)


def personHeightFromBox(box, config):
    """Height of a fully visible person standing on flat ground, or None when it cannot be measured."""
    if config.get("cameraMountHeight") is None:
        return None
    box = validBox(box)
    feetAngle, headAngle, topCut, bottomCut = rayAngles(box, config, focalLengthPixels(config))
    if topCut or bottomCut or feetAngle < minRayAngle:
        return None
    height = float(config["cameraMountHeight"]) * (1 + math.tan(headAngle) / math.tan(feetAngle))
    return height if heightRange[0] <= height <= heightRange[1] else None
