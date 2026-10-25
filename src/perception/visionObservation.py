import math

from perception.detector import personDetector
from perception.estimator import estimateTarget
from perception.targetSelector import targetSelector
from perception.tracker import personTracker


def toRoverFrame(bearingError, distance, config):
    # the estimator measures from the camera, the controller expects range from the rover center
    offset = config.get("cameraForwardOffset", 0.0)
    forward = distance * math.cos(bearingError) + offset
    left = distance * math.sin(bearingError)
    return math.atan2(left, forward), math.hypot(forward, left)


def boxToObservation(box, config, fovScale):
    if box is None:
        return False, 0.0, 0.0
    bearingError, distance = toRoverFrame(*estimateTarget(box, config), config)
    # narrower cone while searching, same as the ground-truth observation
    visible = abs(bearingError) < config["cameraHalfFov"] * fovScale
    return visible, bearingError, distance


def makeVisionObservation(config):
    detector = personDetector(config)
    tracker = personTracker(config)
    selector = targetSelector(config)

    def visionObservation(rover, config, fovScale=1.0):
        frame = rover.getFrame()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        box = selector.select(tracks, frame)
        return boxToObservation(box, config, fovScale)

    return visionObservation


def makeDirectVisionObservation(config):
    # temporary integration stand-in: no tracking or identity, just the most central person box
    detector = personDetector(config)
    centerX = config["cameraWidth"] / 2

    def directVisionObservation(rover, config, fovScale=1.0):
        detections = detector.detect(rover.getFrame())
        box = min(detections, key=lambda d: abs((d[0] + d[2]) / 2 - centerX), default=None)
        return boxToObservation(box and box[:4], config, fovScale)

    return directVisionObservation
