from perception.detector import personDetector
from perception.estimator import estimateTarget
from perception.targetSelector import targetSelector
from perception.tracker import personTracker


def makeVisionObservation(config):
    detector = personDetector(config)
    tracker = personTracker(config)
    selector = targetSelector(config)

    def visionObservation(rover, config, fovScale=1.0):
        frame = rover.getFrame()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        box = selector.select(tracks, frame)
        if box is None:
            return False, 0.0, 0.0
        bearingError, distance = estimateTarget(box, config)
        return True, bearingError, distance

    return visionObservation