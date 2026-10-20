class personTracker:
    def __init__(self, config):
        self.config = config

    def update(self, detections, frame):
        # return [(trackId, x1, y1, x2, y2), ...] with stable integer ids
        raise NotImplementedError