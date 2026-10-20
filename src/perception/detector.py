class personDetector:
    def __init__(self, config):
        self.config = config
        # load the model here: person class only, detectorInputSize

    def detect(self, frame):
        # frame: BGR uint8 array (height, width, 3)
        # return [(x1, y1, x2, y2, confidence), ...] in pixels
        raise NotImplementedError