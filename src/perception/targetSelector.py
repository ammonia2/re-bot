class targetSelector:
    def __init__(self, config):
        self.config = config
        self.targetId = None

    def select(self, tracks, frame):
        # enroll the track nearest the image center on the first call that has tracks
        # return (x1, y1, x2, y2) of the target, or None when it is not visible
        raise NotImplementedError

    def reset(self):
        self.targetId = None
