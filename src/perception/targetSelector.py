from collections import deque

import cv2

from perception.tracker import iouMatrix


def appearanceOf(frame, box):
    # hue-saturation histograms of the torso and the legs, inset to keep out background
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box[:4]
    inset = (x2 - x1) * 0.2
    left, right = int(max(x1 + inset, 0)), int(min(x2 - inset, width))
    parts = []
    for top, bottom in ((0.15, 0.55), (0.55, 0.95)):
        upper, lower = int(max(y1 + (y2 - y1) * top, 0)), int(min(y1 + (y2 - y1) * bottom, height))
        if right - left < 2 or lower - upper < 2:
            return None
        hsv = cv2.cvtColor(frame[upper:lower, left:right], cv2.COLOR_BGR2HSV)
        parts.append(cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]))
    return parts


def appearanceSimilarity(featureA, featureB):
    # 1 is identical, 0 shares no colour
    distances = [cv2.compareHist(partA, partB, cv2.HISTCMP_BHATTACHARYYA) for partA, partB in zip(featureA, featureB)]
    return 1.0 - sum(distances) / len(distances)


class targetSelector:
    def __init__(self, config):
        self.config = config
        self.matchThreshold = float(config.get("selectorMatchThreshold", 0.65))
        self.rejectThreshold = float(config.get("selectorRejectThreshold", 0.35))
        self.margin = float(config.get("selectorMargin", 0.1))
        self.confirmFrames = int(config.get("selectorConfirmFrames", 3))
        self.gallerySize = int(config.get("selectorGallerySize", 20))
        self.reset()

    def select(self, tracks, frame):
        # enroll the track nearest the image center on the first call that has tracks
        # return (x1, y1, x2, y2) of the target, or None when it is not visible
        boxes = {trackId: tuple(float(value) for value in box) for trackId, *box in tracks}
        if not boxes:
            self.candidateId, self.candidateCount = None, 0
            return None
        if self.targetId is None and not self.gallery:
            return self._enroll(boxes, frame)
        if self.targetId in boxes:
            return self._follow(boxes, frame)
        return self._reacquire(boxes, frame)

    def reset(self):
        self.targetId = None
        self.gallery = deque(maxlen=self.gallerySize)
        self.otherIds = set()
        self.candidateId, self.candidateCount = None, 0

    def _enroll(self, boxes, frame):
        centerX = frame.shape[1] / 2
        self.targetId = min(boxes, key=lambda trackId: abs((boxes[trackId][0] + boxes[trackId][2]) / 2 - centerX))
        feature = appearanceOf(frame, boxes[self.targetId])
        if feature is not None:
            self.gallery.append(feature)
        return boxes[self.targetId]

    def _follow(self, boxes, frame):
        box = boxes[self.targetId]
        # anyone visible at the same time as the target is someone else
        self.otherIds.update(trackId for trackId in boxes if trackId != self.targetId)
        if not self._isClean(box, boxes, frame):
            return box
        feature = appearanceOf(frame, box)
        similarity = self._similarity(feature)
        if self.gallery and similarity < self.rejectThreshold:
            # the tracker likely moved this id onto someone else, so ids can no longer be trusted
            self.targetId = None
            self.otherIds.clear()
            return None
        if similarity >= self.matchThreshold or not self.gallery:
            self.gallery.append(feature)
        return box

    def _reacquire(self, boxes, frame):
        similarities = {trackId: self._similarity(appearanceOf(frame, box)) for trackId, box in boxes.items()}
        candidates = [trackId for trackId in boxes if trackId not in self.otherIds]
        bestId = max(candidates, key=similarities.get, default=None)
        rival = max((value for trackId, value in similarities.items() if trackId != bestId), default=0.0)
        if bestId is None or similarities[bestId] < self.matchThreshold or similarities[bestId] - rival < self.margin:
            self.candidateId, self.candidateCount = None, 0
            return None
        if bestId != self.candidateId:
            self.candidateId, self.candidateCount = bestId, 0
        self.candidateCount += 1
        if self.candidateCount < self.confirmFrames:
            return None
        self.targetId = bestId
        self.candidateId, self.candidateCount = None, 0
        return boxes[bestId]

    def _isClean(self, box, boxes, frame):
        # a box cut by the frame edge or overlapped by someone else gives a misleading appearance
        height, width = frame.shape[:2]
        if box[0] <= 1 or box[1] <= 1 or box[2] >= width - 1 or box[3] >= height - 1:
            return False
        others = [other for trackId, other in boxes.items() if trackId != self.targetId]
        return not others or iouMatrix([box], others).max() == 0

    def _similarity(self, feature):
        if feature is None or not self.gallery:
            return 0.0
        return max(appearanceSimilarity(feature, sample) for sample in self.gallery)
