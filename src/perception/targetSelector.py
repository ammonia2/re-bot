from collections import deque

import cv2
import numpy as np

from perception.estimator import personHeightFromBox
from perception.tracker import iouMatrix

# height samples needed before the learned height replaces personHeightMeters
minHeightSamples = 5


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


# OpenCV hue ranges (0 to 180) for the shirt colours a person can be picked by
colourHues = {
    "red": ((0, 8), (170, 180)),
    "orange": ((8, 20),),
    "yellow": ((20, 35),),
    "green": ((35, 85),),
    "blue": ((90, 130),),
    "purple": ((130, 160),),
}


def colourShare(frame, box, colour):
    # share of the torso that is the named colour, 0 to 1
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box[:4]
    inset = (x2 - x1) * 0.2
    left, right = int(max(x1 + inset, 0)), int(min(x2 - inset, width))
    upper, lower = int(max(y1 + (y2 - y1) * 0.15, 0)), int(min(y1 + (y2 - y1) * 0.55, height))
    if right - left < 2 or lower - upper < 2:
        return 0.0
    hsv = cv2.cvtColor(frame[upper:lower, left:right], cv2.COLOR_BGR2HSV)
    hue, vivid = hsv[..., 0], (hsv[..., 1] >= 90) & (hsv[..., 2] >= 50)
    inRange = np.zeros(hue.shape, dtype=bool)
    for low, high in colourHues[colour]:
        inRange |= (hue >= low) & (hue < high)
    return float((inRange & vivid).mean())


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
        self.colourMinShare = float(config.get("selectorColourMinShare", 0.3))
        self.reset()

    def select(self, tracks, frame):
        # enroll the track nearest the image center on the first call that has tracks
        # return (x1, y1, x2, y2) of the target, or None when it is not visible
        boxes = {trackId: tuple(float(value) for value in box) for trackId, *box in tracks}
        if not boxes:
            self.candidateId, self.candidateCount = None, 0
            return None
        if self.request is not None:
            return self._enrollRequested(boxes, frame)
        if self.targetId is None and not self.gallery:
            return self._enroll(boxes, frame)
        if self.targetId in boxes:
            return self._follow(boxes, frame)
        return self._reacquire(boxes, frame)

    def reset(self):
        self.targetId = None
        self.gallery = deque(maxlen=self.gallerySize)
        self.heights = deque(maxlen=50)
        self.otherIds = set()
        self.candidateId, self.candidateCount = None, 0
        self.request = None
        self.targetColour = None
        self.avoidId, self.avoidGallery = None, []

    def requestTarget(self, mode, colour=None):
        # drop the current target and enroll a new one as soon as a matching person is in view
        # mode "center": the person nearest the image center
        # mode "colour": the person whose shirt is the named colour (a key of colourHues)
        # mode "other": anyone but the current target
        if mode not in ("center", "colour", "other") or (mode == "colour" and colour not in colourHues):
            raise ValueError(f"unknown target request: {mode} {colour}")
        avoidId, avoidGallery = self.targetId, list(self.gallery)
        self.reset()
        self.request = (mode, colour)
        self.targetColour = colour if mode == "colour" else None
        if mode == "other":
            self.avoidId, self.avoidGallery = avoidId, avoidGallery

    def hasTarget(self):
        return self.request is None and (self.targetId is not None or bool(self.gallery))

    def _enrollRequested(self, boxes, frame):
        mode, colour = self.request
        if mode == "colour":
            shares = {trackId: colourShare(frame, box, colour) for trackId, box in boxes.items()}
            bestId = max(shares, key=shares.get)
            choices = [bestId] if shares[bestId] >= self.colourMinShare else []
        elif mode == "other":
            choices = [trackId for trackId, box in boxes.items() if trackId != self.avoidId and not self._looksAvoided(frame, box)]
        else:
            choices = list(boxes)
        if not choices:
            self.candidateId, self.candidateCount = None, 0
            return None
        centerX = frame.shape[1] / 2
        bestId = min(choices, key=lambda trackId: abs((boxes[trackId][0] + boxes[trackId][2]) / 2 - centerX))
        if bestId != self.candidateId:
            self.candidateId, self.candidateCount = bestId, 0
        self.candidateCount += 1
        if mode != "center" and self.candidateCount < self.confirmFrames:
            return None
        self.request = None
        self.candidateId, self.candidateCount = None, 0
        return self._enroll(boxes, frame, bestId)

    def _looksAvoided(self, frame, box):
        feature = appearanceOf(frame, box)
        if feature is None or not self.avoidGallery:
            return False
        return max(appearanceSimilarity(feature, sample) for sample in self.avoidGallery) >= self.matchThreshold

    def _enroll(self, boxes, frame, targetId=None):
        centerX = frame.shape[1] / 2
        nearestCenter = min(boxes, key=lambda trackId: abs((boxes[trackId][0] + boxes[trackId][2]) / 2 - centerX))
        self.targetId = nearestCenter if targetId is None else targetId
        feature = appearanceOf(frame, boxes[self.targetId])
        if feature is not None:
            self.gallery.append(feature)
            if self._isClean(boxes[self.targetId], boxes, frame):
                self._learnHeight(boxes[self.targetId])
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
            self._learnHeight(box)
        return box

    def _reacquire(self, boxes, frame):
        similarities = {trackId: self._similarity(appearanceOf(frame, box)) for trackId, box in boxes.items()}
        if self.targetColour is not None:
            # a target picked by shirt colour can also be found again by it, whatever the lighting does
            for trackId, box in boxes.items():
                similarities[trackId] = max(similarities[trackId], colourShare(frame, box, self.targetColour))
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

    def targetHeight(self):
        # median of the target's measured heights, or None until there are enough
        return float(np.median(self.heights)) if len(self.heights) >= minHeightSamples else None

    def similarityTo(self, frame, box):
        # appearance similarity of a box to the enrolled target, 0 to 1
        return self._similarity(appearanceOf(frame, box))

    def _learnHeight(self, box):
        height = personHeightFromBox(box, self.config)
        if height is not None:
            self.heights.append(height)

    def _similarity(self, feature):
        if feature is None or not self.gallery:
            return 0.0
        return max(appearanceSimilarity(feature, sample) for sample in self.gallery)
