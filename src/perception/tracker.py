import numpy as np


def boxToMeasurement(box):
    x1, y1, x2, y2 = box[:4]
    width, height = x2 - x1, y2 - y1
    return np.array([x1 + width / 2, y1 + height / 2, width * height, width / height])


def measurementToBox(state):
    centerX, centerY, area, ratio = state[:4]
    if area <= 0 or ratio <= 0:
        return (np.nan,) * 4
    width = np.sqrt(area * ratio)
    height = area / width
    return (centerX - width / 2, centerY - height / 2, centerX + width / 2, centerY + height / 2)


def iouMatrix(boxesA, boxesB):
    boxesA = np.asarray(boxesA, dtype=float).reshape(-1, 4)[:, None]
    boxesB = np.asarray(boxesB, dtype=float).reshape(-1, 4)[None]
    width = (np.minimum(boxesA[..., 2], boxesB[..., 2]) - np.maximum(boxesA[..., 0], boxesB[..., 0])).clip(0)
    height = (np.minimum(boxesA[..., 3], boxesB[..., 3]) - np.maximum(boxesA[..., 1], boxesB[..., 1])).clip(0)
    overlap = width * height
    areaA = (boxesA[..., 2] - boxesA[..., 0]) * (boxesA[..., 3] - boxesA[..., 1])
    areaB = (boxesB[..., 2] - boxesB[..., 0]) * (boxesB[..., 3] - boxesB[..., 1])
    return overlap / np.maximum(areaA + areaB - overlap, 1e-9)


def directionMatrix(fromBoxes, toBoxes):
    # unit vectors from each fromBox center (rows) to each toBox center (columns)
    fromBoxes = np.asarray(fromBoxes, dtype=float).reshape(-1, 4)[:, None]
    toBoxes = np.asarray(toBoxes, dtype=float).reshape(-1, 4)[None]
    delta = (toBoxes[..., :2] + toBoxes[..., 2:]) / 2 - (fromBoxes[..., :2] + fromBoxes[..., 2:]) / 2
    return delta / (np.linalg.norm(delta, axis=2, keepdims=True) + 1e-6)


def greedyMatch(scores, overlaps, threshold):
    matches, usedRows, usedColumns = [], set(), set()
    for flatIndex in np.argsort(-scores, axis=None, kind="stable"):
        row, column = (int(index) for index in np.unravel_index(flatIndex, scores.shape))
        if row in usedRows or column in usedColumns or overlaps[row, column] < threshold:
            continue
        matches.append((row, column))
        usedRows.add(row)
        usedColumns.add(column)
    return matches


class boxKalmanFilter:
    def __init__(self, box):
        # state: [centerX, centerY, area, ratio, velocityX, velocityY, velocityArea]
        self.state = np.zeros(7)
        self.state[:4] = boxToMeasurement(box)
        self.transition = np.eye(7)
        self.transition[[0, 1, 2], [4, 5, 6]] = 1
        self.observationMatrix = np.eye(4, 7)
        self.covariance = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.processNoise = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])
        self.measurementNoise = np.diag([1.0, 1.0, 10.0, 10.0])
        self.lastBox = tuple(float(value) for value in box[:4])
        self.frozen = None
        self.missed = 0

    def predict(self):
        if self.state[2] + self.state[6] <= 0:
            self.state[6] = 0
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.processNoise
        return measurementToBox(self.state)

    def update(self, box):
        if box is None:
            if self.frozen is None:
                self.frozen = (self.state.copy(), self.covariance.copy())
            self.missed += 1
            return
        if self.frozen is not None:
            self._replayGap(box)
        else:
            self._correct(boxToMeasurement(box))
        self.lastBox = tuple(float(value) for value in box[:4])
        self.missed = 0

    def _correct(self, measurement):
        residual = measurement - self.observationMatrix @ self.state
        residualCovariance = self.observationMatrix @ self.covariance @ self.observationMatrix.T + self.measurementNoise
        gain = self.covariance @ self.observationMatrix.T @ np.linalg.inv(residualCovariance)
        self.state = self.state + gain @ residual
        self.covariance = (np.eye(7) - gain @ self.observationMatrix) @ self.covariance

    def _replayGap(self, box):
        # ORU: rewind to the first missed frame, then refit along a straight line to the new box
        self.state, self.covariance = self.frozen
        self.frozen = None
        start, end = np.array(self.lastBox), np.array(box[:4], dtype=float)
        steps = self.missed + 1
        for step in range(1, steps + 1):
            self._correct(boxToMeasurement(start + (end - start) * step / steps))
            if step < steps:
                self.predict()


class personTrack:
    def __init__(self, trackId, box, frameIndex, deltaT):
        self.trackId = trackId
        self.filter = boxKalmanFilter(box)
        self.deltaT = deltaT
        self.observations = {frameIndex: self.filter.lastBox}
        self.velocity = np.zeros(2)
        self.hitStreak = 0
        self.timeSinceUpdate = 0

    def previousBox(self, frameIndex):
        # the observation about deltaT frames back, used for the OCM direction
        for gap in range(self.deltaT, 0, -1):
            if frameIndex - gap in self.observations:
                return self.observations[frameIndex - gap]
        return self.filter.lastBox

    def predict(self):
        if self.timeSinceUpdate > 0:
            self.hitStreak = 0
        self.timeSinceUpdate += 1
        return self.filter.predict()

    def hit(self, box, frameIndex):
        self.velocity = directionMatrix(self.previousBox(frameIndex), box[:4])[0, 0]
        self.filter.update(box)
        self.observations[frameIndex] = self.filter.lastBox
        self.observations = {index: value for index, value in self.observations.items() if index >= frameIndex - self.deltaT}
        self.hitStreak += 1
        self.timeSinceUpdate = 0

    def miss(self):
        self.filter.update(None)


class personTracker:
    # slimmed OC-SORT: Kalman + IoU, OCM direction term, OCR second round, ORU on re-match
    def __init__(self, config):
        self.config = config
        self.maxAge = int(config.get("trackerMaxAge", 30))
        self.minHits = int(config.get("trackerMinHits", 3))
        self.iouThreshold = float(config.get("trackerIouThreshold", 0.3))
        self.deltaT = int(config.get("trackerDeltaT", 3))
        self.inertia = float(config.get("trackerInertia", 0.2))
        self.tracks = []
        self.nextId = 1
        self.frameCount = 0

    def update(self, detections, frame):
        # return [(trackId, x1, y1, x2, y2), ...] with stable integer ids
        self.frameCount += 1
        predicted = [track.predict() for track in self.tracks]
        alive = [index for index, box in enumerate(predicted) if np.all(np.isfinite(box))]
        self.tracks = [self.tracks[index] for index in alive]
        predicted = np.array([predicted[index] for index in alive]).reshape(-1, 4)

        rows = np.asarray(detections, dtype=float).reshape(-1, 5)
        boxes, scores = rows[:, :4], rows[:, 4]
        matches, unmatchedDetections, unmatchedTracks = self._associate(boxes, scores, predicted)

        for detectionIndex, trackIndex in matches:
            self.tracks[trackIndex].hit(boxes[detectionIndex], self.frameCount)
        for trackIndex in unmatchedTracks:
            self.tracks[trackIndex].miss()
        for detectionIndex in unmatchedDetections:
            self.tracks.append(personTrack(self.nextId, boxes[detectionIndex], self.frameCount, self.deltaT))
            self.nextId += 1

        output = [
            (track.trackId, *track.filter.lastBox)
            for track in self.tracks
            if track.timeSinceUpdate == 0 and (track.hitStreak >= self.minHits or self.frameCount <= self.minHits)
        ]
        self.tracks = [track for track in self.tracks if track.timeSinceUpdate <= self.maxAge]
        return output

    def _associate(self, boxes, scores, predicted):
        overlaps = iouMatrix(boxes, predicted)
        consistency = np.zeros_like(overlaps)
        if overlaps.size:
            previous = [track.previousBox(self.frameCount) for track in self.tracks]
            directions = directionMatrix(previous, boxes)
            velocities = np.array([track.velocity for track in self.tracks])
            cosine = np.clip((directions * velocities[:, None]).sum(axis=2), -1, 1)
            angleScore = (np.pi / 2 - np.arccos(cosine)) / np.pi
            consistency = self.inertia * angleScore.T * scores[:, None]
        matches = greedyMatch(overlaps + consistency, overlaps, self.iouThreshold)

        # OCR: give lost tracks a second chance against their last real box
        unmatchedDetections = [index for index in range(len(boxes)) if index not in {match[0] for match in matches}]
        unmatchedTracks = [index for index in range(len(self.tracks)) if index not in {match[1] for match in matches}]
        if unmatchedDetections and unmatchedTracks:
            lastBoxes = [self.tracks[index].filter.lastBox for index in unmatchedTracks]
            rescueOverlaps = iouMatrix(boxes[unmatchedDetections], lastBoxes)
            for row, column in greedyMatch(rescueOverlaps, rescueOverlaps, self.iouThreshold):
                matches.append((unmatchedDetections[row], unmatchedTracks[column]))
            unmatchedDetections = [index for index in unmatchedDetections if index not in {match[0] for match in matches}]
            unmatchedTracks = [index for index in unmatchedTracks if index not in {match[1] for match in matches}]
        return matches, unmatchedDetections, unmatchedTracks
