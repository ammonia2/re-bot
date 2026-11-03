import unittest

import numpy as np

from perception.tracker import boxKalmanFilter, iouMatrix, personTracker


def personBox(centerX, confidence=0.9, top=100.0, width=60.0, height=180.0):
    return (centerX - width / 2, top, centerX + width / 2, top + height, confidence)


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.tracker = personTracker({})

    def run(self, result=None):
        with np.errstate(all="raise"):
            return super().run(result)

    def idsOf(self, output):
        return [track[0] for track in output]

    def testKalmanLearnsConstantVelocity(self):
        kalman = boxKalmanFilter(personBox(100)[:4])
        for step in range(1, 20):
            kalman.predict()
            kalman.update(personBox(100 + 5 * step)[:4])
        predicted = kalman.predict()
        self.assertAlmostEqual((predicted[0] + predicted[2]) / 2, 200, delta=1.0)

    def testIouOfIdenticalAndDisjointBoxes(self):
        overlaps = iouMatrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (20, 20, 30, 30)])
        np.testing.assert_allclose(overlaps, [[1.0, 0.0]])

    def testEmptyDetectionsGiveEmptyOutput(self):
        self.assertEqual(self.tracker.update([], self.frame), [])
        self.tracker.update([personBox(300)], self.frame)
        self.assertEqual(self.tracker.update([], self.frame), [])

    def testOutputIsDetectionBoxWithIntegerId(self):
        detection = personBox(300)
        output = self.tracker.update([detection], self.frame)
        self.assertEqual(output, [(1, *detection[:4])])
        self.assertIsInstance(output[0][0], int)
        self.assertTrue(all(isinstance(value, float) for value in output[0][1:]))

    def testWalkingPersonKeepsOneId(self):
        for step in range(60):
            output = self.tracker.update([personBox(100 + 6 * step)], self.frame)
            self.assertEqual(self.idsOf(output), [1])

    def testNewTrackWaitsForMinHitsAfterWarmup(self):
        for step in range(10):
            self.tracker.update([personBox(100 + 4 * step)], self.frame)
        outputs = [self.tracker.update([personBox(100 + 4 * step), personBox(550)], self.frame) for step in range(10, 14)]
        self.assertEqual([self.idsOf(output) for output in outputs], [[1], [1], [1], [1, 2]])

    def testConfirmedTrackIsReportedThroughFlicker(self):
        for step in range(10):
            self.tracker.update([personBox(100 + 4 * step)], self.frame)
        for step in range(10, 30):
            detections = [personBox(100 + 4 * step)] if step % 2 else []
            output = self.tracker.update(detections, self.frame)
            self.assertEqual(self.idsOf(output), [1] if step % 2 else [])

    def testShortOcclusionKeepsId(self):
        for step in range(20):
            self.tracker.update([personBox(100 + 6 * step)], self.frame)
        for step in range(20, 26):
            self.assertEqual(self.tracker.update([], self.frame), [])
        for step in range(26, 30):
            output = self.tracker.update([personBox(100 + 6 * step)], self.frame)
        self.assertEqual(self.idsOf(output), [1])

    def testIdIsNeverReusedAfterDeletion(self):
        for step in range(10):
            self.tracker.update([personBox(300)], self.frame)
        for step in range(self.tracker.maxAge + 1):
            self.tracker.update([], self.frame)
        self.assertEqual(self.tracker.tracks, [])
        for step in range(5):
            output = self.tracker.update([personBox(300)], self.frame)
        self.assertEqual(self.idsOf(output), [2])

    def testCrossingPeopleKeepTheirIds(self):
        idByDirection = {}
        for step in range(60):
            rightward, leftward = personBox(100 + 8 * step), personBox(580 - 8 * step, top=110.0)
            output = self.tracker.update([rightward, leftward], self.frame)
            for trackId, x1, y1, *rest in output:
                direction = "right" if y1 == rightward[1] else "left"
                idByDirection.setdefault(direction, set()).add(trackId)
        self.assertEqual(idByDirection, {"right": {1}, "left": {2}})

    def testOccludedCrossingKeepsIds(self):
        # the leftward person is hidden behind the other while they overlap
        idByDirection = {}
        for step in range(60):
            rightward, leftward = personBox(100 + 8 * step), personBox(580 - 8 * step, top=110.0)
            hidden = abs(rightward[0] - leftward[0]) < 60
            output = self.tracker.update([rightward] if hidden else [rightward, leftward], self.frame)
            for trackId, x1, y1, *rest in output:
                direction = "right" if y1 == rightward[1] else "left"
                idByDirection.setdefault(direction, set()).add(trackId)
        self.assertEqual(idByDirection, {"right": {1}, "left": {2}})


if __name__ == "__main__":
    unittest.main()
