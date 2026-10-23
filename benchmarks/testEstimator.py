import math
import unittest

from perception.estimator import estimateTarget, focalLengthPixels


class TestEstimator(unittest.TestCase):
    def setUp(self):
        self.config = {"cameraWidth": 640, "cameraFov": math.pi / 4, "personHeightMeters": 1.838331}

    def testKnownFocalLength(self):
        self.assertAlmostEqual(focalLengthPixels(self.config), 772.5483399593904)

    def testProjectedTargetsRecoverBearingAndRange(self):
        focal = focalLengthPixels(self.config)
        for bearing in (-0.35, -0.1, 0.0, 0.1, 0.35):
            for distance in (2.0, 4.0, 6.0):
                with self.subTest(bearing=bearing, distance=distance):
                    depth = distance * math.cos(bearing)
                    center = 320 - focal * math.tan(bearing)
                    height = focal * self.config["personHeightMeters"] / depth
                    box = (center - 20, 240 - height / 2, center + 20, 240 + height / 2)
                    actualBearing, actualDistance = estimateTarget(box, self.config)
                    self.assertAlmostEqual(actualBearing, bearing)
                    self.assertAlmostEqual(actualDistance, distance)

    def testRightOfCenterHasNegativeBearing(self):
        bearing, distance = estimateTarget((400, 100, 500, 400), self.config)
        self.assertLess(bearing, 0)
        self.assertGreater(distance, 0)

    def testInvalidBoxesAreRejected(self):
        for box in ((0, 0, 0, 10), (0, 5, 10, 5), (10, 0, 0, 10), (0, 0, math.inf, 10)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                estimateTarget(box, self.config)

    def testInvalidCameraCalibrationIsRejected(self):
        for key, value in (("cameraWidth", 0), ("cameraFov", 0), ("cameraFov", math.pi), ("cameraFov", math.nan)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                focalLengthPixels({**self.config, key: value})

    def testInvalidPersonHeightIsRejected(self):
        for height in (0, -1, math.nan):
            with self.subTest(height=height), self.assertRaises(ValueError):
                estimateTarget((0, 0, 20, 100), {**self.config, "personHeightMeters": height})

    def testCroppingOverestimatesDistance(self):
        full = estimateTarget((300, 0, 340, 400), self.config)[1]
        cropped = estimateTarget((300, 0, 340, 200), self.config)[1]
        self.assertAlmostEqual(cropped, full * 2)


if __name__ == "__main__":
    unittest.main()
