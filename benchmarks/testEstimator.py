import math
import unittest

from perception.estimator import estimateTarget, estimateTargetWithCrop, focalLengthPixels, personHeightFromBox

mountedConfig = {"cameraWidth": 640, "cameraHeight": 480, "cameraFov": math.pi / 4,
                 "personHeightMeters": 1.84, "cameraMountHeight": 0.9975}


def projectPerson(depth, height, config=mountedConfig):
    # box of an upright person standing on flat ground, clipped to the frame like the detector does
    focal = focalLengthPixels(config)
    top = 240 - focal * (height - config["cameraMountHeight"]) / depth
    bottom = 240 + focal * config["cameraMountHeight"] / depth
    return (300.0, max(top, 0.0), 340.0, min(bottom, 480.0))


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


class TestCropAwareEstimator(unittest.TestCase):
    def testFullBodyMatchesHeightFormula(self):
        box = projectPerson(5.0, 1.84)
        self.assertAlmostEqual(estimateTargetWithCrop(box, mountedConfig)[1], 5.0)
        self.assertAlmostEqual(estimateTargetWithCrop(box, mountedConfig)[1], estimateTarget(box, mountedConfig)[1])

    def testFeetCutUsesHeadRay(self):
        box = projectPerson(2.9, 1.84)
        self.assertEqual(box[3], 480.0)
        self.assertAlmostEqual(estimateTargetWithCrop(box, mountedConfig)[1], 2.9)
        # the plain height formula would think the person is further than they are
        self.assertGreater(estimateTarget(box, mountedConfig)[1], 2.9)

    def testHeadCutUsesFeetRayWhateverTheHeight(self):
        config = {**mountedConfig, "cameraHeight": 480}
        for height in (1.84, 2.4):
            with self.subTest(height=height):
                box = projectPerson(3.5, height, config)
                box = (box[0], 0.0, box[2], box[3])
                self.assertAlmostEqual(estimateTargetWithCrop(box, config, personHeight=1.0)[1], 3.5)

    def testCutAtBothEndsHasNoRange(self):
        self.assertIsNone(estimateTargetWithCrop((300, 0, 340, 480), mountedConfig)[1])

    def testCroppedBoxWithoutMountHeightHasNoRange(self):
        config = {key: value for key, value in mountedConfig.items() if key != "cameraMountHeight"}
        self.assertIsNone(estimateTargetWithCrop(projectPerson(2.9, 1.84), config)[1])
        self.assertAlmostEqual(estimateTargetWithCrop(projectPerson(5.0, 1.84), config)[1], 5.0)

    def testMeasuresHeightOfFullyVisiblePeople(self):
        for height in (1.1, 1.6, 1.84, 2.0):
            for depth in (4.0, 6.0):
                with self.subTest(height=height, depth=depth):
                    self.assertAlmostEqual(personHeightFromBox(projectPerson(depth, height), mountedConfig), height)

    def testNoHeightFromCroppedBoxOrWithoutMount(self):
        self.assertIsNone(personHeightFromBox(projectPerson(2.9, 1.84), mountedConfig))
        config = {key: value for key, value in mountedConfig.items() if key != "cameraMountHeight"}
        self.assertIsNone(personHeightFromBox(projectPerson(5.0, 1.84), config))

    def testLearnedHeightFixesRangeForShortPerson(self):
        box = projectPerson(5.0, 1.6)
        self.assertAlmostEqual(estimateTargetWithCrop(box, mountedConfig)[1], 5.0 * 1.84 / 1.6)
        self.assertAlmostEqual(estimateTargetWithCrop(box, mountedConfig, personHeight=1.6)[1], 5.0)

    def testPitchedCameraRecoversRange(self):
        pitch = 0.1
        config = {**mountedConfig, "cameraPitch": pitch}
        focal = focalLengthPixels(config)
        depth, height = 5.0, 1.84
        feet = 240 + focal * math.tan(math.atan(config["cameraMountHeight"] / depth) - pitch)
        head = 240 - focal * math.tan(math.atan((height - config["cameraMountHeight"]) / depth) + pitch)
        self.assertAlmostEqual(estimateTargetWithCrop((300, head, 340, feet), config)[1], depth)
        self.assertAlmostEqual(personHeightFromBox((300, head, 340, feet), config), height)


if __name__ == "__main__":
    unittest.main()
