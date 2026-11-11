import math
import unittest
from pathlib import Path

import numpy as np
import yaml

from core.followStateMachine import followStateMachine, roverState
from core.obstacleAvoider import obstacleAvoider
from core.odometry import odometry

config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/robotConfig.yaml").read_text())
angles = np.radians(np.linspace(120, -120, 241))


def scanOf(points, maxRange=8.0):
    # range scan of small round posts at (x, y) in the rover frame, x forward and y to the left
    ranges = np.full(len(angles), np.inf)
    for x, y in points:
        distance, bearing = math.hypot(x, y), math.atan2(y, x)
        width = math.asin(min(1.0, 0.06 / distance))
        hit = np.abs(angles - bearing) <= max(width, math.radians(0.6))
        ranges[hit] = np.minimum(ranges[hit], distance)
    return angles, ranges


def wallWithGap(distance, gapWidth):
    # a wall across the path with an opening centred straight ahead
    return scanOf([(distance, y) for y in np.arange(-4, 4.001, 0.05) if abs(y) > gapWidth / 2])


class TestObstacleAvoider(unittest.TestCase):
    def setUp(self):
        self.avoider = obstacleAvoider(config)

    def testOpenSpaceDrivesStraightAtTheGoal(self):
        steer, speedScale, blocked = self.avoider.plan(scanOf([]), 0.2, 5.0)
        self.assertEqual((steer, speedScale, blocked), (0.2, 1.0, False))

    def testSteersAroundAnObstacleInTheWay(self):
        steer, _, blocked = self.avoider.plan(scanOf([(1.2, 0.0), (1.2, 0.1), (1.2, -0.1)]), 0.0, 5.0)
        self.assertFalse(blocked)
        self.assertGreater(abs(steer), math.radians(15))

    def testDoorwayIsPassable(self):
        steer, _, blocked = self.avoider.plan(wallWithGap(1.5, 1.5), 0.0, 5.0)
        self.assertFalse(blocked)
        self.assertLess(abs(steer), math.radians(5))

    def testGapNarrowerThanTheRoverIsRefused(self):
        _, speedScale, blocked = self.avoider.plan(wallWithGap(1.5, 0.5), 0.0, 5.0)
        self.assertTrue(blocked)
        self.assertEqual(speedScale, 0.0)

    def testObstaclesBeyondTheGoalAreIgnored(self):
        steer, _, blocked = self.avoider.plan(scanOf([(1.5, 0.0)]), 0.0, 1.0)
        self.assertEqual((steer, blocked), (0.0, False))

    def testSomethingRightInFrontStopsTheRover(self):
        _, _, blocked = self.avoider.plan(scanOf([(0.3, 0.05)]), 1.0, 5.0)
        self.assertTrue(blocked)


class TestOdometry(unittest.TestCase):
    def testStraightLineAndTurnOnTheSpot(self):
        tracker = odometry(config)
        tracker.update(0.0, 0.0)
        x, y, yaw = tracker.update(10.0, 10.0)
        self.assertAlmostEqual(x, 10.0 * config["wheelRadius"])
        self.assertAlmostEqual(y, 0.0)
        quarterTurn = (math.pi / 2) * (config["wheelBase"] / 2) / config["wheelRadius"]
        x, y, yaw = tracker.update(10.0 - quarterTurn, 10.0 + quarterTurn)
        self.assertAlmostEqual(yaw, math.pi / 2)
        self.assertAlmostEqual(x, 10.0 * config["wheelRadius"])


class TestFollowStateMachine(unittest.TestCase):
    def setUp(self):
        self.machine = followStateMachine(config)
        self.pose = (0.0, 0.0, 0.0)
        self.clear = scanOf([])

    def testStaysStillUntilTold(self):
        self.assertEqual(self.machine.update(True, 0.0, 6.0, self.pose, self.clear), (0.0, 0.0))
        self.assertEqual(self.machine.state, roverState.idle)

    def testFollowsAndKeepsItsDistance(self):
        self.machine.resume()
        left, right = self.machine.update(True, 0.0, 6.0, self.pose, self.clear)
        self.assertEqual(self.machine.state, roverState.following)
        self.assertGreater(left, 0)
        self.assertEqual(left, right)
        self.assertEqual(self.machine.update(True, 0.0, config["targetFollowDistance"] - 0.5, self.pose, self.clear), (0.0, 0.0))

    def testStopCommandHaltsAndGoResumes(self):
        self.machine.resume()
        self.machine.pause()
        self.assertEqual(self.machine.update(True, 0.0, 6.0, self.pose, self.clear), (0.0, 0.0))
        self.machine.resume()
        self.assertGreater(self.machine.update(True, 0.0, 6.0, self.pose, self.clear)[0], 0)

    def testGoesToWhereTheTargetWasLastSeenThenSearches(self):
        self.machine.resume()
        self.machine.update(True, 0.0, 5.0, self.pose, self.clear)
        for _ in range(config["lostGraceSteps"] + 1):
            left, right = self.machine.update(False, 0.0, 0.0, self.pose, self.clear)
        self.assertEqual(self.machine.state, roverState.pursuing)
        self.assertGreater(left + right, 0)
        self.machine.update(False, 0.0, 0.0, (4.8, 0.0, 0.0), self.clear)
        self.assertEqual(self.machine.state, roverState.searching)

    def testWaitsAtAGapItCannotFitThrough(self):
        self.machine.resume()
        left, right = self.machine.update(True, 0.0, 6.0, self.pose, wallWithGap(1.5, 0.5))
        self.assertEqual(self.machine.state, roverState.blocked)
        self.assertEqual(left + right, 0.0)

    def testGivesUpSearchingEventually(self):
        self.machine.resume()
        for _ in range(config["searchTimeoutSteps"] + 2):
            speeds = self.machine.update(False, 0.0, 0.0, self.pose, self.clear)
        self.assertEqual((self.machine.state, speeds), (roverState.idle, (0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
