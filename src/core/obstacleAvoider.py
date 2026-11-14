import math

import numpy as np

candidateStep = math.radians(2.5)
candidateLimit = math.radians(100)


def wrapAngles(angles):
    return np.arctan2(np.sin(angles), np.cos(angles))


class obstacleAvoider:
    # picks the drivable direction closest to where the rover wants to go, from one range scan
    def __init__(self, config):
        self.clearance = float(config["robotRadius"]) + float(config["safetyMargin"])
        self.tightClearance = float(config["robotRadius"]) + float(config["tightMargin"])
        self.lookahead = float(config["avoidLookahead"])
        self.maxDetour = float(config["avoidMaxDetour"])
        self.stopDistance = float(config["obstacleStopDistance"])
        self.slowDistance = float(config["obstacleSlowDistance"])
        self.candidates = np.arange(-candidateLimit, candidateLimit + 1e-9, candidateStep)
        self.previousSteer = 0.0

    def plan(self, scan, desiredBearing, goalDistance):
        # scan: (angles, ranges), angles in radians, positive to the left of the rover heading
        # returns (steerBearing, speedScale, blocked)
        angles, ranges = (np.asarray(values, dtype=float) for values in scan)
        horizon = min(self.lookahead, max(goalDistance, self.clearance))
        near = np.isfinite(ranges) & (ranges < horizon)
        angles, ranges = angles[near], ranges[near]
        if abs(desiredBearing) > math.pi / 2 or not len(ranges):
            # turning on the spot toward a goal behind the rover, or nothing in the way
            self.previousSteer = max(-candidateLimit, min(candidateLimit, desiredBearing))
            return desiredBearing, 1.0, False

        detours = np.abs(self.candidates - desiredBearing)
        relative = self.candidates[:, None] - angles[None, :]
        along, across = ranges * np.cos(relative), np.abs(ranges * np.sin(relative))
        # how close each direction would bring the rover to each point; points beside or behind it
        # only block a direction that closes in on them, so passing something close by is fine
        closest = np.where(along > 0, across, ranges)
        closingIn = closest < ranges - 0.03
        # a swerve only has to stay clear until the rover is past whatever blocks the straight way to the goal,
        # otherwise the side walls of a corridor would rule out every way around a box standing in it
        straight = int(np.argmin(detours))
        inTheWay = (closest[straight] < self.clearance) & closingIn[straight]
        pastIt = float(along[straight][inTheWay].max()) + self.clearance if inTheWay.any() else horizon
        reach = np.clip(pastIt / np.maximum(np.cos(detours), 1e-3), 2 * self.clearance, horizon)
        closingIn &= along < reach[:, None]
        # first look for a way through with the full safety margin; failing that, squeeze past slowly
        # a gap narrower than the rover leaves no free direction through it
        for clearance, pace in ((self.clearance, 1.0), (self.tightClearance, 0.5)):
            usable = ~((closest < clearance) & closingIn).any(axis=1) & (detours <= self.maxDetour)
            if usable.any():
                break
        # something right in front of the bumper (a person stepping in, say): stop rather than swerve
        inFront = (np.abs(angles) < math.pi / 4) & (np.abs(ranges * np.sin(angles)) < self.tightClearance)
        if not usable.any() or (inFront & (ranges < self.stopDistance)).any():
            return desiredBearing, 0.0, True

        # prefer the direction nearest the goal, with a little stickiness so it does not flip sides
        cost = detours + 0.3 * np.abs(self.candidates - self.previousSteer)
        cost[~usable] = np.inf
        steer = float(self.candidates[int(np.argmin(cost))])
        if detours.min() < candidateStep and usable[int(np.argmin(detours))]:
            steer = desiredBearing
        self.previousSteer = steer
        # the chosen direction is clear; slow down only for things that pass close to the rover's path
        index = int(np.argmin(np.abs(self.candidates - steer)))
        inPath = (along[index] > 0.05) & (across[index] < self.clearance + 0.1)
        pathRange = float(along[index][inPath].min()) if inPath.any() else math.inf
        span = max(self.slowDistance - self.stopDistance, 1e-6)
        speedScale = pace * max(0.3, min(1.0, (pathRange - self.stopDistance) / span))
        return steer, speedScale, False
