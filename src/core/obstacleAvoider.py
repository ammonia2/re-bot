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

        ahead = np.abs(angles) < math.pi / 4
        nearestAhead = float(ranges[ahead].min()) if ahead.any() else math.inf
        detours = np.abs(self.candidates - desiredBearing)
        offsets = np.abs(wrapAngles(self.candidates[:, None] - angles[None, :]))
        # first look for a way through with the full safety margin; failing that, squeeze past slowly
        for clearance, pace in ((self.clearance, 1.0), (self.tightClearance, 0.5)):
            # an obstacle point blocks every direction whose driving corridor would touch it,
            # so a gap narrower than the rover leaves no free direction through it
            halfWidths = np.arcsin(np.clip(clearance / ranges, 0.0, 1.0))
            usable = ~(offsets < halfWidths[None, :]).any(axis=1) & (detours <= self.maxDetour)
            if usable.any():
                break
        if not usable.any() or nearestAhead < self.stopDistance:
            return desiredBearing, 0.0, True

        # prefer the direction nearest the goal, with a little stickiness so it does not flip sides
        cost = detours + 0.3 * np.abs(self.candidates - self.previousSteer)
        cost[~usable] = np.inf
        steer = float(self.candidates[int(np.argmin(cost))])
        if detours.min() < candidateStep and usable[int(np.argmin(detours))]:
            steer = desiredBearing
        self.previousSteer = steer
        span = max(self.slowDistance - self.stopDistance, 1e-6)
        speedScale = pace * max(0.3, min(1.0, (nearestAhead - self.stopDistance) / span))
        return steer, speedScale, False
