import math
from collections import deque
from enum import Enum

from core.followController import clamp, wrapAngle
from core.obstacleAvoider import obstacleAvoider


class roverState(Enum):
    idle = "idle"
    following = "following"
    pursuing = "pursuing"
    searching = "searching"
    blocked = "blocked"


class followStateMachine:
    def __init__(self, config):
        self.config = config
        self.avoider = obstacleAvoider(config)
        self.state = roverState.idle
        self.active = False
        self.forgetTarget()

    def pause(self):
        self.active = False
        self.state = roverState.idle

    def resume(self):
        self.active = True
        self.lostSteps, self.searchSteps = 0, 0
        self.gaveUp = False

    def forgetTarget(self):
        # a new target was asked for: drop where the old one was last seen
        self.lastSeen = None
        self.heading = None
        self.trail = deque(maxlen=int(self.config["headingWindowSteps"]))
        self.waypoints = None
        self.gaveUp = False
        self.searchSign = 1.0
        self.lostSteps, self.searchSteps = 0, 0
        self.bestRange, self.stalledSteps = math.inf, 0

    def update(self, targetVisible, bearingError, distance, pose, scan):
        # pose: (x, y, yaw) from odometry; scan: (angles, ranges) from the range sensor
        # returns (leftSpeed, rightSpeed)
        if targetVisible:
            self._remember(bearingError, distance, pose)
        if not self.active:
            self.state = roverState.idle
            return 0.0, 0.0

        if targetVisible:
            self.lostSteps, self.searchSteps = 0, 0
            self.waypoints, self.gaveUp = None, False
            self.bestRange, self.stalledSteps = math.inf, 0
            reach = distance - self.config["targetFollowDistance"]
            # the target itself is not an obstacle, so only look for things in front of it
            return self._drive(roverState.following, bearingError, reach, distance - self.config["targetClearance"], scan)

        self.lostSteps += 1
        if self.lastSeen is not None and self.lostSteps <= self.config["lostGraceSteps"]:
            # brief dropout: carry on toward where the target just was
            bearing, range_ = self._toPoint(self.lastSeen, pose)
            reach = range_ - self.config["targetFollowDistance"]
            return self._drive(roverState.following, bearing, reach, range_ - self.config["targetClearance"], scan)
        if self.waypoints is None and self.lastSeen is not None:
            # go to where the target was last seen (e.g. the doorway it used), then a little further the way it was walking
            self.waypoints = [self.lastSeen]
            if self.heading is not None:
                lead = self.config["pursueLead"]
                self.waypoints.append((self.lastSeen[0] + lead * math.cos(self.heading), self.lastSeen[1] + lead * math.sin(self.heading)))
        while self.waypoints:
            bearing, range_ = self._toPoint(self.waypoints[0], pose)
            firstLeg = self.waypoints[0] == self.lastSeen
            patience = self.config["blockedPatienceSteps"] if firstLeg else self.config["leadPatienceSteps"]
            # no progress for a while means the way there is shut (a wall, or a gap too narrow)
            if range_ < self.bestRange - 0.05:
                self.bestRange, self.stalledSteps = range_, 0
            else:
                self.stalledSteps += 1
            if self.lostSteps > self.config["pursueTimeoutSteps"]:
                self.waypoints = []
            elif range_ <= self.config["arrivalRadius"] or self.stalledSteps > patience:
                self.waypoints.pop(0)
                self.bestRange, self.stalledSteps = math.inf, 0
            else:
                return self._drive(roverState.pursuing, bearing, range_, range_, scan)
            if not self.waypoints and self.heading is not None:
                # look the way the target was walking
                self.searchSign = 1.0 if wrapAngle(self.heading - pose[2]) >= 0 else -1.0

        self.searchSteps += 1
        if self.gaveUp or self.searchSteps > self.config["searchTimeoutSteps"]:
            self.gaveUp = True
            self.state = roverState.idle
            return 0.0, 0.0
        self.state = roverState.searching
        if self.lastSeen is None and self.searchSteps <= self.config["searchDelaySteps"]:
            # just asked to find someone: look straight ahead for a moment before turning away
            return 0.0, 0.0
        turnSpeed = self.config["searchTurnSpeed"] * self.searchSign
        return -turnSpeed, turnSpeed

    def _remember(self, bearingError, distance, pose):
        x, y, yaw = pose
        if self.lostSteps > self.config["lostGraceSteps"]:
            self.trail.clear()
        self.lastSeen = (x + distance * math.cos(yaw + bearingError), y + distance * math.sin(yaw + bearingError))
        self.trail.append(self.lastSeen)
        self.searchSign = 1.0 if bearingError >= 0 else -1.0
        # which way the target is walking: from the average of the oldest sightings to the average of the newest
        third = len(self.trail) // 3
        self.heading = None
        if third >= 8:
            points = list(self.trail)
            startX, startY = (sum(point[axis] for point in points[:third]) / third for axis in (0, 1))
            endX, endY = (sum(point[axis] for point in points[-third:]) / third for axis in (0, 1))
            if math.hypot(endX - startX, endY - startY) > self.config["headingBaseline"]:
                self.heading = math.atan2(endY - startY, endX - startX)

    def _toPoint(self, point, pose):
        x, y, yaw = pose
        deltaX, deltaY = point[0] - x, point[1] - y
        return wrapAngle(math.atan2(deltaY, deltaX) - yaw), math.hypot(deltaX, deltaY)

    def _drive(self, state, bearing, reach, obstacleHorizon, scan):
        maxSpeed = self.config["maxWheelSpeed"]
        forwardSpeed = clamp(max(0.0, self.config["distanceGain"] * reach), maxSpeed)
        steer, speedScale, blocked = self.avoider.plan(scan, bearing, obstacleHorizon)
        if blocked and forwardSpeed > 0:
            # no way through that fits the rover: hold position and keep facing the goal
            self.state = roverState.blocked
            forwardSpeed = 0.0
        else:
            self.state = state
            forwardSpeed *= speedScale * max(0.0, math.cos(steer))  # slow down while turning
        turnSpeed = clamp(self.config["bearingGain"] * steer, self.config["maxTurnSpeed"])
        return clamp(forwardSpeed - turnSpeed, maxSpeed), clamp(forwardSpeed + turnSpeed, maxSpeed)
