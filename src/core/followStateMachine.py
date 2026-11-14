import math
from collections import deque
from enum import Enum

import numpy as np

from core.followController import clamp, wrapAngle
from core.obstacleAvoider import obstacleAvoider, wrapAngles


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
        self.roamGoal, self.sweepSteps = None, 0
        self.visited = []

    def forgetTarget(self):
        # a new target was asked for: drop where the old one was last seen
        self.lastSeen = None
        self.heading, self.targetSpeed = None, 0.0
        self.trail = deque(maxlen=int(self.config["headingWindowSteps"]))
        self.crumbs = deque(maxlen=20)
        self.waypoints, self.leadPoint = None, None
        self.gaveUp = False
        self.searchSign = 1.0
        self.lostSteps, self.searchSteps = 0, 0
        self.bestRange, self.stalledSteps = math.inf, 0
        self.roamGoal, self.sweepSteps = None, 0
        self.visited = []

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
            self.roamGoal, self.sweepSteps = None, 0
            self.visited = []
            self.bestRange, self.stalledSteps = math.inf, 0
            reach = distance - self.config["targetFollowDistance"]
            # the target itself is not an obstacle, so only look for things in front of it
            return self._drive(roverState.following, bearingError, reach, distance - self.config["targetClearance"], scan,
                               feedForward=self._walkingAway(bearingError, pose) if reach > -0.3 else 0.0)

        self.lostSteps += 1
        if self.lastSeen is not None and self.lostSteps <= self.config["lostGraceSteps"]:
            # brief dropout: carry on toward where the target just was
            bearing, range_ = self._toPoint(self.lastSeen, pose)
            reach = range_ - self.config["targetFollowDistance"]
            return self._drive(roverState.following, bearing, reach, range_ - self.config["targetClearance"], scan)
        if self.waypoints is None and self.lastSeen is not None:
            # walk the target's own path to where it was last seen (e.g. through the doorway it used),
            # then a little further the way it was walking
            crumbs = list(self.crumbs)
            nearest = min(range(len(crumbs)), key=lambda index: math.dist(crumbs[index], pose[:2]), default=0)
            self.waypoints = crumbs[nearest + 1:] + [self.lastSeen]
            self.leadPoint = None
            if self.heading is not None:
                lead = self.config["pursueLead"]
                self.leadPoint = (self.lastSeen[0] + lead * math.cos(self.heading), self.lastSeen[1] + lead * math.sin(self.heading))
                self.waypoints.append(self.leadPoint)
        while self.waypoints:
            bearing, range_ = self._toPoint(self.waypoints[0], pose)
            onLead = self.waypoints[0] == self.leadPoint
            patience = self.config["leadPatienceSteps"] if onLead else self.config["blockedPatienceSteps"]
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
        return self._search(pose, scan)

    def _search(self, pose, scan):
        # look all the way around while circling slowly, then drive to open space nearby and look again
        if self.roamGoal is None and self.sweepSteps < self.config["searchSweepSteps"]:
            self.sweepSteps += 1
            return self._sweep(scan)
        if self.roamGoal is None:
            self.roamGoal = self._openPoint(pose, scan)
            self.visited.append((pose[0], pose[1]))
            self.bestRange, self.stalledSteps = math.inf, 0
        bearing, range_ = self._toPoint(self.roamGoal, pose)
        if range_ < self.bestRange - 0.05:
            self.bestRange, self.stalledSteps = range_, 0
        else:
            self.stalledSteps += 1
        if range_ > self.config["arrivalRadius"] and self.stalledSteps <= self.config["leadPatienceSteps"]:
            speeds = self._drive(roverState.searching, bearing, range_, range_, scan, self.config["searchRoamSpeed"])
            if self.state != roverState.blocked:
                return speeds
        self.state = roverState.searching
        self.roamGoal, self.sweepSteps = None, 0
        return self._sweep(scan)

    def _sweep(self, scan):
        angles, ranges = (np.asarray(values, dtype=float) for values in scan)
        ahead = np.abs(angles) < math.radians(35)
        roomAhead = not ahead.any() or float(np.min(ranges[ahead])) > self.config["obstacleSlowDistance"]
        forwardSpeed = self.config["searchCreepSpeed"] if roomAhead else 0.0
        turnSpeed = self.config["searchTurnSpeed"] * self.searchSign
        return forwardSpeed - turnSpeed, forwardSpeed + turnSpeed

    def _openPoint(self, pose, scan):
        # a point up to searchRoamDistance away that the rover fits all the way to,
        # preferring the way the target was walking and places it has not searched yet
        x, y, yaw = pose
        angles, ranges = (np.asarray(values, dtype=float) for values in scan)
        hit = np.isfinite(ranges)
        angles, ranges = angles[hit], ranges[hit]
        clearance = self.avoider.clearance
        roam = self.config["searchRoamDistance"]
        candidates = np.radians(np.arange(-110, 111, 5))
        relative = angles[None, :] - candidates[:, None]
        along, across = ranges * np.cos(relative), np.abs(ranges * np.sin(relative))
        blocking = (along > 0) & (across < clearance)
        free = np.where(blocking, along, np.inf).min(axis=1, initial=np.inf)
        reach = np.clip(free - clearance, 0.0, roam)
        preferred = 0.0 if self.heading is None else wrapAngle(self.heading - yaw)
        score = reach - 0.6 * np.abs(wrapAngles(candidates - preferred))
        goalX, goalY = x + reach * np.cos(yaw + candidates), y + reach * np.sin(yaw + candidates)
        for visitedX, visitedY in self.visited:
            score -= 1.5 * (np.hypot(goalX - visitedX, goalY - visitedY) < roam * 0.6)
        best = int(np.argmax(score))
        return float(goalX[best]), float(goalY[best])

    def _remember(self, bearingError, distance, pose):
        x, y, yaw = pose
        if self.lostSteps > self.config["lostGraceSteps"]:
            # found again after a while, maybe somewhere else: the old path no longer leads to them
            self.trail.clear()
            self.crumbs.clear()
        self.lastSeen = (x + distance * math.cos(yaw + bearingError), y + distance * math.sin(yaw + bearingError))
        self.trail.append(self.lastSeen)
        if not self.crumbs or math.dist(self.crumbs[-1], self.lastSeen) >= self.config["crumbSpacing"]:
            self.crumbs.append(self.lastSeen)
        self.searchSign = 1.0 if bearingError >= 0 else -1.0
        # which way the target is walking: from the average of the oldest sightings to the average of the newest
        third = len(self.trail) // 3
        self.heading, self.targetSpeed = None, 0.0
        if third >= 8:
            points = list(self.trail)
            startX, startY = (sum(point[axis] for point in points[:third]) / third for axis in (0, 1))
            endX, endY = (sum(point[axis] for point in points[-third:]) / third for axis in (0, 1))
            walked = math.hypot(endX - startX, endY - startY)
            if walked > self.config["headingBaseline"]:
                self.heading = math.atan2(endY - startY, endX - startX)
                # meters per second, from the steps between the two averages
                self.targetSpeed = walked / ((len(points) - third) * self.config["controlStepSeconds"])

    def _walkingAway(self, bearingError, pose):
        # wheel speed that matches how fast the target is walking away, so the rover keeps up instead of lagging behind
        if self.heading is None:
            return 0.0
        away = self.targetSpeed * math.cos(self.heading - (pose[2] + bearingError))
        return max(0.0, away) / self.config["wheelRadius"]

    def _toPoint(self, point, pose):
        x, y, yaw = pose
        deltaX, deltaY = point[0] - x, point[1] - y
        return wrapAngle(math.atan2(deltaY, deltaX) - yaw), math.hypot(deltaX, deltaY)

    def _drive(self, state, bearing, reach, obstacleHorizon, scan, speedLimit=None, feedForward=0.0):
        maxSpeed = self.config["maxWheelSpeed"]
        forwardSpeed = clamp(max(0.0, self.config["distanceGain"] * reach + feedForward), speedLimit or maxSpeed)
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
