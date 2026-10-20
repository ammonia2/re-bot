from enum import Enum

from core.followController import computeWheelSpeeds


class roverState(Enum):
    idle = "idle"
    following = "following"
    searching = "searching"
    stopped = "stopped"


class followStateMachine:
    def __init__(self, config):
        self.config = config
        self.state = roverState.following
        self.lastBearingSign = 1.0
        self.lostSteps = 0
        self.lastSpeeds = (0.0, 0.0)

    def update(self, targetVisible, bearingError, distance, nearestObstacle):
        # safety override always wins
        if nearestObstacle < self.config["obstacleStopDistance"]:
            self.state = roverState.stopped
            return 0.0, 0.0

        if targetVisible:
            self.state = roverState.following
            self.lostSteps = 0
            self.lastBearingSign = 1.0 if bearingError >= 0 else -1.0
            self.lastSpeeds = computeWheelSpeeds(bearingError, distance, self.config)
            return self.lastSpeeds

        self.lostSteps += 1
        graceSteps = self.config["lostGraceSteps"]

        # brief dropout: keep the last command
        if self.lostSteps <= graceSteps:
            return self.lastSpeeds

        if self.lostSteps > graceSteps + self.config["searchTimeoutSteps"]:
            self.state = roverState.idle
            return 0.0, 0.0

        # spin toward where the target was last seen
        self.state = roverState.searching
        turnSpeed = self.config["searchTurnSpeed"] * self.lastBearingSign
        return -turnSpeed, turnSpeed