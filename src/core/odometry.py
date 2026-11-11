import math


class odometry:
    # dead reckoning from the wheel encoders: pose is relative to where the rover started
    def __init__(self, config):
        self.wheelRadius = float(config["wheelRadius"])
        self.wheelBase = float(config["wheelBase"])
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.lastAngles = None

    def update(self, leftAngle, rightAngle):
        if self.lastAngles is not None and not (math.isnan(leftAngle) or math.isnan(rightAngle)):
            leftTravel = (leftAngle - self.lastAngles[0]) * self.wheelRadius
            rightTravel = (rightAngle - self.lastAngles[1]) * self.wheelRadius
            travel = (leftTravel + rightTravel) / 2
            turn = (rightTravel - leftTravel) / self.wheelBase
            self.x += travel * math.cos(self.yaw + turn / 2)
            self.y += travel * math.sin(self.yaw + turn / 2)
            self.yaw = math.atan2(math.sin(self.yaw + turn), math.cos(self.yaw + turn))
        if not (math.isnan(leftAngle) or math.isnan(rightAngle)):
            self.lastAngles = (leftAngle, rightAngle)
        return self.pose()

    def pose(self):
        return self.x, self.y, self.yaw
