import math
from pathlib import Path

import yaml

from adapters.webotsRobot import webotsRobot
from core.followController import computeWheelSpeeds, wrapAngle

configPath = Path(__file__).resolve().parents[4] / "config" / "robotConfig.yaml"
config = yaml.safe_load(configPath.read_text())

rover = webotsRobot()

while rover.step():
    roverX, roverY, roverYaw = rover.getSelfPose()
    personX, personY = rover.getPersonPosition()
    deltaX, deltaY = personX - roverX, personY - roverY
    distance = math.hypot(deltaX, deltaY)
    bearingError = wrapAngle(math.atan2(deltaY, deltaX) - roverYaw)
    leftSpeed, rightSpeed = computeWheelSpeeds(bearingError, distance, config)
    rover.setWheelSpeeds(leftSpeed, rightSpeed)