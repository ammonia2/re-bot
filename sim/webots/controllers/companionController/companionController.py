import math
from pathlib import Path

import yaml

from adapters.webotsRobot import webotsRobot
from core.followController import wrapAngle
from core.followStateMachine import followStateMachine, roverState

# repo root is four levels above this file
configPath = Path(__file__).resolve().parents[4] / "config" / "robotConfig.yaml"
config = yaml.safe_load(configPath.read_text())
 
rover = webotsRobot()
machine = followStateMachine(config)
previousState = machine.state

while rover.step():
    roverX, roverY, roverYaw = rover.getSelfPose()
    personX, personY = rover.getPersonPosition()
    deltaX, deltaY = personX - roverX, personY - roverY
    distance = math.hypot(deltaX, deltaY)
    bearingError = wrapAngle(math.atan2(deltaY, deltaX) - roverYaw)

        # smaller cone while searching, so it re-locks well inside the view
    halfFov = config["cameraHalfFov"]
    if machine.state == roverState.searching:
        halfFov *= config["reacquireFovFactor"]

    targetVisible = abs(bearingError) < halfFov and distance < config["maxDetectionRange"]
    
    leftSpeed, rightSpeed = machine.update(
        targetVisible, bearingError, distance, float("inf")
    )
    rover.setWheelSpeeds(leftSpeed, rightSpeed)

    if machine.state != previousState:
        print(f"state: {previousState.value} -> {machine.state.value}")
        previousState = machine.state