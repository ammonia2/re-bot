import math
from pathlib import Path

import yaml

from adapters.webotsRobot import webotsRobot
from core.followController import rampSpeed, wrapAngle
from core.followStateMachine import followStateMachine, roverState
from perception.visionObservation import makeVisionObservation

configPath = Path(__file__).resolve().parents[4] / "config" / "robotConfig.yaml"
config = yaml.safe_load(configPath.read_text())

# ensure the detector OBS function uses same params!
def groundTruthObservation(rover, config, fovScale=1.0):
    roverX, roverY, roverYaw = rover.getSelfPose()
    personX, personY = rover.getPersonPosition()
    deltaX, deltaY = personX - roverX, personY - roverY
    distance = math.hypot(deltaX, deltaY)
    bearingError = wrapAngle(math.atan2(deltaY, deltaX) - roverYaw)
    halfFov = config["cameraHalfFov"] * fovScale
    visible = abs(bearingError) < halfFov and distance < config["maxDetectionRange"]
    return visible, bearingError, distance

if config["observationSource"] == "vision":
    observe = makeVisionObservation(config)
else:
    observe = groundTruthObservation

rover = webotsRobot()
machine = followStateMachine(config)
previousState = machine.state
leftCommand, rightCommand = 0.0, 0.0

while rover.step():
    # narrower cone while searching, so it re-locks well inside the view
    searching = machine.state == roverState.searching
    fovScale = config["reacquireFovFactor"] if searching else 1.0
    targetVisible, bearingError, distance = observe(rover, config, fovScale)

    leftSpeed, rightSpeed = machine.update(
        targetVisible, bearingError, distance, float("inf")
    )

    if machine.state == roverState.stopped:
        # safety stop is instant, never ramped
        leftCommand, rightCommand = 0.0, 0.0
    else:
        leftCommand = rampSpeed(leftCommand, leftSpeed, config["maxWheelAccel"])
        rightCommand = rampSpeed(rightCommand, rightSpeed, config["maxWheelAccel"])
    rover.setWheelSpeeds(leftCommand, rightCommand)

    if machine.state != previousState:
        print(f"state: {previousState.value} -> {machine.state.value}")
        previousState = machine.state