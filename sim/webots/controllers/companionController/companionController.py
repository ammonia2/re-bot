import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from adapters.webotsRobot import webotsRobot
from core.followController import rampSpeed, wrapAngle
from core.followStateMachine import followStateMachine, roverState
from core.odometry import odometry
from perception.visionObservation import makeDirectVisionObservation, makeVisionObservation
from voice.commands import parseCommand
from voice.speechListener import speechListener

configPath = Path(__file__).resolve().parents[4] / "config" / "robotConfig.yaml"
config = yaml.safe_load(configPath.read_text())
config["observationSource"] = os.environ.get("REBOT_OBSERVATION", config["observationSource"])

people = {"red": "personRed", "blue": "personBlue", "green": "personGreen", "yellow": "personYellow"}
keyCommands = {"F": "follow me", "R": "follow red", "B": "follow blue", "G": "follow green", "Y": "follow yellow",
               "N": "next", "S": "stop", "C": "go"}


class groundTruthTargets:
    # stands in for the vision selector when observationSource is groundTruth
    def __init__(self, rover):
        self.rover = rover
        self.defName = people["red"]

    def requestTarget(self, mode, colour=None):
        names = list(people.values())
        if mode == "colour":
            self.defName = people.get(colour, self.defName)
        elif mode == "other":
            self.defName = names[(names.index(self.defName) + 1) % len(names)]
        else:
            self.defName = min(names, key=lambda name: abs(self.locate(name)[0]))

    def locate(self, defName):
        roverX, roverY, roverYaw = self.rover.getSelfPose()
        personX, personY = self.rover.getPersonPosition(defName)
        deltaX, deltaY = personX - roverX, personY - roverY
        return wrapAngle(math.atan2(deltaY, deltaX) - roverYaw), math.hypot(deltaX, deltaY)


def makeGroundTruthObservation(rover):
    targets = groundTruthTargets(rover)

    def groundTruthObservation(rover, config, fovScale=1.0):
        bearingError, distance = targets.locate(targets.defName)
        halfFov = config["cameraHalfFov"] * fovScale
        visible = abs(bearingError) < halfFov and distance < config["maxDetectionRange"]
        if visible:
            # a wall between the rover and the person hides them, as it would from the camera
            angles, ranges = rover.getRangeScan()
            visible = ranges[int(np.argmin(np.abs(angles - bearingError)))] > distance - 0.6
        return visible, bearingError, distance

    groundTruthObservation.selector = targets
    return groundTruthObservation


def scriptedCommands():
    # "5:follow red;40:next" in REBOT_COMMANDS plays commands at simulated seconds, for headless test runs
    entries = [entry.split(":", 1) for entry in os.environ.get("REBOT_COMMANDS", "").split(";") if entry.strip()]
    return sorted((float(when), text) for when, text in entries)


rover = webotsRobot()
if config["observationSource"] == "vision":
    observe = makeVisionObservation(config)
elif config["observationSource"] == "visionDirect":
    observe = makeDirectVisionObservation(config)
else:
    observe = makeGroundTruthObservation(rover)
selector = getattr(observe, "selector", None)

machine = followStateMachine(config)
wheelOdometry = odometry(config)
script = scriptedCommands()
# test runs set these; they must not open the microphone
headless = bool(script) or "REBOT_LOG" in os.environ
listener = None
if config.get("voiceEnabled", True) and not headless:
    listener = speechListener(config)
    listener.start()
print("commands: click the 3D view, then F follow me, R/B/G/Y follow that colour, N next person, S stop, C go")

logPath = os.environ.get("REBOT_LOG")
logFile = open(logPath, "w") if logPath else None
framesFolder = os.environ.get("REBOT_FRAMES")
runSeconds = float(os.environ.get("REBOT_SECONDS", "inf"))

targetName = "nobody"
lastHeard = "-"


def applyCommand(text, source):
    global targetName, lastHeard
    command = parseCommand(text)
    if command is None:
        return
    kind, colour = command
    lastHeard = text
    print(f"{source}: '{text}'")
    if kind == "stop":
        machine.pause()
        return
    if kind != "go":
        if selector is None:
            print("this observation source cannot change target")
        else:
            mode = {"followMe": "center", "followColour": "colour", "followOther": "other"}[kind]
            selector.requestTarget(mode, colour)
            targetName = {"followMe": "person in front", "followColour": f"{colour} shirt", "followOther": "someone else"}[kind]
        machine.forgetTarget()
    machine.resume()


previousState = machine.state
shownStatus = None
forwardCommand, turnCommand = 0.0, 0.0

while rover.step():
    now = rover.getTime()
    while script and script[0][0] <= now:
        applyCommand(script.pop(0)[1], "script")
    for key in rover.getPressedKeys():
        if key in keyCommands:
            applyCommand(keyCommands[key], "keyboard")
    for sentence in listener.poll() if listener else []:
        applyCommand(sentence, "voice")

    # narrower cone while searching, so it re-locks well inside the view
    searching = machine.state == roverState.searching
    fovScale = config["reacquireFovFactor"] if searching else 1.0
    targetVisible, bearingError, distance = observe(rover, config, fovScale)
    pose = wheelOdometry.update(*rover.getWheelAngles())
    scan = rover.getRangeScan()

    leftSpeed, rightSpeed = machine.update(targetVisible, bearingError, distance, pose, scan)

    # ramp driving and turning separately, so braking never holds back a change of direction
    forwardTarget, turnTarget = (leftSpeed + rightSpeed) / 2, (rightSpeed - leftSpeed) / 2
    if machine.state in (roverState.blocked, roverState.idle):
        # safety stop is instant, never ramped
        forwardCommand = forwardTarget
    else:
        forwardCommand = rampSpeed(forwardCommand, forwardTarget, config["maxWheelAccel"])
    turnCommand = rampSpeed(turnCommand, turnTarget, config["maxTurnAccel"])
    leftCommand, rightCommand = forwardCommand - turnCommand, forwardCommand + turnCommand
    rover.setWheelSpeeds(leftCommand, rightCommand)

    if machine.state != previousState:
        print(f"state: {previousState.value} -> {machine.state.value}")
        previousState = machine.state
    status = f"{machine.state.value} | target: {targetName} | last command: {lastHeard}"
    if status != shownStatus:
        rover.showStatus(status)
        shownStatus = status

    if logFile:
        truth = {name: rover.getPersonPosition(defName) for name, defName in people.items()}
        logFile.write(json.dumps({
            "time": now, "state": machine.state.value, "visible": bool(targetVisible), "distance": distance,
            "rover": rover.getSelfPose(), "odometry": pose, "people": truth,
            "nearestObstacle": float(np.min(scan[1])), "nearestAngle": float(scan[0][int(np.argmin(scan[1]))]),
            "bearing": bearingError, "target": targetName, "targetId": getattr(selector, "targetId", None),
            "tracks": [(track[0], round(selector.similarityTo(observe.lastFrame, track[1:]), 2))
                       for track in getattr(observe, "lastTracks", [])],
        }) + "\n")
    if framesFolder and round(now * 1000) % 1000 < rover.timeStep:
        cv2.imwrite(str(Path(framesFolder) / f"frame{now:06.1f}.png"), rover.getFrame())
    if now >= runSeconds:
        break

if logFile:
    logFile.close()
    rover.robot.simulationQuit(0)
