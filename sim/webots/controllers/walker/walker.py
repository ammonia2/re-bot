# Walks a Pedestrian PROTO around a closed loop of waypoints, optionally standing still at some of them.
# The walking gait tables come from the Webots "pedestrian" controller,
# Copyright 1996-2024 Cyberbotics Ltd., licensed under the Apache License, Version 2.0.
import argparse
import math

from controller import Supervisor

rootHeight = 1.27
cycleToDistanceRatio = 0.22
jointNames = [
    "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
    "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
    "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
    "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
    "headAngle",
]
heightOffsets = [-0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03]
gaitAngles = [
    [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],
    [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
    [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],
    [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],
    [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38],
    [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],
    [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],
    [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],
    [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],
    [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],
    [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],
    [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],
    [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09],
]


def parseTrajectory(text):
    # "x y, x y:seconds, ..." where :seconds makes the walker stand at that waypoint
    waypoints = []
    for entry in text.split(","):
        point, _, pause = entry.strip().partition(":")
        x, y = (float(value) for value in point.split())
        waypoints.append((x, y, float(pause) if pause else 0.0))
    return waypoints


def buildSchedule(waypoints, speed):
    # list of (startTime, kind, data) covering one lap, plus the lap duration
    schedule, time, walked = [], 0.0, 0.0
    for index, (x, y, pause) in enumerate(waypoints):
        nextX, nextY, _ = waypoints[(index + 1) % len(waypoints)]
        heading = math.atan2(nextY - y, nextX - x)
        if pause > 0:
            schedule.append((time, "stand", (x, y, heading, walked)))
            time += pause
        length = math.hypot(nextX - x, nextY - y)
        schedule.append((time, "walk", (x, y, nextX, nextY, heading, walked, length)))
        time += length / speed
        walked += length
    return schedule, time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--speed", type=float, default=0.6)
    args = parser.parse_args()
    waypoints = parseTrajectory(args.trajectory)
    schedule, lapTime = buildSchedule(waypoints, args.speed)

    robot = Supervisor()
    timeStep = int(robot.getBasicTimeStep())
    root = robot.getSelf()
    translationField = root.getField("translation")
    rotationField = root.getField("rotation")
    jointFields = [root.getField(name) for name in jointNames]

    while robot.step(timeStep) != -1:
        lapClock = robot.getTime() % lapTime
        startTime, kind, data = [entry for entry in schedule if entry[0] <= lapClock][-1]
        if kind == "stand":
            x, y, heading, _ = data
            angles, heightOffset = [0.0] * len(jointNames), 0.0
        else:
            fromX, fromY, toX, toY, heading, walkedBefore, length = data
            covered = min((lapClock - startTime) * args.speed, length)
            ratio = covered / length if length else 1.0
            x, y = fromX + (toX - fromX) * ratio, fromY + (toY - fromY) * ratio
            cycle = (walkedBefore + covered) / cycleToDistanceRatio
            sequence, blend = int(cycle) % 8, cycle - int(cycle)
            following = (sequence + 1) % 8
            angles = [row[sequence] * (1 - blend) + row[following] * blend for row in gaitAngles]
            heightOffset = heightOffsets[sequence] * (1 - blend) + heightOffsets[following] * blend
        for field, angle in zip(jointFields, angles):
            field.setSFFloat(angle)
        translationField.setSFVec3f([x, y, rootHeight + heightOffset])
        rotationField.setSFRotation([0, 0, 1, heading])


main()
