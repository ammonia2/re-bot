import math


def wrapAngle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, limit):
    return max(-limit, min(limit, value))


def computeWheelSpeeds(bearingError, distance, config):
    distanceError = distance - config["targetFollowDistance"]
    forwardSpeed = max(0.0, config["distanceGain"] * distanceError)
    forwardSpeed *= max(0.0, math.cos(bearingError))  # slow down while turning
    turnSpeed = config["bearingGain"] * bearingError
    maxSpeed = config["maxWheelSpeed"]
    leftSpeed = clamp(forwardSpeed - turnSpeed, maxSpeed)
    rightSpeed = clamp(forwardSpeed + turnSpeed, maxSpeed)
    return leftSpeed, rightSpeed