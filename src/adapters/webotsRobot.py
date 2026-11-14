import math

import numpy as np
from controller import Supervisor

from core.robotInterface import robotInterface

# meters from the lidar to the edge of the Pioneer's body
bodyRange = 0.2


class webotsRobot(robotInterface):
    def __init__(self):
        self.robot = Supervisor()
        self.timeStep = int(self.robot.getBasicTimeStep())

        self.leftMotor = self.robot.getDevice("left wheel")
        self.rightMotor = self.robot.getDevice("right wheel")
        for motor in (self.leftMotor, self.rightMotor):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        self.wheelSensors = [self.robot.getDevice("left wheel sensor"), self.robot.getDevice("right wheel sensor")]
        for sensor in self.wheelSensors:
            sensor.enable(self.timeStep)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timeStep)

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timeStep)
        fov, count = self.lidar.getFov(), self.lidar.getHorizontalResolution()
        # the first ray is the leftmost one
        self.scanAngles = np.linspace(fov / 2, -fov / 2, count)

        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timeStep)
        self.heldKeys = set()

    def step(self):
        return self.robot.step(self.timeStep) != -1

    def getTime(self):
        return self.robot.getTime()

    def getFrame(self):
        raw = self.camera.getImage()
        height, width = self.camera.getHeight(), self.camera.getWidth()
        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
        return frame[:, :, :3].copy()  # BGR, ready for OpenCV

    def getRangeScan(self):
        ranges = np.array(self.lidar.getRangeImage(), dtype=float)
        # nothing real can be inside the rover's own body: such readings are the lidar seeing itself or glitches
        ranges[ranges < bodyRange] = np.inf
        return self.scanAngles, ranges

    def getWheelAngles(self):
        return tuple(sensor.getValue() for sensor in self.wheelSensors)

    def setWheelSpeeds(self, leftSpeed, rightSpeed):
        self.leftMotor.setVelocity(leftSpeed)
        self.rightMotor.setVelocity(rightSpeed)

    def getPressedKeys(self):
        # letters newly pressed in the 3D view since the last step
        held = set()
        key = self.keyboard.getKey()
        while key != -1:
            held.add(key & 0xFFFF)
            key = self.keyboard.getKey()
        pressed = held - self.heldKeys
        self.heldKeys = held
        return [chr(key) for key in sorted(pressed) if 32 <= key < 127]

    def showStatus(self, text):
        self.robot.setLabel(0, text, 0.01, 0.01, 0.07, 0xFFFFFF, 0.0, "Arial")

    # simulator-only ground truth, for debugging and for measuring how well the rover does
    def getSelfPose(self):
        node = self.robot.getSelf()
        x, y, _ = node.getPosition()
        orientation = node.getOrientation()
        yaw = math.atan2(orientation[3], orientation[0])
        return x, y, yaw

    def getPersonPosition(self, defName="personRed"):
        x, y, _ = self.robot.getFromDef(defName).getPosition()
        return x, y
