import numpy as np
from controller import Robot, Supervisor
from core.robotInterface import robotInterface
import math

class webotsRobot(robotInterface):
    def __init__(self):
        self.robot = Supervisor()  # was Robot()
        self.timeStep = int(self.robot.getBasicTimeStep())

        self.leftMotor = self.robot.getDevice("left wheel")
        self.rightMotor = self.robot.getDevice("right wheel")
        for motor in (self.leftMotor, self.rightMotor):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timeStep)

        self.sonars = [self.robot.getDevice(f"so{i}") for i in range(16)]
        for sonar in self.sonars:
            sonar.enable(self.timeStep)

    def step(self):
        return self.robot.step(self.timeStep) != -1

    def getSelfPose(self):
        node = self.robot.getSelf()
        x, y, _ = node.getPosition()
        orientation = node.getOrientation()
        yaw = math.atan2(orientation[3], orientation[0])
        return x, y, yaw

    def getPersonPosition(self, defName="person"):
        x, y, _ = self.robot.getFromDef(defName).getPosition()
        return x, y

    def getFrame(self):
        raw = self.camera.getImage()
        height, width = self.camera.getHeight(), self.camera.getWidth()
        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
        return frame[:, :, :3].copy()  # BGR, ready for OpenCV

    def getDistanceReadings(self):
        return [sonar.getValue() for sonar in self.sonars]

    def setWheelSpeeds(self, leftSpeed, rightSpeed):
        self.leftMotor.setVelocity(leftSpeed)
        self.rightMotor.setVelocity(rightSpeed)