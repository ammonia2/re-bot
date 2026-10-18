import sys
import cv2
from adapters.webotsRobot import webotsRobot

rover = webotsRobot()
print("python:", sys.version)
frameSaved = False

while rover.step():
    rover.setWheelSpeeds(2.0, 2.0)
    if not frameSaved:
        cv2.imwrite("firstFrame.png", rover.getFrame())
        print("sonars:", [round(value, 2) for value in rover.getDistanceReadings()])
        frameSaved = True