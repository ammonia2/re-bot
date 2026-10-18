class robotInterface:
    def getFrame(self):
        raise NotImplementedError

    def getDistanceReadings(self):
        raise NotImplementedError

    def setWheelSpeeds(self, leftSpeed, rightSpeed):
        raise NotImplementedError
