class robotInterface:
    def getFrame(self):
        raise NotImplementedError

    def getRangeScan(self):
        # (angles, ranges): radians positive to the left of the heading, meters
        raise NotImplementedError

    def getWheelAngles(self):
        # (left, right) wheel rotation in radians
        raise NotImplementedError

    def setWheelSpeeds(self, leftSpeed, rightSpeed):
        raise NotImplementedError
