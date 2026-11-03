import unittest

import numpy as np

from perception.targetSelector import appearanceOf, targetSelector

red, blue, green, yellow = (0, 0, 220), (220, 0, 0), (0, 200, 0), (0, 220, 220)


def personBox(centerX, top=100.0, width=60.0, height=200.0):
    return (centerX - width / 2, top, centerX + width / 2, top + height)


def drawFrame(people):
    # people: [(box, shirtColour, trouserColour)] on a grey background
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    for (x1, y1, x2, y2), shirt, trousers in people:
        middle = int(y1 + (y2 - y1) * 0.55)
        frame[int(y1):middle, int(x1):int(x2)] = shirt
        frame[middle:int(y2), int(x1):int(x2)] = trousers
    return frame


class TestTargetSelector(unittest.TestCase):
    def setUp(self):
        self.selector = targetSelector({})
        self.target, self.other = personBox(320), personBox(500)

    def selectScene(self, tracks, people):
        return self.selector.select(tracks, drawFrame(people))

    def enrollRedBlue(self, frames=5):
        for step in range(frames):
            box = self.selectScene([(1, *self.target)], [(self.target, red, blue)])
        return box

    def testEnrollsTrackNearestCenter(self):
        tracks = [(4, *personBox(100)), (7, *personBox(330)), (9, *personBox(560))]
        box = self.selector.select(tracks, drawFrame([]))
        self.assertEqual(self.selector.targetId, 7)
        self.assertEqual(box, personBox(330))

    def testNoTracksGivesNoneAndKeepsTarget(self):
        self.enrollRedBlue()
        self.assertIsNone(self.selector.select([], drawFrame([])))
        self.assertEqual(self.selector.targetId, 1)

    def testFollowsTargetIdAmongOthers(self):
        self.enrollRedBlue()
        tracks = [(1, *self.target), (2, *self.other)]
        box = self.selectScene(tracks, [(self.target, red, blue), (self.other, green, yellow)])
        self.assertEqual(box, self.target)

    def testReacquiresSamePersonUnderNewIdAfterConfirmFrames(self):
        self.enrollRedBlue()
        self.selector.select([], drawFrame([]))
        returning = personBox(200)
        boxes = [self.selectScene([(5, *returning)], [(returning, red, blue)]) for step in range(4)]
        self.assertEqual(boxes, [None, None, returning, returning])
        self.assertEqual(self.selector.targetId, 5)

    def testDifferentPersonIsNeverLockedOn(self):
        self.enrollRedBlue()
        for step in range(10):
            self.assertIsNone(self.selectScene([(5, *self.other)], [(self.other, green, yellow)]))
        self.assertNotEqual(self.selector.targetId, 5)

    def testLookalikeSeenWithTargetIsExcluded(self):
        self.enrollRedBlue()
        tracks = [(1, *self.target), (2, *self.other)]
        self.selectScene(tracks, [(self.target, red, blue), (self.other, red, blue)])
        for step in range(5):
            self.assertIsNone(self.selectScene([(2, *self.other)], [(self.other, red, blue)]))

    def testTwoEqualCandidatesAreAmbiguous(self):
        self.enrollRedBlue()
        left, right = personBox(150), personBox(500)
        for step in range(5):
            box = self.selectScene([(5, *left), (6, *right)], [(left, red, blue), (right, red, blue)])
            self.assertIsNone(box)

    def testIdSwitchOntoDifferentPersonIsDropped(self):
        self.enrollRedBlue()
        self.assertIsNone(self.selectScene([(1, *self.other)], [(self.other, green, yellow)]))
        self.assertIsNone(self.selector.targetId)
        boxes = [self.selectScene([(3, *self.target)], [(self.target, red, blue)]) for step in range(3)]
        self.assertEqual(boxes[-1], self.target)

    def testOverlappedTargetIsStillFollowedButNotLearned(self):
        self.enrollRedBlue()
        size = len(self.selector.gallery)
        occluder = personBox(350)
        tracks = [(1, *self.target), (2, *occluder)]
        box = self.selectScene(tracks, [(self.target, red, blue), (occluder, green, yellow)])
        self.assertEqual(box, self.target)
        self.assertEqual(len(self.selector.gallery), size)

    def testResetEnrollsAgain(self):
        self.enrollRedBlue()
        self.selector.reset()
        self.selectScene([(8, *self.other)], [(self.other, green, yellow)])
        self.assertEqual(self.selector.targetId, 8)

    def testTinyOrOffFrameBoxHasNoAppearance(self):
        frame = drawFrame([])
        self.assertIsNone(appearanceOf(frame, (100, 100, 102, 103)))
        self.assertIsNone(appearanceOf(frame, (700, 100, 760, 300)))


if __name__ == "__main__":
    unittest.main()
