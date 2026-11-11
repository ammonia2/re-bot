import unittest

import numpy as np

from perception.targetSelector import colourShare, targetSelector
from voice.commands import grammarPhrases, parseCommand

red, blue, green, grey = (0, 0, 220), (220, 0, 0), (0, 200, 0), (90, 90, 90)


def personBox(centerX, top=100.0, width=60.0, height=200.0):
    return (centerX - width / 2, top, centerX + width / 2, top + height)


def drawFrame(people):
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    for (x1, y1, x2, y2), shirt in people:
        middle = int(y1 + (y2 - y1) * 0.55)
        frame[int(y1):middle, int(x1):int(x2)] = shirt
        frame[middle:int(y2), int(x1):int(x2)] = grey
    return frame


class TestParseCommand(unittest.TestCase):
    def testSentences(self):
        cases = {
            "follow me": ("followMe", None), "robot come here": ("followMe", None),
            "follow red": ("followColour", "red"), "follow the blue one": ("followColour", "blue"),
            "next": ("followOther", None), "follow someone else": ("followOther", None),
            "stop": ("stop", None), "stop following me": ("stop", None), "wait": ("stop", None),
            "go": ("go", None), "continue": ("go", None),
            "": None, "what a nice day": None,
        }
        for sentence, expected in cases.items():
            self.assertEqual(parseCommand(sentence), expected, sentence)

    def testEveryGrammarPhraseIsACommand(self):
        for phrase in grammarPhrases():
            if phrase != "[unk]":
                self.assertIsNotNone(parseCommand(phrase), phrase)


class TestTargetRequests(unittest.TestCase):
    def setUp(self):
        self.selector = targetSelector({})
        self.boxes = {1: personBox(320), 2: personBox(500), 3: personBox(120)}
        self.frame = drawFrame([(self.boxes[1], red), (self.boxes[2], blue), (self.boxes[3], green)])
        self.tracks = [(trackId, *box) for trackId, box in self.boxes.items()]

    def select(self, times=1, tracks=None):
        for _ in range(times):
            box = self.selector.select(self.tracks if tracks is None else tracks, self.frame)
        return box

    def testColourShare(self):
        self.assertGreater(colourShare(self.frame, self.boxes[2], "blue"), 0.9)
        self.assertEqual(colourShare(self.frame, self.boxes[2], "red"), 0.0)

    def testFollowColourPicksThatShirtEvenOffCenter(self):
        self.select()
        self.selector.requestTarget("colour", "blue")
        self.assertEqual(self.select(5), self.boxes[2])
        self.assertEqual(self.selector.targetId, 2)

    def testFollowColourWaitsUntilThatPersonIsInView(self):
        self.selector.requestTarget("colour", "blue")
        withoutBlue = [track for track in self.tracks if track[0] != 2]
        self.assertIsNone(self.select(5, withoutBlue))
        self.assertFalse(self.selector.hasTarget())
        self.assertEqual(self.select(5), self.boxes[2])

    def testFollowOtherNeverPicksTheOldTarget(self):
        self.assertEqual(self.select(), self.boxes[1])
        self.selector.requestTarget("other")
        self.assertIsNone(self.select(5, [self.tracks[0]]))
        self.assertIn(self.select(5), (self.boxes[2], self.boxes[3]))

    def testFollowMePicksThePersonInFront(self):
        self.selector.requestTarget("colour", "green")
        self.select(5)
        self.selector.requestTarget("center")
        self.assertEqual(self.select(), self.boxes[1])

    def testUnknownColourIsRejected(self):
        with self.assertRaises(ValueError):
            self.selector.requestTarget("colour", "sparkly")


if __name__ == "__main__":
    unittest.main()
