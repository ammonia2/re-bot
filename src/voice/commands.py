import re

from perception.targetSelector import colourHues

# every command is (kind, colour): kind is one of followMe, followColour, followOther, stop, go
followWords = ("follow", "track", "come")
stopWords = ("stop", "wait", "stay", "halt", "freeze")
goWords = ("go", "continue", "resume")
otherWords = ("next", "switch", "other", "else", "another")


def parseCommand(text):
    # turns a heard or typed sentence into a command, or None when it is not one
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    if any(word in stopWords for word in words):
        return ("stop", None)
    colour = next((word for word in words if word in colourHues), None)
    if colour is not None:
        return ("followColour", colour)
    if any(word in otherWords for word in words):
        return ("followOther", None)
    if any(word in followWords for word in words):
        return ("followMe", None)
    if any(word in goWords for word in words):
        return ("go", None)
    return None


def grammarPhrases():
    # the only sentences the speech recogniser listens for, which keeps it small and accurate
    phrases = ["follow me", "come here", "stop", "wait", "stay", "go", "continue", "next", "switch", "follow someone else"]
    for colour in colourHues:
        phrases += [f"follow {colour}", f"follow the {colour} one"]
    return phrases + ["[unk]"]
