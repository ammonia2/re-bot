import argparse
import json
from pathlib import Path
import time
import tomllib

import cv2
import numpy as np
import yaml

from perception.targetSelector import targetSelector
from perception.tracker import greedyMatch, iouMatrix, personTracker

root = Path(__file__).resolve().parents[1]
fps, shrink, minHeight, matchIou = 30, 0.5, 20, 0.5
# ground truth is annotated every 3rd frame, so everything runs at 10 fps (close to the robot's rate)
step = 3
selectorStarts = (0, 50, 100, 150, 200)


class centralSelector:
    # the visionDirect baseline: most central box every frame, no identity
    def __init__(self, config):
        pass

    def select(self, tracks, frame):
        centerX = frame.shape[1] / 2
        return min((tuple(track[1:]) for track in tracks), key=lambda box: abs((box[0] + box[2]) / 2 - centerX), default=None)


def loadScenarios(catalogue, split):
    # a scenario without a dataset key is aicity2025, as in the source catalogue
    scenarios = [{"dataset": "aicity2025", **tomllib.loads(path.read_text())} for path in sorted(Path(catalogue).glob("*.toml"))]
    return [scenario for scenario in scenarios if scenario["split"].startswith(split) and scenario["dataset"].startswith("aicity")]


def loadGroundTruth(dataRoot, scenario):
    # {clipFrame: [(personId, (x1, y1, x2, y2)), ...]} in clip pixels
    boxes = np.load(Path(dataRoot) / scenario["dataset"] / "boxes" / f"{scenario.get('source_split', scenario['split'])}_{scenario['scene']}.npz")
    camera = list(boxes["cams"]).index(scenario["cameras"][0])
    first = int(scenario["start_s"] * fps)
    rows = (boxes["cam"] == camera) & (boxes["f"] >= first) & (boxes["f"] < first + int(scenario["duration_s"] * fps))
    truth = {}
    for frame, personId, *box in zip(boxes["f"][rows], boxes["pid"][rows], boxes["x0"][rows], boxes["y0"][rows], boxes["x1"][rows], boxes["y1"][rows]):
        box = tuple(float(value) * shrink for value in box)
        if box[3] - box[1] >= minHeight:
            truth.setdefault(int(frame) - first, []).append((int(personId), box))
    return truth


def matchFrame(truthRows, boxes):
    # greedy one-to-one match of ground truth to boxes at IoU >= matchIou
    if not truthRows or not boxes:
        return []
    overlaps = iouMatrix([box for _, box in truthRows], boxes)
    return greedyMatch(overlaps, overlaps, matchIou)


def detectionsFor(scenario, frames, detector, cacheFolder):
    cache = cacheFolder / f"{scenario['name']}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    detections = [detector.detect(frame) for frame in frames]
    cacheFolder.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(detections))
    return detections


def readFrames(dataRoot, scenario):
    capture = cv2.VideoCapture(str(Path(dataRoot) / scenario["dataset"] / "clips" / scenario["name"] / f"{scenario['cameras'][0]}.mp4"))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            return frames
        frames.append(frame)


def evaluateClip(scenario, frames, truth, detections, config, selectorType):
    tracker = personTracker(config)
    selectors = {}
    lastTrackOf, switches, matchedCount, truthCount, predictedCount = {}, 0, 0, 0, 0
    pairCounts = {}
    selection = {start: [] for start in selectorStarts}
    trackSeconds = 0.0
    for index, frame in enumerate(frames):
        truthRows = truth.get(index, [])
        started = time.perf_counter()
        tracks = tracker.update(detections[index], frame)
        trackSeconds += time.perf_counter() - started
        truthCount += len(truthRows)
        predictedCount += len(tracks)
        for row, column in matchFrame(truthRows, [track[1:] for track in tracks]):
            personId, trackId = truthRows[row][0], tracks[column][0]
            matchedCount += 1
            pairCounts[personId, trackId] = pairCounts.get((personId, trackId), 0) + 1
            if lastTrackOf.get(personId, trackId) != trackId:
                switches += 1
            lastTrackOf[personId] = trackId

        for start in selectorStarts:
            if index < start:
                continue
            if start not in selectors:
                if not tracks:
                    continue
                selectors[start] = (selectorType(config), None)
            selector, targetPerson = selectors[start]
            box = selector.select(tracks, frame)
            if targetPerson is None:
                # the person the selector enrolled, from the ground truth under its first box
                matched = matchFrame(truthRows, [box]) if box is not None else []
                if not matched:
                    del selectors[start]
                    continue
                targetPerson = truthRows[matched[0][0]][0]
                selectors[start] = (selector, targetPerson)
            targetBox = next((personBox for personId, personBox in truthRows if personId == targetPerson), None)
            if box is None:
                outcome = "none"
            elif targetBox is not None and iouMatrix([box], [targetBox])[0, 0] >= matchIou:
                outcome = "correct"
            else:
                outcome = "wrong"
            selection[start].append((outcome, targetBox is not None))

    # IDF1 with a greedy one-to-one person-to-track assignment by shared frames
    usedPeople, usedTracks, idTruePositives = set(), set(), 0
    for (personId, trackId), count in sorted(pairCounts.items(), key=lambda item: -item[1]):
        if personId not in usedPeople and trackId not in usedTracks:
            usedPeople.add(personId)
            usedTracks.add(trackId)
            idTruePositives += count
    return {
        "name": scenario["name"], "frames": len(frames), "truthBoxes": truthCount, "people": len(lastTrackOf),
        "recall": matchedCount / max(truthCount, 1), "idSwitches": switches,
        "idf1": 2 * idTruePositives / max(truthCount + predictedCount, 1),
        "trackerMsPerFrame": 1000 * trackSeconds / max(len(frames), 1),
        "selection": [runs for runs in selection.values() if runs],
    }


def selectionSummary(runs):
    outcomes = [outcome for run in runs for outcome, _ in run]
    visible = [outcome for run in runs for outcome, targetVisible in run if targetVisible]
    # a loss counts only if the target is visible again later, otherwise it cannot be re-found
    losses = recovered = 0
    for run in runs:
        for position in range(1, len(run)):
            if run[position - 1][0] == "correct" and run[position][0] != "correct":
                if any(targetVisible for _, targetVisible in run[position + 1:]):
                    losses += 1
                    recovered += any(outcome == "correct" for outcome, _ in run[position:])
    share = lambda items, value: sum(1 for item in items if item == value) / max(len(items), 1)
    return {
        "runs": len(runs), "frames": len(outcomes),
        "correctWhenVisible": share(visible, "correct"), "wrong": share(outcomes, "wrong"),
        "lossEvents": losses, "reacquisitionRate": recovered / losses if losses else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Tracker and target selector on AI City 2024/2025 single-camera clips")
    parser.add_argument("--catalogue", type=Path, required=True, help="folder of tier1 scenario .toml files")
    parser.add_argument("--data", type=Path, required=True, help="folder holding aicity2024/ and aicity2025/, each with boxes/ and clips/")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--boxes", default="gt", choices=("gt", "det"))
    parser.add_argument("--inputSize", type=int, default=640, help="detector input size for --boxes det")
    parser.add_argument("--selector", default="target", choices=("target", "central"))
    parser.add_argument("--set", action="append", default=[], metavar="key=value", help="override a config value")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=root / "data/aicityEval")
    args = parser.parse_args()

    config = yaml.safe_load((root / "config/robotConfig.yaml").read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        config[key] = yaml.safe_load(value)
    detector = None
    if args.boxes == "det":
        from perception.detector import personDetector

        detector = personDetector({**config, "detectorInputSize": args.inputSize})
    selectorType = targetSelector if args.selector == "target" else centralSelector

    results = []
    for scenario in loadScenarios(args.catalogue, args.split)[:args.limit]:
        frames = readFrames(args.data, scenario)[::step]
        truth = {index // step: rows for index, rows in loadGroundTruth(args.data, scenario).items() if index % step == 0}
        if args.boxes == "gt":
            detections = [[(*box, 1.0) for _, box in truth.get(index, [])] for index in range(len(frames))]
        else:
            detections = detectionsFor(scenario, frames, detector, args.output / f"dets{args.inputSize}")
        result = evaluateClip(scenario, frames, truth, detections, config, selectorType)
        results.append(result)
        summary = selectionSummary(result["selection"])
        print(f"{result['name']:40s} people {result['people']:3d} recall {result['recall']:.2f} idsw {result['idSwitches']:4d}"
              f" idf1 {result['idf1']:.3f} | target correct {summary['correctWhenVisible']:.2f} wrong {summary['wrong']:.2f}", flush=True)

    allRuns = [run for result in results for run in result["selection"]]
    total = {
        "split": args.split, "boxes": args.boxes, "selector": args.selector, "overrides": args.set, "clips": len(results),
        "idSwitches": sum(result["idSwitches"] for result in results),
        "idSwitchesPerPerson": sum(result["idSwitches"] for result in results) / max(sum(result["people"] for result in results), 1),
        "meanIdf1": float(np.mean([result["idf1"] for result in results])),
        "meanRecall": float(np.mean([result["recall"] for result in results])),
        "trackerMsPerFrame": float(np.mean([result["trackerMsPerFrame"] for result in results])),
        **selectionSummary(allRuns),
    }
    print(json.dumps(total, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"{args.split}_{args.boxes}_{args.selector}" + "".join(f"_{item}" for item in args.set)
    (args.output / f"{name}.json").write_text(json.dumps({"total": total, "clips": [{**r, "selection": None} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
