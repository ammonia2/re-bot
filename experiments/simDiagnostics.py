import argparse
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

root = Path(__file__).resolve().parents[1]
defaultWebots = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Webots/msys64/mingw64/bin/webots.exe"


def runDiagnostics():
    # runs the real vision loop and logs every stage next to the ground truth
    import cv2
    import yaml
    from adapters.webotsRobot import webotsRobot
    from core.followController import rampSpeed, wrapAngle
    from core.followStateMachine import followStateMachine
    from perception.detector import personDetector
    from perception.targetSelector import appearanceOf, targetSelector
    from perception.tracker import personTracker
    from perception.visionObservation import boxToObservation

    output = Path(os.environ["REBOT_DIAGNOSTICS_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    seconds = float(os.environ["REBOT_DIAGNOSTICS_SECONDS"])
    config = yaml.safe_load((root / "config/robotConfig.yaml").read_text())
    rover = webotsRobot()
    detector, tracker, selector = personDetector(config), personTracker(config), targetSelector(config)
    machine = followStateMachine(config)
    leftCommand, rightCommand = 0.0, 0.0
    rows, savedMisses = [], 0
    for step in range(int(seconds * 1000 / rover.timeStep)):
        if not rover.step():
            break
        frame = rover.getFrame()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        box = selector.select(tracks, frame)
        visible, bearingError, distance = boxToObservation(box, config, 1.0)
        leftSpeed, rightSpeed = machine.update(visible, bearingError, distance, float("inf"))
        leftCommand = rampSpeed(leftCommand, leftSpeed, config["maxWheelAccel"])
        rightCommand = rampSpeed(rightCommand, rightSpeed, config["maxWheelAccel"])
        rover.setWheelSpeeds(leftCommand, rightCommand)

        roverX, roverY, roverYaw = rover.getSelfPose()
        personX, personY = rover.getPersonPosition()
        truthDistance = math.hypot(personX - roverX, personY - roverY)
        truthBearing = wrapAngle(math.atan2(personY - roverY, personX - roverX) - roverYaw)
        inView = abs(truthBearing) < config["cameraHalfFov"]
        if inView and not detections and savedMisses < 10 and step % 10 == 0:
            cv2.imwrite(str(output / f"miss{step:05d}_{truthDistance:.1f}m.png"), frame)
            savedMisses += 1
        rows.append({
            "step": step, "truthDistance": truthDistance, "truthBearing": truthBearing, "inView": inView,
            "detections": len(detections), "maxConfidence": max((d[4] for d in detections), default=0.0),
            "boxHeight": max((d[3] - d[1] for d in detections), default=0.0),
            "trackIds": [track[0] for track in tracks], "targetId": selector.targetId,
            "similarities": [selector._similarity(appearanceOf(frame, track[1:])) for track in tracks],
            "selected": box is not None, "visible": visible, "estimatedDistance": distance,
            "state": machine.state.value,
        })
    (output / "steps.json").write_text(json.dumps(rows) + "\n")
    rover.robot.simulationQuit(0)


def summarize(rows):
    def share(subset, key):
        return f"{sum(1 for row in subset if row[key]) / len(subset):.0%}" if subset else "n/a"

    inView = [row for row in rows if row["inView"]]
    print(f"steps: {len(rows)}, person in camera view: {len(inView)} ({len(inView) / max(len(rows), 1):.0%})")
    for low, high in ((0, 2), (2, 3), (3, 4), (4, 5), (5, 99)):
        band = [row for row in inView if low <= row["truthDistance"] < high]
        print(f"  in view {low}-{high} m: {len(band):4d} steps | detected {share(band, 'detections')}"
              f" | tracked {share([r for r in band if r['detections']], 'trackIds')}"
              f" | selected {share(band, 'selected')} | visible {share(band, 'visible')}")
    trackIds = sorted({trackId for row in rows for trackId in row["trackIds"]})
    targetIds = [row["targetId"] for row in rows]
    changes = sum(1 for before, after in zip(targetIds, targetIds[1:]) if after != before)
    print(f"track ids seen: {trackIds}")
    print(f"selector target id changes: {changes}, final target id: {targetIds[-1] if targetIds else None}")
    states = [row["state"] for row in rows]
    print("time per state: " + ", ".join(f"{state} {states.count(state) / len(states):.0%}" for state in sorted(set(states))))
    selected = [row for row in rows if row["visible"]]
    if selected:
        errors = [abs(row["estimatedDistance"] - row["truthDistance"]) / row["truthDistance"] for row in selected]
        print(f"distance error when visible: median {sorted(errors)[len(errors) // 2]:.0%}")


def main():
    parser = argparse.ArgumentParser(description="Run the vision loop headless in Webots and report where the person is lost")
    parser.add_argument("--seconds", type=float, default=60.0, help="Simulated seconds")
    parser.add_argument("--output", type=Path, default=root / "data/simDiagnostics")
    parser.add_argument("--webots", type=Path, default=defaultWebots)
    args = parser.parse_args()
    output = args.output.resolve()
    world = (root / "sim/webots/worlds/companionArena.wbt").read_text()
    world = world.replace('controller "companionController"', 'controller "simDiagnostics"', 1)
    environment = dict(os.environ)
    environment.update({"REBOT_DIAGNOSTICS_OUTPUT": str(output), "REBOT_DIAGNOSTICS_SECONDS": str(args.seconds),
                        "YOLO_CONFIG_DIR": str(root / "data/ultralyticsSettings"), "YOLO_AUTOINSTALL": "false"})
    with TemporaryDirectory(prefix="rebotDiagnostics-") as folder:
        project = Path(folder)
        (project / "worlds").mkdir()
        controller = project / "controllers/simDiagnostics"
        controller.mkdir(parents=True)
        (project / "worlds/diagnostics.wbt").write_text(world)
        wrapper = (f"import sys, runpy\nsys.path.insert(0, {str(root / 'src')!r})\n"
                   f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__webots_controller__')\n")
        (controller / "simDiagnostics.py").write_text(wrapper)
        subprocess.run([str(args.webots), "--batch", "--mode=fast", "--stdout", "--stderr", "--minimize",
                        str(project / "worlds/diagnostics.wbt")], env=environment, check=True, timeout=1800)
    summarize(json.loads((output / "steps.json").read_text()))


if __name__ == "__webots_controller__":
    runDiagnostics()
elif __name__ == "__main__":
    main()
