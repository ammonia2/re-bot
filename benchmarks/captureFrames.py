import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


def isolateWebotsSettings(environment, folder):
    # run Webots on a copy of the user's settings so the temporary world never lands in their recent files
    settings = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Cyberbotics"
    copy = Path(folder) / "settings"
    if settings.is_dir():
        shutil.copytree(settings, copy / "Cyberbotics")
    environment["XDG_CONFIG_HOME"] = str(copy)


def captureFrames():
    import cv2
    import numpy as np
    import yaml
    from adapters.webotsRobot import webotsRobot
    from perception.detector import personDetector
    from perception.estimator import estimateTarget

    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ["REBOT_CAPTURE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((root / "config/robotConfig.yaml").read_text())
    rover = webotsRobot()
    detector = personDetector(config) if os.environ.get("REBOT_CAPTURE_DETECT") == "1" else None
    selfNode = rover.robot.getSelf()
    cameraNode = rover.robot.getFromDef("captureCamera")
    samples = []
    for distance in (3.0, 4.0, 5.0):
        for _ in range(8):
            if not rover.step():
                raise RuntimeError("Simulation ended before capture")
            # look back down the corridor at the red person, who stands still for the capture
            personX, personY = rover.getPersonPosition("personRed")
            selfNode.getField("translation").setSFVec3f([personX + distance, personY, 0.0975])
            selfNode.getField("rotation").setSFRotation([0, 0, 1, math.pi])
            selfNode.resetPhysics()
        if not rover.step():
            raise RuntimeError("Simulation ended before capture")
        frame = rover.getFrame()
        name = f"person{int(distance)}m.png"
        if not cv2.imwrite(str(output / name), frame):
            raise RuntimeError(f"Unable to save {name}")
        cameraPose = np.asarray(cameraNode.getPose()).reshape(4, 4)
        personPosition = rover.robot.getFromDef("personRed").getPosition()
        relative = cameraPose[:3, :3].T @ (np.asarray(personPosition) - cameraPose[:3, 3])
        truthBearing = math.atan2(relative[1], relative[0])
        truthRange = math.hypot(relative[0], relative[1])
        row = {
            "frame": name, "width": rover.camera.getWidth(), "height": rover.camera.getHeight(),
            "horizontalFov": rover.camera.getFov(), "requestedSeparationMeters": distance,
            "cameraPose": cameraPose.tolist(), "personRootPosition": personPosition,
            "trueBearing": truthBearing, "trueHorizontalRange": truthRange,
        }
        if detector is not None:
            detections = detector.detect(frame)
            row["detections"] = detections
            if detections:
                actualConfig = {**config, "cameraWidth": rover.camera.getWidth(),
                                "cameraHeight": rover.camera.getHeight(), "cameraFov": rover.camera.getFov()}
                bearing, estimatedRange = estimateTarget(detections[0][:4], actualConfig)
                row.update({"estimatedBearing": bearing, "estimatedRange": estimatedRange,
                            "bearingAbsoluteError": abs(bearing - truthBearing),
                            "rangeRelativeError": abs(estimatedRange - truthRange) / truthRange})
        samples.append(row)
    (output / "captureMetadata.json").write_text(json.dumps(samples, indent=2) + "\n")
    rover.robot.simulationQuit(0)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Capture real adapter frames in an isolated Webots project")
    parser.add_argument("--variant", choices=("baseline", "calibrated"), default="baseline")
    parser.add_argument("--revision", default="HEAD",
                        help="Read the committed world at this revision, or 'worktree' for the local file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detect", action="store_true", help="Run the detector live during capture")
    args = parser.parse_args()
    defaultFolder = "detectorFeasibility" if args.variant == "baseline" else "calibratedFeasibility"
    output = (args.output or root / "data" / defaultFolder).resolve()
    worldPath = "sim/webots/worlds/companionArena.wbt"
    if args.revision == "worktree":
        world = (root / worldPath).read_text()
    else:
        world = subprocess.check_output(["git", "show", f"{args.revision}:{worldPath}"], cwd=root, text=True)
    world = world.replace('controller "companionController"', 'controller "captureFrames"', 1)
    # only the red person, standing still: the others would walk into the shots
    world = re.sub(r"DEF person(Blue|Green|Yellow) Pedestrian \{.*?\n\}\n", "", world, flags=re.DOTALL)
    world = world.replace('controller "walker"', 'controller "<none>"')
    if args.variant == "calibrated":
        camera = lambda match: "DEF captureCamera Camera { translation 0.3 0 0.9 width 640 height 480 fieldOfView 0.7854 }"
    else:
        camera = lambda match: "DEF captureCamera " + match.group(0)
    world, count = re.subn(r"Camera\s*\{[^{}]*\}", camera, world, count=1)
    if count != 1:
        parser.error("Capture expects one Camera node without child nodes; update this benchmark for the new world")
    environment = dict(os.environ)
    environment.update({"REBOT_CAPTURE_OUTPUT": str(output), "REBOT_CAPTURE_DETECT": str(int(args.detect)),
                        "YOLO_CONFIG_DIR": str(root / "data/ultralyticsSettings"), "YOLO_AUTOINSTALL": "false"})
    startedAt = time.time_ns()
    with TemporaryDirectory(prefix="rebotCapture-") as folder:
        isolateWebotsSettings(environment, folder)
        project = Path(folder)
        (project / "worlds").mkdir()
        controller = project / "controllers/captureFrames"
        controller.mkdir(parents=True)
        (project / "worlds/capture.wbt").write_text(world)
        wrapper = (f"import sys, runpy\nsys.path.insert(0, {str(root / 'src')!r})\n"
                   f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__webots_controller__')\n")
        (controller / "captureFrames.py").write_text(wrapper)
        subprocess.run(["webots", "--batch", "--mode=fast", "--stdout", "--stderr", "--no-rendering",
                        str(project / "worlds/capture.wbt")], env=environment, check=True, timeout=240)
    metadataPath = output / "captureMetadata.json"
    if not metadataPath.is_file() or metadataPath.stat().st_mtime_ns < startedAt:
        raise RuntimeError("Webots exited without writing capture metadata; inspect its controller console")
    print(metadataPath.read_text())


if __name__ == "__webots_controller__":
    captureFrames()
elif __name__ == "__main__":
    main()
