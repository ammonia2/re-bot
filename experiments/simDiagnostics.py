import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

root = Path(__file__).resolve().parents[1]
world = root / "sim/webots/worlds/companionArena.wbt"
defaultCommands = "3:follow red;150:follow blue;260:follow yellow;400:stop"
# the storage room, reachable only through a gap too narrow for the rover: (xMin, xMax, yMin, yMax)
noGoZone = (3.1, 10.0, -10.0, -2.1)


def isolateWebotsSettings(environment, folder):
    # run Webots on a copy of the user's settings so headless runs never change what the user's Webots reopens
    settings = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Cyberbotics"
    copy = Path(folder) / "settings"
    if settings.is_dir():
        shutil.copytree(settings, copy / "Cyberbotics")
    environment["XDG_CONFIG_HOME"] = str(copy)


def findWebots():
    onPath = shutil.which("webots")
    if onPath:
        return Path(onPath)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Webots/msys64/mingw64/bin/webots.exe"
    if sys.platform == "darwin":
        return Path("/Applications/Webots.app/Contents/MacOS/webots")
    return Path("/usr/local/webots/webots")


def targetOf(command):
    words = command.lower().split()
    return next((colour for colour in ("red", "blue", "green", "yellow") if colour in words), None)


def summarize(rows, commands, robotRadius):
    print(f"simulated {rows[-1]['time']:.0f} s, {len(rows)} steps")
    states = [row["state"] for row in rows]
    print("time per state: " + ", ".join(f"{state} {states.count(state) / len(states):.0%}" for state in sorted(set(states))))
    driftX, driftY = rows[-1]["odometry"][0] - (rows[-1]["rover"][0] - rows[0]["rover"][0]), rows[-1]["odometry"][1] - (rows[-1]["rover"][1] - rows[0]["rover"][1])
    travelled = sum(math.dist(a["rover"][:2], b["rover"][:2]) for a, b in zip(rows, rows[1:]))
    print(f"rover travelled {travelled:.0f} m, odometry drift {math.hypot(driftX, driftY):.2f} m")
    nearest = min(row["nearestObstacle"] for row in rows)
    touching = sum(1 for row in rows if row["nearestObstacle"] < robotRadius)
    print(f"closest the range sensor ever got to anything: {nearest:.2f} m ({touching} steps inside the rover radius {robotRadius} m)")
    xMin, xMax, yMin, yMax = noGoZone
    inside = sum(1 for row in rows if xMin < row["rover"][0] < xMax and yMin < row["rover"][1] < yMax)
    print(f"steps inside the storage room (must be 0): {inside}")
    bounds = commands + [(math.inf, "")]
    for (start, command), (end, _) in zip(bounds, bounds[1:]):
        span = [row for row in rows if start <= row["time"] < end]
        colour = targetOf(command)
        if not span or colour is None:
            continue
        gaps = [math.dist(row["rover"][:2], row["people"][colour]) for row in span]
        seen = sum(1 for row in span if row["visible"]) / len(span)
        spanStates = [row["state"] for row in span]
        mix = ", ".join(f"{state} {spanStates.count(state) / len(span):.0%}" for state in sorted(set(spanStates)))
        print(f"  '{command}' {start:.0f}-{span[-1]['time']:.0f} s: target in view {seen:.0%}, "
              f"distance to {colour} median {sorted(gaps)[len(gaps) // 2]:.1f} m, max {max(gaps):.1f} m, end {gaps[-1]:.1f} m | {mix}")


def main():
    parser = argparse.ArgumentParser(description="Run the companion world headless with scripted commands and report how the rover did")
    parser.add_argument("--seconds", type=float, default=420.0, help="Simulated seconds")
    parser.add_argument("--commands", default=defaultCommands, help="'seconds:sentence;...' played as if spoken")
    parser.add_argument("--observation", choices=("vision", "visionDirect", "groundTruth"), help="Override observationSource")
    parser.add_argument("--output", type=Path, default=root / "data/simDiagnostics")
    parser.add_argument("--webots", type=Path, default=findWebots())
    parser.add_argument("--summarizeOnly", action="store_true", help="Re-read the last run instead of simulating")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "steps.jsonl"
    if not args.summarizeOnly:
        environment = dict(os.environ)
        environment.update({"REBOT_LOG": str(log), "REBOT_SECONDS": str(args.seconds), "REBOT_COMMANDS": args.commands,
                            "YOLO_CONFIG_DIR": str(root / "data/ultralyticsSettings"), "YOLO_AUTOINSTALL": "false"})
        if args.observation:
            environment["REBOT_OBSERVATION"] = args.observation
        with TemporaryDirectory(prefix="rebotDiagnostics-") as folder:
            isolateWebotsSettings(environment, folder)
            subprocess.run([str(args.webots), "--batch", "--mode=fast", "--stdout", "--stderr", "--minimize", "--no-rendering",
                            str(world)], env=environment, check=True, timeout=3600)
    import yaml

    config = yaml.safe_load((root / "config/robotConfig.yaml").read_text())
    rows = [row for row in map(json.loads, log.read_text().splitlines()) if "state" in row]
    commands = sorted((float(when), text) for when, text in (entry.split(":", 1) for entry in args.commands.split(";") if entry.strip()))
    summarize(rows, commands, config["robotRadius"])


if __name__ == "__main__":
    main()
