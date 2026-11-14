# re-bot
A companion robot that follows you around

The robot is simulated in [Webots](https://cyberbotics.com/) as a differential-drive rover with a camera, a 2D lidar and wheel encoders, in an indoor world with walls, doorways, obstacles and four walking people. You tell it whom to follow by voice or keyboard. Controllers are written in Python and run with the project's own virtual environment.

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| [Webots](https://github.com/cyberbotics/webots/releases/tag/R2025a) | **R2025a** | Use exactly this version so world files open the same for everyone. |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | latest | Manages Python and the project's dependencies. |
| Python | 3.12 | uv installs this for you. You don't need a system Python. |

## Setup

### 1. Install Webots R2025a

Download R2025a for your OS from the [GitHub release page](https://github.com/cyberbotics/webots/releases/tag/R2025a) and install it the usual way:

- **Ubuntu:** the `.deb` package or the `.tar.bz2` archive.
- **Windows 11:** the `.exe` installer.

Open Webots once to check that it starts.

### 2. Install uv and the project environment

Install uv using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/) for your OS. Then, from the repo root:

```
uv sync
```

This creates `.venv/` with Python 3.12 and every dependency, including the dev group.

### 3. Point Webots at the project's Python

In Webots, open **Tools → Preferences → General** and set **Python command** to the **absolute path** of the venv interpreter:

- **Ubuntu:** `<repo>/.venv/bin/python`
- **Windows 11:** `<repo>\.venv\Scripts\python.exe`

Replace `<repo>` with the folder where you cloned the project. A relative path won't work.

If you skip this, Webots runs controllers with whatever system Python it finds, and imports like `cv2` or `onnxruntime` will fail.

## Project layout

```
config/            robot configuration (robotConfig.yaml)
src/               robot code: core (driving, navigation), perception (camera), voice (commands), adapters (Webots)
sim/webots/
  worlds/          Webots world files (.wbt)
  controllers/     Python controllers (companionController for the rover, walker for the people)
models/            model weights
experiments/       headless test runner and unit tests
benchmarks/        detector speed and range-estimate checks
```

## Running the simulation

From the repository root, run:

```
uv sync
webots sim/webots/worlds/companionArena.wbt
# Webots > Preferences > Python command: <repo>/.venv/bin/python
# after any world file change: File > Reload World
```

1. Open Webots and choose **File → Open World…**, then pick a world in `sim/webots/worlds/`.
2. Press **▶** if the clock is not running. The rover starts idle and waits for a command. Controller output shows up in the **Console** panel, and the rover's state is printed in the top-left corner of the 3D view.

### The world

A 20 × 20 m floor with a central corridor, three rooms joined by doorways, pillars, boxes and furniture. The storage room in the bottom-right corner can only be entered through a 0.5 m gap. The rover is wider than that, so it must never go in. Four people walk their own routes and stop now and then. They wear red, blue, green and yellow shirts.

### Commands

Click once inside the 3D view so it has keyboard focus, then press a key. You can also say the command out loud.

| Key | Say | What the rover does |
|-----|-----|---------------------|
| F | "follow me" | follows the person nearest the middle of its camera view |
| R / B / G / Y | "follow red" (blue, green, yellow) | finds and follows the person in that shirt |
| N | "next", "follow someone else" | switches to a different person |
| S | "stop", "wait" | stops where it is |
| C | "go", "continue" | carries on with the same person |

Speech recognition is offline ([Vosk](https://alphacephei.com/vosk/), small English model). The first launch downloads the model (about 40 MB) into `models/`. If no microphone is found the keyboard still works.

### What the rover does

| State | Meaning |
|-------|---------|
| `idle` | waiting for a command, or gave up searching |
| `following` | the person is in view: it drives to stay about 2.5 m behind them and matches their walking speed |
| `pursuing` | the person just went out of view: it retraces the path they walked (for example through a doorway) to where they were last seen, then a little further in the direction they were heading |
| `searching` | still not found: it circles slowly to look all the way around, then drives a few meters into open space it has not searched yet and looks again. After about a minute it gives up and goes idle |
| `blocked` | the only way on is too narrow or something is right in front of it: it waits and keeps facing the person |

Sensing uses only what a real robot would have: the camera (person detection, tracking and recognising the same person again by clothing colour), a 2D lidar for obstacles, and wheel encoders for its own position. Ground-truth positions from the simulator are used only for logging and scoring.

### Headless test runs

`experiments/simDiagnostics.py` runs the world without a window, plays scripted commands as if they were spoken, and prints a report: time spent in each state, distance to the followed person, how close the rover came to obstacles, and whether it ever entered the storage room.

```
uv run python experiments/simDiagnostics.py                                  # default scenario, 420 s
uv run python experiments/simDiagnostics.py --seconds 200 --commands "3:follow red"
uv run python experiments/simDiagnostics.py --observation groundTruth        # perfect sensing, to test driving only
uv run python experiments/simDiagnostics.py --summarizeOnly                  # re-print the last report
```

The step-by-step log is written to `data/simDiagnostics/steps.jsonl`.

### Tests

```
uv run python -m pytest experiments/test*.py benchmarks/test*.py
```

### Tuning

All settings live in `config/robotConfig.yaml`. The ones you are most likely to change:

| Key | Effect |
|-----|--------|
| `targetFollowDistance` | how far behind the person the rover stays (m) |
| `distanceGain`, `maxWheelSpeed`, `maxWheelAccel` | how hard it drives to close the gap, its top wheel speed (rad/s) and how quickly it speeds up |
| `searchTurnSpeed`, `searchCreepSpeed` | turning and forward wheel speed while circling to look around |
| `searchRoamDistance`, `searchRoamSpeed` | how far and how fast it drives between look-arounds |
| `searchTimeoutSteps` | how long it searches before giving up (steps of 32 ms) |
| `robotRadius`, `safetyMargin` | the rover's size and the space it keeps from obstacles |
| `voiceEnabled` | turn the microphone off |

## Troubleshooting

- **`ModuleNotFoundError` in the console:** the Python command preference is wrong or empty (see step 3), or you haven't run `uv sync` yet.
- **Webots looks broken on Linux with Wayland:** the official launcher already forces X11 (`QT_QPA_PLATFORM=xcb`). Launch Webots with the `webots` command, not by running `webots-bin` directly.
- **Lots of `Fontconfig` warnings on Linux at startup:** these are harmless and you can ignore them.
