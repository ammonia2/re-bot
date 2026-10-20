# re-bot
A companion robot that follows you around

The robot is simulated in [Webots](https://cyberbotics.com/) as a differential-drive rover with a camera, distance sensors, and pedestrians. Controllers are written in Python and run with the project's own virtual environment.

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
src/               robot code: core, perception, voice, adapters
sim/webots/
  worlds/          Webots world files (.wbt)
  controllers/     Python controllers (companionController)
models/            model weights
tests/             pytest tests
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
2. The simulation starts and runs the robot's controller automatically. Controller output shows up in the **Console** panel.

## Troubleshooting

- **`ModuleNotFoundError` in the console:** the Python command preference is wrong or empty (see step 3), or you haven't run `uv sync` yet.
- **Webots looks broken on Linux with Wayland:** the official launcher already forces X11 (`QT_QPA_PLATFORM=xcb`). Launch Webots with the `webots` command, not by running `webots-bin` directly.
- **Lots of `Fontconfig` warnings on Linux at startup:** these are harmless and you can ignore them.
