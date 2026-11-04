# Perception Contract

Project: re-bot, Edge-AI companion robot
Version: 1.0
Owner: Cocomo (integration). Changes to this file need the owner's approval through a pull request.

## 1. Goal

Replace the ground-truth target observation with a camera-only pipeline. It runs in the Webots simulation first and on the Raspberry Pi later.

Pipeline: frame -> detections -> tracks -> target box -> (bearingError, distance)

## 2. The seam

The controller calls one function every step:

    observe(rover, config, fovScale) -> (visible, bearingError, distance)

- `groundTruthObservation` already exists in `companionController.py` and uses simulator data.
- `visionObservation` is the camera-only replacement. It is built by `makeVisionObservation(config)` in `src/perception/visionObservation.py`.
- The config key `observationSource` (`groundTruth` or `vision`) chooses between them.

`fovScale` can be ignored by the vision version.

## 3. Conventions

| Item | Rule |
| --- | --- |
| Units | meters, radians, pixels |
| Image origin | top-left, x to the right, y downward |
| Box | `(x1, y1, x2, y2)` in pixels, `x1 < x2`, `y1 < y2` |
| Frame | numpy uint8 BGR array, shape `(height, width, 3)`, from `rover.getFrame()` |
| bearingError | radians, positive when the target is to the LEFT of the robot heading. A target right of the image center gives a negative value |
| distance | meters from the camera to the target |
| Target not visible | return `(False, 0.0, 0.0)`, never a guess |
| Naming | camelCase everywhere: files, functions, variables, constants, config keys |
| Comments | minimal |

## 4. No simulator-only data in perception

Nothing under `src/perception/` may import `controller` (Webots), call supervisor functions, or read the person's true position. Inputs are the frame and the config only. This code must run unchanged on the Pi.

## 5. Module interfaces

### detector: `src/perception/detector.py`

    class personDetector:
        def __init__(self, config)
        def detect(self, frame) -> list of (x1, y1, x2, y2, confidence)

- Person class only. Drop detections below `detectorConfidence`.
- Input size comes from `detectorInputSize`.
- The inference backend must be swappable. Ultralytics is for development only. The Pi runtime uses ONNX Runtime.

### tracker: `src/perception/tracker.py`

    class personTracker:
        def __init__(self, config)
        def update(self, detections, frame) -> list of (trackId, x1, y1, x2, y2)

- `trackId` is an int, stable across frames.
- An ID is never reused for a different person within one run.

### target selection: `src/perception/targetSelector.py`

    class targetSelector:
        def __init__(self, config)
        def select(self, tracks, frame) -> (x1, y1, x2, y2) or None
        def reset(self)

- The first call that has tracks enrolls the track nearest the image center.
- After a loss, re-find the same person with an appearance cue.
- Never lock onto a different person without a strong match. Returning `None` is better than a wrong target.

### estimator: `src/perception/estimator.py`

    def estimateTarget(box, config) -> (bearingError, distance)

A reference version is already written (pinhole model, monocular distance from box height). Its owner validates it against ground truth and fixes it if needed.

    def estimateTargetWithCrop(box, config, personHeight=None) -> (bearingError, distance or None)
    def personHeightFromBox(box, config) -> height or None

- `estimateTargetWithCrop` uses only box edges inside the frame. Full body: height formula. Feet cut: head ray and `personHeight - cameraMountHeight`. Head cut: feet ray on flat ground. Both cut: `None`.
- `personHeightFromBox` measures a fully visible person's height from the feet and head rays.
- The selector keeps the target's measured heights (`targetHeight()`, the median once there are enough samples). The vision observation passes it in, so people of any height get the right range.
- When the range is `None`, the vision observation reports the target visible at `targetFollowDistance`, so the rover keeps facing them but does not drive closer.

## 6. Config keys added for perception

    observationSource: vision
    cameraWidth: 640
    cameraHeight: 480
    cameraFov: 0.7854
    personHeightMeters: 1.84
    cameraMountHeight: 0.9975
    cameraPitch: 0.0
    detectorConfidence: 0.35
    trackerMaxAge: 30
    trackerMinHits: 3
    trackerIouThreshold: 0.3
    trackerDeltaT: 3
    trackerInertia: 0.2
    selectorMatchThreshold: 0.65
    selectorRejectThreshold: 0.35
    selectorMargin: 0.1
    selectorConfirmFrames: 3
    selectorGallerySize: 20

- `cameraWidth`, `cameraHeight` and `cameraFov` (horizontal, radians) must match the `Camera` node in the world.
- `personHeightMeters` must match the Pedestrian model. On the vision path it is only the starting guess until the selector has measured the target.
- `cameraMountHeight` is the camera's height above the floor in meters, and `cameraPitch` its downward tilt in radians. Both must match the world (and later the real mounting). Without `cameraMountHeight`, cropped boxes give no range and no height is learned.
- `cameraForwardOffset` is how far the camera sits ahead of the rover center, in meters. Vision converts camera range to rover-center range with it.
- `observationSource: visionDirect` is a temporary integration stand-in: detector plus estimator on the most central box, with no tracking or identity. `vision` is the real pipeline.
- `detectorInputSize` already exists.
- `tracker*` keys tune the OC-SORT tracker: frames a lost track survives (`trackerMaxAge`), hits before a track is reported (`trackerMinHits`), minimum IoU to match (`trackerIouThreshold`), frames back for the motion direction (`trackerDeltaT`), weight of the direction term (`trackerInertia`).
- `selector*` keys tune re-acquisition by appearance (colour histogram similarity, 0 to 1): minimum similarity to re-lock (`selectorMatchThreshold`), similarity below which the target's own track is dropped as an ID switch (`selectorRejectThreshold`), lead the best match needs over everyone else (`selectorMargin`), frames in a row before re-locking (`selectorConfirmFrames`), appearance samples kept (`selectorGallerySize`).
- All `tracker*` and `selector*` keys are optional; the code falls back to the values above.
- Anyone adding a key tells the owner and documents it here.

## 7. Ownership

| Path | Owner |
| --- | --- |
| `src/perception/detector.py` | Teammate A |
| `src/perception/estimator.py` (validation and tests) | Teammate A |
| `benchmarks/` | Teammate A |
| `src/perception/tracker.py` | Teammate B |
| `src/perception/targetSelector.py` | Teammate B |
| `experiments/` | Teammate B |
| `src/perception/visionObservation.py`, `src/core/`, `src/adapters/`, `companionController.py`, `config/`, the world file | Cocomo |

Edit only your own files. If you need a change elsewhere, ask the owner or open a pull request.

## 8. Definition of done

Targets below are proposed, and the team confirms or changes them at kickoff.

| Module | Done when |
| --- | --- |
| Detector | Runs live in the simulation. Person-only. An ONNX export script exists. FPS and latency are measured and written to `benchmarks/` |
| Tracker and selector | In the scripted crossing scenario the target keeps its ID, and the robot re-acquires the right person after the target leaves view or is occluded. Identity switches and re-acquisition rate are reported |
| Estimator | Unit tests pass. Bearing within 0.05 rad and distance within 15 percent of ground truth over a recorded run |
| Integration | With `observationSource: vision`, the rover follows the pedestrian for one full loop of the arena |

## 9. Known risks

1. **The detector may not fire on Webots pedestrians.** They are low-poly and COCO-trained models can miss them. Day-one check: save frames from `rover.getFrame()` with the pedestrian in view and run the detector on them. If detection is poor, tell the owner the same day. The tracker and identity work then moves to recorded real video or MOT17 clips, and the simulation is used only for the control loop.
2. **Monocular distance** assumed the full body is in frame and the person height is known. The crop-aware estimator and learned target height now cover cropping and different heights, but they assume flat ground and an upright person. Sitting, crouching, or a body hidden behind furniture with both box edges still in frame is not detected and still reads as further away.
3. **Obstacle sensing** for the real robot is undecided (owner).

## 10. Workflow

- Branches: `detector`, `tracker`, `estimator`, `benchmarks`, `experiments`. Merge into `main` through pull requests.
- `main` runs with `observationSource: vision`. `groundTruth` stays available for debugging the control loop.
- Never commit weights, videos or recorded data. Share them through the team Drive folder.
- Setup: `uv sync`, then open `sim/webots/worlds/companionArena.wbt` in Webots. Webots' Python command must point to `.venv/bin/python` in the repo. After any change to the world file use File > Reload World. Reset the simulation before saving the world.
- Contract changes: a pull request that edits this file, approved by the owner.