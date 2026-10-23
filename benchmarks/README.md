# Detector and estimator work

Ownership: detector, estimator validation/tests and `benchmarks/`. Integration, config, tracker, selector and the shared world stay with their owners. The contract currently lives at `CONTRACT.md`, not `docs/contract.md`.

## Setup

Run from the repository root with Webots R2025a installed:

```bash
uv sync --frozen
uv pip install --python .venv/bin/python -r benchmarks/exportRequirements.txt
curl -fL --retry 2 -o models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

Webots' Python command must point to this repository's `.venv/bin/python`. The default dev dependencies include large CUDA packages even on a CPU-only laptop. ONNX is an export-only requirement installed separately; the shared dependency files are unchanged. After another `uv sync`, reinstall the export requirements if needed.

Weights and exported models belong under ignored `models/`. Frames, metadata, settings and annotations belong under ignored `data/`. Do not commit them or recorded videos. Share recorded inputs through the team Drive; none have been uploaded by these scripts.

## First: simulation feasibility

```bash
.venv/bin/python benchmarks/captureFrames.py --variant baseline --detect
.venv/bin/python benchmarks/captureFrames.py --variant calibrated --detect
```

Each command copies the committed world into a temporary project, uses the real `webotsRobot.getFrame()`, positions the rover 3/4/5 m from the walking pedestrian, saves three PNGs and records detections. The calibrated variant changes only the temporary camera: 640×480, horizontal FOV 0.7854, translation `(0.3, 0, 0.9)`. The baseline preserves the committed camera. No project world, configuration or controller is edited. Live scene viewing remains available in the separate Webots window.

The commands read the world at `HEAD`; use `--revision <commit>` to reproduce an older scene. Output defaults to `data/detectorFeasibility/` and `data/calibratedFeasibility/`; `--output` selects another ignored data folder. Existing captures at that location are replaced. The benchmark expects the original empty camera node and fails clearly if the world structure changes. Simulator data is used only here for validation, never inside perception.

Initial check on 2026-10-23: baseline 0/3 frames detected; calibrated view 1/3. The owner was informed immediately. See [calibrationReport.md](calibrationReport.md). This is a feasibility warning, not a successful following milestone. If the team follows the contract fallback, obtain recorded real video/MOT17 inputs through the owner before evaluating tracking; this work did not download a dataset.

## Detector interface

```python
from perception.detector import personDetector

detector = personDetector(config)  # local models/yolo11n.pt, development CPU backend
onnxDetector = personDetector(config, backend="onnx")  # models/yolo11n.onnx
detections = onnxDetector.detect(frame)
```

Both return `(x1, y1, x2, y2, confidence)` tuples in original-frame pixels, with COCO class 0 only, thresholding and NMS. Invalid frames and incompatible models raise errors. Missing weights raise `FileNotFoundError`; there are no silent model downloads. Existing `detectorInputSize` and `detectorConfidence` config keys are used; no new shared keys were introduced.

The native backend performs BGR→RGB conversion, square letterboxing, FP32 NCHW normalization, raw YOLO11 decoding, coordinate restoration, clipping and NMS. It expects a COCO YOLO11 export `(1, 84, N)` with `nms=False`, not an end-to-end/NMS export, YOLOX output or quantized input. It imports neither PyTorch nor Ultralytics. Backend/model/thread selection uses optional constructor arguments; the integration owner must wire that selection into the observation factory for Pi deployment. The current factory still uses the development default.

## Tests, measurement and export

```bash
.venv/bin/python -B -m unittest discover -s benchmarks -p 'test*.py' -v
.venv/bin/python -B benchmarks/measurePedestrian.py > benchmarks/pedestrianMeasurement.json
.venv/bin/python -B benchmarks/exportOnnx.py
.venv/bin/python -B benchmarks/checkBackends.py
```

The height script reads cached standard R2025a mesh vertices at neutral joint angles. It requires the head and both feet to have been cached by Webots. Mesh hashes identify the assets; it does not measure a walking pose's instantaneous height. Export produces a static batch-1, 320×320 FP32 model at ONNX opset 17 with no embedded NMS and validates it with `onnx.checker`.

Tests cover projected geometry, bearing sign, off-axis range, bad calibration/boxes, cropped-body bias, letterbox reversal, RGB normalization, filtering, NMS and empty/incompatible outputs. Synthetic pinhole tests do not establish real-scene accuracy.

## CPU benchmark

Run each backend separately, with no other benchmark or simulation running:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B benchmarks/benchmarkDetector.py --backend onnx --output benchmarks/laptopOnnx.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B benchmarks/benchmarkDetector.py --backend ultralytics --output benchmarks/laptopUltralytics.json
```

The Linux benchmark uses three calibrated frames loaded into RAM, 10 warmup calls and 100 measured calls. It measures synchronous detection including preprocessing/postprocessing. It excludes disk reads, model initialization, camera capture, tracking and movement. Model initialization time is reported separately. CPU utilization is process user+system CPU time over wall time; 100% means one occupied core. Machine-normalized CPU divides by logical CPU count. RAM reports RSS after measurement and process high-water RSS including startup/warmup, before result serialization.

Recorded laptop: Intel Core i5-5200U, 2 physical/4 logical cores, 14.53 GiB RAM, Python 3.12.15, one inference thread, 320 input, confidence 0.35. Sequential single trials on three repeated images are preliminary measurements, not statistical comparisons or accuracy evaluations.

| Backend | FPS | Mean / p95 latency | Process CPU | RSS / peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Native ONNX Runtime | 26.26 | 38.08 / 44.36 ms | 100.04% | 136.08 / 136.08 MiB |
| Ultralytics/PyTorch | 18.56 | 53.88 / 58.74 ms | 99.86% | 764.32 / 764.32 MiB |

Both produced detections on 33 of 100 repeated inputs because only one of the three input frames was detected. This is not recall on an independent dataset. Raw JSON includes package versions, model/frame hashes, timing scope and hardware. PyTorch and native ONNX agreed on all six initial baseline/calibrated frames, including the five misses; positive box/confidence values agreed within the checked tolerance (0.02 absolute plus 0.01% relative).

Pi results remain pending arrival of hardware. Re-run the native benchmark there with CPU temperature, power supply, throttling state, OS and sustained-run conditions recorded. No Pi or OpenVINO speedup is inferred from these laptop measurements. See [modelComparison.md](modelComparison.md) for the separately sourced model/runtime comparison.
