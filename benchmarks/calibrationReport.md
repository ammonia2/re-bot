# Camera, pedestrian and estimator validation

Measured on 2026-10-23 using Webots R2025a and the world committed at `7be4c1c`. The user's existing saved-world edits were preserved and excluded from commits. Capture scripts use a separate temporary project and the actual `webotsRobot.getFrame()` method. Frames and raw simulator records remain under ignored `data/`; the initial detector check preceded detector implementation.

## Constants checked against the implementation

| Quantity | Shared config | Measured / inspected | Assessment |
| --- | ---: | ---: | --- |
| Camera width | 640 px | 64 px from `getWidth()` | Mismatch |
| Camera height | 480 px | 64 px from `getHeight()` | Mismatch |
| Horizontal FOV | 0.7854 rad | 0.785398 rad from `getFov()` | Matches to rounding |
| Person height | 1.8 m | 1.838331 m neutral-pose mesh extent | Config is about 2.09% below neutral height |
| Detector input | 320 | Actual model tensor 320×320 | Matches |

The empty `Camera {}` in the world inherits simulator defaults. Changing width/height in YAML does not change the camera. The baseline camera origin is at the rover origin, approximately 0.0975 m above the floor in the controlled capture; the rover body occupies a large part of its image. The shared world/config were not corrected because they belong to integration.

Neutral mesh minimum Z is -1.279010 m (right foot), maximum Z 0.559321 m (head), giving 1.838331 m. `benchmarks/measurePedestrian.py` records per-part mesh hashes and extents in `pedestrianMeasurement.json`. The standard neutral mesh parts have no extra part translations at zero joint angles. The walking controller changes joint angles/root height, so 1.838331 m is not a constant observed silhouette height. The pedestrian's root translation of 1.27 m is not its full height.

The separate diagnostic variant sets camera width/height to 640×480, FOV 0.7854 and translation `(0.3, 0, 0.9)` in the temporary world. The measured camera height there is 0.9975 m. Neither that mounting nor those dimensions have been applied to the project world.

## First detector gate and live verification

Official YOLO11n COCO weights, SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`; Ultralytics 8.4.182, CPU, input 320, person class only, confidence 0.35, IoU threshold 0.45, square letterbox.

| Rover-to-pedestrian setup | Original camera | Temporary calibrated camera |
| --- | --- | --- |
| 3 m | No detection | No detection |
| 4 m | No detection | No detection |
| 5 m | No detection | Person, confidence 0.848893 |

This is three controlled frames per variant, not a labelled validation dataset or a recall estimate. The owner was notified immediately of the poor detection result. PyTorch and native ONNX agree on all six frames; reproducing missed detections is not a correctness claim about detection quality.

The implemented detector was subsequently run live inside each temporary Webots capture controller. The counts above were reproduced. Live metadata is in `data/liveBaseline/` and `data/liveCalibrated/`. This verifies detector execution on real simulator frames, not successful following or identity tracking. The main observation source remains ground truth.

## Estimator changes and evidence

The reference height formula gives optical-axis depth `Z = f × H / boxHeight`. The controller consumes a horizontal range and bearing, so the estimator now returns `Z / cos(bearing)`. At zero bearing the result is unchanged. This is horizontal camera-to-person range, not the 3D distance to a body point and not rover-center range when the camera is offset. Invalid boxes/calibration/heights now raise `ValueError` instead of turning zero-height boxes into huge distances.

Unit tests project known upright people at several signed bearings and ranges, verify the recovered range/bearing, and test degenerate boxes, invalid calibration and cropping bias. These validate the idealized pinhole calculation, not body pose or detector box accuracy.

One positive live diagnostic frame allowed a scene comparison:

| Metric | Value |
| --- | ---: |
| Camera-relative ground-truth horizontal range | 4.700027 m |
| Estimated range using actual intrinsics and configured 1.8 m height | 4.434777 m |
| Relative range error | 5.6436% |
| Ground-truth bearing | 0.003404 rad |
| Estimated bearing | 0.006036 rad |
| Absolute bearing error | 0.002631 rad |

Ground truth is computed from the camera pose and pedestrian root position, transformed into the camera frame. It does not use the simulator's recognition bounding box, whose shape-derived extents can overestimate the rendered silhouette. Perception itself has no supervisor access.

This single detected observation is within the proposed 0.05 rad / 15% tolerances. The other frames are missed observations. **The recorded-run accuracy milestone is not established by this one observation**, and no full-loop vision integration or Pi evaluation has been completed.

## Integration decisions still required

- The integration owner must align actual camera dimensions, FOV and mounting with config, then decide how to handle cropped bodies near the 1.5 m following distance. With the configured focal length, an ideal full 1.8 m body at 1.5 m optical depth projects to roughly 927 pixels vertically, exceeding a 480-pixel frame even when centered.
- Re-evaluate the synthetic pedestrian domain before relying on detection; the raised-camera diagnostic alone did not make detection reliable. Follow the contract's real-video/MOT17 fallback if the owner chooses it.
- Obtain a sufficiently varied recorded run with reliable full-body boxes before claiming the estimator's accuracy milestone. Include pose, cropping, bearing and range coverage, and distinguish camera range from rover-center separation.
- The pedestrian's trajectory and speed remain bundled into one `controllerArgs` item. The standard parser treats speed as part of the trajectory string, leaving its speed option at the default. This report did not change that owner-controlled world setting.
