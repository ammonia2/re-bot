import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from perception.detector import personDetector


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check PyTorch/native ONNX output agreement on captured frames")
    parser.add_argument("frames", nargs="*", type=Path)
    args = parser.parse_args()
    folders = args.frames or [root / "data/detectorFeasibility", root / "data/calibratedFeasibility"]
    paths = [path for folder in folders for path in sorted(folder.glob("*.png"))]
    if not paths:
        parser.error("Capture frames first")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "data/ultralyticsSettings"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    config = yaml.safe_load((root / "config/robotConfig.yaml").read_text())
    development = personDetector(config)
    native = personDetector(config, backend="onnx")
    positiveFrames = 0
    for path in paths:
        frame = cv2.imread(str(path))
        expected, actual = development.detect(frame), native.detect(frame)
        if len(expected) != len(actual):
            raise AssertionError(f"Detection count mismatch: {path}")
        if expected:
            np.testing.assert_allclose(actual, expected, atol=.02, rtol=1e-4, err_msg=str(path))
            positiveFrames += 1
        print(f"{path.name}: {len(actual)} person detections")
    if not positiveFrames:
        raise AssertionError("No positive frame; empty outputs alone cannot establish box agreement")
    print(f"PASS: {len(paths)} frames agree, including {positiveFrames} positive frames")


if __name__ == "__main__":
    main()
