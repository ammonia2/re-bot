import argparse
import os
from pathlib import Path

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export raw FP32 COCO YOLO11n for the native detector")
    parser.add_argument("--model", type=Path, default=root / "models/yolo11n.pt")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("Download the official weights into models/ first; see benchmarks/README.md")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "data/ultralyticsSettings"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import onnx
    from ultralytics import YOLO

    config = yaml.safe_load((root / "config/robotConfig.yaml").read_text())
    model = YOLO(str(args.model.resolve()), task="detect")
    output = model.export(
        format="onnx", imgsz=config["detectorInputSize"], batch=1,
        dynamic=False, simplify=False, half=False, nms=False, device="cpu", opset=17,
    )
    onnx.checker.check_model(str(output))
    print(f"Validated ONNX model: {output}")


if __name__ == "__main__":
    main()
