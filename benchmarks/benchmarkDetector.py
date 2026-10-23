import argparse
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import resource
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import psutil
import yaml

from perception.detector import personDetector


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Measure synchronous CPU detection including pre/postprocessing")
    parser.add_argument("--backend", choices=("ultralytics", "onnx"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--frames", type=Path, default=root / "data/calibratedFeasibility")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1 or args.threads < 1:
        parser.error("iterations, warmup and threads must be positive")
    framePaths = sorted(args.frames.glob("*.png"))
    frames = [cv2.imread(str(path)) for path in framePaths]
    if not frames or any(frame is None for frame in frames):
        parser.error("frames must contain readable PNG images")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "data/ultralyticsSettings"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    cv2.setNumThreads(args.threads)
    config = yaml.safe_load((root / "config/robotConfig.yaml").read_text())
    start = time.perf_counter()
    detector = personDetector(config, backend=args.backend, modelPath=args.model, inferenceThreads=args.threads)
    initializationMs = (time.perf_counter() - start) * 1000
    detector.detect(frames[0])
    if args.backend == "ultralytics":
        import torch

        torch.set_num_threads(args.threads)
    for index in range(args.warmup):
        detector.detect(frames[index % len(frames)])

    process = psutil.Process()
    cpuBefore = process.cpu_times()
    latencies, counts = [], []
    start = time.perf_counter()
    for index in range(args.iterations):
        tick = time.perf_counter()
        detections = detector.detect(frames[index % len(frames)])
        latencies.append((time.perf_counter() - tick) * 1000)
        counts.append(len(detections))
    elapsed = time.perf_counter() - start
    cpuAfter = process.cpu_times()
    cpuSeconds = cpuAfter.user + cpuAfter.system - cpuBefore.user - cpuBefore.system
    packageVersions = {}
    for name in ("numpy", "opencv-python", "onnxruntime", "onnx", "ultralytics", "torch"):
        try:
            packageVersions[name] = version(name)
        except PackageNotFoundError:
            packageVersions[name] = None
    cpuName = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpuName = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")), cpuName)
    rssAfterMiB = process.memory_info().rss / 2 ** 20
    peakRss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peakRssMiB = max(rssAfterMiB, peakRss / (1024 ** 2 if platform.system() == "Darwin" else 1024))
    results = {
        "measuredAt": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
        "backend": args.backend, "device": "cpu", "cpu": cpuName,
        "platform": platform.platform(), "python": platform.python_version(),
        "physicalCores": psutil.cpu_count(logical=False), "logicalCores": psutil.cpu_count(),
        "ramGiB": psutil.virtual_memory().total / 2 ** 30,
        "threads": args.threads, "packages": packageVersions,
        "model": detector.modelPath.name,
        "modelSha256": hashlib.sha256(detector.modelPath.read_bytes()).hexdigest(),
        "inputSize": config["detectorInputSize"], "confidence": config["detectorConfidence"],
        "frames": [{"name": path.name, "shape": list(frame.shape),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path, frame in zip(framePaths, frames)],
        "iterations": args.iterations, "warmup": args.warmup,
        "initializationMs": initializationMs, "elapsedSeconds": elapsed,
        "fps": args.iterations / elapsed,
        "latencyMeanMs": float(np.mean(latencies)), "latencyP50Ms": float(np.median(latencies)),
        "latencyP95Ms": float(np.percentile(latencies, 95)),
        "processCpuPercent": 100 * cpuSeconds / elapsed,
        "machineNormalizedCpuPercent": 100 * cpuSeconds / elapsed / psutil.cpu_count(),
        "rssAfterMiB": rssAfterMiB, "processPeakRssMiB": peakRssMiB,
        "framesWithDetections": sum(count > 0 for count in counts),
        "scope": "Single process; synchronous detect(), pre/postprocessing included; disk IO, model load, camera and tracking excluded. Peak RSS includes startup/warmup. CPU percent: one fully occupied core equals 100%.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
