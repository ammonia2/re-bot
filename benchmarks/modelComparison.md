# YOLO families and Intel OpenVINO for re-bot

Research date: 2026-10-23. Published model results below are reference values, not measurements on this laptop. The local experiments benchmark YOLO11n through PyTorch and ONNX Runtime only. OpenVINO, YOLOX, YOLOv8 and YOLOv9 have not been benchmarked locally.

## Architecture and runtime are separate choices

OpenVINO is an inference optimization/runtime toolkit, not a competing YOLO architecture. A useful comparison keeps the detector weights fixed and compares PyTorch, ONNX Runtime and OpenVINO; a separate experiment compares different detector architectures under the same runtime. Intel's verified-model list includes YOLO11, YOLOv8n and YOLOX variants, with validation coverage varying by device and precision. A listed result does not guarantee every export/device combination works. [Intel verified models](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html).

## Small-model candidates

| Candidate | Published input | COCO validation AP50–95 | Parameters | Published FLOPs | Relevance |
| --- | ---: | ---: | ---: | ---: | --- |
| YOLOX-Nano | 416 | 25.8 | 0.91 M | 1.08 G | Small alternative with a separate deployment/output decoder |
| YOLOX-Tiny | 416 | 32.8 | 5.06 M | 6.45 G | Larger YOLOX baseline |
| YOLOv8n | 640 | 37.3 | 3.2 M | 8.7 G | Mature Ultralytics baseline with the same development API |
| YOLOv9-T | 640 | 38.3 | 2.0 M | 7.7 G | Small variant from the original YOLOv9 implementation |
| YOLO11n | 640 | 39.5 | 2.6 M | 6.5 G | Required project baseline; implemented here |

Sources: [YOLOX authors' model table](https://github.com/Megvii-BaseDetection/YOLOX#benchmark), [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8), [original YOLOv9](https://github.com/WongKinYiu/yolov9#performance), [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11).

These are different training recipes and, for YOLOX, different input sizes. They do not establish person-only accuracy, Webots recall, latency at 320 pixels, or Pi performance. This project filters COCO class 0 after inference; it still computes the original multi-class network. Filtering does not turn it into a trained single-class network.

YOLOX uses an anchor-free design with a decoupled head and SimOTA training assignment. Its deployment examples support ONNX Runtime and OpenVINO, but its outputs/preprocessing must not be sent through this detector's YOLO11 decoder. [YOLOX paper](https://arxiv.org/abs/2107.08430), [authors' deployment examples](https://github.com/Megvii-BaseDetection/YOLOX#deployment).

YOLOv8 supplies an anchor-free split head and a broad export workflow. YOLOv9 introduces Programmable Gradient Information; identify the exact original or Ultralytics implementation and inference/reparameterized export before comparing it. YOLO11 is the requested baseline and has supported detection exports. Architecture names alone do not specify the artifact's input/output contract. [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8), [YOLOv9 authors](https://github.com/WongKinYiu/yolov9), [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11).

## OpenVINO evaluation

Ultralytics provides an OpenVINO export route, including calibrated INT8 options. Begin with the same YOLO11n weights, input 320, confidence 0.35, batch 1 and FP32 precision as the recorded ONNX result. Measure CPU first, then any supported Intel GPU/NPU separately; record driver/runtime versions and compilation time. GPU/NPU require suitable hardware and drivers. Do not count model compilation as per-frame latency, or omit preprocessing/NMS from only one backend. [Ultralytics OpenVINO integration](https://docs.ultralytics.com/integrations/openvino), [Intel requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html).

The measured laptop is a fifth-generation i5-5200U. The current Intel requirements list sixth-generation and newer Core CPUs; this laptop is outside that listed Core generation range. OpenVINO compatibility on it must be established experimentally. Do not assume a recent Intel NPU is present. Current documentation also lists ARM CPU support, but that does not establish compatibility or performance on an unspecified Raspberry Pi/OS. Keep the contract's ONNX Runtime deployment baseline until the actual Pi is measured. [Intel CPU support matrix](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html).

INT8 calibration needs representative real camera inputs, including near/far people and relevant backgrounds. Validate person misses, box-height bias and range errors after quantization; a detector that is faster but crops the legs differently can degrade distance estimation. This is an experimental recommendation, not a measured INT8 result. [Ultralytics export options](https://docs.ultralytics.com/modes/export).

## Proposed fair experiment

1. Resolve camera calibration with the integration owner and obtain a labelled real-camera evaluation set alongside simulation frames. Preserve identities/occlusions for downstream testing.
2. Compare YOLO11n, YOLOv8n, YOLOv9-T and YOLOX-Nano at matched input sizes and precision, with the correct decoder for each export. Retain each original model's published-size run as a separate reference.
3. Report person precision/recall and misses by range, crop and occlusion; report box-height/bearing errors because those drive robot commands. COCO all-class AP alone cannot select this robot's model.
4. Measure single-frame p50/p95 latency, throughput, CPU and process RAM with equal thread limits. Repeat runs and include sustained temperature/throttling tests on the Pi.
5. For supported Intel hardware, compare fixed YOLO11n weights through ONNX Runtime and OpenVINO FP32, then try FP16/INT8 as separate accuracy/performance experiments.

Current recommendation: retain YOLO11n as the contract baseline and native ONNX as the portable deployment path. Investigate OpenVINO as an additional runtime on verified Intel hardware. The immediate blocker is detection reliability: baseline Webots captures yielded 0/3 detections, and even the temporary calibrated view yielded 1/3. No model family has been demonstrated to solve that domain gap in this work.
