import os
from pathlib import Path

import cv2
import numpy as np


class personDetector:
    def __init__(self, config, backend="ultralytics", modelPath=None, inferenceThreads=1):
        self.config = dict(config)
        self.inputSize = int(config["detectorInputSize"])
        self.confidence = float(config["detectorConfidence"])
        self.nmsThreshold = 0.45
        self.backend = backend
        if self.inputSize < 32 or self.inputSize % 32 or self.inputSize != config["detectorInputSize"]:
            raise ValueError("detectorInputSize must be a positive multiple of 32")
        if not 0 <= self.confidence <= 1:
            raise ValueError("detectorConfidence must be between zero and one")
        if backend not in ("ultralytics", "onnx"):
            raise ValueError("backend must be 'ultralytics' or 'onnx'")
        suffix = ".pt" if backend == "ultralytics" else ".onnx"
        defaultPath = Path(__file__).resolve().parents[2] / "models" / f"yolo11n{suffix}"
        self.modelPath = Path(modelPath) if modelPath is not None else defaultPath
        if not self.modelPath.is_file():
            raise FileNotFoundError(f"Model not found: {self.modelPath}; see benchmarks/README.md")

        if backend == "ultralytics":
            # inference is CPU-only; hiding CUDA stops a CUDA build of torch probing NVML and warning on GPU-less machines
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            from ultralytics import YOLO

            self.model = YOLO(str(self.modelPath), task="detect")
        else:
            import onnxruntime as ort

            if not isinstance(inferenceThreads, int) or inferenceThreads < 1:
                raise ValueError("inferenceThreads must be a positive integer")
            options = ort.SessionOptions()
            options.intra_op_num_threads = inferenceThreads
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(self.modelPath), sess_options=options, providers=["CPUExecutionProvider"]
            )
            modelInput = self.session.get_inputs()[0]
            if modelInput.type != "tensor(float)" or len(modelInput.shape) != 4:
                raise ValueError("Expected a float32 NCHW ONNX input")
            for actual, expected in zip(modelInput.shape, (1, 3, self.inputSize, self.inputSize)):
                if isinstance(actual, int) and actual != expected:
                    raise ValueError("ONNX input shape does not match detectorInputSize")
            self.inputName = modelInput.name

    def detect(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("frame must be a uint8 BGR numpy array")
        if frame.ndim != 3 or frame.shape[2] != 3 or not all(frame.shape[:2]):
            raise ValueError("frame must have shape (height, width, 3) with nonzero dimensions")
        if self.backend == "ultralytics":
            prediction = self.model.predict(
                frame, imgsz=self.inputSize, conf=self.confidence, classes=[0],
                iou=self.nmsThreshold, device="cpu", rect=False, verbose=False,
            )[0]
            rows = prediction.boxes.data.cpu().numpy()
        else:
            rows = self._detectOnnx(frame)
        return self._filterDetections(rows, frame.shape)

    def _detectOnnx(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.inputSize / width, self.inputSize / height)
        resizedWidth, resizedHeight = round(width * scale), round(height * scale)
        leftPad = round((self.inputSize - resizedWidth) / 2 - 0.1)
        topPad = round((self.inputSize - resizedHeight) / 2 - 0.1)
        resized = cv2.resize(frame, (resizedWidth, resizedHeight), interpolation=cv2.INTER_LINEAR)
        image = np.full((self.inputSize, self.inputSize, 3), 114, dtype=np.uint8)
        image[topPad:topPad + resizedHeight, leftPad:leftPad + resizedWidth] = resized
        tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        output = self.session.run(None, {self.inputName: tensor})[0]
        if output.ndim != 3 or output.shape[:2] != (1, 84):
            raise ValueError("Expected raw COCO YOLO11 output (1, 84, N); export with nms=False")
        predictions = output[0].T
        classIds = predictions[:, 4:].argmax(axis=1)
        selected = predictions[(classIds == 0) & (predictions[:, 4] >= self.confidence)]
        rows = np.zeros((len(selected), 6), dtype=np.float32)
        rows[:, 0] = (selected[:, 0] - selected[:, 2] / 2 - leftPad) / scale
        rows[:, 1] = (selected[:, 1] - selected[:, 3] / 2 - topPad) / scale
        rows[:, 2] = (selected[:, 0] + selected[:, 2] / 2 - leftPad) / scale
        rows[:, 3] = (selected[:, 1] + selected[:, 3] / 2 - topPad) / scale
        rows[:, 4] = selected[:, 4]
        return rows

    def _filterDetections(self, rows, frameShape):
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6).copy()
        rows = rows[np.isfinite(rows).all(axis=1)]
        rows = rows[(rows[:, 5] == 0) & (rows[:, 4] >= self.confidence) & (rows[:, 4] <= 1)]
        height, width = frameShape[:2]
        rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, width)
        rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, height)
        rows = rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]
        if not len(rows):
            return []
        boxes = np.column_stack((rows[:, :2], rows[:, 2:4] - rows[:, :2])).tolist()
        scores = rows[:, 4].tolist()
        nmsScores = [max(score, float(np.finfo(np.float32).tiny)) for score in scores]
        indices = cv2.dnn.NMSBoxes(boxes, nmsScores, 0.0, self.nmsThreshold)
        indices = sorted(np.asarray(indices).reshape(-1), key=lambda index: scores[index], reverse=True)
        return [tuple(float(value) for value in rows[index, :5]) for index in indices]
