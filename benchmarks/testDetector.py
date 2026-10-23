from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from perception.detector import personDetector


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.config = {"detectorInputSize": 320, "detectorConfidence": 0.35}

    def makeOnnxDetector(self, output):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        modelPath = Path(temporary.name) / "model.onnx"
        modelPath.touch()
        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 320, 320])],
            run=lambda names, inputs: [output],
        )
        with patch("onnxruntime.InferenceSession", return_value=session):
            return personDetector(self.config, backend="onnx", modelPath=modelPath), session

    def testLetterboxIsUndoneAndOnlyPersonsAreReturned(self):
        output = np.zeros((1, 84, 4), dtype=np.float32)
        output[0, :4, :] = np.array([[160, 160, 80, 100], [160, 160, 160, 160], [100, 100, 50, 50], [100, 100, 100, 100]])
        output[0, 4, :] = [.9, .8, .7, .2]
        output[0, 6, 2] = .95
        detector, session = self.makeOnnxDetector(output)
        tensors = []
        session.run = lambda names, inputs: tensors.append(inputs["images"]) or [output]
        frame = np.full((480, 640, 3), (10, 20, 30), dtype=np.uint8)
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(detections[0], (220, 140, 420, 340, .9), atol=1e-5)
        self.assertEqual(tensors[0].shape, (1, 3, 320, 320))
        self.assertEqual(tensors[0].dtype, np.float32)
        np.testing.assert_allclose(tensors[0][0, :, 100, 100], np.array([30, 20, 10]) / 255)
        np.testing.assert_allclose(tensors[0][0, :, 0, 0], np.full(3, 114 / 255))
        np.testing.assert_array_equal(frame[0, 0], [10, 20, 30])
        self.assertTrue(all(isinstance(value, float) for value in detections[0]))

    def testEmptyPredictions(self):
        detector, _ = self.makeOnnxDetector(np.zeros((1, 84, 10), dtype=np.float32))
        self.assertEqual(detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)), [])

    def testBoxesAreClippedAndInvalidResultsDropped(self):
        detector, _ = self.makeOnnxDetector(np.zeros((1, 84, 0), dtype=np.float32))
        rows = [[-5, -10, 70, 80, .9, 0], [1, 1, 10, 10, .8, 1],
                [2, 2, 2, 5, .8, 0], [0, 0, 10, 10, float("nan"), 0]]
        self.assertEqual(detector._filterDetections(rows, (64, 64, 3))[:1][0][:4], (0, 0, 64, 64))
        self.assertEqual(len(detector._filterDetections(rows, (64, 64, 3))), 1)

    def testInvalidFrames(self):
        detector, _ = self.makeOnnxDetector(np.zeros((1, 84, 0), dtype=np.float32))
        for frame in (None, np.zeros((3, 3, 3)), np.zeros((3, 3), dtype=np.uint8), np.zeros((0, 3, 3), dtype=np.uint8)):
            with self.subTest(shape=getattr(frame, "shape", None)), self.assertRaises(ValueError):
                detector.detect(frame)

    def testIncompatibleExportFailsClearly(self):
        detector, _ = self.makeOnnxDetector(np.zeros((1, 10, 6), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "nms=False"):
            detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    def testMissingWeightsDoNotTriggerDownload(self):
        with self.assertRaises(FileNotFoundError):
            personDetector(self.config, modelPath="/tmp/rebot-nonexistent-weights.pt")

    def testUltralyticsUsesPersonOnlyAndOriginalPixelBoxes(self):
        rows = np.array([[10, 20, 40, 60, .8, 0]], dtype=np.float32)
        boxes = SimpleNamespace(data=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: rows)))
        model = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(boxes=boxes)]))
        with TemporaryDirectory() as folder:
            modelPath = Path(folder) / "model.pt"
            modelPath.touch()
            with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model))}):
                detector = personDetector(self.config, modelPath=modelPath)
                detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
        np.testing.assert_allclose(detections, [[10, 20, 40, 60, .8]], atol=1e-6)
        self.assertEqual(model.predict.call_args.kwargs["classes"], [0])
        self.assertEqual(model.predict.call_args.kwargs["imgsz"], 320)
        self.assertEqual(model.predict.call_args.kwargs["conf"], .35)

    def testInvalidDetectorConfigIsRejectedBeforeLoading(self):
        for values in ({"detectorInputSize": 319}, {"detectorConfidence": -1}, {"detectorConfidence": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                personDetector({**self.config, **values})

    def testConfidenceThresholdIsInclusive(self):
        detector, _ = self.makeOnnxDetector(np.zeros((1, 84, 0), dtype=np.float32))
        detector.confidence = 0.0
        self.assertEqual(detector._filterDetections([[1, 2, 3, 4, 0, 0]], (64, 64, 3)), [(1, 2, 3, 4, 0)])


if __name__ == "__main__":
    unittest.main()
