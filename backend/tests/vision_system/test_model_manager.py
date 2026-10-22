import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from model_manager import (
    ModelError,
    ModelManager,
    UltralyticsAdapter,
    aggregate_detections,
)


class FakeAdapter:
    family = "fake"

    def __init__(self, detections=None):
        self.detections = detections or []

    def predict(self, image, confidence, iou):
        return self.detections

    def get_class_names(self):
        return {0: "red_cube", 1: "blue_cube"}

    def get_model_info(self):
        return {"model_family": "fake", "model_task": "detect", "input_size": 640,
                "device": "cpu", "device_fallback": None}

    def is_obb(self):
        return False


def jpeg_data_url():
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode()


def test_status_when_no_model_is_loaded(tmp_path):
    status = ModelManager(tmp_path / "models").status()
    assert status["model_loaded"] is False
    assert status["model_name"] is None


def test_available_model_listing_filters_formats(tmp_path):
    manager = ModelManager(tmp_path / "models")
    (manager.model_dir / "best.pt").write_bytes(b"checkpoint")
    (manager.model_dir / "cube.onnx").write_bytes(b"onnx")
    (manager.model_dir / "notes.txt").write_text("ignore")
    assert [item["filename"] for item in manager.list_models()] == ["best.pt", "cube.onnx"]


@pytest.mark.parametrize("filename", ["../best.pt", "/tmp/best.pt", "nested/best.pt"])
def test_model_directory_traversal_is_rejected(tmp_path, filename):
    with pytest.raises(ModelError, match="approved models directory"):
        ModelManager(tmp_path / "models").resolve_model(filename)


def test_unsupported_model_format_is_rejected(tmp_path):
    manager = ModelManager(tmp_path / "models")
    (manager.model_dir / "bad.engine").write_bytes(b"x")
    with pytest.raises(ModelError, match="Unsupported model format"):
        manager.resolve_model("bad.engine")


def test_missing_model_is_rejected(tmp_path):
    with pytest.raises(ModelError, match="does not exist"):
        ModelManager(tmp_path / "models").resolve_model("missing.pt")


def test_invalid_image_input(tmp_path):
    manager = ModelManager(tmp_path / "models")
    manager._adapter = FakeAdapter()
    with pytest.raises(ModelError, match="valid base64"):
        manager.predict("not base64!", 0.25, 0.45)


@pytest.mark.parametrize(("confidence", "iou", "message"), [
    (0, 0.45, "Confidence"), (1.1, 0.45, "Confidence"),
    (0.25, 0, "IoU"), (0.25, 2, "IoU"),
])
def test_threshold_validation(tmp_path, confidence, iou, message):
    manager = ModelManager(tmp_path / "models")
    manager._adapter = FakeAdapter()
    with pytest.raises(ModelError, match=message):
        manager.predict(jpeg_data_url(), confidence, iou)


def test_total_and_per_class_aggregation():
    detections = [
        {"class_name": "red_cube"}, {"class_name": "blue_cube"}, {"class_name": "red_cube"},
    ]
    assert aggregate_detections(detections) == (3, {"blue_cube": 1, "red_cube": 2})


class Tensor:
    def __init__(self, values): self.values = np.asarray(values)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.values


def test_ultralytics_normal_box_response_formatting():
    adapter = UltralyticsAdapter("cpu")
    adapter.names = {0: "cube"}
    adapter.model = type("Model", (), {"predict": lambda self, **kwargs: [type("Result", (), {
        "names": {0: "cube"}, "obb": None,
        "boxes": type("Boxes", (), {"xyxy": Tensor([[1, 2, 11, 12]]), "cls": Tensor([0]), "conf": Tensor([0.9])})(),
    })()]})()
    result = adapter.predict(np.zeros((20, 20, 3), dtype=np.uint8), 0.25, 0.45)
    assert result == [{"class_id": 0, "class_name": "cube", "confidence": 0.9,
                       "box_type": "xyxy", "box": [1.0, 2.0, 11.0, 12.0]}]


def test_ultralytics_obb_response_formatting():
    adapter = UltralyticsAdapter("cpu")
    adapter.names = {0: "cube"}
    obb = type("OBB", (), {"xyxyxyxy": Tensor([[[1, 2], [11, 2], [11, 12], [1, 12]]]),
                           "cls": Tensor([0]), "conf": Tensor([0.8])})()
    adapter.model = type("Model", (), {"predict": lambda self, **kwargs: [type("Result", (), {
        "names": {0: "cube"}, "obb": obb, "boxes": None,
    })()]})()
    result = adapter.predict(np.zeros((20, 20, 3), dtype=np.uint8), 0.25, 0.45)
    assert result[0]["box_type"] == "obb"
    assert result[0]["points"] == [[1.0, 2.0], [11.0, 2.0], [11.0, 12.0], [1.0, 12.0]]


def test_prediction_count_response(tmp_path):
    detections = [
        {"class_id": 0, "class_name": "red_cube", "confidence": .9, "box_type": "xyxy", "box": [1, 2, 3, 4]},
        {"class_id": 1, "class_name": "blue_cube", "confidence": .8, "box_type": "xyxy", "box": [5, 6, 7, 8]},
    ]
    manager = ModelManager(tmp_path / "models")
    manager._adapter = FakeAdapter(detections)
    manager._model_path = Path("fake.pt")
    result = manager.predict(jpeg_data_url(), .25, .45)
    assert result["total_count"] == 2
    assert result["class_counts"] == {"blue_cube": 1, "red_cube": 1}
    assert result["image_width"] == 32

