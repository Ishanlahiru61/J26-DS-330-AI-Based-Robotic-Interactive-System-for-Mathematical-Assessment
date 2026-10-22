import base64
import io

import cv2
import numpy as np
import pytest

from app import create_app
from model_manager import ModelManager


class FakeAdapter:
    def predict(self, image, confidence, iou):
        return [{"class_id": 0, "class_name": "cube", "confidence": .91,
                 "box_type": "xyxy", "box": [1, 2, 10, 12]}]

    def get_class_names(self): return {0: "cube"}
    def get_model_info(self):
        return {"model_family": "fake", "model_task": "detect", "input_size": 640,
                "device": "cpu", "device_fallback": None, "framework_version": "test",
                "runtime_python": "test", "checkpoint_metadata": {}}
    def is_obb(self): return False


def image_data():
    ok, encoded = cv2.imencode(".jpg", np.zeros((20, 30, 3), dtype=np.uint8))
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode()


@pytest.fixture
def app(tmp_path):
    application = create_app(tmp_path, tmp_path / "models")
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def test_homepage_and_empty_model_status(client):
    assert client.get("/").status_code == 200
    status = client.get("/api/model/status").get_json()
    assert status["model_loaded"] is False


def test_models_endpoint(client, app):
    manager: ModelManager = app.config["MODEL_MANAGER"]
    (manager.model_dir / "best.pt").write_bytes(b"x")
    response = client.get("/api/models").get_json()
    assert response["approved_directory"] == "models/"
    assert response["models"][0]["filename"] == "best.pt"


@pytest.mark.parametrize("filename", ["../best.pt", "bad.txt", "missing.pt"])
def test_invalid_model_load(client, filename):
    response = client.post("/api/model/load", json={"filename": filename, "family": "auto"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_upload_rejects_unsupported_format(client):
    response = client.post("/api/model/upload", data={"model": (io.BytesIO(b"x"), "bad.txt")})
    assert response.status_code == 400


def test_prediction_without_loaded_model(client):
    response = client.post("/api/model/predict", json={
        "image": image_data(), "confidence": .25, "iou": .45, "session_id": "no_model_1",
    })
    assert response.status_code == 409
    assert response.get_json()["error"] == "No model is loaded."


def test_mocked_api_prediction_and_frame_pairing(client, app):
    manager: ModelManager = app.config["MODEL_MANAGER"]
    manager._adapter = FakeAdapter()
    manager._model_path = manager.model_dir / "mock.pt"
    response = client.post("/api/model/predict", json={
        "image": image_data(), "confidence": .25, "iou": .45,
        "session_id": "mock_session", "frame_id": "frame_7",
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["session_id"] == "mock_session" and data["frame_id"] == "frame_7"
    assert data["total_count"] == 1 and data["class_counts"] == {"cube": 1}


def test_api_rejects_invalid_image_with_loaded_mock(client, app):
    manager: ModelManager = app.config["MODEL_MANAGER"]
    manager._adapter = FakeAdapter()
    response = client.post("/api/model/predict", json={
        "image": "not-base64", "confidence": .25, "iou": .45, "session_id": "bad_image",
    })
    assert response.status_code == 400
    assert "base64" in response.get_json()["error"]


@pytest.mark.parametrize(("field", "value", "message"), [
    ("confidence", "bad", "numeric"), ("confidence", 0, "Confidence"), ("iou", 1.5, "IoU"),
])
def test_invalid_prediction_thresholds(client, field, value, message):
    payload = {"image": image_data(), "confidence": .25, "iou": .45,
               "session_id": f"invalid_{field}_{str(value).replace('.', '_')}"}
    payload[field] = value
    response = client.post("/api/model/predict", json=payload)
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_existing_annotation_create_stats_export_and_delete(client):
    save = client.post("/api/save", json={
        "image": image_data(), "img_width": 30, "img_height": 20,
        "boxes": [{"cx": 10, "cy": 10, "w": 8, "h": 8, "angle": 0, "class_id": 0}],
    })
    assert save.status_code == 201
    filename = save.get_json()["filename"]
    stats = client.get("/api/stats").get_json()
    assert stats["total_images"] == 1 and stats["total_cubes"] == 1
    exported = client.post("/api/export_yaml").get_json()["content"]
    assert "0: cube" in exported
    assert client.delete(f"/api/annotations/{filename}").status_code == 200
    assert client.get("/api/stats").get_json()["total_images"] == 0
