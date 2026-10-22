"""Secure, adapter-based model loading and normalized cube inference."""

from __future__ import annotations

import base64
import binascii
import importlib.metadata
import os
import platform
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".pt", ".onnx"}
SUPPORTED_FAMILIES = {"auto", "ultralytics", "onnx", "yolov5", "yolov7"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class ModelError(RuntimeError):
    """A model or inference error that is safe to show to the client."""


class DetectionModel(ABC):
    family = "unknown"

    @abstractmethod
    def load(self, model_path: Path) -> None: ...

    @abstractmethod
    def predict(self, image: np.ndarray, confidence: float, iou: float) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_class_names(self) -> dict[int, str]: ...

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]: ...

    @abstractmethod
    def is_obb(self) -> bool: ...


def select_torch_device() -> str:
    """Select CUDA, then Apple MPS, then CPU without requiring torch."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


class UltralyticsAdapter(DetectionModel):
    """Adapter for checkpoints accepted by the installed Ultralytics package."""

    family = "ultralytics"

    def __init__(self, device: str | None = None) -> None:
        self.device = device or select_torch_device()
        self.model: Any = None
        self.names: dict[int, str] = {}
        self.task = "unknown"
        self.input_size: Any = None
        self.device_fallback: str | None = None
        self.checkpoint_metadata: dict[str, Any] = {}

    def load(self, model_path: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc
        try:
            self.model = YOLO(str(model_path))
            self.names = _normalise_names(getattr(self.model, "names", {}))
            self.task = str(getattr(self.model, "task", "unknown") or "unknown")
            overrides = getattr(self.model, "overrides", {}) or {}
            checkpoint = getattr(self.model, "ckpt", {}) or {}
            train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
            self.input_size = overrides.get("imgsz") or train_args.get("imgsz")
            self.checkpoint_metadata = {
                key: value for key, value in {
                    "ultralytics_version": checkpoint.get("version") if isinstance(checkpoint, dict) else None,
                    "checkpoint_date": checkpoint.get("date") if isinstance(checkpoint, dict) else None,
                    "python_version": checkpoint.get("python") if isinstance(checkpoint, dict) else None,
                    "torch_version": checkpoint.get("torch") if isinstance(checkpoint, dict) else None,
                    "training_confidence": train_args.get("conf"),
                    "training_iou": train_args.get("iou"),
                }.items() if value is not None
            }
        except Exception as exc:
            raise ModelError(
                "The checkpoint is not compatible with the installed Ultralytics version. "
                "It may be an original YOLOv5/YOLOv7 checkpoint."
            ) from exc

    def _run(self, image: np.ndarray, confidence: float, iou: float) -> Any:
        return self.model.predict(
            source=image, conf=confidence, iou=iou, device=self.device, verbose=False
        )

    def predict(self, image: np.ndarray, confidence: float, iou: float) -> list[dict[str, Any]]:
        try:
            results = self._run(image, confidence, iou)
        except Exception as exc:
            if self.device == "mps":
                self.device_fallback = f"MPS inference failed; using CPU ({type(exc).__name__})"
                self.device = "cpu"
                try:
                    results = self._run(image, confidence, iou)
                except Exception as cpu_exc:
                    raise ModelError("Inference failed on MPS and CPU.") from cpu_exc
            else:
                raise ModelError("Model inference failed.") from exc

        if not results:
            return []
        result = results[0]
        names = _normalise_names(getattr(result, "names", self.names)) or self.names
        detections: list[dict[str, Any]] = []

        obb = getattr(result, "obb", None)
        if obb is not None and getattr(obb, "xyxyxyxy", None) is not None:
            points_array = obb.xyxyxyxy.detach().cpu().numpy()
            classes = obb.cls.detach().cpu().numpy()
            scores = obb.conf.detach().cpu().numpy()
            for points, class_id, score in zip(points_array, classes, scores):
                cid = int(class_id)
                detections.append({
                    "class_id": cid,
                    "class_name": names.get(cid, str(cid)),
                    "confidence": round(float(score), 6),
                    "box_type": "obb",
                    "points": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
                })
            return detections

        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []
        coordinates = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        for box, class_id, score in zip(coordinates, classes, scores):
            cid = int(class_id)
            detections.append({
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
                "confidence": round(float(score), 6),
                "box_type": "xyxy",
                "box": [round(float(value), 3) for value in box],
            })
        return detections

    def get_class_names(self) -> dict[int, str]:
        return dict(self.names)

    def get_model_info(self) -> dict[str, Any]:
        try:
            framework_version = importlib.metadata.version("ultralytics")
        except importlib.metadata.PackageNotFoundError:
            framework_version = None
        return {
            "model_family": self.family,
            "model_task": self.task,
            "input_size": self.input_size,
            "device": self.device,
            "device_fallback": self.device_fallback,
            "framework_version": framework_version,
            "runtime_python": platform.python_version(),
            "checkpoint_metadata": self.checkpoint_metadata,
        }

    def is_obb(self) -> bool:
        return self.task == "obb"


class ONNXAdapter(DetectionModel):
    """ONNX Runtime adapter for common exported YOLO axis-aligned detectors."""

    family = "onnx"

    def __init__(self) -> None:
        self.session: Any = None
        self.input_name = ""
        self.input_height = 640
        self.input_width = 640
        self.names: dict[int, str] = {}
        self.device = "cpu"
        self.metadata: dict[str, str] = {}

    def load(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError(
                "ONNX Runtime is not installed. Run: pip install -r requirements.txt"
            ) from exc
        available = ort.get_available_providers()
        preferred = [name for name in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider") if name in available]
        try:
            self.session = ort.InferenceSession(str(model_path), providers=preferred)
        except Exception as exc:
            raise ModelError("The ONNX model could not be loaded by ONNX Runtime.") from exc
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = input_meta.shape
        if len(shape) != 4:
            raise ModelError("Only 4D NCHW YOLO ONNX inputs are supported.")
        if isinstance(shape[2], int):
            self.input_height = shape[2]
        if isinstance(shape[3], int):
            self.input_width = shape[3]
        active = self.session.get_providers()[0]
        self.device = {"CUDAExecutionProvider": "cuda", "CoreMLExecutionProvider": "mps"}.get(active, "cpu")
        metadata = self.session.get_modelmeta().custom_metadata_map or {}
        self.metadata = {str(key): str(value) for key, value in metadata.items()
                         if key in {"version", "description", "author", "date"}}
        raw_names = metadata.get("names", "")
        if raw_names:
            try:
                import ast
                self.names = _normalise_names(ast.literal_eval(raw_names))
            except (ValueError, SyntaxError):
                self.names = {}

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        height, width = image.shape[:2]
        ratio = min(self.input_width / width, self.input_height / height)
        new_width, new_height = round(width * ratio), round(height * ratio)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_width - new_width) / 2
        pad_y = (self.input_height - new_height) / 2
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        left, top = round(pad_x - 0.1), round(pad_y - 0.1)
        canvas[top:top + new_height, left:left + new_width] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, 0), ratio, float(left), float(top)

    def predict(self, image: np.ndarray, confidence: float, iou: float) -> list[dict[str, Any]]:
        blob, ratio, pad_x, pad_y = self._letterbox(image)
        try:
            output = np.asarray(self.session.run(None, {self.input_name: blob})[0])
        except Exception as exc:
            raise ModelError("ONNX inference failed.") from exc
        output = np.squeeze(output)
        if output.ndim != 2:
            raise ModelError("Unsupported ONNX output shape; expected a 2D YOLO detection tensor.")
        # Ultralytics exports [features, anchors]; YOLOv5 commonly exports [anchors, features].
        if output.shape[0] < output.shape[1] and output.shape[0] <= 256:
            output = output.T
        if output.shape[1] < 6:
            raise ModelError("Unsupported ONNX detection output.")

        # Handle end-to-end exports with x1,y1,x2,y2,score,class directly.
        is_end_to_end = (
            output.shape[1] == 6
            and np.all(output[:, 5] >= 0)
            and np.allclose(output[:, 5], np.round(output[:, 5]), atol=1e-5)
        )
        boxes_xywh: list[list[float]] = []
        boxes_xyxy: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in output:
            if is_end_to_end:
                score, cid = float(row[4]), int(row[5])
                if score < confidence:
                    continue
                boxes_xyxy.append([float(value) for value in row[:4]])
                boxes_xywh.append([float(row[0]), float(row[1]), float(row[2] - row[0]), float(row[3] - row[1])])
            else:
                # YOLOv8: 4 + classes. YOLOv5: 4 + objectness + classes.
                is_yolov5_raw = (
                    (self.names and output.shape[1] == len(self.names) + 5)
                    or (not self.names and output.shape[1] in {6, 85})
                )
                if is_yolov5_raw:
                    objectness = float(row[4])
                    class_scores = row[5:]
                    cid = int(np.argmax(class_scores))
                    score = objectness * float(class_scores[cid])
                else:
                    class_scores = row[4:]
                    cid = int(np.argmax(class_scores))
                    score = float(class_scores[cid])
                if score < confidence:
                    continue
                cx, cy, width, height = (float(value) for value in row[:4])
                boxes_xywh.append([cx - width / 2, cy - height / 2, width, height])
                boxes_xyxy.append([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
            scores.append(score)
            class_ids.append(cid)

        if not boxes_xywh:
            return []
        kept_indices: list[int] = []
        # NMS is class-aware so overlapping cubes of different colours survive.
        for class_id in sorted(set(class_ids)):
            candidates = [index for index, value in enumerate(class_ids) if value == class_id]
            class_boxes = [boxes_xywh[index] for index in candidates]
            class_scores = [scores[index] for index in candidates]
            selected = cv2.dnn.NMSBoxes(class_boxes, class_scores, confidence, iou)
            kept_indices.extend(candidates[int(index)] for index in np.asarray(selected).reshape(-1))
        detections: list[dict[str, Any]] = []
        image_height, image_width = image.shape[:2]
        for index in kept_indices:
            box = boxes_xyxy[int(index)]
            scaled = [
                max(0.0, min(float(image_width), (box[0] - pad_x) / ratio)),
                max(0.0, min(float(image_height), (box[1] - pad_y) / ratio)),
                max(0.0, min(float(image_width), (box[2] - pad_x) / ratio)),
                max(0.0, min(float(image_height), (box[3] - pad_y) / ratio)),
            ]
            cid = class_ids[int(index)]
            detections.append({
                "class_id": cid,
                "class_name": self.names.get(cid, str(cid)),
                "confidence": round(float(scores[int(index)]), 6),
                "box_type": "xyxy",
                "box": [round(value, 3) for value in scaled],
            })
        return detections

    def get_class_names(self) -> dict[int, str]:
        return dict(self.names)

    def get_model_info(self) -> dict[str, Any]:
        try:
            framework_version = importlib.metadata.version("onnxruntime")
        except importlib.metadata.PackageNotFoundError:
            framework_version = None
        return {
            "model_family": self.family,
            "model_task": "detect",
            "input_size": [self.input_width, self.input_height],
            "device": self.device,
            "device_fallback": None,
            "framework_version": framework_version,
            "runtime_python": platform.python_version(),
            "checkpoint_metadata": self.metadata,
        }

    def is_obb(self) -> bool:
        return False


def aggregate_detections(detections: list[dict[str, Any]]) -> tuple[int, dict[str, int]]:
    counts = Counter(str(item["class_name"]) for item in detections)
    return len(detections), dict(sorted(counts.items()))


class ModelManager:
    """Own the active model, approved directory, decoding, and inference lock."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._adapter: DetectionModel | None = None
        self._model_path: Path | None = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {"filename": path.name, "size": path.stat().st_size, "extension": path.suffix.lower()}
            for path in sorted(self.model_dir.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    def resolve_model(self, filename: str) -> Path:
        if not filename or Path(filename).name != filename:
            raise ModelError("Model path must be a filename inside the approved models directory.")
        path = (self.model_dir / filename).resolve()
        try:
            path.relative_to(self.model_dir)
        except ValueError as exc:
            raise ModelError("Model path is outside the approved models directory.") from exc
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ModelError("Unsupported model format. Supported formats: .pt, .onnx")
        if not path.is_file():
            raise ModelError("Model file does not exist.")
        return path

    def save_upload(self, storage: Any) -> str:
        from werkzeug.utils import secure_filename

        filename = secure_filename(storage.filename or "")
        if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ModelError("Unsupported model format. Supported formats: .pt, .onnx")
        destination = self.model_dir / filename
        if destination.exists():
            raise ModelError("A model with that filename already exists; rename it before uploading.")
        temporary = self.model_dir / f".{filename}.{os.getpid()}.upload"
        try:
            storage.save(temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return filename

    def load(self, filename: str, family: str = "auto") -> dict[str, Any]:
        family = (family or "auto").lower()
        if family not in SUPPORTED_FAMILIES:
            raise ModelError("Unknown model family selection.")
        path = self.resolve_model(filename)
        if family in {"yolov5", "yolov7"}:
            raise ModelError(
                f"Original {family.upper()} checkpoints are not enabled because no matching local "
                "training repository was found. Try Auto detect/Ultralytics, or export with the original repository."
            )
        if family == "onnx" and path.suffix.lower() != ".onnx":
            raise ModelError("ONNX was selected, but the chosen file is not .onnx.")
        if family == "ultralytics" and path.suffix.lower() != ".pt":
            raise ModelError("Ultralytics YOLO was selected, but the chosen file is not .pt.")
        adapter: DetectionModel
        if path.suffix.lower() == ".onnx" or family == "onnx":
            adapter = ONNXAdapter()
        else:
            adapter = UltralyticsAdapter()
        try:
            adapter.load(path)
        except ModelError as exc:
            self._last_error = str(exc)
            raise
        with self._lock:
            self._adapter = adapter
            self._model_path = path
            self._last_error = None
        return self.status()

    def status(self) -> dict[str, Any]:
        adapter = self._adapter
        base = {
            "model_loaded": adapter is not None,
            "model_name": self._model_path.name if self._model_path else None,
            "loading_status": "loaded" if adapter else "not_loaded",
            "loading_error": self._last_error,
        }
        if adapter is None:
            return {**base, "model_family": None, "model_task": None, "box_type": None,
                    "class_names": {}, "input_size": None, "device": select_torch_device(),
                    "device_fallback": None, "framework_version": None,
                    "runtime_python": platform.python_version(), "checkpoint_metadata": {}}
        info = adapter.get_model_info()
        return {**base, **info, "box_type": "obb" if adapter.is_obb() else "xyxy",
                "class_names": adapter.get_class_names()}

    @staticmethod
    def decode_image(image_data: str) -> np.ndarray:
        if not isinstance(image_data, str) or not image_data:
            raise ModelError("No image data provided.")
        encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ModelError("Input image is not valid base64 data.") from exc
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ModelError("Input image is empty or exceeds the 12 MB limit.")
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ModelError("Input image could not be decoded.")
        height, width = image.shape[:2]
        if width * height > MAX_IMAGE_PIXELS:
            raise ModelError("Input image dimensions exceed the 20 megapixel limit.")
        return image

    def predict(self, image_data: str, confidence: float, iou: float) -> dict[str, Any]:
        if not 0.0 < confidence <= 1.0:
            raise ModelError("Confidence must be greater than 0 and at most 1.")
        if not 0.0 < iou <= 1.0:
            raise ModelError("IoU must be greater than 0 and at most 1.")
        image = self.decode_image(image_data)
        if self._adapter is None:
            raise ModelError("No model is loaded.")
        with self._lock:
            if self._adapter is None:
                raise ModelError("No model is loaded.")
            started = time.perf_counter()
            detections = self._adapter.predict(image, confidence, iou)
            inference_ms = (time.perf_counter() - started) * 1000
            status = self.status()
        total, class_counts = aggregate_detections(detections)
        return {
            **status,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "total_count": total,
            "class_counts": class_counts,
            "inference_ms": round(inference_ms, 2),
            "detections": detections,
        }
