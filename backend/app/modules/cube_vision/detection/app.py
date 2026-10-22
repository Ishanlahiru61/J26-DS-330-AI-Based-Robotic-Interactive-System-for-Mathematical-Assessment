

from __future__ import annotations

import argparse
import base64
import binascii
import logging
import re
import tempfile
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from werkzeug.exceptions import RequestEntityTooLarge

if __package__ in (None, ""):
    from dataset_manager import DatasetManager
    from model_manager import ModelError, ModelManager
    from camera_manager import CameraManager
else:
    from .dataset_manager import DatasetManager
    from .model_manager import ModelError, ModelManager
    from .camera_manager import CameraManager


def create_app(
    dataset_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> Flask:
    """Create and configure the Flask application."""
    if dataset_dir is None:
        dataset_dir = Path(__file__).parent

    app = Flask(__name__)
    manager = DatasetManager(dataset_dir)
    models = ModelManager(model_dir or (Path(dataset_dir) / "models"))
    app.config["MANAGER"] = manager
    app.config["MODEL_MANAGER"] = models
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

    # -- temp storage for captured frames before annotation -----------------
    temp_dir = Path(tempfile.mkdtemp(prefix="cube_capture_"))
    app.config["TEMP_DIR"] = temp_dir

    # -- Camera manager (AVFoundation / OpenCV) ----------------------------
    camera_mgr = CameraManager()
    # Shared mutable state for the active MJPEG stream
    _stream_lock = threading.Lock()
    _active_cap: list = [None]       # list-cell so the closure can mutate it
    _active_index: list = [-1]

    # Browser-session inference guard: one request at a time and at most 5/sec.
    prediction_guard = threading.Lock()
    prediction_active: set[str] = set()
    prediction_last: dict[str, float] = {}
    session_pattern = re.compile(r"^[A-Za-z0-9_-]{1,100}$")

    def json_error(message: str, status: int = 400):
        return jsonify({"error": message}), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return json_error("Upload exceeds the 512 MB request limit.", 413)

   

    @app.get("/")
    def index():
        return render_template("index.html")

    # -- Camera enumeration and streaming routes ----------------------------

    @app.get("/api/cameras")
    def api_cameras():
        """Return all detected video input devices.

        Uses FFmpeg AVFoundation listing for real device names, then probes
        each with cv2.CAP_AVFOUNDATION to confirm it is usable.
        Set ?probe=0 to skip frame probing for a faster response.
        """
        probe = request.args.get("probe", "1") != "0"
        try:
            cameras = camera_mgr.list_cameras(probe_frames=probe)
        except Exception as exc:
            logging.getLogger(__name__).error("Camera enumeration failed: %s", exc)
            return jsonify({"cameras": [], "error": str(exc)})
        return jsonify({"cameras": [c.as_dict() for c in cameras]})

    @app.get("/api/camera/stream/<int:cam_index>")
    def api_camera_stream(cam_index: int):
        """MJPEG stream from the selected camera (AVFoundation backend).

        Opens cv2.VideoCapture(cam_index, CAP_AVFOUNDATION), encodes each
        frame as JPEG, and sends a multipart/x-mixed-replace stream.

        Only one stream is allowed at a time.  Opening a new index stops the
        previous one automatically.
        """
        import cv2  # local import keeps startup clean on systems without display

        def _release_current():
            with _stream_lock:
                if _active_cap[0] is not None:
                    CameraManager.release(_active_cap[0])
                    _active_cap[0] = None
                    _active_index[0] = -1

        def _generate():
            _release_current()

            cap = camera_mgr.open_camera(cam_index)
            if cap is None:
                # Send a single error boundary so the browser knows it failed
                error_msg = (
                    f"--frame\r\n"
                    f"Content-Type: text/plain\r\n\r\n"
                    f"Camera {cam_index} could not be opened. "
                    f"Check macOS camera permissions.\r\n"
                )
                yield error_msg.encode()
                return

            with _stream_lock:
                _active_cap[0] = cap
                _active_index[0] = cam_index

            try:
                while True:
                    with _stream_lock:
                        if _active_index[0] != cam_index:
                            break
                        current_cap = _active_cap[0]
                    if current_cap is None:
                        break
                    ok, frame = current_cap.read()
                    if not ok or frame is None:
                        time.sleep(0.02)
                        continue
                    ret, jpeg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    if not ret:
                        continue
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n"
                    )
            finally:
                # Only release if we are still the owner
                with _stream_lock:
                    if _active_index[0] == cam_index:
                        CameraManager.release(_active_cap[0])
                        _active_cap[0] = None
                        _active_index[0] = -1

        return Response(
            stream_with_context(_generate()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/camera/stop")
    def api_camera_stop():
        """Stop the active MJPEG stream and release the camera."""
        with _stream_lock:
            if _active_cap[0] is not None:
                CameraManager.release(_active_cap[0])
                _active_cap[0] = None
                stopped = _active_index[0]
                _active_index[0] = -1
                return jsonify({"stopped": stopped})
        return jsonify({"stopped": None})

    @app.get("/api/camera/frame/<int:cam_index>")
    def api_camera_frame(cam_index: int):
        """Capture a single JPEG frame from the camera and return it as base64.

        Useful for the Capture view when using a server-side (USB) camera.
        Opens, grabs one frame, releases immediately.
        """
        import cv2
        cap = camera_mgr.open_camera(cam_index)
        if cap is None:
            return json_error(
                f"Camera {cam_index} could not be opened. "
                "Verify it is connected and that macOS has granted camera access."
            )
        ok, frame = cap.read()
        CameraManager.release(cap)
        if not ok or frame is None:
            return json_error("Camera opened but could not read a frame.")
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret:
            return json_error("Failed to encode frame as JPEG.")
        b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
        return jsonify({
            "image": f"data:image/jpeg;base64,{b64}",
            "width": frame.shape[1],
            "height": frame.shape[0],
        })



   

    @app.post("/api/capture")
    def api_capture():
        """Receive a base64-encoded JPEG frame from the browser camera."""
        mgr: DatasetManager = app.config["MANAGER"]
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image", "")

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except (binascii.Error, ValueError):
            return jsonify({"error": "Invalid base64 image data"}), 400

       
        temp_path = temp_dir / "capture_temp.jpg"
        with open(temp_path, "wb") as fh:
            fh.write(image_bytes)

        return jsonify({
            "message": "Image captured",
            "temp_path": str(temp_path),
            "size": len(image_bytes),
        })

   

    @app.post("/api/save")
    def api_save():
        """Save the captured image with bounding box annotations."""
        mgr: DatasetManager = app.config["MANAGER"]
        payload = request.get_json(silent=True) or {}

        image_data = payload.get("image", "")
        boxes = payload.get("boxes", [])
        img_width = int(payload.get("img_width", 0))
        img_height = int(payload.get("img_height", 0))

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        if img_width <= 0 or img_height <= 0:
            return jsonify({"error": "Invalid image dimensions"}), 400

        
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except (binascii.Error, ValueError):
            return jsonify({"error": "Invalid base64 image data"}), 400

        record = mgr.save(image_bytes, boxes, img_width, img_height)

        return jsonify({
            "message": f"Saved {record.filename} to {record.split}",
            "filename": record.filename,
            "split": record.split,
            "num_boxes": len(boxes),
        }), 201



    @app.get("/api/annotations")
    def api_annotations():
        """List all saved annotations."""
        mgr: DatasetManager = app.config["MANAGER"]
        return jsonify(mgr.list_annotations())

    @app.get("/api/annotations/<name>")
    def api_annotation_detail(name: str):
        """Get annotation details for one image."""
        mgr: DatasetManager = app.config["MANAGER"]
        record = mgr.get_annotation(name)
        if record is None:
            return jsonify({"error": "Not found"}), 404
        return jsonify(record)

    @app.delete("/api/annotations/<name>")
    def api_annotation_delete(name: str):
        """Delete an image and its label."""
        mgr: DatasetManager = app.config["MANAGER"]
        if mgr.delete_annotation(name):
            return jsonify({"message": f"Deleted {name}"})
        return jsonify({"error": "Not found"}), 404



    @app.get("/api/stats")
    def api_stats():
        """Return dataset statistics."""
        mgr: DatasetManager = app.config["MANAGER"]
        return jsonify(mgr.stats())

   

    @app.post("/api/export_yaml")
    def api_export_yaml():
        """Generate data.yaml for YOLO training."""
        mgr: DatasetManager = app.config["MANAGER"]
        content = mgr.export_yaml()
        return jsonify({"message": "data.yaml exported", "content": content})

  

    @app.get("/api/image/<split>/<filename>")
    def api_serve_image(split: str, filename: str):
        """Serve a saved image from the dataset."""
        mgr: DatasetManager = app.config["MANAGER"]
        if split == "train":
            img_dir = mgr.images_train
        elif split == "val":
            img_dir = mgr.images_val
        else:
            return jsonify({"error": "Invalid split"}), 400

        img_path = img_dir / filename
        if not img_path.exists():
            return jsonify({"error": "Image not found"}), 404

        return send_file(str(img_path), mimetype="image/jpeg")

    # -- Trained model management and inference ---------------------------

    @app.get("/api/models")
    def api_models():
        return jsonify({"models": models.list_models(), "approved_directory": "models/"})

    @app.get("/api/model/status")
    def api_model_status():
        return jsonify(models.status())

    @app.post("/api/model/upload")
    def api_model_upload():
        upload = request.files.get("model")
        if upload is None:
            return json_error("No model file provided.")
        try:
            filename = models.save_upload(upload)
        except ModelError as exc:
            return json_error(str(exc))
        return jsonify({"message": "Model uploaded", "filename": filename}), 201

    @app.post("/api/model/load")
    def api_model_load():
        payload = request.get_json(silent=True) or {}
        try:
            status = models.load(str(payload.get("filename", "")), str(payload.get("family", "auto")))
        except ModelError as exc:
            return json_error(str(exc))
        return jsonify(status)

    @app.post("/api/model/predict")
    def api_model_predict():
        payload = request.get_json(silent=True) or {}
        session_id = str(payload.get("session_id", "single-frame"))
        if not session_pattern.fullmatch(session_id):
            return json_error("Invalid validation session ID.")
        try:
            confidence = float(payload.get("confidence", 0.25))
            iou = float(payload.get("iou", 0.45))
        except (TypeError, ValueError):
            return json_error("Confidence and IoU must be numeric.")

        now = time.monotonic()
        with prediction_guard:
            if session_id in prediction_active:
                return json_error("An inference request is already running for this session.", 429)
            if now - prediction_last.get(session_id, 0.0) < 0.18:
                return json_error("Inference requests are arriving too quickly.", 429)
            prediction_active.add(session_id)
            prediction_last[session_id] = now
        try:
            result = models.predict(str(payload.get("image", "")), confidence, iou)
            result["session_id"] = session_id
            result["frame_id"] = payload.get("frame_id")
            return jsonify(result)
        except ModelError as exc:
            status = 409 if str(exc) == "No model is loaded." else 400
            return json_error(str(exc), status)
        finally:
            with prediction_guard:
                prediction_active.discard(session_id)

    return app



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Cube Dataset Collection and Annotation web app."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).parent),
        help="Base directory for the dataset (default: cube_counting/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(dataset_dir=args.dataset_dir)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(f"\n  Cube Dataset App running at http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
