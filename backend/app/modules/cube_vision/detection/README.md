# Cube Annotator and Live Model Validation

This Flask application preserves the original camera-capture, OBB annotation, dataset browser, statistics, deletion, label generation, and `data.yaml` export workflow. It also provides a separate **Live Model Validation** view for trusted trained-model loading, live camera inference, timed cube counting, and expected-answer checking.

## What was found in the original project

- The dataset is an Ultralytics-style OBB dataset with one class: `0: cube`.
- Two annotated images are included (`dataset/images/train/cube_0001.jpg` and `dataset/images/val/cube_0002.jpg`).
- No `.pt`, `.pth`, `.onnx`, `.tflite`, or `.engine` model exists in the project.
- No training script, notebook, run directory, console log, export, or recorded Python/PyTorch/Ultralytics version exists.
- Consequently, the actual trained model family, task, input size, classes, confidence/IoU settings, and original environment cannot be identified until the checkpoint is added. The application never substitutes a pretrained model or fake prediction.

## Installation and launch

Python 3.10–3.12 is recommended. Create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python app.py --host 127.0.0.1 --port 5002
```

Open <http://127.0.0.1:5002>. Browsers permit camera access on localhost. On macOS, also grant camera permission to the browser in System Settings if prompted.

Run automated checks with:

```bash
pytest -q
python -m py_compile app.py dataset_manager.py model_manager.py
node --check static/validation_logic.js
node --check static/script.js
```

## Loading a model

Use only a checkpoint you trust. PyTorch `.pt` files may deserialize Python objects and can execute code while loading.

1. Copy a `.pt` or `.onnx` file into `models/`, or upload it from **Live Model Validation**.
2. Select the filename and model family. Start with **Auto detect**.
3. Press **Load Model**. The status panel reports the detected family, task, box type, classes, input size, and device—or a safe compatibility error.

Uploads are filename-sanitized, restricted to the project `models/` directory, written atomically, limited to 512 MB, and never overwrite an existing file silently. Browser-provided filesystem paths are not accepted.

### Supported adapters

- **Ultralytics `.pt`:** checkpoints accepted by the installed Ultralytics `YOLO` loader, including its normal detection and OBB result formats. The actual `model.names` and `model.task` are used.
- **ONNX `.onnx`:** common static/dynamic NCHW YOLO detection exports with axis-aligned output. Class names are read from ONNX metadata when present.
- **Original YOLOv5/YOLOv7 `.pt`:** not claimed as supported in this repository. No matching legacy training repository or model was present for validation. The family choices return a clear error; first try Auto detect because some later-compatible checkpoints load through Ultralytics. Export legacy checkpoints using their original repository rather than an unverified command.

Device preference is CUDA, Apple MPS, then CPU. Ultralytics inference automatically retries on CPU if MPS execution fails and reports the fallback. ONNX Runtime chooses CUDA, CoreML, then CPU from installed providers.

## Timed validation

1. Load the trained model and start the validation camera.
2. Enter the expected cube count.
3. Choose 5, 10, 15, 30, or 60 seconds, or a custom whole-number duration from 3–300 seconds.
4. Optionally adjust confidence and IoU thresholds.
5. Press **Start Timed Validation**.

Frames are sampled at roughly 2–3 requests per second, with no overlapping request in a validation session. The countdown is computed from monotonic timestamps, not decremented counters. The expected answer, duration, and model controls are locked while running.

The last five counts form the stabilization window. A stable result requires at least three matching counts. `Correct` means the stable count exactly equals the expected count; a differing stable count is `Incorrect`; otherwise it is `Uncertain`. The model output is never adjusted to match the answer.

At timeout, new requests stop. An in-flight request may finish, after which its captured JPEG, detections, counts, and metadata become the frozen final result. A generation token, session ID, and frame ID prevent old responses from changing a new, stopped, or completed session. Manual stop is labeled `Stopped`. If no request completed, the result is `No valid result` rather than a fabricated count.

**Save Final Frame** downloads the annotated PNG only when clicked. **Run New Validation** clears the result/history and creates a fresh session while retaining the loaded model and available camera.

## API

- `GET /api/models` — allowed model files
- `GET /api/model/status` — current model and device information
- `POST /api/model/upload` — multipart field `model`
- `POST /api/model/load` — JSON `{ "filename": "best.pt", "family": "auto" }`
- `POST /api/model/predict` — base64 image, thresholds, session ID, and frame ID

Existing dataset APIs remain unchanged: `/api/save`, `/api/annotations`, `/api/stats`, `/api/export_yaml`, and `/api/image/...`.

## Limits and manual verification

- No physical camera can be tested from a headless automated environment. Test camera permission, framing, timing, and MacBook M1 performance locally through the steps above.
- Performance depends on checkpoint size, input size, backend, camera resolution, and MPS operator support.
- ONNX exports vary. Unsupported output shapes fail explicitly; OBB ONNX output is not claimed as supported.
- Once the real checkpoint is placed in `models/`, load it in the UI and test a saved dataset image or live frame. Its exact family/task/classes/input size will then appear in status. No real-model result is available while the model file is absent.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries and inference flow.

