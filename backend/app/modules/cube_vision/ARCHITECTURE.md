# Architecture

## Components

- `app.py` creates the Flask app, retains the annotation endpoints, validates model API requests, enforces upload size, and limits per-session inference concurrency/rate.
- `dataset_manager.py` remains responsible for images, OBB labels, metadata, statistics, deletion, and YAML export.
- `model_manager.py` owns the approved model directory, model adapters, device choice, secure path resolution, image decoding/limits, thread-safe single model instance, normalized detections, and count aggregation.
- `static/script.js` owns camera streams, frame capture, inference scheduling, overlays, timestamp countdown, frozen final output, and downloads.
- `static/validation_logic.js` contains browser-independent duration, stabilization, answer, final-outcome, and stale-response rules.

## Adapter contract

Every active adapter exposes `load`, `predict`, `get_class_names`, `get_model_info`, and `is_obb`. Predictions are normalized to either:

```json
{"class_id": 0, "class_name": "cube", "confidence": 0.9, "box_type": "xyxy", "box": [10, 20, 30, 40]}
```

or:

```json
{"class_id": 0, "class_name": "cube", "confidence": 0.9, "box_type": "obb", "points": [[10, 20], [30, 20], [30, 40], [10, 40]]}
```

Routes do not depend on Ultralytics result objects.

## Timed request lifecycle

1. The browser creates a new generation and session ID and captures a specific frame.
2. It posts that JPEG with a unique frame ID. A new request is never scheduled while the prior request is active.
3. The backend validates thresholds/image limits and serializes model access with a lock.
4. The response echoes session and frame IDs. The browser accepts it only if all identifiers still match the current generation.
5. The accepted image data URL and its exact response are stored together.
6. When monotonic elapsed time reaches the requested duration, scheduling stops. Any request already in flight is allowed to settle.
7. The most recent accepted image/response pair is drawn into the final canvas, and the latest stable count determines the answer status.

The camera stream stays available after normal completion. A manual stop invalidates the generation immediately, so its delayed response cannot mutate the frozen result.

## Security boundaries

Models are selected by basename inside resolved `models/`; traversal and unsupported extensions are rejected. Uploads use safe names and atomic replacement from a temporary file, but replacement is refused if the destination exists. Images use strict base64 decoding, a 12 MB encoded-image payload limit after decoding, and a 20-megapixel decoded size limit. Browser errors contain no stack traces.
