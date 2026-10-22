"""macOS camera enumeration and OpenCV capture management.

Design
------
* Enumerates video devices via FFmpeg's AVFoundation backend to get real device
  names (e.g. "FaceTime HD Camera", "USB Camera"), then cross-references with
  OpenCV cv2.CAP_AVFOUNDATION probes to confirm each index actually opens.
* Falls back to plain OpenCV probing when FFmpeg is unavailable.
* Built-in camera heuristics: names containing "facetime", "isight", or
  "apple camera" are tagged as built-in; all others are treated as external.
* Never opens more than one capture at a time — each caller must release().
* All public functions are safe to call without a real camera attached (they
  return empty lists / None instead of raising).

Usage
-----
    from camera_manager import CameraManager
    mgr = CameraManager()
    cameras = mgr.list_cameras()        # list[CameraDevice]
    cap    = mgr.open_camera(index=1)   # cv2.VideoCapture | None
    ok, frame = cap.read()
    mgr.release(cap)
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import cv2

logger = logging.getLogger(__name__)

# Maximum index to probe when enumerating cameras via OpenCV alone.
_MAX_PROBE_INDEX = 10
# Seconds to wait for a VideoCapture to open before giving up.
_OPEN_TIMEOUT_S = 2.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraDevice:
    """Represents one physical (or virtual) video capture device."""
    index: int
    name: str
    is_builtin: bool
    can_open: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    backend: str = "AVFoundation"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "is_builtin": self.is_builtin,
            "can_open": self.can_open,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "backend": self.backend,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BUILTIN_KEYWORDS = ("facetime", "isight", "apple camera", "built-in", "builtin")


def _is_builtin(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _BUILTIN_KEYWORDS)


def _ffmpeg_list_devices() -> list[tuple[int, str]]:
    """Return [(avf_index, device_name), ...] from ffmpeg -list_devices.

    Returns an empty list if ffmpeg is not installed or fails.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=8,
        )
        # ffmpeg writes device list to stderr regardless of exit code
        output = result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

    devices: list[tuple[int, str]] = []
    # Lines look like: [AVFoundation indev @ 0x...] [0] FaceTime HD Camera
    in_video_section = False
    for line in output.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if not in_video_section:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)", line)
        if match:
            idx = int(match.group(1))
            name = match.group(2).strip()
            # Skip screen-capture entries
            if "capture screen" in name.lower():
                continue
            devices.append((idx, name))

    return devices


def _probe_opencv(index: int, backend: int = cv2.CAP_AVFOUNDATION) -> CameraDevice | None:
    """Try to open index with the given backend and read one frame.

    Returns a CameraDevice on success, None on failure.
    Releases the capture before returning.
    """
    try:
        cap = cv2.VideoCapture(index, backend)
    except Exception:
        return None

    if not cap.isOpened():
        cap.release()
        return None

    # Give the driver a moment to initialise
    deadline = time.monotonic() + _OPEN_TIMEOUT_S
    ok = False
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            break
        time.sleep(0.05)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not ok:
        # Device opened but couldn't produce a frame — still report it
        return CameraDevice(
            index=index, name=f"Camera {index}", is_builtin=(index == 0),
            can_open=True, width=w, height=h, fps=fps,
        )

    return CameraDevice(
        index=index, name=f"Camera {index}", is_builtin=(index == 0),
        can_open=True, width=w, height=h, fps=fps,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CameraManager:
    """Enumerate and open macOS cameras via cv2.CAP_AVFOUNDATION."""

    def __init__(self) -> None:
        self._backend = cv2.CAP_AVFOUNDATION

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_cameras(self, *, probe_frames: bool = True) -> list[CameraDevice]:
        """Return all available video capture devices.

        Steps:
        1. Ask FFmpeg for device names (most reliable on macOS).
        2. For each named device, attempt an OpenCV open to confirm it works
           and to read resolution/fps.
        3. If FFmpeg is unavailable, fall back to brute-force OpenCV probing
           for indices 0 … _MAX_PROBE_INDEX.

        ``probe_frames`` controls whether a frame-read is attempted; set
        False for faster enumeration when frame verification is not needed.
        """
        named = _ffmpeg_list_devices()

        if named:
            return self._enumerate_from_names(named, probe_frames=probe_frames)

        # Fallback: brute-force OpenCV probe
        logger.warning(
            "ffmpeg not found or returned no devices — falling back to "
            "OpenCV index probe (indices 0–%d).", _MAX_PROBE_INDEX
        )
        return self._enumerate_by_probe(probe_frames=probe_frames)

    def _enumerate_from_names(
        self,
        named: list[tuple[int, str]],
        *,
        probe_frames: bool,
    ) -> list[CameraDevice]:
        devices: list[CameraDevice] = []
        for avf_index, name in named:
            device = CameraDevice(
                index=avf_index,
                name=name,
                is_builtin=_is_builtin(name),
            )
            if probe_frames:
                probed = _probe_opencv(avf_index, self._backend)
                if probed is not None:
                    device.can_open = probed.can_open
                    device.width = probed.width
                    device.height = probed.height
                    device.fps = probed.fps
            devices.append(device)
        return devices

    def _enumerate_by_probe(self, *, probe_frames: bool) -> list[CameraDevice]:
        devices: list[CameraDevice] = []
        for idx in range(_MAX_PROBE_INDEX):
            probed = _probe_opencv(idx, self._backend)
            if probed is None:
                continue
            probed.name = f"Camera {idx}" + (" (Built-in)" if idx == 0 else "")
            probed.is_builtin = idx == 0
            devices.append(probed)
        return devices

    # ------------------------------------------------------------------
    # Open / release
    # ------------------------------------------------------------------

    def open_camera(
        self,
        index: int,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> cv2.VideoCapture | None:
        """Open the camera at ``index`` using AVFoundation.

        Returns a ready ``cv2.VideoCapture`` on success, or ``None`` if the
        device cannot be opened or does not produce frames.

        The caller is responsible for calling ``release()`` when done.
        """
        try:
            cap = cv2.VideoCapture(index, self._backend)
        except Exception as exc:
            logger.error("VideoCapture(%d) raised: %s", index, exc)
            return None

        if not cap.isOpened():
            logger.warning("VideoCapture(%d) did not open.", index)
            cap.release()
            return None

        # Apply optional resolution / fps hints *before* reading frames
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)

        # Confirm we can actually read a frame
        deadline = time.monotonic() + _OPEN_TIMEOUT_S
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap
            time.sleep(0.05)

        logger.error(
            "VideoCapture(%d) opened but produced no frames within %.1fs. "
            "Check macOS camera permissions for this process.",
            index, _OPEN_TIMEOUT_S,
        )
        cap.release()
        return None

    @staticmethod
    def release(cap: cv2.VideoCapture | None) -> None:
        """Safely release a VideoCapture (accepts None)."""
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

