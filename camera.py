"""Camera adapters and selection.

`Camera` is the backend-neutral interface findr relies on: a preview JPEG, a
still JPEG, a description of the device, and a set of adjustable controls.
Concrete adapters live in `camera_picamera2`, `camera_opencv` and
`camera_dummy`; `list_cameras` enumerates what the machine offers and
`open_camera` picks one, falling back to the dummy when hardware is missing.

Keeping the seam free of picamera2 concepts is what lets the same app drive a
Raspberry Pi CSI camera and a USB webcam on a laptop.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class Camera(Protocol):
    """The camera interface the findr app relies on."""

    @property
    def properties(self):
        """Device description: `Model` and `PixelArraySize`."""
        ...

    @property
    def controls(self):
        """Adjustable controls, each a `(min, max, default)` range."""
        ...

    def capture_preview(self) -> bytes:
        """Return the live-preview frame as JPEG bytes."""
        ...

    def capture_still(self) -> bytes:
        """Return a full still frame as JPEG bytes."""
        ...

    def set_controls(self, controls):
        """Apply the given controls, clamping to the device's ranges."""
        ...

    def close(self):
        """Release the device."""
        ...


@dataclass(frozen=True)
class CameraDescriptor:
    """A camera the machine can offer, as the UI and factory see it."""

    id: str
    label: str
    backend: str  # "picamera2" | "opencv" | "dummy"
    index: int = 0
    api: int = 0  # OpenCV backend preference; ignored by the others


def clamp_controls(controls, available):
    """Clamp each value to its advertised range, dropping unknown names.

    `available` maps a control name to `(min, max, ...)`. Values that are not
    comparable to the bounds (e.g. a `ScalerCrop` tuple) pass through unchanged.
    """
    safe = {}
    for name, value in controls.items():
        if name not in available:
            continue
        spec = available[name]
        try:
            lo, hi = spec[0], spec[1]
            safe[name] = min(max(value, lo), hi)
        except (TypeError, ValueError):
            safe[name] = value
    return safe


def list_cameras():
    """The cameras this machine can offer, as descriptors."""
    descriptors = _dedupe(_picamera2_cameras() + _opencv_cameras())
    descriptors.append(
        CameraDescriptor("dummy", "Test camera (dummy)", "dummy")
    )
    return descriptors


def _dedupe(descriptors):
    """Drop cameras two backends both report, keeping the first seen."""
    seen = set()
    unique = []
    for descriptor in descriptors:
        key = descriptor.label.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(descriptor)
    return unique


def _picamera2_cameras():
    """The libcamera CSI cameras picamera2 reports, if it is installed.

    USB webcams are excluded: libcamera exposes them too, but its uvcvideo
    pipeline cannot give the two streams `Picamera2Camera` configures, so the
    OpenCV adapter owns them instead.
    """
    try:
        from picamera2 import Picamera2
        return [
            CameraDescriptor(
                id=f"picamera2:{info['Num']}",
                label=info.get("Model", f"Camera {info['Num']}"),
                backend="picamera2",
                index=info["Num"],
            )
            for info in Picamera2.global_camera_info()
            if "usb" not in str(info.get("Id", "")).lower()
        ]
    except Exception as e:
        logger.info("picamera2 unavailable (%s); no libcamera cameras.", e)
        return []


def _opencv_cameras():
    """The cameras OpenCV can enumerate, if OpenCV is installed."""
    try:
        import cv2
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception as e:
        logger.info("OpenCV camera enumeration unavailable (%s).", e)
        return []

    api = _preferred_opencv_api(cv2)
    try:
        infos = enumerate_cameras(api)
    except Exception as e:
        logger.info("OpenCV enumeration failed (%s); probing all backends.", e)
        try:
            infos = enumerate_cameras(cv2.CAP_ANY)
        except Exception:
            return []

    descriptors = []
    for info in infos:
        # On Linux the V4L2 enumeration also reports platform capture-pipeline
        # nodes (the Pi's rp1-cfe/pispbe) that are not user cameras; they have
        # no USB vendor id, unlike every real webcam.
        if sys.platform.startswith("linux") and getattr(info, "vid", None) is None:
            continue
        descriptors.append(
            CameraDescriptor(
                id=f"opencv:{info.backend}:{info.index}",
                label=info.name or f"OpenCV camera {info.index}",
                backend="opencv",
                index=info.index,
                api=info.backend,
            )
        )
    return descriptors


def _preferred_opencv_api(cv2):
    """The one OpenCV backend to enumerate, so a device is not listed twice."""
    if sys.platform.startswith("linux"):
        return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    if sys.platform == "darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    if sys.platform.startswith("win"):
        return getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)
    return cv2.CAP_ANY


def make_camera(descriptor):
    """Build the Camera described by a CameraDescriptor."""
    if descriptor.backend == "picamera2":
        from camera_picamera2 import Picamera2Camera
        return Picamera2Camera(descriptor.index)
    if descriptor.backend == "opencv":
        from camera_opencv import OpenCvCamera
        return OpenCvCamera(
            descriptor.index, descriptor.api, label=descriptor.label
        )
    if descriptor.backend == "dummy":
        from camera_dummy import DummyCamera
        return DummyCamera()
    raise ValueError(f"Unknown camera backend: {descriptor.backend}")


def open_camera(preferred=None):
    """Open the preferred camera, else the first real one, else the dummy.

    Any failure to open the chosen device falls back to the dummy camera so the
    app still starts on a machine with no hardware.
    """
    descriptors = list_cameras()
    chosen = _choose_descriptor(descriptors, preferred)
    try:
        camera = make_camera(chosen)
        print(f"Camera opened: {chosen.label} ({chosen.id}).")
        return camera
    except Exception as e:
        logger.warning(
            "Could not open %s (%s). Falling back to dummy camera.",
            chosen.label, e,
        )
        from camera_dummy import DummyCamera
        return DummyCamera()


def open_camera_manager(preferred=None):
    """Build a CameraManager over the machine's cameras.

    The preferred camera is opened eagerly; a failure falls back to the dummy
    but leaves every real camera selectable from the UI.
    """
    from camera_manager import CameraManager
    from camera_dummy import DummyCamera

    descriptors = list_cameras()
    chosen = _choose_descriptor(descriptors, preferred)
    try:
        camera = make_camera(chosen)
        print(f"Camera opened: {chosen.label} ({chosen.id}).")
    except Exception as e:
        logger.warning(
            "Could not open %s (%s). Falling back to dummy camera.",
            chosen.label, e,
        )
        camera = DummyCamera()
        chosen = next(d for d in descriptors if d.backend == "dummy")
    return CameraManager(
        descriptors=descriptors, camera=camera, current_id=chosen.id
    )


def _choose_descriptor(descriptors, preferred=None):
    """Pick the preferred descriptor, else the first real one, else the dummy."""
    if preferred is not None:
        chosen = next((d for d in descriptors if d.id == preferred), None)
        if chosen is not None:
            return chosen
    chosen = next((d for d in descriptors if d.backend != "dummy"), None)
    if chosen is None:
        chosen = next((d for d in descriptors if d.backend == "dummy"), None)
    if chosen is None:
        raise RuntimeError("No cameras available")
    return chosen
