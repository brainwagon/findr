"""Camera adapters.

`open_camera` probes for picamera2 and falls back to the dummy camera on a
non-Pi machine. `Camera` is the interface the findr app relies on, so the
adapters and test fakes all satisfy it.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

STILL_CONFIGURATION = {
    "main": {"size": (1456, 1088), "format": "RGB888"},
    "lores": {"size": (640, 480), "format": "YUV420"},
}

INITIAL_CONTROLS = {
    "AnalogueGain": 1.0,
    "ExposureTime": 10000,
    "Brightness": 0.0,
    "Contrast": 1.0,
    "Sharpness": 1.0,
    "ExposureValue": 0.0,
}


class Camera(Protocol):
    """The camera interface the findr app relies on."""

    @property
    def camera_properties(self):
        ...

    @property
    def camera_controls(self):
        ...

    def create_still_configuration(self, **kwargs):
        ...

    def configure(self, config):
        ...

    def start(self):
        ...

    def close(self):
        ...

    def capture_file(self, buffer, name=None, format="jpeg"):
        ...

    def set_controls(self, controls):
        ...


def open_camera():
    """Open, configure and start the camera, falling back to the dummy."""
    camera = _probe_camera()
    config = camera.create_still_configuration(**STILL_CONFIGURATION)
    camera.configure(config)
    camera.start()
    _set_initial_controls(camera)
    return camera


def _probe_camera():
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        # Trigger an internal check to see if libcamera is actually available
        _ = camera.camera_properties
        print("Picamera2 initialized successfully.")
        return camera
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Picamera2 or libcamera not found ({e}). Falling back to dummy camera.")
    except Exception as e:
        print(f"Unexpected error initializing Picamera2: {e}. Falling back to dummy camera.")
    from camera_dummy import Picamera2
    return Picamera2()


def _set_initial_controls(camera):
    available = camera.camera_controls
    safe = {k: v for k, v in INITIAL_CONTROLS.items() if k in available}
    if safe:
        camera.set_controls(safe)
