"""The libcamera/picamera2 camera adapter.

Used on a Raspberry Pi, where picamera2 exposes the CSI camera through
libcamera. Two streams are configured: a low-resolution `lores` for the live
preview (and the plate solve that consumes it) and a full-resolution `main`
for stills.
"""

import io

from camera import clamp_controls

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


class Picamera2Camera:
    """A Camera backed by a libcamera camera via picamera2."""

    def __init__(self, index=0, picamera2=None):
        if picamera2 is None:
            from picamera2 import Picamera2 as picamera2
        self._camera = picamera2(index)
        config = self._camera.create_still_configuration(**STILL_CONFIGURATION)
        self._camera.configure(config)
        self._camera.start()
        self.set_controls(INITIAL_CONTROLS)

    @property
    def properties(self):
        return self._camera.camera_properties

    @property
    def controls(self):
        return self._camera.camera_controls

    def capture_preview(self):
        return self._capture("lores")

    def capture_still(self):
        return self._capture("main")

    def set_controls(self, controls):
        safe = clamp_controls(controls, self._camera.camera_controls)
        if safe:
            self._camera.set_controls(safe)

    def close(self):
        self._camera.close()

    def _capture(self, name):
        buffer = io.BytesIO()
        self._camera.capture_file(buffer, name=name, format="jpeg")
        return buffer.getvalue()
