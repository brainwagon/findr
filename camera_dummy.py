"""A dummy camera for development without hardware.

Implements the `Camera` interface directly so the app can run, and the tests
can exercise the pipeline, on a machine with no camera at all.
"""

import io
import time

from PIL import Image, ImageDraw, ImageFont

from camera import clamp_controls

DUMMY_CONTROLS = {
    "AnalogueGain": (1.0, 251.1886444091797, 1.0),
    "ExposureTime": (29, 15534385, 20000),
    "Brightness": (-1.0, 1.0, 0.0),
    "Contrast": (0.0, 32.0, 1.0),
    "Sharpness": (0.0, 16.0, 1.0),
    "ExposureValue": (-8.0, 8.0, 0.0),
    "AeExposureMode": (0, 3, 0),
}


class DummyCamera:
    """A Camera that renders a timestamped placeholder frame."""

    def __init__(self):
        self.font = ImageFont.load_default()
        self._controls = {}

    @property
    def properties(self):
        return {"Model": "dummy", "PixelArraySize": (640, 480)}

    @property
    def controls(self):
        return dict(DUMMY_CONTROLS)

    def capture_preview(self):
        return self._render()

    def capture_still(self):
        return self._render()

    def set_controls(self, controls):
        self._controls.update(clamp_controls(controls, self.controls))

    def close(self):
        pass

    def _render(self):
        img = Image.new("RGB", (640, 480), color="darkgrey")
        d = ImageDraw.Draw(img)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        controls_str = "\n".join(
            f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in self._controls.items()
        )
        d.text(
            (10, 10),
            f"Dummy Camera Feed\n{timestamp}\n\n{controls_str}",
            fill=(255, 255, 0),
            font=self.font,
        )
        buffer = io.BytesIO()
        img.save(buffer, format="jpeg")
        return buffer.getvalue()
