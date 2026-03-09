import io
import time
from PIL import Image, ImageDraw, ImageFont

class Picamera2:
    """A dummy class that mimics the picamera2 library for development on non-Raspberry Pi machines."""

    def __init__(self):
        self.font = ImageFont.load_default()
        self._controls = {}

    @property
    def controls(self):
        class Controls:
            def __init__(self, controls_dict):
                for key, value in controls_dict.items():
                    setattr(self, key, value)

        return Controls(self._controls)

    @property
    def camera_properties(self):
        """Returns a dictionary of dummy camera properties."""
        return {
            "Model": "dummy",
            "PixelArraySize": (640, 480)
        }

        return {
            "AeEnable": (False, True, True),
            "AnalogueGain": (1.0, 251.1886444091797, 1.0),
            "ExposureTime": (29, 15534385, 20000),
            "Brightness": (-1.0, 1.0, 0.0),
            "Contrast": (0.0, 32.0, 1.0),
            "Sharpness": (0.0, 16.0, 1.0),
            "ExposureValue": (-8.0, 8.0, 0.0),
            "AeExposureMode": (0, 3, 0)
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def configure(self, config):
        pass

    def create_video_configuration(self, main={"size": (640, 480)}):
        return {"main": main}

    def start(self):
        pass

    def set_controls(self, controls):
        """Prints the controls for debugging."""
        safe_controls = {k: v for k, v in controls.items() if k in self.camera_controls}
        self._controls.update(safe_controls)
        print("Dummy camera controls set:", self._controls)

    def capture_file(self, buffer, format='jpeg'):
        """Generates a dummy image with a timestamp and writes it to the buffer."""
        img = Image.new('RGB', (640, 480), color = 'darkgrey')
        d = ImageDraw.Draw(img)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        controls_str = "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in self._controls.items()])
        d.text((10,10), f"Dummy Camera Feed\n{timestamp}\n\n{controls_str}", fill=(255,255,0), font=self.font)
        img.save(buffer, format=format)
