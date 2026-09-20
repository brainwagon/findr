"""The OpenCV camera adapter.

Drives a USB webcam (or a laptop's built-in camera) through
`cv2.VideoCapture`, which is portable across Linux (V4L2), macOS
(AVFoundation) and Windows (Media Foundation / DirectShow). A UVC camera
cannot give the two streams the CSI adapter uses, so a single 640x480 stream
serves both the preview and the still.

Controls are the point of this adapter for an astronomical camera: exposure
and gain must be settable by hand. On Linux they are driven through the
kernel's V4L2 controls (see `camera_v4l2`), because OpenCV's
`CAP_PROP_EXPOSURE` is unreliable on some UVC cameras — the Orbbec webcam used
here blacks out and ignores it. Elsewhere the adapter falls back to
`CAP_PROP_*`, which is best-effort.

Setting exposure or gain takes the camera off its auto exposure (V4L2 manual
mode), so the values the UI sends are what the sensor uses.
"""

import logging
import sys

logger = logging.getLogger(__name__)

PREVIEW_SIZE = (640, 480)

_UNSET = object()

# The gain the UI's 1..20 maps onto the device's full range.
_GAIN_CANONICAL = (1.0, 20.0)

# canonical -> (CAP_PROP name, canonical (min, max), device (min, max)); the
# non-Linux fallback.
_CAP_PROP_CONTROLS = {
    "AnalogueGain": ("CAP_PROP_GAIN", _GAIN_CANONICAL, (0.0, 100.0)),
    "ExposureTime": ("CAP_PROP_EXPOSURE", (0.0, 1000000.0), (0.0, 10000.0)),
    "Brightness": ("CAP_PROP_BRIGHTNESS", (-1.0, 1.0), (0.0, 255.0)),
    "Contrast": ("CAP_PROP_CONTRAST", (0.0, 2.0), (0.0, 255.0)),
    "Sharpness": ("CAP_PROP_SHARPNESS", (0.0, 1.0), (0.0, 255.0)),
}

# canonical -> (camera_v4l2 attribute, canonical (min, max), device (min, max))
_V4L2_CONTROLS = {
    "Brightness": ("BRIGHTNESS", (-1.0, 1.0), (0.0, 255.0)),
    "Contrast": ("CONTRAST", (0.0, 2.0), (0.0, 255.0)),
    "Sharpness": ("SHARPNESS", (0.0, 1.0), (0.0, 255.0)),
}


class OpenCvCamera:
    """A Camera backed by `cv2.VideoCapture`."""

    def __init__(self, index=0, api=0, cv2=None, label=None, v4l2=_UNSET):
        if cv2 is None:
            import cv2
        self._cv2 = cv2
        self._label = label or f"OpenCV camera {index}"
        self._cap = cv2.VideoCapture(index, api)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera {index}")
        self._v4l2 = _open_v4l2(index) if v4l2 is _UNSET else v4l2
        self._gain_max = 100
        self._exposure_max = 10000
        self._gain = 0
        self._exposure = 0
        self._read_v4l2_ranges()
        self._configure()

    @property
    def properties(self):
        cv2 = self._cv2
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or PREVIEW_SIZE[0]
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or PREVIEW_SIZE[1]
        return {"Model": self._label, "PixelArraySize": (width, height)}

    @property
    def controls(self):
        if self._v4l2 is not None:
            return {
                "AnalogueGain": (_GAIN_CANONICAL[0], _GAIN_CANONICAL[1], 1.0),
                "ExposureTime": (0.0, float(self._exposure_max * 100), 0.0),
                "Brightness": (-1.0, 1.0, 0.0),
                "Contrast": (0.0, 2.0, 1.0),
                "Sharpness": (0.0, 1.0, 0.0),
            }
        return {
            name: (cmin, cmax, cmin)
            for name, (_, (cmin, cmax), _) in _CAP_PROP_CONTROLS.items()
        }

    def capture_preview(self):
        return self._read_jpeg()

    def capture_still(self):
        return self._read_jpeg()

    def set_controls(self, controls):
        if self._v4l2 is not None:
            self._set_controls_v4l2(controls)
        else:
            self._set_controls_cap_prop(controls)

    def close(self):
        if self._v4l2 is not None:
            try:
                self._v4l2.close()
            except Exception as e:
                logger.warning("Error closing V4L2 controls: %s", e)
        self._cap.release()

    def _set_controls_v4l2(self, controls):
        v4l2 = _v4l2_module()
        if "AnalogueGain" in controls:
            self._gain = _scale(
                controls["AnalogueGain"], _GAIN_CANONICAL, (0.0, self._gain_max)
            )
        if "ExposureTime" in controls:
            units = int(round(float(controls["ExposureTime"]) / v4l2.EXPOSURE_UNIT_US))
            self._exposure = _clamp(units, 0, self._exposure_max)

        # Setting either control takes the camera off auto exposure. Both are
        # written together so manual mode reflects the last known values rather
        # than dropping to the sensor's minimum.
        if "ExposureTime" in controls or "AnalogueGain" in controls:
            self._v4l2.set(v4l2.EXPOSURE_AUTO, v4l2.EXPOSURE_MODE_MANUAL)
            self._v4l2.set(v4l2.EXPOSURE_ABSOLUTE, self._exposure)
            self._write_gain(self._gain)

        for name, value in controls.items():
            if name in _V4L2_CONTROLS:
                attr, canonical, device = _V4L2_CONTROLS[name]
                self._v4l2.set(
                    getattr(v4l2, attr), _scale(value, canonical, device)
                )

    def _write_gain(self, value):
        """Write gain, nudging first because this driver ignores a same value."""
        v4l2 = _v4l2_module()
        value = int(round(value))
        try:
            if value > 0 and self._v4l2.get(v4l2.GAIN) == value:
                self._v4l2.set(v4l2.GAIN, value - 1)
        except Exception:
            pass
        self._v4l2.set(v4l2.GAIN, value)

    def _set_controls_cap_prop(self, controls):
        for name, value in controls.items():
            spec = _CAP_PROP_CONTROLS.get(name)
            if spec is None:
                continue
            prop_name, canonical, device = spec
            self._cap.set(
                getattr(self._cv2, prop_name), _scale(value, canonical, device)
            )

    def _read_v4l2_ranges(self):
        if self._v4l2 is None:
            return
        v4l2 = _v4l2_module()
        try:
            self._gain_max = self._v4l2.query(v4l2.GAIN)[1]
            self._exposure_max = self._v4l2.query(v4l2.EXPOSURE_ABSOLUTE)[1]
            # Remember what auto exposure chose, so the first manual setting
            # starts from a usable gain rather than the sensor minimum.
            self._gain = self._v4l2.get(v4l2.GAIN)
            self._exposure = self._v4l2.get(v4l2.EXPOSURE_ABSOLUTE)
        except Exception as e:
            logger.info("V4L2 control ranges unavailable (%s); using CAP_PROP.", e)
            self._v4l2 = None

    def _configure(self):
        cv2 = self._cv2
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_SIZE[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_SIZE[1])
        # Deliberately leave the pixel format at the device default: forcing
        # MJPG makes some UVC cameras hand back the same frame forever.

    def _read_jpeg(self):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("OpenCV camera returned no frame")
        ok, buffer = self._cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("OpenCV could not encode the frame")
        return buffer.tobytes()


def _open_v4l2(index):
    """Open a control handle on /dev/videoN, or None if unavailable."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        from camera_v4l2 import V4L2Controls
        return V4L2Controls(f"/dev/video{index}")
    except Exception as e:
        logger.info("V4L2 controls unavailable for /dev/video%d: %s", index, e)
        return None


def _v4l2_module():
    import camera_v4l2
    return camera_v4l2


def _clamp(value, low, high):
    return min(max(value, low), high)


def _scale(value, canonical, device):
    cmin, cmax = canonical
    dmin, dmax = device
    value = _clamp(float(value), cmin, cmax)
    span = cmax - cmin
    return dmin if span == 0 else dmin + (value - cmin) / span * (dmax - dmin)
