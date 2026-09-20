"""Direct V4L2 control access for USB cameras on Linux.

OpenCV's `CAP_PROP_EXPOSURE` is unreliable on some UVC cameras — the Orbbec
webcam used here blacks out and ignores it — so the OpenCV adapter drives
exposure, gain and the image controls through the kernel's V4L2 controls
instead. V4L2 lets the device be opened a second time for control while OpenCV
holds it for capture.

Linux only; on other platforms the OpenCV adapter falls back to `CAP_PROP_*`.
"""

import fcntl
import os
import struct

# Control ids (linux/v4l2-controls.h).
BRIGHTNESS = 0x00980900
CONTRAST = 0x00980901
SATURATION = 0x00980902
GAIN = 0x00980913
SHARPNESS = 0x0098091B
EXPOSURE_AUTO = 0x009A0901
EXPOSURE_ABSOLUTE = 0x009A0902

# `exposure_time_absolute` is in 100us units; `EXPOSURE_AUTO` is a menu.
EXPOSURE_UNIT_US = 100
EXPOSURE_MODE_AUTO = 0
EXPOSURE_MODE_MANUAL = 1

# ioctl numbers: _IOWR('V', n, struct)
_VIDIOC_G_CTRL = 0xC008561B
_VIDIOC_S_CTRL = 0xC008561C
_VIDIOC_QUERYCTRL = 0xC0445624

_QUERYCTRL_SIZE = 68  # struct v4l2_queryctrl


class V4L2Controls:
    """Read and write a video device's controls through V4L2 ioctls."""

    def __init__(self, device):
        self._fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)

    def get(self, control_id):
        buffer = bytearray(struct.pack("Ii", control_id, 0))
        fcntl.ioctl(self._fd, _VIDIOC_G_CTRL, buffer)
        return struct.unpack("Ii", buffer)[1]

    def set(self, control_id, value):
        fcntl.ioctl(
            self._fd, _VIDIOC_S_CTRL,
            struct.pack("Ii", control_id, int(value)),
        )

    def query(self, control_id):
        """Return `(minimum, maximum, step, default)` for a control."""
        buffer = bytearray(_QUERYCTRL_SIZE)
        struct.pack_into("I", buffer, 0, control_id)
        fcntl.ioctl(self._fd, _VIDIOC_QUERYCTRL, buffer)
        return struct.unpack_from("iiii", buffer, 40)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
