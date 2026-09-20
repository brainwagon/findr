"""The camera manager.

Holds the active Camera and the cameras the machine offers, and is the single
place that owns the camera lifecycle. Captures and control changes delegate to
the active camera under a lock, and `set_camera` swaps it live: the new camera
is built before the old one is closed, so a failed switch leaves the working
camera in place. The capture thread and the web layer never touch a Camera
directly, so a switch cannot close one mid-capture.
"""

import logging
import threading

from camera import list_cameras, make_camera

logger = logging.getLogger(__name__)


class CameraManager:
    """Selects the active Camera and delegates to it."""

    def __init__(self, descriptors=None, camera=None, current_id=None,
                 factory=None):
        self._lock = threading.RLock()
        self._switch_lock = threading.Lock()
        self._factory = factory or make_camera
        self._descriptors = (
            list(descriptors) if descriptors is not None else list_cameras()
        )
        self._camera = camera
        self._current_id = current_id
        if self._camera is None:
            descriptor = self._default_descriptor()
            self._camera = self._factory(descriptor)
            self._current_id = descriptor.id

    def available_cameras(self):
        """The cameras the machine offers, as descriptors."""
        return list(self._descriptors)

    def get_current_camera_id(self):
        """The id of the active camera."""
        return self._current_id

    def current_label(self):
        """The label of the active camera, or its id when unknown."""
        for descriptor in self._descriptors:
            if descriptor.id == self._current_id:
                return descriptor.label
        return self._current_id or ""

    def get_current_camera(self):
        """The active Camera instance."""
        return self._camera

    @property
    def properties(self):
        with self._lock:
            return self._camera.properties

    @property
    def controls(self):
        with self._lock:
            return self._camera.controls

    def capture_preview(self):
        with self._lock:
            return self._camera.capture_preview()

    def capture_still(self):
        with self._lock:
            return self._camera.capture_still()

    def set_controls(self, controls):
        with self._lock:
            self._camera.set_controls(controls)

    def set_camera(self, camera_id):
        """Switch to the camera with this id, closing the previous one."""
        with self._switch_lock:
            descriptor = next(
                (d for d in self._descriptors if d.id == camera_id), None
            )
            if descriptor is None:
                raise ValueError(f"Unknown camera: {camera_id}")
            if camera_id == self._current_id:
                return descriptor

            # Build first: if the device cannot be opened the old camera, and
            # the captures using it, are left untouched.
            new_camera = self._factory(descriptor)
            with self._lock:
                previous = self._camera
                self._camera = new_camera
                self._current_id = descriptor.id
            _close_quietly(previous)
            logger.info("Switched camera to %s (%s)", descriptor.label, camera_id)
            return descriptor

    def close(self):
        with self._lock:
            _close_quietly(self._camera)

    def _default_descriptor(self):
        descriptor = next(
            (d for d in self._descriptors if d.backend != "dummy"), None
        )
        if descriptor is None:
            descriptor = next(
                (d for d in self._descriptors if d.backend == "dummy"), None
            )
        if descriptor is None:
            raise RuntimeError("No cameras available")
        return descriptor


def _close_quietly(camera):
    if camera is None:
        return
    try:
        camera.close()
    except Exception as e:
        logger.warning("Error closing camera: %s", e)
