import threading
import unittest

from camera import CameraDescriptor
from camera_manager import CameraManager


class FakeCamera:
    """A Camera adapter that records what was asked of it."""

    def __init__(self, name="fake"):
        self.name = name
        self.closed = False
        self.applied = {}
        self.previews = 0
        self.stills = 0

    @property
    def properties(self):
        return {"Model": self.name, "PixelArraySize": (640, 480)}

    @property
    def controls(self):
        return {"AnalogueGain": (1.0, 20.0, 1.0)}

    def capture_preview(self):
        self.previews += 1
        return b"preview"

    def capture_still(self):
        self.stills += 1
        return b"still"

    def set_controls(self, controls):
        self.applied.update(controls)

    def close(self):
        self.closed = True


DESCRIPTORS = [
    CameraDescriptor("picamera2:0", "imx296", "picamera2", 0),
    CameraDescriptor("opencv:200:8", "USB webcam", "opencv", 8, 200),
    CameraDescriptor("dummy", "Test camera (dummy)", "dummy"),
]


class FakeFactory:
    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.built = []
        self.cameras = {}

    def __call__(self, descriptor):
        if descriptor.id in self.fail_for:
            raise RuntimeError(f"cannot open {descriptor.id}")
        camera = FakeCamera(descriptor.label)
        self.built.append(descriptor.id)
        self.cameras[descriptor.id] = camera
        return camera


def make_manager(fail_for=()):
    factory = FakeFactory(fail_for)
    manager = CameraManager(descriptors=DESCRIPTORS, factory=factory)
    return manager, factory


class TestCameraManager(unittest.TestCase):
    def test_opens_the_first_real_camera_by_default(self):
        manager, factory = make_manager()
        self.assertEqual(manager.get_current_camera_id(), "picamera2:0")
        self.assertEqual(factory.built, ["picamera2:0"])

    def test_available_cameras_lists_descriptors(self):
        manager, _ = make_manager()
        self.assertEqual(
            [d.id for d in manager.available_cameras()],
            [d.id for d in DESCRIPTORS],
        )

    def test_capture_and_controls_delegate(self):
        manager, factory = make_manager()
        self.assertEqual(manager.capture_preview(), b"preview")
        self.assertEqual(manager.capture_still(), b"still")
        manager.set_controls({"AnalogueGain": 3})
        camera = factory.cameras["picamera2:0"]
        self.assertEqual(camera.previews, 1)
        self.assertEqual(camera.stills, 1)
        self.assertEqual(camera.applied, {"AnalogueGain": 3})

    def test_properties_and_controls_delegate(self):
        manager, _ = make_manager()
        self.assertEqual(manager.properties["Model"], "imx296")
        self.assertEqual(manager.controls["AnalogueGain"], (1.0, 20.0, 1.0))

    def test_switch_builds_new_and_closes_old(self):
        manager, factory = make_manager()
        old = factory.cameras["picamera2:0"]
        descriptor = manager.set_camera("opencv:200:8")
        self.assertEqual(descriptor.label, "USB webcam")
        self.assertEqual(manager.get_current_camera_id(), "opencv:200:8")
        self.assertTrue(old.closed)
        self.assertFalse(factory.cameras["opencv:200:8"].closed)

    def test_switch_to_current_is_a_noop(self):
        manager, factory = make_manager()
        manager.set_camera("picamera2:0")
        self.assertEqual(factory.built, ["picamera2:0"])
        self.assertFalse(factory.cameras["picamera2:0"].closed)

    def test_switch_to_unknown_raises(self):
        manager, _ = make_manager()
        with self.assertRaises(ValueError):
            manager.set_camera("nope")

    def test_failed_switch_keeps_the_working_camera(self):
        manager, factory = make_manager(fail_for=["opencv:200:8"])
        old = factory.cameras["picamera2:0"]
        with self.assertRaises(RuntimeError):
            manager.set_camera("opencv:200:8")
        self.assertEqual(manager.get_current_camera_id(), "picamera2:0")
        self.assertFalse(old.closed)
        self.assertEqual(manager.capture_preview(), b"preview")

    def test_current_label(self):
        manager, _ = make_manager()
        self.assertEqual(manager.current_label(), "imx296")
        manager.set_camera("dummy")
        self.assertEqual(manager.current_label(), "Test camera (dummy)")

    def test_close_closes_the_active_camera(self):
        manager, factory = make_manager()
        manager.close()
        self.assertTrue(factory.cameras["picamera2:0"].closed)

    def test_capture_during_a_switch_does_not_raise(self):
        manager, _ = make_manager()
        errors = []

        def capture():
            try:
                for _ in range(50):
                    manager.capture_preview()
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        thread = threading.Thread(target=capture)
        thread.start()
        manager.set_camera("opencv:200:8")
        manager.set_camera("dummy")
        thread.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
