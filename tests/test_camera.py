import io
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import camera as camera_module
from camera import (
    CameraDescriptor,
    clamp_controls,
    list_cameras,
    make_camera,
    open_camera,
)
from camera_dummy import DummyCamera
import camera_v4l2
from camera_opencv import OpenCvCamera
from camera_picamera2 import Picamera2Camera, STILL_CONFIGURATION


def jpeg_size(data):
    return Image.open(io.BytesIO(data)).size


class FakePicamera2:
    """A picamera2.Picamera2 stand-in recording what the adapter asks of it."""

    def __init__(self, index=0):
        self.index = index
        self.configured = None
        self.started = False
        self.closed = False
        self.captures = []
        self.control_calls = []
        self._controls = {
            "AnalogueGain": (1.0, 4.0, 1.0),
            "ExposureTime": (0, 650000, 10000),
            "Brightness": (-1.0, 0.9921875, 0.0),
            "Sharpness": (0.0, 16.0, 1.0),
        }

    @property
    def camera_properties(self):
        return {"Model": "fake-picam", "PixelArraySize": (2560, 1440)}

    @property
    def camera_controls(self):
        return dict(self._controls)

    def create_still_configuration(self, **kwargs):
        return {"kwargs": kwargs}

    def configure(self, config):
        self.configured = config

    def start(self):
        self.started = True

    def capture_file(self, buffer, name=None, format="jpeg"):
        self.captures.append((name, format))
        Image.new("RGB", (8, 8), "grey").save(buffer, format=format)

    def set_controls(self, controls):
        self.control_calls.append(dict(controls))

    def close(self):
        self.closed = True


class TestClampControls(unittest.TestCase):
    def test_clamps_to_range(self):
        safe = clamp_controls(
            {"AnalogueGain": 999, "ExposureTime": -5},
            {"AnalogueGain": (1.0, 4.0, 1.0), "ExposureTime": (0, 650000, 0)},
        )
        self.assertEqual(safe, {"AnalogueGain": 4.0, "ExposureTime": 0})

    def test_drops_unknown_names(self):
        safe = clamp_controls({"Nope": 1}, {"AnalogueGain": (1.0, 4.0, 1.0)})
        self.assertEqual(safe, {})

    def test_passes_uncomparable_values_through(self):
        safe = clamp_controls(
            {"ScalerCrop": [0, 0, 640, 480]}, {"ScalerCrop": (0, 0, 0)}
        )
        self.assertEqual(safe, {"ScalerCrop": [0, 0, 640, 480]})


class TestDummyCamera(unittest.TestCase):
    def test_properties(self):
        camera = DummyCamera()
        self.assertEqual(camera.properties["Model"], "dummy")
        self.assertEqual(camera.properties["PixelArraySize"], (640, 480))

    def test_capture_preview_and_still_are_jpegs(self):
        camera = DummyCamera()
        self.assertEqual(jpeg_size(camera.capture_preview()), (640, 480))
        self.assertEqual(jpeg_size(camera.capture_still()), (640, 480))

    def test_set_controls_clamps_and_ignores_unknown(self):
        camera = DummyCamera()
        camera.set_controls(
            {"AnalogueGain": 999, "ExposureTime": 0, "Nope": 1}
        )
        self.assertEqual(
            camera._controls["AnalogueGain"], 251.1886444091797
        )
        self.assertEqual(camera._controls["ExposureTime"], 29)
        self.assertNotIn("Nope", camera._controls)

    def test_close_is_a_noop(self):
        DummyCamera().close()


class TestPicamera2Camera(unittest.TestCase):
    def test_opens_configures_and_starts(self):
        fake = FakePicamera2()
        camera = Picamera2Camera(0, picamera2=lambda index: fake)
        self.assertTrue(fake.started)
        self.assertEqual(
            fake.configured, {"kwargs": STILL_CONFIGURATION}
        )

    def test_preview_and_still_use_the_two_streams(self):
        fake = FakePicamera2()
        camera = Picamera2Camera(0, picamera2=lambda index: fake)
        camera.capture_preview()
        camera.capture_still()
        self.assertEqual(fake.captures, [("lores", "jpeg"), ("main", "jpeg")])

    def test_initial_controls_are_applied(self):
        fake = FakePicamera2()
        Picamera2Camera(0, picamera2=lambda index: fake)
        self.assertEqual(fake.control_calls[0]["AnalogueGain"], 1.0)
        self.assertEqual(fake.control_calls[0]["ExposureTime"], 10000)

    def test_set_controls_clamps_and_filters(self):
        fake = FakePicamera2()
        camera = Picamera2Camera(0, picamera2=lambda index: fake)
        camera.set_controls({"AnalogueGain": 100, "Nope": 1})
        self.assertEqual(fake.control_calls[-1], {"AnalogueGain": 4.0})

    def test_close_releases_the_device(self):
        fake = FakePicamera2()
        Picamera2Camera(0, picamera2=lambda index: fake).close()
        self.assertTrue(fake.closed)


class FakeVideoCapture:
    def __init__(self, index, api):
        self.index = index
        self.api = api
        self.opened = True
        self.released = False
        self.sets = []

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def get(self, prop):
        return {FakeCv2.CAP_PROP_FRAME_WIDTH: 640,
                FakeCv2.CAP_PROP_FRAME_HEIGHT: 480}.get(prop, 0)

    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _Encoded:
    def __init__(self, data):
        self._data = data

    def tobytes(self):
        return self._data


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FOURCC = 6
    CAP_PROP_BRIGHTNESS = 10
    CAP_PROP_CONTRAST = 11
    CAP_PROP_GAIN = 14
    CAP_PROP_EXPOSURE = 15
    CAP_PROP_SHARPNESS = 20
    CAP_PROP_AUTO_EXPOSURE = 21
    CAP_V4L2 = 200
    CAP_AVFOUNDATION = 1200
    CAP_MSMF = 1400
    CAP_ANY = 0

    def __init__(self):
        self.captures = []

    def VideoCapture(self, index, api):
        capture = FakeVideoCapture(index, api)
        self.captures.append(capture)
        return capture

    def VideoWriter_fourcc(self, *chars):
        return 1

    def imencode(self, ext, frame):
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="jpeg")
        return True, _Encoded(buffer.getvalue())


class FakeV4L2:
    """A V4L2Controls stand-in recording what the adapter asks of it."""

    def __init__(self, gain_max=100, exposure_max=6500):
        self.sets = []
        self.closed = False
        self._ranges = {
            camera_v4l2.GAIN: (0, gain_max, 1, 5),
            camera_v4l2.EXPOSURE_ABSOLUTE: (0, exposure_max, 1, 100),
        }

    def query(self, control_id):
        return self._ranges.get(control_id, (0, 255, 1, 0))

    def get(self, control_id):
        return 0

    def set(self, control_id, value):
        self.sets.append((control_id, int(value)))

    def close(self):
        self.closed = True


class TestOpenCvCamera(unittest.TestCase):
    def make_camera(self, cv2=None, v4l2=None):
        cv2 = cv2 or FakeCv2()
        v4l2 = v4l2 or FakeV4L2()
        camera = OpenCvCamera(
            0, cv2.CAP_V4L2, cv2=cv2, label="USB webcam", v4l2=v4l2
        )
        return camera, cv2, v4l2

    def test_configures_size(self):
        camera, cv2, _ = self.make_camera()
        capture = cv2.captures[0]
        self.assertIn((FakeCv2.CAP_PROP_FRAME_WIDTH, 640), capture.sets)
        self.assertIn((FakeCv2.CAP_PROP_FRAME_HEIGHT, 480), capture.sets)

    def test_leaves_the_pixel_format_at_the_device_default(self):
        camera, cv2, _ = self.make_camera()
        props = [prop for prop, _ in cv2.captures[0].sets]
        self.assertNotIn(FakeCv2.CAP_PROP_FOURCC, props)

    def test_properties(self):
        camera, _, _ = self.make_camera()
        self.assertEqual(camera.properties["Model"], "USB webcam")
        self.assertEqual(camera.properties["PixelArraySize"], (640, 480))

    def test_preview_and_still_are_jpegs(self):
        camera, _, _ = self.make_camera()
        self.assertEqual(jpeg_size(camera.capture_preview()), (640, 480))
        self.assertEqual(jpeg_size(camera.capture_still()), (640, 480))

    def test_controls_report_exposure_and_gain_from_the_device(self):
        camera, _, _ = self.make_camera()
        self.assertEqual(camera.controls["AnalogueGain"], (1.0, 20.0, 1.0))
        self.assertEqual(camera.controls["ExposureTime"], (0.0, 650000.0, 0.0))

    def test_exposure_enters_manual_and_converts_microseconds(self):
        camera, _, v4l2 = self.make_camera()
        camera.set_controls({"ExposureTime": 3000})
        self.assertIn(
            (camera_v4l2.EXPOSURE_AUTO, camera_v4l2.EXPOSURE_MODE_MANUAL),
            v4l2.sets,
        )
        self.assertIn((camera_v4l2.EXPOSURE_ABSOLUTE, 30), v4l2.sets)

    def test_gain_enters_manual_and_scales_to_the_device(self):
        camera, _, v4l2 = self.make_camera()
        camera.set_controls({"AnalogueGain": 20})
        self.assertIn((camera_v4l2.EXPOSURE_AUTO, 1), v4l2.sets)
        self.assertIn((camera_v4l2.GAIN, 100), v4l2.sets)

    def test_exposure_is_clamped_to_the_device_maximum(self):
        camera, _, v4l2 = self.make_camera(v4l2=FakeV4L2(exposure_max=6500))
        camera.set_controls({"ExposureTime": 1000000})
        self.assertIn((camera_v4l2.EXPOSURE_ABSOLUTE, 6500), v4l2.sets)

    def test_image_controls_do_not_touch_auto_exposure(self):
        camera, _, v4l2 = self.make_camera()
        camera.set_controls({"Brightness": 1.0})
        self.assertIn((camera_v4l2.BRIGHTNESS, 255), v4l2.sets)
        self.assertNotIn((camera_v4l2.EXPOSURE_AUTO, 1), v4l2.sets)

    def test_set_controls_ignores_unknown_names(self):
        camera, _, v4l2 = self.make_camera()
        camera.set_controls({"Nope": 1})
        self.assertEqual(v4l2.sets, [])

    def test_close_releases_the_device_and_the_controls(self):
        camera, cv2, v4l2 = self.make_camera()
        camera.close()
        self.assertTrue(cv2.captures[0].released)
        self.assertTrue(v4l2.closed)

    def test_unopenable_device_raises(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = lambda index, api: _UnopenableCapture()
        with self.assertRaises(RuntimeError):
            OpenCvCamera(0, cv2.CAP_V4L2, cv2=cv2, v4l2=None)


class TestOpenCvCameraFallback(unittest.TestCase):
    """The non-Linux path: controls go through CAP_PROP_*."""

    def make_camera(self, cv2=None):
        cv2 = cv2 or FakeCv2()
        return OpenCvCamera(
            0, cv2.CAP_V4L2, cv2=cv2, label="USB webcam", v4l2=None
        ), cv2

    def test_controls_use_assumed_ranges(self):
        camera, _ = self.make_camera()
        self.assertIn("AnalogueGain", camera.controls)
        self.assertIn("ExposureTime", camera.controls)

    def test_set_controls_uses_cap_props(self):
        camera, cv2 = self.make_camera()
        camera.set_controls({"Brightness": 1.0, "Contrast": 2.0})
        sets = cv2.captures[0].sets
        self.assertIn((FakeCv2.CAP_PROP_BRIGHTNESS, 255.0), sets)
        self.assertIn((FakeCv2.CAP_PROP_CONTRAST, 255.0), sets)

    def test_set_controls_ignores_unknown_names(self):
        camera, cv2 = self.make_camera()
        before = len(cv2.captures[0].sets)
        camera.set_controls({"Nope": 1})
        self.assertEqual(len(cv2.captures[0].sets), before)


class _UnopenableCapture:
    def isOpened(self):
        return False


class _FakeCameraInfo:
    def __init__(self, index, name, backend, vid=None):
        self.index = index
        self.name = name
        self.backend = backend
        self.vid = vid


class _FakeEnumerateModule:
    def __init__(self, infos):
        self._infos = infos

    def enumerate_cameras(self, api):
        return self._infos


class TestOpenCvEnumeration(unittest.TestCase):
    def test_opencv_cameras_are_listed(self):
        cv2 = FakeCv2()
        enum = _FakeEnumerateModule([
            _FakeCameraInfo(8, "USB webcam", cv2.CAP_V4L2, vid=0x2BC5)
        ])
        with mock.patch.dict(sys.modules, {
            "cv2": cv2, "cv2_enumerate_cameras": enum,
        }):
            descriptors = camera_module._opencv_cameras()
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0].backend, "opencv")
        self.assertEqual(descriptors[0].index, 8)
        self.assertEqual(descriptors[0].api, cv2.CAP_V4L2)
        self.assertIn("opencv:", descriptors[0].id)

    def test_linux_pipeline_nodes_without_a_vendor_id_are_dropped(self):
        cv2 = FakeCv2()
        enum = _FakeEnumerateModule([
            _FakeCameraInfo(8, "USB webcam", cv2.CAP_V4L2, vid=0x2BC5),
            _FakeCameraInfo(0, "rp1-cfe-csi2_ch0", cv2.CAP_V4L2, vid=None),
        ])
        with mock.patch.dict(sys.modules, {
            "cv2": cv2, "cv2_enumerate_cameras": enum,
        }):
            descriptors = camera_module._opencv_cameras()
        self.assertEqual([d.index for d in descriptors], [8])

    def test_missing_opencv_is_not_an_error(self):
        with mock.patch.dict(sys.modules, {"cv2": None}):
            self.assertEqual(camera_module._opencv_cameras(), [])


class TestCameraSelection(unittest.TestCase):
    def test_list_cameras_always_offers_the_dummy(self):
        ids = [d.id for d in list_cameras()]
        self.assertIn("dummy", ids)

    def test_make_camera_builds_the_dummy(self):
        camera = make_camera(CameraDescriptor("dummy", "d", "dummy"))
        self.assertIsInstance(camera, DummyCamera)

    def test_make_camera_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            make_camera(CameraDescriptor("x", "x", "nope"))

    def test_open_camera_returns_a_usable_camera(self):
        camera = open_camera()
        try:
            self.assertIsInstance(camera.capture_preview(), bytes)
        finally:
            camera.close()

    def test_open_camera_ignores_an_unknown_preference(self):
        camera = open_camera(preferred="does-not-exist")
        try:
            self.assertIsInstance(camera.capture_preview(), bytes)
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main()
