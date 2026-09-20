import os
import random
import tempfile
import unittest

from PIL import Image

from solve import CameraImageSource, ImageSourceError, TestImageSource


class TestTestImageSource(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        Image.new("RGB", (8, 8), "black").save(
            os.path.join(self.directory, "a.jpg")
        )
        Image.new("RGB", (9, 9), "black").save(
            os.path.join(self.directory, "b.jpeg")
        )
        with open(os.path.join(self.directory, "notes.txt"), "w") as f:
            f.write("not an image")

    def test_selects_deterministically_with_injected_rng(self):
        listing = [
            f for f in os.listdir(self.directory)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]
        picked = random.Random(7).choice(listing)
        expected_size = {"a.jpg": (8, 8), "b.jpeg": (9, 9)}[picked]

        image = TestImageSource(self.directory, rng=random.Random(7)).acquire()
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, expected_size)

    def test_ignores_non_images(self):
        source = TestImageSource(self.directory, rng=random.Random(0))
        for _ in range(5):
            self.assertIsInstance(source.acquire(), Image.Image)

    def test_empty_directory_raises(self):
        with self.assertRaises(ImageSourceError):
            TestImageSource(tempfile.mkdtemp()).acquire()

    def test_missing_directory_raises(self):
        with self.assertRaises(ImageSourceError):
            TestImageSource(os.path.join(self.directory, "nope")).acquire()


class FakeCamera:
    def __init__(self, image):
        self._image = image

    def capture_file(self, buffer, name=None, format="jpeg"):
        self._image.save(buffer, format=format)


class TestCameraImageSource(unittest.TestCase):
    def test_acquires_from_camera(self):
        source = CameraImageSource(
            FakeCamera(Image.new("RGB", (16, 12), "grey"))
        )
        image = source.acquire()
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (16, 12))


if __name__ == "__main__":
    unittest.main()
