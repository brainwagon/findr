import io
import os
import random
import tempfile
import unittest

from PIL import Image

from solve import ImageSourceError, PreviewFrameSource, TestImageSource


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


def jpeg_bytes(size=(16, 12)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "grey").save(buffer, format="jpeg")
    return buffer.getvalue()


class TestPreviewFrameSource(unittest.TestCase):
    def test_acquires_the_latest_frame(self):
        source = PreviewFrameSource(lambda: jpeg_bytes())
        image = source.acquire()
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (16, 12))

    def test_waits_for_the_first_frame(self):
        frames = [None, None, jpeg_bytes()]
        source = PreviewFrameSource(
            lambda: frames.pop(0), timeout=1.0, interval=0.01
        )
        self.assertIsInstance(source.acquire(), Image.Image)

    def test_raises_when_no_frame_arrives(self):
        source = PreviewFrameSource(
            lambda: None, timeout=0.05, interval=0.01
        )
        with self.assertRaises(ImageSourceError):
            source.acquire()


if __name__ == "__main__":
    unittest.main()
