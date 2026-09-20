import unittest
from unittest import mock

from PIL import Image, ImageFont

import overlay
from catalog import Catalog
from overlay import render
from solver import SolverResult


def make_catalog(star_names=None, boundaries=None):
    return Catalog(
        star_names=star_names or {},
        boundaries=boundaries or {},
        font=ImageFont.load_default(),
    )


def black_image():
    return Image.new("RGB", (100, 100), "black")


class TestRender(unittest.TestCase):
    def test_labels_matched_stars(self):
        result = SolverResult(
            ra=0, dec=0, roll=0, fov=0,
            matched_cat_ids=[1], matched_centroids=[(50, 50)],
        )
        image = black_image()
        before = image.tobytes()
        returned = render(image, result, make_catalog({1: "alf Cen"}), "Cen")
        self.assertIs(returned, image)
        self.assertNotEqual(image.tobytes(), before)

    def test_no_matches_leaves_image_untouched(self):
        result = SolverResult(ra=0, dec=0, roll=0, fov=0)
        image = black_image()
        before = image.tobytes()
        render(image, result, make_catalog(), "Cen")
        self.assertEqual(image.tobytes(), before)

    def test_unknown_constellation_draws_no_boundaries(self):
        result = SolverResult(ra=0, dec=0, roll=0, fov=0)
        image = black_image()
        before = image.tobytes()
        render(image, result, make_catalog(boundaries={}), "Nope")
        self.assertEqual(image.tobytes(), before)

    def test_boundaries_flag_controls_boundary_drawing(self):
        result = SolverResult(
            ra=0, dec=0, roll=0, fov=0,
            matched_cat_ids=[1], matched_centroids=[(50, 50)],
        )
        catalog = make_catalog({1: "alf Cen"})

        with mock.patch.object(overlay, "draw_boundaries") as draw:
            render(black_image(), result, catalog, "Cen", boundaries=False)
            draw.assert_not_called()

        with mock.patch.object(overlay, "draw_boundaries") as draw:
            render(black_image(), result, catalog, "Cen", boundaries=True)
            draw.assert_called_once()

    def test_boundary_failure_is_logged_not_raised(self):
        # One matched star is too few to fit a WCS, so astropy raises; render
        # must swallow it and still return the image.
        result = SolverResult(
            ra=0, dec=0, roll=0, fov=0,
            matched_stars=[(0.0, 0.0)], matched_centroids=[(10, 10)],
        )
        catalog = make_catalog(boundaries={"CEN": [(0.0, 0.0), (1.0, 1.0)]})
        image = black_image()
        with self.assertLogs("overlay", level="WARNING"):
            returned = render(image, result, catalog, "Cen")
        self.assertIs(returned, image)


if __name__ == "__main__":
    unittest.main()
