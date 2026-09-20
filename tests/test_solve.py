import unittest

import ephem
from PIL import Image, ImageFont

from catalog import Catalog
from solve import format_radec_fixed_width, run_solve
from solver import BaseSolver, SolverResult


SOLUTION = SolverResult(
    ra=186.0,
    dec=-60.0,
    roll=12.5,
    fov=3.2,
    matched_stars_count=42,
)


class FakeSolver(BaseSolver):
    """A Solver backend that returns a canned result."""

    def __init__(self, result):
        self._result = result

    def solve(self, image):
        return self._result


class RaisingSolver(BaseSolver):
    def solve(self, image):
        raise RuntimeError("boom")


def make_catalog():
    return Catalog(star_names={}, boundaries={}, font=ImageFont.load_default())


def make_observer():
    observer = ephem.Observer()
    observer.lat = "0"
    observer.lon = "0"
    return observer


def fixed_clock():
    return ephem.Date("2026/01/01 00:00:00")


def black_image():
    return Image.new("RGB", (100, 100), "black")


class TestRunSolve(unittest.TestCase):
    def test_success_populates_outcome(self):
        outcome = run_solve(
            black_image(), FakeSolver(SOLUTION), make_observer(),
            make_catalog(), clock=fixed_clock,
        )
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.ra, 186.0)
        self.assertEqual(outcome.dec, -60.0)
        self.assertEqual(outcome.roll, 12.5)
        self.assertEqual(outcome.fov, 3.2)
        self.assertEqual(outcome.matched_stars_count, 42)
        self.assertEqual(outcome.ra_hms, "12:24:00.0")
        self.assertIsNotNone(outcome.alt)
        self.assertIsNotNone(outcome.az)
        self.assertTrue(outcome.constellation)
        self.assertIsNotNone(outcome.image)

    def test_failure_sets_error_and_keeps_image(self):
        image = black_image()
        outcome = run_solve(
            image, FakeSolver(None), make_observer(), make_catalog(),
            clock=fixed_clock,
        )
        self.assertFalse(outcome.ok)
        self.assertIsNotNone(outcome.error)
        self.assertIs(outcome.image, image)
        self.assertEqual(outcome.to_json(), {"error": outcome.error})

    def test_solver_exception_becomes_failure(self):
        image = black_image()
        outcome = run_solve(
            image, RaisingSolver(), make_observer(), make_catalog(),
            clock=fixed_clock,
        )
        self.assertFalse(outcome.ok)
        self.assertIn("boom", outcome.error)
        self.assertIs(outcome.image, image)

    def test_to_json_has_stable_keys(self):
        outcome = run_solve(
            black_image(), FakeSolver(SOLUTION), make_observer(),
            make_catalog(), clock=fixed_clock,
        )
        payload = outcome.to_json()
        for key in (
            "ra", "dec", "roll", "ra_hms", "dec_dms",
            "alt", "az", "constellation", "matched_stars_count",
        ):
            self.assertIn(key, payload)


class TestFormatRadecFixedWidth(unittest.TestCase):
    def test_ra_is_padded(self):
        self.assertEqual(
            format_radec_fixed_width(
                ephem.hours("1:30:00"), is_ra=True, total_width=10,
                decimal_places=1,
            ),
            "01:30:00.0",
        )


if __name__ == "__main__":
    unittest.main()
