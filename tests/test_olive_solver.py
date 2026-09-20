import os
import unittest

from solver import BACKENDS, BaseSolver, make_solver


@unittest.skipUnless('olive-solve' in BACKENDS, 'olive-solve backend not built')
class TestOliveSolver(unittest.TestCase):
    def setUp(self):
        self.solver = make_solver('olive-solve')
        self.sample_image_path = 'test-images/lores_jpeg_2025-11-07T03_03_46.674Z.jpg'

    def test_is_base_solver(self):
        """Test that OliveSolver implements BaseSolver."""
        self.assertIsInstance(self.solver, BaseSolver)

    def test_initialization(self):
        """Test that the solver initializes correctly."""
        self.assertIsNotNone(self.solver.t3)

    def test_backend_key(self):
        """The normalized result identifies the backend that produced it."""
        self.assertEqual(self.solver.backend.key, 'olive-solve')

    def test_solve_successful(self):
        """Test a successful plate solve with a known good image."""
        if not os.path.exists(self.sample_image_path):
            self.skipTest(f"Sample image not found: {self.sample_image_path}")

        result = self.solver.solve(self.sample_image_path)
        self.assertIsNotNone(result)
        self.assertGreater(result.ra, 0)
        self.assertGreater(result.dec, 0)
        self.assertEqual(result.solver_type, 'olive-solve')

    def test_solve_failure(self):
        """Test solver behavior with a non-star image."""
        black_image_path = 'static/black_640x480.jpg'
        if os.path.exists(black_image_path):
            self.assertIsNone(self.solver.solve(black_image_path))

    def test_invalid_path(self):
        """Test solver behavior with an invalid image path."""
        self.assertIsNone(self.solver.solve('non_existent_image.jpg'))


if __name__ == '__main__':
    unittest.main()
