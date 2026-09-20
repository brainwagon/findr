import os
import unittest

from solver import BACKENDS, LibrarySolver, SolverManager


class TestCedarSolver(unittest.TestCase):
    def setUp(self):
        self.solver = LibrarySolver(BACKENDS['cedar-solve'])
        self.sample_image_path = 'test-images/lores_jpeg_2025-11-07T03_03_46.674Z.jpg'

    def test_initialization(self):
        """Test that the solver initializes correctly."""
        self.assertIsNotNone(self.solver.t3)

    def test_backend_key(self):
        """The normalized result identifies the backend that produced it."""
        self.assertEqual(self.solver.backend.key, 'cedar-solve')

    def test_solve_successful(self):
        """Test a successful plate solve with a known good image."""
        if not os.path.exists(self.sample_image_path):
            self.skipTest(f"Sample image not found: {self.sample_image_path}")

        result = self.solver.solve(self.sample_image_path)
        self.assertIsNotNone(result)
        self.assertGreater(result.ra, 0)
        self.assertGreater(result.dec, 0)
        self.assertEqual(result.solver_type, 'cedar-solve')

    def test_solve_failure(self):
        """Test solver behavior with a non-star image (if available or empty)."""
        black_image_path = 'static/black_640x480.jpg'
        if os.path.exists(black_image_path):
            self.assertIsNone(self.solver.solve(black_image_path))

    def test_invalid_path(self):
        """Test solver behavior with an invalid image path."""
        self.assertIsNone(self.solver.solve('non_existent_image.jpg'))

    def test_get_solver_singleton(self):
        """Test that get_solver returns the same singleton instance."""
        from solver import get_solver
        s1 = get_solver()
        s2 = get_solver()
        self.assertIs(s1, s2)
        self.assertIsInstance(s1, SolverManager)
        # It should still act as a solver (duck typing or interface)
        self.assertTrue(hasattr(s1, 'solve'))


if __name__ == '__main__':
    unittest.main()
