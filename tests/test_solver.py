import unittest
import os
from solver import PlateSolver, CedarSolver, get_solver

class TestPlateSolver(unittest.TestCase):
    def setUp(self):
        self.solver = PlateSolver()
        # Adjusted path if necessary, assuming it's run from project root
        self.sample_image_path = 'test-images/lores_jpeg_2025-11-07T03_03_46.674Z.jpg'

    def test_initialization(self):
        """Test that the solver initializes correctly."""
        self.assertIsNotNone(self.solver.t3)

    def test_solve_successful(self):
        """Test a successful plate solve with a known good image."""
        if not os.path.exists(self.sample_image_path):
            self.skipTest(f"Sample image not found: {self.sample_image_path}")
        
        result = self.solver.solve(self.sample_image_path)
        self.assertIsNotNone(result)
        self.assertIn('ra', result)
        self.assertIn('dec', result)
        self.assertIn('fov', result)
        self.assertGreater(result['ra'], 0)
        self.assertGreater(result['dec'], 0)

    def test_solve_failure(self):
        """Test solver behavior with a non-star image (if available or empty)."""
        # We'll use a black image for this test if it exists.
        black_image_path = 'static/black_640x480.jpg'
        if os.path.exists(black_image_path):
            result = self.solver.solve(black_image_path)
            self.assertIsNone(result)

    def test_invalid_path(self):
        """Test solver behavior with an invalid image path."""
        result = self.solver.solve('non_existent_image.jpg')
        self.assertIsNone(result)

    def test_get_solver_singleton(self):
        """Test that get_solver returns the same singleton instance."""
        s1 = get_solver()
        s2 = get_solver()
        self.assertIs(s1, s2)
        self.assertIsInstance(s1, CedarSolver)
        # It should still act as a solver (duck typing or interface)
        self.assertTrue(hasattr(s1, 'solve'))

if __name__ == '__main__':
    unittest.main()
