import unittest
import os
from solver import Tetra3Solver, BaseSolver

class TestTetra3Solver(unittest.TestCase):
    def setUp(self):
        self.solver = Tetra3Solver()
        self.sample_image_path = 'test-images/lores_jpeg_2025-11-07T03_03_46.674Z.jpg'

    def test_is_base_solver(self):
        """Test that Tetra3Solver implements BaseSolver."""
        self.assertTrue(isinstance(self.solver, BaseSolver))

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

    def test_solve_failure(self):
        """Test solver behavior with a non-star image."""
        black_image_path = 'static/black_640x480.jpg'
        if os.path.exists(black_image_path):
            result = self.solver.solve(black_image_path)
            self.assertIsNone(result)

    def test_invalid_path(self):
        """Test solver behavior with an invalid image path."""
        result = self.solver.solve('non_existent_image.jpg')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
