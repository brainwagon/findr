import unittest
from abc import ABC
from solver import BaseSolver

class TestBaseSolver(unittest.TestCase):
    def test_is_abstract(self):
        """Test that BaseSolver is an abstract base class."""
        self.assertTrue(issubclass(BaseSolver, ABC))
        with self.assertRaises(TypeError):
            BaseSolver()

    def test_abstract_methods(self):
        """Test that BaseSolver has the required abstract methods."""
        class MockSolver(BaseSolver):
            pass
        
        with self.assertRaises(TypeError):
            MockSolver()

    def test_solve_interface(self):
        """Test that a valid subclass can be instantiated and has a solve method."""
        class ValidSolver(BaseSolver):
            def solve(self, image):
                return {"ra": 0, "dec": 0}
        
        solver = ValidSolver()
        self.assertTrue(hasattr(solver, 'solve'))
        result = solver.solve(None)
        self.assertEqual(result["ra"], 0)

if __name__ == '__main__':
    unittest.main()
