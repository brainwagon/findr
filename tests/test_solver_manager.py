import unittest
from solver import SolverManager, Tetra3Solver, BaseSolver

class MockSolver(BaseSolver):
    def solve(self, image):
        return {"ra": 100, "dec": 100}

class TestSolverManager(unittest.TestCase):
    def setUp(self):
        self.manager = SolverManager()

    def test_default_solver(self):
        """Test that the default solver is Tetra3Solver."""
        self.assertIsInstance(self.manager.get_current_solver(), Tetra3Solver)

    def test_set_solver(self):
        """Test switching solvers."""
        # For now, we only have tetra3, but we can register a mock for testing.
        self.manager.register_solver('mock', MockSolver)
        self.manager.set_solver('mock')
        self.assertIsInstance(self.manager.get_current_solver(), MockSolver)
        
        result = self.manager.solve(None)
        self.assertEqual(result['ra'], 100)

    def test_invalid_solver(self):
        """Test setting an invalid solver type."""
        with self.assertRaises(ValueError):
            self.manager.set_solver('unknown_solver')

    def test_get_solver_singleton(self):
        """Test the global get_solver returns the same manager/solver instance logic."""
        from solver import get_solver
        # get_solver should now return the manager or the current solver depending on implementation.
        # If it returns the manager, we check it acts as a solver.
        solver = get_solver()
        self.assertTrue(hasattr(solver, 'solve'))

if __name__ == '__main__':
    unittest.main()
