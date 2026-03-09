import unittest
from solver import SolverManager, Tetra3Solver, CedarSolver, BaseSolver

class MockSolver(BaseSolver):
    def solve(self, image):
        return {"ra": 100, "dec": 100}

class TestSolverManager(unittest.TestCase):
    def setUp(self):
        self.manager = SolverManager()

    def test_default_solver(self):
        """Test that the default solver is CedarSolver."""
        self.assertIsInstance(self.manager.get_current_solver(), CedarSolver)

    def test_set_solver(self):
        """Test switching solvers."""
        self.manager.set_solver('tetra3')
        self.assertIsInstance(self.manager.get_current_solver(), Tetra3Solver)
        
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
        solver = get_solver()
        self.assertTrue(hasattr(solver, 'solve'))

if __name__ == '__main__':
    unittest.main()
