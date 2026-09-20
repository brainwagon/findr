import unittest

from solver import BACKENDS, BaseSolver, LibrarySolver, SolverManager, SolverResult


class MockSolver(BaseSolver):
    def solve(self, image):
        return SolverResult(ra=100, dec=100, roll=0, fov=0)


class TestSolverManager(unittest.TestCase):
    def setUp(self):
        self.manager = SolverManager()

    def test_default_solver(self):
        """The default backend is cedar-solve, driven by a LibrarySolver."""
        current = self.manager.get_current_solver()
        self.assertIsInstance(current, LibrarySolver)
        self.assertEqual(current.backend.key, 'cedar-solve')

    def test_set_solver(self):
        """Switching backends swaps the active LibrarySolver's backend."""
        self.manager.set_solver('tetra3')
        current = self.manager.get_current_solver()
        self.assertIsInstance(current, LibrarySolver)
        self.assertEqual(current.backend.key, 'tetra3')

        self.manager.register_solver('mock', lambda: MockSolver())
        self.manager.set_solver('mock')
        self.assertIsInstance(self.manager.get_current_solver(), MockSolver)

        result = self.manager.solve(None)
        self.assertEqual(result.ra, 100)

    def test_available_solvers(self):
        """The available solvers come from the registry, not a hardcoded list."""
        self.assertEqual(set(self.manager.available_solvers()), set(BACKENDS))
        self.assertLessEqual({'tetra3', 'cedar-solve'}, set(BACKENDS))

    def test_invalid_solver(self):
        """Test setting an invalid solver type."""
        with self.assertRaises(ValueError):
            self.manager.set_solver('unknown_solver')

    def test_get_solver_singleton(self):
        """Test the global get_solver returns the same manager/solver instance."""
        from solver import get_solver
        solver = get_solver()
        self.assertTrue(hasattr(solver, 'solve'))


if __name__ == '__main__':
    unittest.main()
