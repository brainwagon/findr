import sys
import os
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from PIL import Image
import numpy as np

# tetra3 calls np.math.factorial, which NumPy 2.x removed. The submodule is
# pinned, so restore the alias here, before importing it, rather than patching
# the vendored library.
np.math = math

# Add the vendored libraries to sys.path, at the front so the bundled backends
# win over any identically-named package installed in the environment.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'cedar-solve'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tetra3-repo'))

# Both backends expose the same API under distinct package names.
import tetra3
import cedar_solve

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolverBackend:
    """A star-pattern matching library the finder can solve with.

    `tetra3` and `cedar-solve` expose the same API, so a backend is a
    descriptor rather than an adapter: a key, a label, and the module that
    holds the `Tetra3` class.
    """

    key: str
    label: str
    module: object


@dataclass(frozen=True)
class SolverResult:
    """The normalised result of a successful solve by a Solver backend."""

    ra: float
    dec: float
    roll: float
    fov: float
    matched_stars_count: int = 0
    matched_cat_ids: List = field(default_factory=list)
    matched_centroids: List = field(default_factory=list)
    matched_stars: List = field(default_factory=list)
    solver_type: str = ""


# The available Solver backends, keyed by the name the UI uses.
BACKENDS = {
    'tetra3': SolverBackend('tetra3', 'Tetra3', tetra3),
    'cedar-solve': SolverBackend('cedar-solve', 'Cedar-Solve', cedar_solve),
}


class BaseSolver(ABC):
    """The interface every Solver backend implementation satisfies."""

    @abstractmethod
    def solve(self, image_path_or_obj) -> Optional[SolverResult]:
        """Solve the plate for a given image.

        Args:
            image_path_or_obj: Path to an image file or a PIL Image object.

        Returns:
            A SolverResult, or None if no solution was found.
        """
        raise NotImplementedError


class LibrarySolver(BaseSolver):
    """One implementation driving any Solver backend."""

    def __init__(self, backend, database_path='default_database'):
        """Initialize the backend's solver with its star database."""
        self.backend = backend
        logger.info(
            "Initializing %s with database: %s...", backend.label, database_path
        )
        try:
            self.t3 = backend.module.Tetra3(load_database=database_path)
        except Exception as e:
            logger.error("Failed to initialize %s: %s", backend.label, e)
            raise
        logger.info("%s initialized successfully.", backend.label)

    def solve(self, image_path_or_obj):
        """Solve an image and normalise the backend's result."""
        try:
            if isinstance(image_path_or_obj, str):
                image = Image.open(image_path_or_obj)
            else:
                image = image_path_or_obj

            logger.info("Attempting to solve image with %s...", self.backend.label)
            solution = self.t3.solve_from_image(image, return_matches=True)

            if solution['RA'] is None:
                logger.warning("Plate solve failed to find a solution.")
                return None

            logger.info("Plate solve successful.")
            return SolverResult(
                ra=solution['RA'],
                dec=solution['Dec'],
                roll=solution['Roll'],
                fov=solution['FOV'],
                matched_stars_count=solution.get('Matches', 0),
                matched_cat_ids=solution.get('matched_catID', []),
                matched_centroids=solution.get('matched_centroids', []),
                matched_stars=solution.get('matched_stars', []),
                solver_type=self.backend.key,
            )
        except Exception as e:
            logger.error(
                "Error during plate solving with %s: %s", self.backend.label, e
            )
            return None


class SolverManager:
    """Selects the active Solver backend and delegates solves to it."""

    def __init__(self, default_solver_type='cedar-solve'):
        self._solvers: Dict[str, Callable[[], BaseSolver]] = {}
        self._current_solver_type = None
        self._current_solver_instance = None

        # Register the available backends, each behind a LibrarySolver.
        for backend in BACKENDS.values():
            self.register_solver(
                backend.key, lambda b=backend: LibrarySolver(b)
            )

        try:
            self.set_solver(default_solver_type)
        except Exception as e:
            logger.error(
                "Failed to set default solver %s: %s", default_solver_type, e
            )
            # Fall back to tetra3 if cedar-solve fails (e.g. database missing).
            if default_solver_type != 'tetra3':
                logger.info("Falling back to tetra3 solver...")
                self.set_solver('tetra3')

    def register_solver(self, solver_type, factory):
        """Register a factory that builds a solver for this type."""
        self._solvers[solver_type] = factory
        logger.info("Registered solver: %s", solver_type)

    def set_solver(self, solver_type):
        """Switch to a different solver type."""
        if solver_type not in self._solvers:
            raise ValueError(f"Unknown solver type: {solver_type}")

        if solver_type == self._current_solver_type:
            return

        logger.info("Switching solver to: %s...", solver_type)
        self._current_solver_type = solver_type
        # Lazy initialization of the solver instance
        self._current_solver_instance = self._solvers[solver_type]()
        logger.info("Active solver is now: %s", solver_type)

    def available_solvers(self):
        """The registered solver type names."""
        return list(self._solvers.keys())

    def get_current_solver_type(self):
        """Get the current active solver type."""
        return self._current_solver_type

    def get_current_solver(self):
        """Get the current active solver instance."""
        return self._current_solver_instance

    def solve(self, image_path_or_obj):
        """Delegate solving to the current active solver."""
        if self._current_solver_instance:
            return self._current_solver_instance.solve(image_path_or_obj)
        logger.error("No active solver instance to handle solve request.")
        return None


# Simple singleton instance for global use
_solver_manager_instance = None


def get_solver():
    global _solver_manager_instance
    if _solver_manager_instance is None:
        _solver_manager_instance = SolverManager()
    return _solver_manager_instance
