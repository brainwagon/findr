import tetra3
from PIL import Image
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseSolver(ABC):
    @abstractmethod
    def solve(self, image_path_or_obj):
        """
        Solve the plate for a given image.
        :param image_path_or_obj: Path to the image file or a PIL Image object.
        :return: A dictionary containing the solve results or None if failed.
        """
        pass

class Tetra3Solver(BaseSolver):
    def __init__(self, database_path='default_database'):
        """
        Initialize the Tetra3 plate solver.
        :param database_path: Path to a custom tetra3 database. If None, the default is used.
        """
        logger.info(f"Initializing Tetra3 solver with database: {database_path}...")
        try:
            self.t3 = tetra3.Tetra3(load_database=database_path)
            logger.info("Tetra3 solver initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Tetra3 solver: {e}")
            raise

    def solve(self, image_path_or_obj):
        """
        Solve the plate for a given image using tetra3.
        :param image_path_or_obj: Path to the image file or a PIL Image object.
        :return: A dictionary containing the solve results or None if failed.
        """
        try:
            if isinstance(image_path_or_obj, str):
                img = Image.open(image_path_or_obj)
            else:
                img = image_path_or_obj

            logger.info(f"Attempting to solve image with Tetra3...")
            solution = self.t3.solve_from_image(img, return_matches=True)

            if solution['RA'] is not None:
                logger.info("Plate solve successful.")
                return {
                    'ra': solution['RA'],
                    'dec': solution['Dec'],
                    'roll': solution['Roll'],
                    'fov': solution['FOV'],
                    'matched_stars_count': solution.get('Matches', 0),
                    'matched_catID': solution.get('matched_catID', []),
                    'matched_centroids': solution.get('matched_centroids', []),
                    'matched_stars': solution.get('matched_stars', []),
                    'timestamp': solution.get('timestamp')
                }
            else:
                logger.warning("Plate solve failed to find a solution.")
                return None
        except Exception as e:
            logger.error(f"Error during plate solving: {e}")
            return None

class SolverManager(BaseSolver):
    def __init__(self, default_solver_type='tetra3'):
        self._solvers = {}
        self._current_solver_type = None
        self._current_solver_instance = None
        
        # Register available solvers
        self.register_solver('tetra3', Tetra3Solver)
        # cedar-solve will be registered later in Phase 2
        
        try:
            self.set_solver(default_solver_type)
        except Exception as e:
            logger.error(f"Failed to set default solver {default_solver_type}: {e}")

    def register_solver(self, solver_type, solver_class):
        """Register a new solver class."""
        self._solvers[solver_type] = solver_class
        logger.info(f"Registered solver: {solver_type}")

    def set_solver(self, solver_type):
        """Switch to a different solver type."""
        if solver_type not in self._solvers:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        if solver_type == self._current_solver_type:
            return

        logger.info(f"Switching solver to: {solver_type}...")
        self._current_solver_type = solver_type
        # Lazy initialization of the solver instance
        self._current_solver_instance = self._solvers[solver_type]()
        logger.info(f"Active solver is now: {solver_type}")

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

# For backward compatibility
PlateSolver = Tetra3Solver

# Simple singleton instance for global use
_solver_manager_instance = None

def get_solver():
    global _solver_manager_instance
    if _solver_manager_instance is None:
        _solver_manager_instance = SolverManager()
    return _solver_manager_instance
