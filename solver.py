import sys
import os
import logging
from abc import ABC, abstractmethod
from PIL import Image

# Add local library paths to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'cedar-solve'))

# Import the renamed library
import cedar_solve

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

class CedarSolver(BaseSolver):
    def __init__(self, database_path='default_database'):
        """
        Initialize the Cedar-Solve plate solver.
        """
        logger.info(f"Initializing Cedar-Solve with database: {database_path}...")
        try:
            self.t3 = cedar_solve.Tetra3(load_database=database_path)
            logger.info("Cedar-Solve initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Cedar-Solve: {e}")
            raise

    def solve(self, image_path_or_obj):
        try:
            if isinstance(image_path_or_obj, str):
                img = Image.open(image_path_or_obj)
            else:
                img = image_path_or_obj

            logger.info(f"Attempting to solve image with Cedar-Solve...")
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
                    'timestamp': solution.get('timestamp'),
                    'solver_type': 'cedar_solve'
                }
            else:
                logger.warning("Plate solve failed to find a solution.")
                return None
        except Exception as e:
            logger.error(f"Error during plate solving with Cedar-Solve: {e}")
            return None

# Simplified PlateSolver using CedarSolver
PlateSolver = CedarSolver

# Simple singleton instance for global use
_solver_instance = None

def get_solver(database_path='default_database'):
    global _solver_instance
    if _solver_instance is None:
        try:
            _solver_instance = CedarSolver(database_path=database_path)
        except Exception as e:
            logger.error(f"Failed to initialize global solver: {e}")
            return None
    return _solver_instance
