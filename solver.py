"""Core plate-solving abstraction and manager for findr.

Defines the BaseSolver interface and the CedarSolver implementation
using the cedar-solve library.
"""

import logging
import os
import sys
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
    """Abstract base class for plate solvers."""

    @abstractmethod
    def solve(self, image_path_or_obj):
        """Solves the plate for a given image.

        Args:
            image_path_or_obj: Path to the image file or a PIL Image object.

        Returns:
            A dictionary containing the solve results (ra, dec, roll, fov, etc.)
            or None if the solve failed.
        """
        pass

class CedarSolver(BaseSolver):
    """Plate solver implementation using the cedar-solve library."""

    def __init__(self, database_path='default_database'):
        """Initializes the Cedar-Solve plate solver.

        Args:
            database_path: Name or path of the star database to load.

        Raises:
            Exception: If the database fails to load.
        """
        logger.info(f"Initializing Cedar-Solve with database: {database_path}...")
        try:
            self.t3 = cedar_solve.Tetra3(load_database=database_path)
            logger.info("Cedar-Solve initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Cedar-Solve: {e}")
            raise

    def solve(self, image_path_or_obj):
        """Solves the plate for a given image using Cedar-Solve.

        Args:
            image_path_or_obj: Path to the image file or a PIL Image object.

        Returns:
            A dictionary with keys 'ra', 'dec', 'roll', 'fov',
            'matched_stars_count', 'matched_catID', 'matched_centroids',
            'matched_stars', 'timestamp', and 'solver_type'.
            Returns None if no solution is found or an error occurs.
        """
        try:
            if isinstance(image_path_or_obj, str):
                img = Image.open(image_path_or_obj)
            else:
                img = image_path_or_obj

            logger.info("Attempting to solve image with Cedar-Solve...")
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
    """Returns a singleton instance of the CedarSolver.

    Args:
        database_path: Name or path of the star database to load.

    Returns:
        A CedarSolver instance or None if initialization fails.
    """
    global _solver_instance
    if _solver_instance is None:
        try:
            _solver_instance = CedarSolver(database_path=database_path)
        except Exception as e:
            logger.error(f"Failed to initialize global solver: {e}")
            return None
    return _solver_instance
