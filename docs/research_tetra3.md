# Research: tetra3 Library Integration

## Overview
- **Library:** tetra3 (ESA star plate solver)
- **Goal:** FAST "lost-in-space" star plate solver.
- **Accuracy:** ~10 arcseconds.
- **Speed:** ~10ms (excluding centroiding).

## Dependencies
- Python 3.7+
- NumPy
- SciPy
- Pillow (PIL)

## Installation
Recommended: `pip install git+https://github.com/esa/tetra3.git`

## Core API
```python
import tetra3
from PIL import Image

# Initialize (loads default database)
t3 = tetra3.Tetra3()

# Solve
img = Image.open('stars.jpg')
solution = t3.solve_from_image(img, fov_estimate=None)
# Returns dict: {'RA': ..., 'Dec': ..., 'Roll': ..., 'FOV': ..., ...}
```

## Databases
- Uses custom databases for specific FOVs.
- Default database is included or generated from catalogs like Yale Bright Star or Hipparcos.
