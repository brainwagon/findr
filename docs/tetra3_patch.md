# Patch: tetra3 compatibility with NumPy 2.x

## Issue
The `tetra3` library (v0.1) uses `np.math.factorial`, which has been removed in NumPy 2.0. This causes an `AttributeError` when attempting to solve images.

## Resolution
The library was manually patched in the virtual environment to use the standard Python `math` module instead of `np.math`.

### Changes:
1. Added `import math` to `tetra3/tetra3.py`.
2. Replaced all occurrences of `np.math.factorial` with `math.factorial`.

### Verification:
The fix was verified using `scripts/verify_tetra3.py` with a sample image, which now solves correctly:
- **RA:** 315.6706...
- **Dec:** 36.0987...
- **FOV:** 10.847...
