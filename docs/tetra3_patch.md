# Patch: tetra3 compatibility with NumPy 2.x

## Issue
The `tetra3` library (v0.1) uses `np.math.factorial`, which has been removed in NumPy 2.0. This causes an `AttributeError` when attempting to solve images.

## Resolution
The vendored `tetra3-repo` submodule is pinned and still calls `np.math.factorial`
(`tetra3/tetra3.py:1395`), so `solver.py` restores the alias before importing it:

```python
# tetra3 calls np.math.factorial, which NumPy 2.x removed. The submodule is
# pinned, so restore the alias here, before importing it, rather than patching
# the vendored library.
np.math = math
```

This is a runtime shim rather than a source edit, so the submodule stays pristine
and a fresh recursive clone needs no manual patching. `cedar-solve` does not use
`np.math`, so only the tetra3 backend needs the alias.

### Verification
Confirmed the alias is still required under NumPy 2.2.6: the pinned tetra3 still
references `np.math.factorial`, and importing it without the alias raises
`AttributeError: module 'numpy' has no attribute 'math'`. With the alias in place
the solver runs correctly.
