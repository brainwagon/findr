# Specification: Remove tetra3 and Scaffolding

## Overview
This track aims to simplify the project by removing the original `tetra3` library and all the scaffolding (SolverManager, dual-solver logic, UI controls) that allowed the application to switch between `tetra3` and `cedar-solve`. `cedar-solve` will become the sole plate-solving engine.

## Functional Requirements
1. **Remove tetra3 Submodule:** Completely delete the `tetra3-repo` submodule and all its contents.
2. **Refactor solver.py:**
   - Remove `SolverManager` and `Tetra3Solver` classes.
   - Promote `CedarSolver` or integrate its logic directly so it is the only available solver.
   - Update `get_solver()` and other helper functions to return/use `CedarSolver`.
3. **Clean Up UI:**
   - Remove the solver selection dropdown from `templates/index.html`.
   - Remove the backend API endpoint (`/api/v1/solver` if it exists) and corresponding JavaScript in `static/main.js` for switching solvers.
4. **Clean Up Files/Docs:**
   - Delete `patches/tetra3_numpy2.patch`.
   - Delete `docs/tetra3_patch.md`.
   - Delete `docs/research_tetra3.md` (if no longer relevant).
5. **Clean Up Tests:**
   - Delete `tests/test_tetra3_solver.py`.
   - Delete `tests/test_solver_manager.py`.

## Non-Functional Requirements
- **Performance:** Ensure there is no performance regression in switching to a single-solver architecture.
- **Maintainability:** Improve code readability by removing unused abstractions and branching logic.

## Acceptance Criteria
- [x] `tetra3-repo` directory and submodule are gone from the project.
- [x] `solver.py` contains only the `cedar-solve` implementation.
- [x] The web interface no longer shows a solver selection option.
- [x] All tests related to `tetra3` or `SolverManager` have been removed.
- [x] The application successfully performs plate solving using `cedar-solve`.
- [x] Existing project tests pass (excluding the ones that were deleted).

## Out of Scope
- Migrating `cedar-solve` to a different repository.
- Changing the core logic of `cedar-solve`.
