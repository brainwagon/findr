# Implementation Plan: Remove tetra3 and Scaffolding

## Phase 1: Dependency Cleanup
*Goal: Remove all physical and git-tracked files related to the old solver.*

- [ ] Task: Remove `tetra3-repo` submodule
    - [ ] `git submodule deinit -f tetra3-repo`
    - [ ] `rm -rf .git/modules/tetra3-repo`
    - [ ] `git rm -f tetra3-repo`
- [ ] Task: Delete legacy patches and documentation
    - [ ] Delete `patches/tetra3_numpy2.patch`
    - [ ] Delete `docs/tetra3_patch.md`
    - [ ] Delete `docs/research_tetra3.md` (if relevant)
- [ ] Task: Delete legacy tests
    - [ ] Delete `tests/test_tetra3_solver.py`
    - [ ] Delete `tests/test_solver_manager.py`
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Dependency Cleanup' (Protocol in workflow.md)

## Phase 2: Backend Refactoring
*Goal: Simplify solver.py to use CedarSolver directly.*

- [ ] Task: Refactor `solver.py` architecture
    - [ ] Remove `Tetra3Solver` class.
    - [ ] Remove `SolverManager` class.
    - [ ] Promote `CedarSolver` or integrate its logic directly as the primary solver.
    - [ ] Update `get_solver()` and `PlateSolver` alias to use `CedarSolver` logic without the manager.
- [ ] Task: Update `requirements.txt`
    - [ ] Ensure `tetra3` (if listed as a standalone) is removed.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Backend Refactoring' (Protocol in workflow.md)

## Phase 3: Frontend & API Cleanup
*Goal: Remove UI elements and endpoints related to switching solvers.*

- [ ] Task: Clean up UI templates
    - [ ] Remove the solver dropdown from `templates/index.html`.
- [ ] Task: Clean up static assets
    - [ ] Remove JavaScript in `static/main.js` that handles solver selection.
- [ ] Task: Clean up Flask endpoints
    - [ ] Remove any solver switching routes (e.g., `/api/v1/solver` POST) from `app.py`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Frontend & API Cleanup' (Protocol in workflow.md)

## Phase 4: Final Validation
*Goal: Verify single-solver functionality and clean workspace.*

- [ ] Task: Verify existing tests
    - [ ] Run all tests to ensure no regressions: `python3 -m unittest discover tests`
- [ ] Task: End-to-end verification
    - [ ] Manually test plate solving functionality in the web interface.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Validation' (Protocol in workflow.md)
