# Implementation Plan: Remove tetra3 and Scaffolding

## Phase 1: Dependency Cleanup [checkpoint: 3800987]
*Goal: Remove all physical and git-tracked files related to the old solver.*

- [x] Task: Remove `tetra3-repo` submodule (9b8636e)
    - [ ] `git submodule deinit -f tetra3-repo`
    - [ ] `rm -rf .git/modules/tetra3-repo`
    - [ ] `git rm -f tetra3-repo`
- [x] Task: Delete legacy patches and documentation (fbc5148)
    - [ ] Delete `patches/tetra3_numpy2.patch`
    - [ ] Delete `docs/tetra3_patch.md`
    - [ ] Delete `docs/research_tetra3.md` (if relevant)
- [x] Task: Delete legacy tests (4b8d6ef)
    - [ ] Delete `tests/test_tetra3_solver.py`
    - [ ] Delete `tests/test_solver_manager.py`
- [x] Task: Conductor - User Manual Verification 'Phase 1: Dependency Cleanup' (Protocol in workflow.md) (3800987)

## Phase 2: Backend Refactoring [checkpoint: 011fc8d]
*Goal: Simplify solver.py to use CedarSolver directly.*

- [x] Task: Refactor `solver.py` architecture (acda3aa)
    - [ ] Remove `Tetra3Solver` class.
    - [ ] Remove `SolverManager` class.
    - [ ] Promote `CedarSolver` or integrate its logic directly as the primary solver.
    - [ ] Update `get_solver()` and `PlateSolver` alias to use `CedarSolver` logic without the manager.
- [x] Task: Update `requirements.txt` (acda3aa)
    - [ ] Ensure `tetra3` (if listed as a standalone) is removed.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Backend Refactoring' (Protocol in workflow.md) (011fc8d)

## Phase 3: Frontend & API Cleanup
*Goal: Remove UI elements and endpoints related to switching solvers.*

- [~] Task: Clean up UI templates
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
