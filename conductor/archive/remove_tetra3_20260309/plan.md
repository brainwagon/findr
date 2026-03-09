# Implementation Plan: Remove tetra3 and Scaffolding

## Phase 1: Dependency Cleanup [checkpoint: 3800987]
*Goal: Remove all physical and git-tracked files related to the old solver.*

- [x] Task: Remove `tetra3-repo` submodule (9b8636e)
    - [x] `git submodule deinit -f tetra3-repo`
    - [x] `rm -rf .git/modules/tetra3-repo`
    - [x] `git rm -f tetra3-repo`
- [x] Task: Delete legacy patches and documentation (fbc5148)
    - [x] Delete `patches/tetra3_numpy2.patch`
    - [x] Delete `docs/tetra3_patch.md`
    - [x] Delete `docs/research_tetra3.md` (if relevant)
- [x] Task: Delete legacy tests (4b8d6ef)
    - [x] Delete `tests/test_tetra3_solver.py`
    - [x] Delete `tests/test_solver_manager.py`
- [x] Task: Conductor - User Manual Verification 'Phase 1: Dependency Cleanup' (Protocol in workflow.md) (3800987)

## Phase 2: Backend Refactoring [checkpoint: 011fc8d]
*Goal: Simplify solver.py to use CedarSolver directly.*

- [x] Task: Refactor `solver.py` architecture (acda3aa)
    - [x] Remove `Tetra3Solver` class.
    - [x] Remove `SolverManager` class.
    - [x] Promote `CedarSolver` or integrate its logic directly as the primary solver.
    - [x] Update `get_solver()` and `PlateSolver` alias to use `CedarSolver` logic without the manager.
- [x] Task: Update `requirements.txt` (acda3aa)
    - [x] Ensure `tetra3` (if listed as a standalone) is removed.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Backend Refactoring' (Protocol in workflow.md) (011fc8d)

## Phase 3: Frontend & API Cleanup [checkpoint: 4a95304]
*Goal: Remove UI elements and endpoints related to switching solvers.*

- [x] Task: Clean up UI templates (4165e68)
    - [x] Remove the solver dropdown from `templates/index.html`.
- [x] Task: Clean up static assets (6d777c4)
    - [x] Remove JavaScript in `static/main.js` that handles solver selection.
- [x] Task: Clean up Flask endpoints (de4310b)
    - [x] Remove any solver switching routes (e.g., `/api/v1/solver` POST) from `app.py`.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Frontend & API Cleanup' (Protocol in workflow.md) (4a95304)

## Phase 4: Final Validation [checkpoint: 4a95304]
*Goal: Verify single-solver functionality and clean workspace.*

- [x] Task: Verify existing tests (4a95304)
    - [x] Run all tests to ensure no regressions: `python3 -m unittest discover tests`
- [x] Task: End-to-end verification (4a95304)
    - [x] Manually test plate solving functionality in the web interface.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Validation' (Protocol in workflow.md) (4a95304)
