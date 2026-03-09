# Implementation Plan: Modularize Plate Solving

## Phase 1: Foundation & Modularization (ABC & tetra3)
*Goal: Define a common interface and refactor existing tetra3 logic into a concrete class.*

- [ ] Task: Create `BaseSolver` Abstract Base Class in `solver.py`
    - [ ] Create `tests/test_base_solver.py` with failing tests for the interface.
    - [ ] Define `BaseSolver` with an abstract method `solve(image)`.
    - [ ] Verify tests pass for the interface definition.
- [ ] Task: Refactor `tetra3` logic into `Tetra3Solver` class
    - [ ] Create `tests/test_tetra3_solver.py` with failing tests for the `tetra3` solver.
    - [ ] Implement `Tetra3Solver` as a subclass of `BaseSolver`.
    - [ ] Verify tests pass and coverage > 80%.
- [ ] Task: Implement `SolverManager` to handle solver switching
    - [ ] Create `tests/test_solver_manager.py` with failing tests for initialization, switching, and default solver.
    - [ ] Implement `SolverManager` with methods `set_solver(type)`, `get_solver()`, and `solve(image)`.
    - [ ] Update `get_solver()` singleton to use `SolverManager`.
    - [ ] Verify tests pass and coverage > 80%.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Foundation & Modularization' (Protocol in workflow.md)

## Phase 2: cedar-solve Integration
*Goal: Add cedar-solve as an alternative solver.*

- [ ] Task: Install `cedar-solve` and dependencies
    - [ ] Add `cedar-solve` to `requirements.txt`.
    - [ ] Verify installation in development environment.
- [ ] Task: Implement `CedarSolver` concrete class
    - [ ] Create `tests/test_cedar_solver.py` with failing tests (mocking `cedar-solve` if necessary).
    - [ ] Implement `CedarSolver` as a subclass of `BaseSolver` following the migration guide.
    - [ ] Verify tests pass and coverage > 80%.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: cedar-solve Integration' (Protocol in workflow.md)

## Phase 3: Web Interface Update
*Goal: Add a UI control to switch solvers.*

- [ ] Task: Update Backend API to support solver switching
    - [ ] Add a new endpoint `/api/v1/solver` (POST) to set the active solver.
    - [ ] Create `tests/test_app_solver_api.py` with failing tests for the new endpoint.
    - [ ] Implement the endpoint in `app.py`.
    - [ ] Verify tests pass.
- [ ] Task: Update UI to include solver dropdown
    - [ ] Add the solver selection dropdown in `templates/index.html` (sidebar or header).
    - [ ] Implement JavaScript in `static/main.js` to call the backend API when the selection changes.
    - [ ] Ensure the UI reflects the current active solver on page load.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Web Interface Update' (Protocol in workflow.md)

## Phase 4: Final Integration & Validation
*Goal: Ensure everything works together perfectly.*

- [ ] Task: End-to-end testing with both solvers
    - [ ] Perform manual testing of solver switching and solving in the web interface.
    - [ ] Verify that `tetra3` remains the default and persists as expected.
- [ ] Task: Update documentation
    - [ ] Update `README.md` or other documentation to reflect the new solver options.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Integration & Validation' (Protocol in workflow.md)
