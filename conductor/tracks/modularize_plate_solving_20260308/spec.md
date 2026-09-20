# Specification: Modularize Plate Solving (tetra3 & cedar-solve)

## Overview
This track involves modularizing the current plate-solving logic in `findr` to support both the existing `tetra3` library and its fork, `cedar-solve`. A control in the web interface will allow users to switch between these solvers dynamically.

## Functional Requirements
- **Solver Modularity:** Refactor the plate-solving logic into a modular architecture using an Abstract Base Class (ABC). This ABC will define a common interface (e.g., `solve(image)`) that both `tetra3` and `cedar-solve` must implement.
- **`tetra3` Implementation:** Migrate existing `tetra3` logic into a concrete class implementing the Solver ABC.
- **`cedar-solve` Implementation:** Create a new concrete class for `cedar-solve`, following the migration guide (https://github.com/smroid/cedar-solve/blob/master/migrating.rst) to ensure proper integration.
- **UI Control:** Add a dropdown menu in the sidebar or header of the web interface to allow users to select their preferred solver.
- **Default State:** `tetra3` will remain the default solver upon application startup.
- **Persistence (Optional but Recommended):** The selected solver should persist across page refreshes (e.g., via session or a configuration setting).

## Non-Functional Requirements
- **Performance:** Switching solvers should be efficient and not cause significant delays in the web interface.
- **Maintainability:** The modular design should make it easy to add more solvers in the future.
- **Stability:** Existing plate-solving functionality must remain fully supported and stable for both solvers.

## Acceptance Criteria
1. The web interface displays a dropdown to switch between "tetra3" and "cedar-solve".
2. Switching the solver in the UI correctly updates the backend solver instance.
3. Plate solving is successful using the `tetra3` implementation.
4. Plate solving is successful using the `cedar-solve` implementation.
5. The application starts with `tetra3` as the default solver.
6. All existing features (camera control, streaming, etc.) continue to function normally.

## Out of Scope
- Integration of other plate solvers beyond `tetra3` and `cedar-solve`.
- Major UI redesign beyond the solver selection dropdown.
- Changes to the camera driver or MJPEG streaming logic.
