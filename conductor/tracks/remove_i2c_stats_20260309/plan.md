# Implementation Plan: Remove I2C and System Stats Functionality

## Phase 1: Dependency & File Cleanup [checkpoint: 52a0df5]
*Goal: Remove the physical files and external dependencies that are no longer needed.*

- [x] Task: Delete hardware modules (d9a71f8)
    - [ ] `git rm i2c.py`
    - [ ] `git rm ina219_reader.py`
- [x] Task: Update `requirements.txt` (8814d4e)
    - [ ] Remove `smbus2`
- [x] Task: Conductor - User Manual Verification 'Phase 1: Dependency & File Cleanup' (Protocol in workflow.md) (52a0df5)

## Phase 2: Backend Refactoring [checkpoint: afdc2b2]
*Goal: Remove references to the deleted modules and associated routes from the Flask application.*

- [x] Task: Refactor `app.py`
    - [x] Remove imports for `ina219_reader` and `i2c`.
    - [x] Remove `ina219` initialization and `estimate_soc` function.
    - [x] Remove the `@app.route('/system-stats')` endpoint.
    - [x] Ensure any logic that relied on these stats is safely removed or bypassed.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Backend Refactoring' (Protocol in workflow.md) (afdc2b2)

## Phase 3: Frontend & UI Cleanup
*Goal: Clean up the user interface and JavaScript logic.*

- [x] Task: Update `templates/index.html` (d200174)
    - [ ] Remove the system stats display elements (CPU, Voltage, etc.) from the footer.
- [x] Task: Update `static/main.js` (671ac05)
    - [ ] Remove the `updateSystemStats` function.
    - [ ] Remove the `setInterval` call that polls the `/system-stats` endpoint.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Frontend & UI Cleanup' (Protocol in workflow.md) (671ac05)

## Phase 4: Final Validation
*Goal: Ensure the application is stable and free of dead code.*

- [~] Task: Verify application stability
    - [ ] Run the app and ensure it starts without errors.
- [ ] Task: Check for dead references
    - [ ] Search the codebase for any remaining 'ina219', 'i2c', or 'system-stats' strings.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Validation' (Protocol in workflow.md)
