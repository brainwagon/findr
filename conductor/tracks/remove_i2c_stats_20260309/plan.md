# Implementation Plan: Remove I2C and System Stats Functionality

## Phase 1: Dependency & File Cleanup
*Goal: Remove the physical files and external dependencies that are no longer needed.*

- [x] Task: Delete hardware modules (d9a71f8)
    - [ ] `git rm i2c.py`
    - [ ] `git rm ina219_reader.py`
- [x] Task: Update `requirements.txt` (8814d4e)
    - [ ] Remove `smbus2`
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Dependency & File Cleanup' (Protocol in workflow.md)

## Phase 2: Backend Refactoring
*Goal: Remove references to the deleted modules and associated routes from the Flask application.*

- [ ] Task: Refactor `app.py`
    - [ ] Remove imports for `ina219_reader` and `i2c`.
    - [ ] Remove `ina219` initialization and `estimate_soc` function.
    - [ ] Remove the `@app.route('/system-stats')` endpoint.
    - [ ] Ensure any logic that relied on these stats is safely removed or bypassed.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Backend Refactoring' (Protocol in workflow.md)

## Phase 3: Frontend & UI Cleanup
*Goal: Clean up the user interface and JavaScript logic.*

- [ ] Task: Update `templates/index.html`
    - [ ] Remove the system stats display elements (CPU, Voltage, etc.) from the footer.
- [ ] Task: Update `static/main.js`
    - [ ] Remove the `updateSystemStats` function.
    - [ ] Remove the `setInterval` call that polls the `/system-stats` endpoint.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Frontend & UI Cleanup' (Protocol in workflow.md)

## Phase 4: Final Validation
*Goal: Ensure the application is stable and free of dead code.*

- [ ] Task: Verify application stability
    - [ ] Run the app and ensure it starts without errors.
- [ ] Task: Check for dead references
    - [ ] Search the codebase for any remaining 'ina219', 'i2c', or 'system-stats' strings.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Validation' (Protocol in workflow.md)
