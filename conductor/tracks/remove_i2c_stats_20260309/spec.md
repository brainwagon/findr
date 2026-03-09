# Specification: Remove I2C and System Stats Functionality

## Overview
This track involves stripping out the I2C-based hardware monitoring (specifically the INA219 power sensor) and the associated "System Stats" display from the web interface. This will simplify the codebase and remove unnecessary hardware dependencies.

## Functional Requirements
1. **Remove Hardware Modules:**
   - Delete `i2c.py` (low-level I2C communication).
   - Delete `ina219_reader.py` (INA219 sensor logic).
2. **Refactor Backend (app.py):**
   - Remove imports and initialization logic for `ina219`.
   - Remove the `system_stats` route and the `estimate_soc` helper function.
   - Remove the `updateSystemStats` logic if any remains in the backend (e.g., polling threads).
3. **Clean Up UI:**
   - Remove the "System Stats" section from the footer in `templates/index.html`.
   - Remove the `updateSystemStats` function and its associated `setInterval` call in `static/main.js`.
4. **Dependency Management:**
   - Remove `smbus2` from `requirements.txt`.

## Non-Functional Requirements
- **Simplicity:** Reduce the number of background threads and active network requests.
- **Maintainability:** Eliminate dead code related to non-core functionality.

## Acceptance Criteria
- [ ] `i2c.py` and `ina219_reader.py` are deleted.
- [ ] `app.py` starts without errors and has no references to the deleted modules.
- [ ] The web interface no longer displays CPU temperature, load, voltage, or current.
- [ ] No background requests are made to `/system-stats`.
- [ ] `smbus2` is no longer listed in `requirements.txt`.

## Out of Scope
- Removal of other hardware controls (camera parameters).
- Modification of plate-solving logic.
