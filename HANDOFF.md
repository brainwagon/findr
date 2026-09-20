# Handoff — findr

**Date:** 2026-09-19
**Branch:** `main` (in sync with `origin/main` at `ebca801`)
**State:** All work below is **uncommitted** in the working tree. No commits were made this session.

## Summary of this session

Housekeeping, documentation, and small usability fixes. No changes to plate-solving
logic or the camera pipeline.

| File | Change |
| --- | --- |
| `README.md` | Brought up to date: modular `tetra3`/`cedar-solve` solver, Test Mode, FPS/pause, expanded power monitoring, full project tree, recursive submodule clone, solver notes, test command, `PORT` docs. |
| `app.py` | Port is now configurable: `port = int(os.environ.get("PORT", 8080))` (`app.py:832-833`). |
| `camera_dummy.py` | Added no-op `close()` so shutdown `cleanup()` no longer raises `AttributeError`. |
| `static/main.js` | Dark mode now defaults to enabled when no preference is stored (`main.js:442-445`). |
| `templates/index.html` | Inline script applies `dark-mode` before render to avoid a light flash. |

## Verification performed

- `PORT=9090 venv/bin/python app.py` → `GET /` returned HTTP 200.
- `PORT=9091 venv/bin/python app.py` → served HTML contains the dark-mode inline script.
- `python3 -c "from camera_dummy import Picamera2; Picamera2().close()"` → clean.
- Dark-mode toggle logic traced for all four states (first visit / toggle off / reload / toggle on); persisted via `localStorage.darkMode`.

## Environment / how to run

- WSL2, `picamera2` is **not** installed, so the app automatically uses
  `camera_dummy.Picamera2` (grey frame with timestamp). Plate solving only works
  with **Test Mode** enabled (solves against `test-images/`).
- Run: `source venv/bin/activate && python3 app.py`, then `http://localhost:8080`.
- Alternate port: `PORT=9090 python3 app.py`.
- Solver default is `cedar-solve` (`solver.py:119`), falling back to `tetra3` if its
  database fails to load. Both are git submodules.

## Open items / suggested next steps

- **Commit the above changes** (none are committed yet).
- `venv/` shows as untracked and is **not** in `.gitignore` — add it (along with
  `__pycache__/`, already present) before committing to avoid accidental adds.
- `cedar-solve` and `tetra3-repo` submodules show as modified (`m`); confirm whether
  that is intended pinned content or drift before committing.
- README's dev-machine setup tells non-Pi users to `pip install -r requirements.txt`,
  but that file lists legacy `picamera`, which does not install cleanly off-Pi.
  Consider noting that `picamera2`/`picamera` are Pi-only and the dummy camera covers
  dev use.
- `app.py:232 point_stellarium()` is defined but never called — wire up or remove.
- No test run was attempted (solver tests need a star database); `python3 -m unittest
  discover -s tests` is documented but unverified in this environment.
