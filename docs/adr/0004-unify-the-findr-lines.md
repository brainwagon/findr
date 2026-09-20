# The findr lines are unified on the September trunk

Status: accepted

Two lines of work diverged from the same March commit and never recombined. The
**March line** (published as `main`) removed tetra3, removed the I2C and power
monitoring, and promoted a single `CedarSolver` behind a monolithic
`static/main.js`. The **September line** kept and deepened the multi-backend
solve pipeline (ADR 0002), kept the hardware modules, split the frontend into ES
modules, and went on to add a third backend (`olive-solve`) and a backend-neutral
camera seam (ADR 0003). Each line held work the other lacked, so neither could
simply be discarded.

We decided the **September line is the trunk**, and that the March line
contributes only its two performance commits (the persistent solve worker and
frame-age tracking) rather than competing on architecture. The three preserved
backup branches (`backup/wsl-line`, `backup/pi-line`, `backup/pi-camera`) remain
as the record of both lines.

The four fork points and their resolutions:

1. **Solver backends** — keep all three (tetra3, cedar-solve, olive-solve) behind
   the `SolverBackend` descriptor seam. The comparison ability is the point of
   the seam; olive-solve is the newest backend and the reason the seam exists.
2. **I2C / power / system stats** — keep them. The INA219 hardware is still in
   use, so March's removal does not apply.
3. **Frontend** — keep the ES modules in `static/js/`, and port March's
   persistent solve worker and frame-age tracking into them rather than
   reverting to a single `static/main.js`.
4. **Camera** — adopt the backend-neutral `Camera` seam and `CameraManager`
   (ADR 0003), so findr can run on USB and laptop cameras, not only the CSI
   Pi camera.

We rejected rebasing the September commits onto the March line: the two lines
encode opposite decisions, so a replay would surface the same conflicts at every
commit instead of once. We rejected discarding the March line entirely, because
its two performance commits are real work with no September equivalent.

The consequence is that the March removals (tetra3, the hardware modules) are
deliberately reverted in the unified tree, and the March line survives in history
and in `backup/pi-line`. Adding a fourth backend is one descriptor plus, where
the API differs, one adapter.
