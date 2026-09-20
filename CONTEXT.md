# findr

A web-controlled telescope finder running on a Raspberry Pi. It streams the camera,
plate-solves the field to work out where the telescope is pointing, and reports the
result — coordinates, field geometry, constellation and the annotated image.

## Language

**Plate solve** (also **Solve**):
The act of identifying which region of sky a captured image shows, from its star
pattern, producing right ascension, declination, roll and field of view.
_Avoid_: "astrometric solution", "solve plate", "fix".

**Solver backend**:
A concrete star-pattern matching library that performs a Plate solve — currently
`tetra3` and `cedar-solve`. The telescope finder selects one at runtime.
_Avoid_: "engine", "algorithm", "solver" on its own.

**Solver manager**:
The single place that holds the active Solver backend and delegates solves to it.
It is not itself a Solver backend.
_Avoid_: "solver" (ambiguous with a backend), "dispatcher".

**Solve outcome**:
The complete result of one Plate solve: the coordinates, roll and field of view,
site-relative altitude and azimuth, constellation, matched-star count, the annotated
image, and — when the solve failed — the reason it failed.
_Avoid_: "solve result", "solution", "response".

**Image source**:
Where the image to be solved comes from: the live camera, or the pre-loaded
test-image set used in Test mode.
_Avoid_: "capture", "frame provider".

**Camera**:
A device findr can capture from, seen through the backend-neutral `Camera`
interface: a preview JPEG, a still JPEG, a device description and adjustable
controls. Concrete adapters wrap picamera2 (CSI cameras on a Raspberry Pi) or
OpenCV (USB webcams and laptop cameras).
_Avoid_: "webcam" (one kind of camera), "sensor" (the INA219 is also a sensor).

**Camera backend**:
The library a Camera adapter is built on: libcamera via picamera2 for CSI
cameras, or OpenCV for USB and laptop cameras.
_Avoid_: "driver".

**Camera manager**:
The single place that holds the active Camera and the cameras the machine
offers, and that switches between them. It is not itself a Camera.
_Avoid_: "camera" (ambiguous with the device), "dispatcher".

**Catalog**:
The reference data used to label a solved field: star identifiers, constellation
boundaries and the label font. Distinct from a Solver backend's own star database,
which it uses to match the pattern.
_Avoid_: "star database", "database" (ambiguous with the Solver backend's data).

**Overlay**:
The star labels and constellation boundaries drawn onto a solved image.
_Avoid_: annotation, markup, layer.

**Solve store**:
The single place that holds the status of the current solve and its Solve outcome,
so that the solver thread and the web request that reads it do not disagree.
_Avoid_: "solver state", "globals".

**Power reading**:
A snapshot of the finder's power supply: bus voltage and current, whether it is on
AC or battery, the estimated state of charge and time remaining, and whether the
voltage is low.
_Avoid_: "power stats", "battery info".

**Test mode**:
A mode in which the Image source is the pre-loaded test-image set rather than the
live camera, so the Solver backends can be exercised without hardware.
