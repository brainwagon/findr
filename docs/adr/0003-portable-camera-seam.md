# A backend-neutral Camera seam, with a manager that switches cameras live

Status: accepted

The camera was reached through a picamera2-shaped seam: `camera.py` exposed
`camera_properties`, `camera_controls`, `create_still_configuration`,
`configure`, `start` and `capture_file(name=...)`, and `open_camera()` hardcoded
`Picamera2()` (the first, CSI, camera). That shape is libcamera-specific and
cannot drive a USB webcam on a laptop, which is where we want findr to run next.

We decided the app depends on a small backend-neutral interface — `Camera` —
with `properties`, `controls`, `capture_preview()`, `capture_still()`,
`set_controls()` and `close()`. One adapter per backend satisfies it:
`Picamera2Camera` (CSI cameras, two streams), `OpenCvCamera` (USB webcams and
laptop cameras, a single 640x480 stream because a UVC camera cannot provide two),
and `DummyCamera`. `camera.py` also holds `CameraDescriptor`, `list_cameras`,
`make_camera` and `open_camera_manager`.

A `CameraManager` owns the active camera and the descriptor list, mirroring
`SolverManager`: captures and control changes delegate under a lock, and
`set_camera(id)` builds the new camera **before** closing the old one, so a
failed switch leaves the working camera and the capture thread untouched. The
capture thread and the web layer never hold a Camera directly. The UI gained a
Camera dropdown (populated by `GET /cameras`, switched by `POST /set_camera`).

We rejected keeping `capture_file(name=...)`: `lores`/`main` are picamera2 stream
names, and a portable adapter would have to fake them. We rejected a
class-per-backend hierarchy with a shared base: the three adapters share no
implementation, only the interface. We rejected switching cameras by restarting
the process: the point is to compare cameras in one session.

The consequence is that the picamera2 surface is confined to `Picamera2Camera`,
and a new backend is one adapter plus a descriptor. Controls stay best-effort:
OpenCV does not expose a device's ranges, so the canonical values the UI sends
are mapped onto assumed `CAP_PROP_*` ranges and clamped; a backend that ignores a
property (macOS AVFoundation ignores most) simply leaves the camera unchanged.
On Linux, exposure and gain are instead driven through the kernel's V4L2
controls (`camera_v4l2`), because OpenCV's `CAP_PROP_EXPOSURE` is unreliable on
some UVC cameras — the Orbbec webcam used here blacks out and ignores it.
Setting exposure or gain takes the camera off auto exposure. The limits of the
hardware still show through: the same webcam ignores `exposure_time_absolute`
altogether and responds only to gain. `ScalerCrop` zoom is unavailable on USB
cameras and is dropped.
