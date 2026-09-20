<p align="center">
  <img src="findr-logo.svg" alt="findr logo" width="200"/>
</p>

# findr - a prototype telescope finder

**⚠️ Warning: This project is a work in progress and a prototype. It is not fully functional or robust and should not be used in production environments. ⚠️**

This project provides a web-based interface for a Raspberry Pi-based telescope finder. It includes a live camera stream, manual camera controls, and integrated **plate solving** to automatically identify where the telescope is pointing.

## Features

- **Modular Plate Solving:** Supports `tetra3`, its fork `cedar-solve` (the default), and the Rust `olive-solve` behind a `BaseSolver` abstraction, with runtime switching via a `SolverManager` and a dropdown in the web UI.
- **Plate Solving:** Identifies star fields and returns RA/Dec coordinates, roll, FOV, altitude/azimuth, and the constellation.
- **Star Identification:** Annotates solved images with star names (Simbad/Greek designations).
- **Constellation Boundaries:** Draws constellation boundaries on solved fields using `astropy` and `pyephem`. The Overlay and Boundaries annotations can each be toggled from the UI; with boundaries off the (expensive) boundary drawing is skipped at solve time.
- **System Monitoring:** Real-time monitoring of CPU temperature, load, and power stats (via INA219 if available), including bus voltage/current, AC vs. battery source, estimated LiPo state of charge, and time remaining.
- **Web-Based Interface:** Control camera settings (gain, exposure, brightness, contrast), pause/resume the stream, view live/solve FPS, and trigger solves from any browser.
- **Test Mode:** Cycle through pre-loaded images in the `test-images/` directory to verify solver performance without live hardware.
- **Portable Camera Support:** Cameras are driven through a backend-neutral interface with adapters for libcamera/picamera2 (Raspberry Pi CSI cameras) and OpenCV (USB webcams and laptop cameras), falling back to a dummy camera on a machine with no camera.
- **Live Camera Selection:** A dropdown lists the cameras the machine offers (CSI and USB) and switches between them without restarting.

## Hardware

The 3D-printed prototype hardware — enclosure, camera head and dovetail clamp — is
modelled in Onshape. The current assembly lives in the `findr2` document:

- **Onshape project:** <https://cad.onshape.com/documents/aac97b29bc57f6d0845ce97c/w/6b16a86d7e2782d3bee433d0/e/28e892ec8534f3e90436015f>

## Hardware Optimization

If you experience horizontal noise lines in your captured images (especially at high gain), it is likely power supply noise on the 3.3V rail. Adding the following line to `/boot/firmware/config.txt` (or `/boot/config.txt`) often resolves this:

```
dtparam=power_force_3v3_pwm=1
```

## Project Structure

```
.
├── app.py                  # Main Flask application and web server
├── solver.py               # Solver backends (SolverBackend, LibrarySolver, OliveSolver, SolverResult) and SolverManager
├── solve.py                # Solve pipeline: ImageSource, run_solve, SolveOutcome, SolveStore
├── overlay.py              # Draws star labels and constellation boundaries on solved images
├── catalog.py              # Star names, constellation boundaries and label font
├── camera.py               # Camera interface, descriptors, enumeration and selection
├── camera_picamera2.py     # Camera adapter over libcamera/picamera2 (CSI cameras)
├── camera_opencv.py        # Camera adapter over OpenCV (USB/laptop cameras)
├── camera_v4l2.py          # Direct V4L2 control access for USB cameras on Linux
├── camera_dummy.py         # Dummy camera for development without hardware
├── camera_manager.py       # Holds the active camera and switches it live
├── power.py                # INA219 driver, state of charge and PowerReading
├── ina219_reader.py        # Standalone INA219 power monitor utility
├── benchmark.py            # Times each Solver backend over the test frames
├── Makefile                # Build, test and deploy targets
├── requirements.txt        # Python dependencies (excluding system libraries)
├── REQUIREMENTS.md         # Original project requirements document
├── findr.service           # systemd unit for auto-start on boot
├── bound_20.dat            # Constellation boundary data
├── ids.csv                 # Star identification database
├── findr-logo.svg          # Project logo
├── cedar-solve/            # Git submodule: cedar-solve plate solver
├── tetra3-repo/            # Git submodule: pristine tetra3 plate solver
├── olive-solve/            # Git submodule: Rust olive-solve plate solver
├── conductor/              # Project planning and track documents
├── docs/                   # Research notes, agent docs and Architecture Decision Records
├── tests/                  # Unit tests for the solver modules
├── test-images/            # Sample images used by Test Mode
├── static/                 # CSS and JavaScript assets
└── templates/              # HTML templates (Flask)
```

## Installation and Usage

### 1. Clone the Repository
The plate solvers are included as git submodules, so clone recursively:

```bash
git clone --recursive <repository-url>
cd findr
```

If you already cloned without `--recursive`, initialize the submodules:

```bash
git submodule update --init --recursive
```

The solver star databases are stored with [Git LFS](https://git-lfs.com/). Install
`git-lfs` and run `git lfs install` **before** cloning, or the submodules will check
out pointer files instead of the databases and no backend will load. If you
already cloned without it, run `git lfs install && git lfs pull` in each submodule.

```bash
sudo apt install git-lfs
git lfs install
```

### 2. Environment Setup

#### On a Raspberry Pi (Recommended)
To use the actual camera hardware, you must allow the virtual environment to access system-site packages (where `libcamera` and `picamera2` are typically installed).

```bash
# Create venv with system site packages
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### On a Development Machine (Non-Pi)
If you are just working on the UI or solver logic using test images:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python3 app.py
```
The app listens on port **8080**. On the Pi deployment it is fronted by Caddy on
port **80**, so there it is reachable at `http://findr.local/` (see
[Deploying Updates](#deploying-updates)).

To use a different port, set the `PORT` environment variable:

```bash
PORT=9090 python3 app.py
```

## Cameras

findr talks to cameras through a small backend-neutral interface, with one
adapter per backend:

- **libcamera / picamera2** drives the Raspberry Pi CSI camera (a low-resolution
  preview stream and a full-resolution still).
- **OpenCV** drives USB webcams and laptop cameras. A UVC camera cannot give two
  streams, so a single 640x480 stream serves both the preview and the still.
- **Dummy** renders a timestamped placeholder so the app runs on a machine with
  no camera.

### Choosing a camera

On startup findr opens the first real camera it finds. Set `FINDR_CAMERA` to a
camera id to choose a different one:

```bash
FINDR_CAMERA=opencv:200:8 python3 app.py
```

The **Camera** dropdown in Expert Mode lists the cameras the machine offers and
switches between them live, without a restart. `GET /cameras` returns the
current camera and the available list; `POST /set_camera` with
`{"camera": "<id>"}` switches.

### Dependencies and platform notes

Camera capture uses OpenCV (`opencv-python-headless`) plus
`cv2-enumerate-cameras` for enumeration. On Raspberry Pi OS the system
`python3-opencv` package already provides `cv2` (the venv is created with
`--system-site-packages`), so the pip wheel is skipped there.

- **Controls are best-effort off the Pi.** On Linux, exposure and gain are
  driven through the kernel's V4L2 controls (`camera_v4l2`), because OpenCV's
  `CAP_PROP_EXPOSURE` is unreliable on some UVC cameras; setting either takes
  the camera off auto exposure. Elsewhere the adapter falls back to
  `CAP_PROP_*`. Brightness, contrast and sharpness are mapped on both paths.
  Note that some webcams ignore `exposure_time_absolute` entirely — the Orbbec
  webcam used here responds to gain but not exposure. Zoom (`ScalerCrop`) is not
  available on USB cameras and is ignored.
- **macOS** prompts for camera permission the first time; grant it to the
  terminal or the Python process running findr.
- **WSL2** cannot see cameras without USB/IP passthrough, so development there
  uses the dummy camera.

## Network Access (mDNS/Avahi)
This project is configured to work with **Avahi/mDNS**, allowing you to access the web interface using a friendly hostname instead of an IP address. 

To ensure this works:
1.  **Install Avahi** (if not already installed):
    ```bash
    sudo apt update
    sudo apt install avahi-daemon
    ```
2.  **Verify Service Status:**
    ```bash
    sudo systemctl enable --now avahi-daemon
    ```
3.  **Access:** Once running, you can find your device at `http://<your-hostname>.local:8080` (or `http://<your-hostname>.local/` on the Pi, where Caddy serves port 80).

## Auto-start on Boot (systemd)
The application is configured to start automatically on boot using **systemd**.

The bundled `findr.service` runs as `markv` from `/home/markv/findr`, with
`SupplementaryGroups=video i2c gpio spi` so the process can reach the camera and
the I2C bus. Adjust the user, paths and groups for your own deployment.

### Management Commands:
- **Check Status:** `sudo systemctl status findr.service`
- **Restart Service:** `sudo systemctl restart findr.service`
- **Stop Service:** `sudo systemctl stop findr.service`
- **View Logs:** `journalctl -u findr.service -f`
- **Disable Auto-start:** `sudo systemctl disable findr.service`

### Manual Installation (if needed):
1.  Copy `findr.service` to `/etc/systemd/system/`:
    ```bash
    sudo cp findr.service /etc/systemd/system/findr.service
    ```
2.  Reload systemd and enable:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable --now findr.service
    ```

## Deploying Updates

The Pi runs the app from `/home/markv/findr` under systemd. Push a new build
from your development machine with the `Makefile`:

```bash
make deploy        # rsync the working tree to the Pi, then restart the service
```

If `requirements.txt` changed, install them on the Pi first:

```bash
make requirements
make deploy
```

### Make targets:
- **Deploy:** `make deploy` (rsync + restart)
- **Sync only:** `make sync`
- **Restart:** `make restart`
- **Status:** `make status`
- **Logs:** `make logs` (follows the journal)
- **Tests:** `make test`
- **Build olive-solve:** `make olive` (builds the Rust extension into the venv)

The deploy defaults to `markv@findr.local:/home/markv/findr`; override with
`make deploy HOST=other.local USER=pi REMOTE_DIR=/home/pi/findr`. Privileged
targets use `ssh -t` so sudo can prompt for your password.

Note: `make deploy` copies whatever is in the working tree — including
uncommitted changes and submodule contents — and deletes files on the Pi that no
longer exist locally. The venv, `.git`, `olive-solve/` (installed as a wheel, not
synced) and agent tooling are excluded.

## Plate Solver Notes
- The solver is modular: `cedar-solve` is the default, with `tetra3` available as a fallback and selectable from the UI. All three backends are bundled as submodules and loaded from their local repositories, so ensure the submodules are initialized.
- `olive-solve`, a Rust port of the cedar-solve algorithms, is a third backend when its Python extension is built. It is much faster (see the benchmark below) and is registered automatically when importable.
- Each solver loads a valid star database (it will attempt to load the default one if available).
- Use **Test Mode** in the web interface to cycle through pre-loaded images in the `test-images/` directory to verify solver performance without live hardware.

### Building olive-solve
`olive-solve` ships as source in the `olive-solve/` submodule. Build its Python
extension into the active venv with:

```bash
make olive          # builds from source with maturin (needs a Rust toolchain)
```

On the Raspberry Pi there is no Rust toolchain, so install the prebuilt aarch64
wheel from the submodule instead (one-time):

```bash
scp olive-solve/dist/olive_solve-*-aarch64.whl findr.local:/tmp/
ssh findr.local '/home/markv/findr/venv/bin/pip install --no-deps /tmp/olive_solve-*-aarch64.whl'
```

## Solver Benchmark

`benchmark.py` times each Solver backend over every frame in `test-images/`:

```bash
python3 benchmark.py            # 3 solves per frame, after a warmup solve
python3 benchmark.py --repeat 5
```

Per-frame times are means in milliseconds; **Database load** is the one-time cost
of loading the backend's star database.

### Development machine (x86_64)

**Machine:** Intel(R) Core(TM) i7-14700F · 28 threads · 23.5 GiB RAM · Ubuntu 22.04.5 LTS (WSL2) · Python 3.10.12

| Frame | tetra3 (ms) | cedar-solve (ms) | olive-solve (ms) |
| --- | ---: | ---: | ---: |
| `lores_jpeg_2025-11-07T01_59_03.175Z.jpg` | 17.8 | 23.0 | 1.6 |
| `lores_jpeg_2025-11-07T02_23_17.329Z.jpg` | 11.1 | 17.1 | 1.6 |
| `lores_jpeg_2025-11-07T02_53_07.536Z.jpg` | 10.4 | 14.2 | 1.8 |
| `lores_jpeg_2025-11-07T02_59_58.735Z.jpg` | 8.9 | 11.9 | 1.7 |
| `lores_jpeg_2025-11-07T03_03_01.032Z.jpg` | 9.7 | 11.8 | 1.7 |
| `lores_jpeg_2025-11-07T03_03_46.674Z.jpg` | 9.7 | 12.3 | 2.0 |
| `lores_jpeg_2025-11-07T03_09_00.164Z.jpg` | 11.1 | 12.5 | 1.9 |
| `lores_jpeg_2025-11-07T03_45_55.786Z.jpg` | 65.2 | 156.4 | 1.5 |
| **Total** | **144** | **259** | **14** |
| **Mean** | **18.0** | **32.4** | **1.7** |
| **Database load** | 0.36 s | 0.13 s | 0.14 s |
| **Frames solved** | 7/8 | 8/8 | 8/8 |

`olive-solve` is roughly 10× faster than `tetra3` and 19× faster than
`cedar-solve`. The last frame is the hardest: `tetra3` fails to solve it (65 ms
to give up), while `cedar-solve` takes 156 ms — but `olive-solve` solves it in
1.5 ms, and solves every frame in under 2 ms.

### Raspberry Pi 5 (`findr.local`)

**Machine:** Raspberry Pi 5 Model B Rev 1.0 · 4 threads · 4.0 GiB RAM · Debian GNU/Linux 13 (trixie) · Python 3.13.5

| Frame | tetra3 (ms) | cedar-solve (ms) | olive-solve (ms) |
| --- | ---: | ---: | ---: |
| `lores_jpeg_2025-11-07T01_59_03.175Z.jpg` | 52.9 | 71.5 | 5.1 |
| `lores_jpeg_2025-11-07T02_23_17.329Z.jpg` | 29.8 | 50.3 | 5.5 |
| `lores_jpeg_2025-11-07T02_53_07.536Z.jpg` | 28.8 | 39.1 | 5.1 |
| `lores_jpeg_2025-11-07T02_59_58.735Z.jpg` | 24.7 | 31.0 | 5.3 |
| `lores_jpeg_2025-11-07T03_03_01.032Z.jpg` | 27.2 | 34.2 | 5.3 |
| `lores_jpeg_2025-11-07T03_03_46.674Z.jpg` | 28.2 | 34.3 | 5.6 |
| `lores_jpeg_2025-11-07T03_09_00.164Z.jpg` | 32.4 | 36.3 | 5.3 |
| `lores_jpeg_2025-11-07T03_45_55.786Z.jpg` | 188.1 | 523.5 | 4.8 |
| **Total** | **412** | **820** | **42** |
| **Mean** | **51.5** | **102.5** | **5.2** |
| **Database load** | 0.49 s | 0.19 s | 0.34 s |
| **Frames solved** | 7/8 | 8/8 | 8/8 |

The same pattern holds on the device: `olive-solve` is about 10× faster than
`tetra3` and 20× faster than `cedar-solve`, solving every frame — including the
hard last one — in about 5 ms.

## Running Tests
Unit tests for the solver abstraction live in `tests/`:

```bash
python3 -m unittest discover -s tests
```
