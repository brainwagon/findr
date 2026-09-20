<p align="center">
  <img src="findr-logo.svg" alt="findr logo" width="200"/>
</p>

# findr - a prototype telescope finder

**⚠️ Warning: This project is a work in progress and a prototype. It is not fully functional or robust and should not be used in production environments. ⚠️**

This project provides a web-based interface for a Raspberry Pi-based telescope finder. It includes a live camera stream, manual camera controls, and integrated **plate solving** to automatically identify where the telescope is pointing.

## Features

- **Modular Plate Solving:** Supports both `tetra3` and its fork `cedar-solve` (the default) behind a `BaseSolver` abstraction, with runtime switching via a `SolverManager` and a dropdown in the web UI.
- **Plate Solving:** Identifies star fields and returns RA/Dec coordinates, roll, FOV, altitude/azimuth, and the constellation.
- **Star Identification:** Annotates solved images with star names (Simbad/Greek designations).
- **Constellation Boundaries:** Automatically draws constellation boundaries on solved fields using `astropy` and `pyephem`.
- **System Monitoring:** Real-time monitoring of CPU temperature, load, and power stats (via INA219 if available), including bus voltage/current, AC vs. battery source, estimated LiPo state of charge, and time remaining.
- **Web-Based Interface:** Control camera settings (gain, exposure, brightness, contrast), pause/resume the stream, view live/solve FPS, and trigger solves from any browser.
- **Test Mode:** Cycle through pre-loaded images in the `test-images/` directory to verify solver performance without live hardware.
- **Hybrid Camera Support:** Automatically uses `Picamera2` on Raspberry Pi hardware, falling back to a dummy camera for development on other platforms.

## Hardware Optimization

If you experience horizontal noise lines in your captured images (especially at high gain), it is likely power supply noise on the 3.3V rail. Adding the following line to `/boot/firmware/config.txt` (or `/boot/config.txt`) often resolves this:

```
dtparam=power_force_3v3_pwm=1
```

## Project Structure

```
.
├── app.py                  # Main Flask application and web server
├── solver.py               # BaseSolver ABC, Tetra3Solver, CedarSolver, SolverManager
├── solve.py                # Solve pipeline: ImageSource, run_solve, SolveOutcome, SolveStore
├── catalog.py              # Star names, constellation boundaries and label font
├── camera_dummy.py         # Dummy camera interface for non-Pi development
├── i2c.py                  # I2C peripheral support module
├── ina219_reader.py        # Standalone INA219 power monitor utility
├── requirements.txt        # Python dependencies (excluding system libraries)
├── REQUIREMENTS.md         # Original project requirements document
├── findr.service           # systemd unit for auto-start on boot
├── bound_20.dat            # Constellation boundary data
├── ids.csv                 # Star identification database
├── findr-logo.svg          # Project logo
├── cedar-solve/            # Git submodule: cedar-solve plate solver
├── tetra3-repo/            # Git submodule: patched tetra3 plate solver
├── conductor/              # Project planning and track documents
├── docs/                   # Research and integration notes
├── patches/                # Patches applied to the bundled solvers
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
Access the interface at `http://findr.local:8080` (or your Pi's actual hostname).

To use a different port, set the `PORT` environment variable:

```bash
PORT=9090 python3 app.py
```

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
3.  **Access:** Once running, you can find your device at `http://<your-hostname>.local:8080`.

## Auto-start on Boot (systemd)
The application is configured to start automatically on boot using **systemd**.

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
    sudo systemctl enable findr.service
    sudo systemctl start findr.service
    ```

## Plate Solver Notes
- The solver is modular: `cedar-solve` is the default, with `tetra3` available as a fallback and selectable from the UI. Both are bundled as submodules and loaded from their local repositories, so ensure the submodules are initialized.
- Each solver loads a valid star database (it will attempt to load the default one if available).
- Use **Test Mode** in the web interface to cycle through pre-loaded images in the `test-images/` directory to verify solver performance without live hardware.

## Running Tests
Unit tests for the solver abstraction live in `tests/`:

```bash
python3 -m unittest discover -s tests
```
