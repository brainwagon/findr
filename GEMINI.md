# findr: Prototype Telescope Finder

`findr` is a prototype telescope finder and remote camera control system built for the Raspberry Pi 5. It provides a web-based interface for live monitoring, camera adjustment, and integrated plate solving to identify the telescope's orientation in the sky.

## Project Overview

- **Purpose:** High-performance, low-latency remote telescope camera control with automated plate solving (RA/Dec identification).
- **Target Platform:** Raspberry Pi 5 with Camera Module 3.
- **Main Technologies:**
    - **Backend:** Python 3 (Flask), `picamera2`, `tetra3`, `cedar-solve`.
    - **Astronomy:** `astropy`, `pyephem` (star mapping and constellation identification).
    - **Frontend:** Vanilla HTML/CSS/JavaScript (Minimalist, MJPEG streaming).
    - **Hardware:** `smbus2` for I2C (INA219 power monitoring).
- **Architecture:** 
    - **Web Server:** Flask-based server providing MJPEG stream and control API.
    - **Solver Layer:** Modular `SolverManager` (`solver.py`) supporting multiple plate solving backends (default: `tetra3`).
    - **Camera Layer:** Abstraction for `Picamera2` with a fallback `DummyCamera` for local development.

## Building and Running

### Installation
```bash
# Recommendation: Use system site packages for libcamera/picamera2 access on Pi
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Application
```bash
python3 app.py
```
The application runs on `http://0.0.0.0:8080`.

### Testing
```bash
# Run all tests
python3 -m unittest discover tests
# Or using pytest if installed
pytest
```

## Development Conventions

- **Frontend:** Adhere to "No Framework" rule. Use Vanilla CSS and standard HTML elements for controls.
- **Hardware Fallback:** Always maintain `camera_dummy.py` compatibility for development on non-Pi systems.
- **Coding Style:** Follow PEP 8 for Python code.
- **Solver Extensions:** To add a new solver backend, inherit from `BaseSolver` in `solver.py` and register it with `SolverManager`.
- **Numpy Compatibility:** Avoid downgrading `numpy`. If libraries like `tetra3` require it, apply patches (see `docs/tetra3_patch.md`).

## Key Files
- `app.py`: Main Flask application, frame capture loop, and API endpoints.
- `solver.py`: Core plate-solving abstraction and manager.
- `camera_dummy.py`: Mock camera implementation for development.
- `i2c.py` & `ina219_reader.py`: Hardware sensor interfaces.
- `templates/` & `static/`: Frontend web interface.
- `cedar-solve/`: Integrated submodule/fork of the plate solver.
- `bound_20.dat`: Constellation boundary database.
- `ids.csv`: Star identification database.
