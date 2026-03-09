<p align="center">
  <img src="findr-logo.svg" alt="findr logo" width="200"/>
</p>

# findr - a prototype telescope finder

**⚠️ Warning: This project is a work in progress and a prototype. It is not fully functional or robust and should not be used in production environments. ⚠️**

This project provides a web-based interface for a Raspberry Pi-based telescope finder. It includes a live camera stream, manual camera controls, and integrated **plate solving** to automatically identify where the telescope is pointing.

## Features

- **Plate Solving:** Uses the `tetra3` library to identify star fields and return RA/Dec coordinates, roll, and FOV.
- **Star Identification:** Annotates solved images with star names (Simbad/Greek designations).
- **Constellation Boundaries:** Automatically draws constellation boundaries on solved fields using `astropy` and `pyephem`.
- **System Monitoring:** Real-time monitoring of CPU temperature, load, and power stats (via INA219 if available).
- **Web-Based Interface:** Control camera settings (gain, exposure, etc.) and trigger solves from any browser.
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
├── solver.py               # PlateSolver abstraction layer
├── camera_dummy.py         # Dummy camera interface for non-Pi development
├── requirements.txt        # Python dependencies (excluding system libraries)
├── bound_20.dat            # Constellation boundary data
├── ids.csv                 # Star identification database
├── static/                 # CSS and JavaScript assets
└── templates/              # HTML templates (Flask)
```

## Installation and Usage

### 1. Clone the Repository
```bash
git clone <repository-url>
cd findr
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
- The solver uses `tetra3`. Ensure you have a valid tetra3 database installed (it will attempt to load the default one if available).
- Use **Test Mode** in the web interface to cycle through pre-loaded images in the `test-images/` directory to verify solver performance without live hardware.
