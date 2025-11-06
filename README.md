# MyFinder - a prototype telescope finder

**⚠️ Warning: This project is a work in progress and a prototype. It is not fully functional or robust and should not be used in production environments. ⚠️**

This project provides a web-based interface to control a Raspberry Pi camera remotely. It runs a Flask web server on the Pi, allowing you to view a live stream and adjust camera settings from any web browser on the same network.

## Features

- **Web-Based Interface:** Control your camera from a simple web page.
- **Live Video Stream:** View a real-time MJPEG stream from the camera.
- **Camera Controls:** Adjust settings like gain, exposure, and white balance.
- **Headless Operation:** Runs on a Raspberry Pi without a monitor or keyboard.
- **Development Mode:** Includes a dummy camera interface for development on non-Pi machines.

## Project Structure

```
.
├── app.py                  # Main Flask application
├── camera_dummy.py         # Dummy camera for development
├── requirements.txt        # Python dependencies
├── static
│   ├── main.js             # Client-side JavaScript
│   └── style.css           # Stylesheet
└── templates
    └── index.html          # Main HTML page
```

## Installation and Usage

There are two ways to run this project: on a development machine (like a laptop) or on a Raspberry Pi.

### On a Development Machine (Non-Pi)

This setup uses a dummy camera feed, allowing you to work on the web interface without a Raspberry Pi.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```

5.  Open your web browser and go to `http://127.0.0.1:8080` to see the interface with the dummy camera feed.

### On a Raspberry Pi

This setup uses the actual Raspberry Pi camera.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the `picamera2` library:**
    ```bash
    pip install picamera2
    ```

5.  **Run the application:**
    ```bash
    python app.py
    ```

6.  Find your Raspberry Pi's IP address (e.g., by running `hostname -I`) and open a web browser on another device on the same network. Go to `http://<your-pi-ip-address>:8080`.
