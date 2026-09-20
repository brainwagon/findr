import logging
from flask import Flask, render_template, Response, request, jsonify
import io
import os
import threading
import time
from PIL import Image
import ephem
import configparser
from solver import get_solver
from catalog import load_catalog
from solve import (
    CameraImageSource,
    ImageSourceError,
    SolveOutcome,
    SolveStore,
    TestImageSource,
    run_solve,
)
import power

logger = logging.getLogger(__name__)

# Try to initialize the INA219 sensor
ina219 = power.open_sensor(1)
if ina219:
    print("INA219 sensor initialized.")
else:
    print("I2C bus not found or smbus2 not installed. INA219 sensor disabled.")

# Create a new ephem observer
observer = ephem.Observer()

# Load configuration from location.ini
config = configparser.ConfigParser()
config.read('location.ini')

# Set observer's location from the configuration file
observer.lat = config.get('location', 'lat', fallback='0')
observer.lon = config.get('location', 'lon', fallback='0')


catalog = load_catalog()

try:
    from picamera2 import Picamera2
    camera = Picamera2()
    # Trigger an internal check to see if libcamera is actually available
    _ = camera.camera_properties
    print("Picamera2 initialized successfully.")
except (ImportError, ModuleNotFoundError) as e:
    print(f"Picamera2 or libcamera not found ({e}). Falling back to dummy camera.")
    from camera_dummy import Picamera2
    camera = Picamera2()
except Exception as e:
    print(f"Unexpected error initializing Picamera2: {e}. Falling back to dummy camera.")
    from camera_dummy import Picamera2
    camera = Picamera2()

app = Flask(__name__)

exposure_times = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

import atexit

def cleanup():
    """Close the camera and I2C bus on exit."""
    camera.close()
    if ina219:
        ina219.bus.close()
    print("Camera and I2C bus closed.")

atexit.register(cleanup)

# Solve status
solve_store = SolveStore()
test_mode = False # Global variable for test mode
is_paused = False # Global variable for pause state

@app.route('/')
def index():
    model = camera.camera_properties.get('Model', 'N/A')
    pixel_array_size = str(camera.camera_properties.get('PixelArraySize', 'N/A'))
    # i just want to pass some variables to the template...
    # i'll fix this later.
    gain = 1
    try:
        exposure_index = exposure_times.index(10000)
    except ValueError:
        exposure_index = 2
    brightness = 50
    contrast = 50
    sharpness = 50

    return render_template('index.html', 
            model=model, 
            pixel_array_size=pixel_array_size,
            gain=gain,
            exposure_index=exposure_index,
            exposure_times=exposure_times,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            test_mode=test_mode
            )

def gen_frames():
    """Generate frames for video stream."""
    while True:
        if latest_frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_bytes + b'\r\n')
        time.sleep(0.05) # control the frame rate

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    """Toggle the paused state."""
    global is_paused
    is_paused = not is_paused
    return jsonify({"is_paused": is_paused})

@app.route('/get_pause_state')
def get_pause_state():
    """Return the pause state."""
    return jsonify({"is_paused": is_paused})

@app.route('/get_fps')
def get_fps():
    """Return the current FPS."""
    return jsonify({"fps": f"{current_fps:.2f}"})

@app.route('/get_solve_fps')
def get_solve_fps():
    """Return the solve FPS."""
    return jsonify({"fps": f"{solve_fps:.2f}"})

@app.route('/set_controls', methods=['POST'])
def set_controls():
    """Set camera controls."""
    data = request.json
    
    # map the controls from the UI to the camera controls
    #
    # UI                Camera
    # ----------------- ----------------
    # gain              AnalogueGain
    # exposure_index    ExposureTime
    # brightness        Brightness (-1.0 to 1.0)
    # contrast          Contrast (0.0 to 2.0)
    #
    
    controls_to_set = {}
    
    if 'gain' in data:
        controls_to_set['AnalogueGain'] = float(data['gain'])
        
    if 'exposure_index' in data and data['exposure_index'] != '':
        try:
            exposure_idx = int(data['exposure_index'])
            if 0 <= exposure_idx < len(exposure_times):
                controls_to_set['ExposureTime'] = exposure_times[exposure_idx]
        except (ValueError, TypeError):
            pass # Ignore if not a valid index

    if 'brightness' in data:
        # scale from 0-100 to -1.0 to 1.0
        controls_to_set['Brightness'] = float(data['brightness']) / 50.0 - 1.0

    if 'contrast' in data:
        # scale from 0-100 to 0.0 to 2.0
        controls_to_set['Contrast'] = float(data['contrast']) / 50.0

    if 'ScalerCrop' in data:
        controls_to_set['ScalerCrop'] = data['ScalerCrop']

    safe_set_controls(controls_to_set)
    return "", 204

@app.route('/capture_lores_jpeg')
def capture_lores_jpeg():
    """Capture a lores JPEG."""
    if latest_frame_bytes:
        return Response(latest_frame_bytes, mimetype='image/jpeg')
    return "No frame available", 404

@app.route('/snapshot')
def snapshot():
    """Capture a full resolution JPEG."""
    buffer = io.BytesIO()
    camera.capture_file(buffer, name='main', format='jpeg')
    return Response(buffer.getvalue(), mimetype='image/jpeg')

@app.route('/solved_field.jpg')
def solved_field():
    """Return the solved field image."""
    image_bytes = solve_store.get_image_bytes()
    if image_bytes:
        return Response(image_bytes, mimetype='image/jpeg')
    # Return a black image if no solved image is available
    img = Image.new('RGB', (640, 480), color = 'black')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return Response(buf.getvalue(), mimetype='image/jpeg')


# Global variables for video feed and FPS
latest_frame_bytes = None
current_fps = 0
last_frame_time = time.time()
frame_count = 0
solve_fps = 0
solve_completed_count = 0

def calculate_solve_fps():
    """Continuously calculates the solve FPS."""
    global solve_fps, solve_completed_count
    while True:
        time.sleep(5)
        solve_fps = solve_completed_count / 5.0
        solve_completed_count = 0

def capture_and_process_frames():
    """Continuously captures frames, calculates FPS, and stores the latest frame."""
    global latest_frame_bytes, current_fps, last_frame_time, frame_count, is_paused
    while True:
        if is_paused:
            time.sleep(0.1)
            continue
        try:
            buffer = io.BytesIO()
            camera.capture_file(buffer, name='lores', format='jpeg')
            frame = buffer.getvalue()

            latest_frame_bytes = frame

            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - last_frame_time

            # Debug prints for frame count and elapsed time
            # print(f"Frame count: {frame_count}, Elapsed time: {elapsed_time:.2f}s")

            if elapsed_time >= 1.0: # Update FPS every second
                current_fps = frame_count / elapsed_time
                frame_count = 0
                last_frame_time = current_time
            time.sleep(0.01) # Small delay to prevent busy-waiting
        except Exception as e:
            print(f"Error capturing frame: {e}")
            # Optionally, you might want to set latest_frame_bytes to a placeholder
            # or handle the error in a way that doesn't crash the thread.
            time.sleep(1) # Wait a bit before retrying to avoid spamming errors

def _encode_jpeg(image):
    """Encode a PIL image to JPEG bytes."""
    if image is None:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def solve_plate():
    """Acquire an image, solve it, and store the outcome."""
    global solve_completed_count
    try:
        if is_paused:
            solve_store.set_status("paused")
            return

        source = TestImageSource() if test_mode else CameraImageSource(camera)
        try:
            image = source.acquire()
        except ImageSourceError as e:
            solve_store.finish(SolveOutcome(error=str(e)))
            return

        try:
            outcome = run_solve(image, get_solver(), observer, catalog)
        except Exception as e:
            logger.error("Error in solve_plate: %s", e)
            outcome = SolveOutcome(error=str(e), image=image)

        solve_store.finish(outcome, _encode_jpeg(outcome.image))
    finally:
        solve_completed_count += 1


@app.route('/solve', methods=['POST'])
def solve():
    """Initiate plate solving in a background thread."""
    solve_store.begin()
    threading.Thread(target=solve_plate).start()
    return jsonify({"status": "solving"})

@app.route('/solve_status')
def get_solve_status():
    """Return the status of the plate solver."""
    status, outcome = solve_store.snapshot()
    if status in ("solved", "failed") and outcome is not None:
        payload = {"status": status, "solved_image_url": "/solved_field.jpg"}
        payload.update(outcome.to_json())
        return jsonify(payload)
    return jsonify({"status": status})


@app.route('/system-stats')
def system_stats():
    """Return system stats as JSON."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000.0
    except IOError:
        temp = 'N/A'

    try:
        with open('/proc/loadavg', 'r') as f:
            load = f.read().split()[0]
    except IOError:
        load = 'N/A'

    reading = power.read_power(ina219)

    return jsonify(cpu_temp=f"{temp:.1f}" if isinstance(temp, float) else temp,
                   cpu_load=load,
                   voltage=f"{reading.voltage:.2f}" if reading else "N/A",
                   current=f"{reading.current:.2f}" if reading else "N/A",
                   low_voltage_warning=reading.low_voltage if reading else False,
                   power_source=reading.source if reading else "N/A",
                   battery_time_remaining=reading.time_remaining if reading else "N/A")


@app.route('/set_test_mode', methods=['POST'])
def set_test_mode():
    """Set the test mode state."""
    global test_mode
    data = request.json
    test_mode = data.get('test_mode', False)
    return "", 204

@app.route('/get_solver')
def get_solver_info():
    """Get the current and available solvers."""
    manager = get_solver()
    return jsonify({
        'current': manager.get_current_solver_type(),
        'available': manager.available_solvers()
    })

@app.route('/set_solver', methods=['POST'])
def set_solver():
    """Switch to a different solver."""
    data = request.get_json()
    solver_type = data.get('solver')
    if not solver_type:
        return jsonify({'error': 'No solver specified'}), 400
    
    manager = get_solver()
    try:
        manager.set_solver(solver_type)
        return jsonify({'status': 'success', 'current': manager.get_current_solver_type()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize camera and set initial controls once
config = camera.create_still_configuration(
    main = {
        "size" : (1456, 1088),
        "format" : "RGB888"
        },
    lores = {
        "size" : (640, 480),
        "format" : "YUV420",
        },
        )

camera.configure(config)
camera.start()



# Set initial controls safely
initial_controls = {

    "AnalogueGain": 1.0,
    "ExposureTime": 10000,
    "Brightness": 0.0,
    "Contrast": 1.0,
    "Sharpness": 1.0,
    "ExposureValue": 0.0 # Explicitly set ExposureValue to 0.0 initially
}

def safe_set_controls(controls):
    """Sets controls only if they are available."""
    available_controls = camera.camera_controls
    safe_controls = {k: v for k, v in controls.items() if k in available_controls}
    if safe_controls:
        camera.set_controls(safe_controls)

safe_set_controls(initial_controls)

frame_capture_thread = threading.Thread(target=capture_and_process_frames)
frame_capture_thread.daemon = True
frame_capture_thread.start()

solve_fps_thread = threading.Thread(target=calculate_solve_fps)
solve_fps_thread.daemon = True
solve_fps_thread.start()
logging.getLogger("werkzeug").setLevel(logging.WARNING)
port = int(os.environ.get("PORT", 8080))
app.run(host='0.0.0.0', port=port, threaded=True)
