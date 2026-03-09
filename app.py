"""findr: Prototype Telescope Finder.

A Raspberry Pi-based telescope finder and remote camera control system
providing a web interface for live monitoring and plate solving.
"""

import atexit
import configparser
import csv
import datetime
import io
import logging
import math
import os
import random
import threading
import time
from math import radians

import ephem
import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageFont

from solver import get_solver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Observer Configuration ---
# Create a new ephem observer
observer = ephem.Observer()

# Load configuration from location.ini
config = configparser.ConfigParser()
config.read('location.ini')

# Set observer's location from the configuration file
observer.lat = config.get('location', 'lat', fallback='0')
observer.lon = config.get('location', 'lon', fallback='0')

# --- Assets ---
font_path = "/usr/share/fonts/truetype/noto/NotoSansDisplay-Regular.ttf"
font_size = 12
try:
    font = ImageFont.truetype(font_path, font_size)
except OSError:
    logger.warning(f"Font not found at {font_path}. Using default.")
    font = ImageFont.load_default()

# --- Star IDs ---
ids = {}
try:
    with open("ids.csv", "r") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if len(row) >= 3:
                a, b, c = row[0], row[1], row[2]
                if b == '':
                    ids[int(a)] = c
                else:
                    ids[int(a)] = b
except FileNotFoundError:
    logger.warning("ids.csv not found. Star identifications will be limited.")

GREEK_MAP = {
    'alf': 'α', 'bet': 'β', 'gam': 'γ', 'del': 'δ', 'eps': 'ε',
    'zet': 'ζ', 'eta': 'η', 'tet': 'θ', 'iot': 'ι', 'kap': 'κ',
    'lam': 'λ', 'mu.': 'μ', 'nu.': 'ν', 'ksi': 'ξ', 'omi': 'ο',
    'pi.': 'π', 'rho': 'ρ', 'sig': 'σ', 'tau': 'τ', 'ups': 'υ',
    'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'ome': 'ω',
}

def decode_simbad_greek(text: str) -> str:
    """Decodes Simbad-style Greek letter codes to UTF-8 characters."""
    result = text
    for code, greek in GREEK_MAP.items():
        if code in result:
            result = result.replace(code, greek)
    return result

# --- Constellation Boundaries ---
constellation_boundaries = {}

def load_constellation_boundaries():
    """Loads constellation boundaries from bound_20.dat."""
    try:
        with open("bound_20.dat", "r") as f:
            for line in f:
                ra_h = float(line[0:10])
                dec_d = float(line[11:22])
                constellation = line[23:27].strip()
                # Convert RA from hours to degrees
                ra_d = ra_h * 15.0
                if constellation not in constellation_boundaries:
                    constellation_boundaries[constellation] = []
                constellation_boundaries[constellation].append((ra_d, dec_d))
    except FileNotFoundError:
        logger.warning("bound_20.dat not found. Constellation boundaries disabled.")

load_constellation_boundaries()

# --- External Integration ---
def point_stellarium(ra_radians: float, dec_radians: float,
                     stellarium_url: str = "http://192.168.1.139:8090") -> bool:
    """Sends pointing information to a Stellarium instance.

    Args:
        ra_radians: Right Ascension in radians.
        dec_radians: Declination in radians.
        stellarium_url: URL of the Stellarium Remote Control API.

    Returns:
        True if the request was successful, False otherwise.
    """
    endpoint = f"{stellarium_url}/api/main/view"
    x = math.cos(dec_radians) * math.cos(ra_radians)
    y = math.cos(dec_radians) * math.sin(ra_radians)
    z = math.sin(dec_radians)
    params = {'j2000': str([x, y, z])}
    try:
        response = requests.post(endpoint, data=params, timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False

# --- Formatting ---
def format_radec_fixed_width(angle_obj: ephem.Angle, is_ra: bool = True,
                             total_width: int = 10,
                             decimal_places: int = 1) -> str:
    """Formats an ephem.Angle object to a fixed-width string.

    Args:
        angle_obj: The angle to format.
        is_ra: Whether the angle is Right Ascension.
        total_width: Total desired width of the string.
        decimal_places: Precision for the seconds component.

    Returns:
        A fixed-width formatted string.
    """
    s = str(angle_obj)
    parts = s.split(':')

    if is_ra:
        # RA: HH:MM:SS.S
        hours = parts[0].zfill(2)
        minutes = parts[1].zfill(2)
        seconds_float = float(parts[2])
        formatted_seconds = f"{seconds_float:0{3+decimal_places}.{decimal_places}f}"
        formatted_time = f"{hours}:{minutes}:{formatted_seconds}"
    else:
        # Dec: sDD:MM:SS.S
        sign = ''
        if parts[0].startswith('-'):
            sign = '-'
            parts[0] = parts[0][1:]
        elif parts[0].startswith('+'):
            sign = '+'
            parts[0] = parts[0][1:]
        degrees = parts[0].zfill(2)
        minutes = parts[1].zfill(2)
        seconds_float = float(parts[2])
        formatted_seconds = f"{seconds_float:0{3+decimal_places}.{decimal_places}f}"
        formatted_time = f"{sign}{degrees}:{minutes}:{formatted_seconds}"

    return formatted_time.ljust(total_width)[:total_width]

# --- Camera Initialization ---
try:
    from picamera2 import Picamera2
    camera = Picamera2()
    # Trigger an internal check to see if libcamera is actually available
    _ = camera.camera_properties
    logger.info("Picamera2 initialized successfully.")
except (ImportError, ModuleNotFoundError, Exception) as e:
    logger.warning(f"Picamera2 initialization failed ({e}). Using dummy camera.")
    from camera_dummy import Picamera2
    camera = Picamera2()

# --- Flask App ---
app = Flask(__name__)

EXPOSURE_TIMES = [
    1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000
]

def cleanup():
    """Performs cleanup on application exit."""
    try:
        if hasattr(camera, 'close'):
            camera.close()
            logger.info("Camera closed.")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

atexit.register(cleanup)

# Global State
solver_status = "idle"
solver_result = {}
test_mode = False
is_paused = False
latest_frame_bytes = None
latest_frame_time = 0.0
current_fps = 0.0
last_frame_time = time.time()
frame_count = 0
solve_fps = 0.0
solve_completed_count = 0
solved_image_bytes = None
solved_image_lock = threading.Lock()

# New: Solve worker state
solve_request_event = threading.Event()
solve_worker_thread = None

@app.route('/')
def index():
# ... (rest of the index function)
    """Renders the main application index page."""
    model = camera.camera_properties.get('Model', 'N/A')
    pixel_array_size = str(camera.camera_properties.get('PixelArraySize', 'N/A'))
    gain = 1
    try:
        exposure_index = EXPOSURE_TIMES.index(10000)
    except ValueError:
        exposure_index = 2
    brightness = 50
    contrast = 50
    sharpness = 50

    return render_template(
        'index.html',
        model=model,
        pixel_array_size=pixel_array_size,
        gain=gain,
        exposure_index=exposure_index,
        exposure_times=EXPOSURE_TIMES,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        test_mode=test_mode
    )

def gen_frames():
    """Generates JPEG frames for the multipart video stream."""
    while True:
        if latest_frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   latest_frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    """MJPEG video streaming route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    """Toggles the camera frame capture pause state."""
    global is_paused
    is_paused = not is_paused
    return jsonify({"is_paused": is_paused})

@app.route('/get_pause_state')
def get_pause_state():
    """Returns the current pause state."""
    return jsonify({"is_paused": is_paused})

@app.route('/get_fps')
def get_fps():
    """Returns the current live stream FPS."""
    return jsonify({"fps": f"{current_fps:.2f}"})

@app.route('/get_solve_fps')
def get_solve_fps():
    """Returns the current plate solving FPS."""
    return jsonify({"fps": f"{solve_fps:.2f}"})

@app.route('/set_controls', methods=['POST'])
def set_controls():
    """Updates camera control parameters."""
    data = request.json
    controls_to_set = {}

    if 'gain' in data:
        controls_to_set['AnalogueGain'] = float(data['gain'])

    if 'exposure_index' in data and data['exposure_index'] != '':
        try:
            idx = int(data['exposure_index'])
            if 0 <= idx < len(EXPOSURE_TIMES):
                controls_to_set['ExposureTime'] = EXPOSURE_TIMES[idx]
        except (ValueError, TypeError):
            pass

    if 'brightness' in data:
        controls_to_set['Brightness'] = float(data['brightness']) / 50.0 - 1.0

    if 'contrast' in data:
        controls_to_set['Contrast'] = float(data['contrast']) / 50.0

    if 'ScalerCrop' in data:
        controls_to_set['ScalerCrop'] = data['ScalerCrop']

    safe_set_controls(controls_to_set)
    return "", 204

@app.route('/capture_lores_jpeg')
def capture_lores_jpeg():
    """Captures and returns the latest low-resolution frame."""
    if latest_frame_bytes:
        return Response(latest_frame_bytes, mimetype='image/jpeg')
    return "No frame available", 404

@app.route('/snapshot')
def snapshot():
    """Captures and returns a full-resolution JPEG frame."""
    buffer = io.BytesIO()
    camera.capture_file(buffer, name='main', format='jpeg')
    return Response(buffer.getvalue(), mimetype='image/jpeg')

@app.route('/solved_field.jpg')
def solved_field():
    """Returns the latest annotated solved field image."""
    with solved_image_lock:
        if solved_image_bytes:
            return Response(solved_image_bytes, mimetype='image/jpeg')
    # Return a black image if no solved image is available
    img = Image.new('RGB', (640, 480), color='black')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return Response(buf.getvalue(), mimetype='image/jpeg')

# --- Background Processing ---
def calculate_solve_fps():
    """Continuously calculates the solve FPS."""
    global solve_fps, solve_completed_count
    while True:
        time.sleep(5)
        solve_fps = solve_completed_count / 5.0
        solve_completed_count = 0

def capture_and_process_frames():
    """Continuously captures frames and calculates live stream FPS."""
    global latest_frame_bytes, latest_frame_time, current_fps, last_frame_time, frame_count
    while True:
        if is_paused:
            time.sleep(0.1)
            continue
        try:
            now = time.time()
            if test_mode:
                test_dir = "test-images"
                if os.path.exists(test_dir):
                    files = [f for f in os.listdir(test_dir)
                             if f.lower().endswith(('.jpg', '.jpeg'))]
                    if files:
                        image_path = os.path.join(test_dir, random.choice(files))
                        with open(image_path, "rb") as f:
                            frame = f.read()
                        latest_frame_bytes = frame
                        latest_frame_time = now
                    else:
                        time.sleep(1)
                        continue
                else:
                    time.sleep(1)
                    continue
            else:
                buffer = io.BytesIO()
                camera.capture_file(buffer, name='lores', format='jpeg')
                frame = buffer.getvalue()
                latest_frame_bytes = frame
                latest_frame_time = now

            frame_count += 1
            dt = now - last_frame_time
            if dt >= 1.0:
                current_fps = frame_count / dt
                frame_count = 0
                last_frame_time = now
            
            # Slow down the loop to be more resource-efficient
            if test_mode:
                time.sleep(0.5)
            else:
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            time.sleep(1)

def solve_worker():
    """Persistent background thread for plate solving."""
    global solver_status, is_paused
    logger.info("Solve worker started.")
    while True:
        # Wait for a request
        if solve_request_event.wait(timeout=0.1):
            solve_request_event.clear()
            if not is_paused:
                solve_plate()
        else:
             time.sleep(0.01)

def solve_plate():
    """Captures an image and performs plate solving.

    Updates global solver_status, solver_result, and solved_image_bytes.
    """
    global solver_status, solver_result, solved_image_bytes, solve_completed_count
    
    t_start = time.time()
    img = None
    frame_timestamp = 0.0
    try:
        t_capture_start = time.time()
        if test_mode:
            test_dir = "test-images"
            if not os.path.exists(test_dir):
                solver_status = "failed"
                solver_result = {"error": "Test directory not found."}
                return
            files = [f for f in os.listdir(test_dir)
                     if f.lower().endswith(('.jpg', '.jpeg'))]
            if not files:
                solver_status = "failed"
                solver_result = {"error": "No test images found."}
                return
            image_path = os.path.join(test_dir, random.choice(files))
            img = Image.open(image_path)
            frame_timestamp = t_capture_start
        else:
            if latest_frame_bytes:
                img = Image.open(io.BytesIO(latest_frame_bytes))
                frame_timestamp = latest_frame_time
            else:
                # Fallback capture if latest_frame_bytes is missing
                buffer = io.BytesIO()
                camera.capture_file(buffer, name='lores', format='jpeg')
                buffer.seek(0)
                img = Image.open(buffer)
                frame_timestamp = time.time()
        t_capture_end = time.time()

        # Check for stale frames (e.g. older than 5 seconds to be safe)
        if not test_mode and (time.time() - frame_timestamp) > 5.0:
            logger.warning(f"Solve skipped: frame is stale (age: {time.time() - frame_timestamp:.2f}s)")
            solver_status = "failed"
            solver_result = {"error": "Stale frame"}
            return

        t_solve_start = time.time()
        solver = get_solver()
        # Use a reasonable max_returned to balance speed and results
        result = solver.solve(img, max_returned=15)
        t_solve_end = time.time()

        if result:
            t_meta_start = time.time()
            ra_val = result['ra']
            dec_val = result['dec']
            roll_val = result['roll']
            fov_val = result['fov']
            
            ra_hms = ephem.hours(radians(ra_val))
            dec_dms = ephem.degrees(radians(dec_val))

            observer.date = ephem.now()
            target = ephem.FixedBody()
            target._ra = ra_hms
            target._dec = dec_dms
            target.compute(observer)

            const_name_code = ephem.constellation((radians(ra_val), radians(dec_val)))[0]
            
            solver_result = {
                "ra": f"{ra_val:.4f}",
                "dec": f"{dec_val:.4f}",
                "roll": f"{roll_val:.4f}",
                "fov": f"{fov_val:.4f}",
                "ra_hms": format_radec_fixed_width(ra_hms),
                "dec_dms": format_radec_fixed_width(dec_dms, is_ra=False),
                "alt": f"{math.degrees(target.alt):.1f}",
                "az": f"{math.degrees(target.az):.1f}",
                "solved_image_url": "/solved_field.jpg",
                "constellation": const_name_code,
                "matched_stars_count": result.get('matched_stars_count', 0),
            }
            t_meta_end = time.time()

            # Annotations and Constellations
            t_draw_start = time.time()
            draw = ImageDraw.Draw(img)
            
            # Limit number of annotated stars to keep it responsive
            cat_ids = result.get('matched_catID', [])
            centroids = result.get('matched_centroids', [])
            num_stars_to_draw = min(len(cat_ids), 30)
            
            for i in range(num_stars_to_draw):
                star_id = cat_ids[i]
                p = centroids[i]
                try:
                    pos = (int(p[1]) + 8, int(p[0]) - 8)
                    id_str = decode_simbad_greek(ids.get(star_id, str(star_id)))
                    if id_str.startswith("* "):
                        id_str = id_str[2:]
                    draw.text(pos, id_str, fill=(255, 255, 255), font=font)
                except Exception:
                    pass
            t_draw_stars_end = time.time()

            # Constellation boundaries using faster method if possible
            t_const_start = time.time()
            try:
                const_upper = const_name_code.upper()
                if const_upper in constellation_boundaries:
                    # Optimized WCS calculation instead of fitting from all points
                    # if we have at least 3 points, it's safer, but 1 point + roll/fov
                    # should be enough if the solver results are accurate.
                    matched_stars = np.array(result.get("matched_stars", []))
                    matched_centroids_arr = np.array(centroids)
                    
                    if len(matched_stars) >= 3:
                        # Only use a subset of points for fitting to save time
                        subset = min(len(matched_stars), 8)
                        star_xy = (matched_centroids_arr[:subset, 1],
                                   matched_centroids_arr[:subset, 0])
                        star_ra = matched_stars[:subset, 0] * u.deg
                        star_dec = matched_stars[:subset, 1] * u.deg
                        world_coords = SkyCoord(ra=star_ra, dec=star_dec, frame='icrs')
                        
                        # fit_wcs_from_points is still slow, but fewer points might help
                        # If we had a robust create_wcs_from_radec_roll it would be better.
                        wcs = fit_wcs_from_points(star_xy, world_coords,
                                                  projection='TAN', sip_degree=0)
                        
                        points = np.array(constellation_boundaries[const_upper])
                        pra_arr = points[:, 0]
                        pdec_arr = points[:, 1]
                        
                        px_arr, py_arr = wcs.world_to_pixel(SkyCoord(ra=pra_arr*u.deg, 
                                                                    dec=pdec_arr*u.deg, 
                                                                    frame='icrs'))
                        line_points = []
                        for i in range(len(px_arr)):
                            if np.isnan(px_arr[i]) or np.isnan(py_arr[i]) or \
                               px_arr[i] < -100 or px_arr[i] > img.width + 100 or \
                               py_arr[i] < -100 or py_arr[i] > img.height + 100:
                                if len(line_points) > 1:
                                    draw.line(line_points, fill="yellow", width=1)
                                line_points = []
                            else:
                                line_points.append((px_arr[i], py_arr[i]))
                        if len(line_points) > 1:
                            draw.line(line_points, fill="yellow", width=1)
            except Exception as e:
                logger.error(f"Error drawing constellations: {e}")
            t_const_end = time.time()

            # Encode to JPEG
            t_encode_start = time.time()
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=80) # Slightly lower quality for speed
            with solved_image_lock:
                solved_image_bytes = buf.getvalue()
            t_encode_end = time.time()
            
            total_duration = (time.time() - t_start) * 1000
            logger.info(f"PERF: Total {total_duration:.1f}ms | "
                        f"Solve: {(t_solve_end-t_solve_start)*1000:.1f}ms | "
                        f"Const: {(t_const_end-t_const_start)*1000:.1f}ms | "
                        f"Encode: {(t_encode_end-t_encode_start)*1000:.1f}ms")
            
            solver_status = "solved"
        else:
            # Failed solve: show raw image anyway
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=75)
            with solved_image_lock:
                solved_image_bytes = buf.getvalue()
            solver_status = "failed"
            solver_result = {"solved_image_url": "/solved_field.jpg"}
    except Exception as e:
        logger.error(f"Critical error in solve_plate: {e}")
        solver_status = "failed"
        solver_result = {"solved_image_url": "/solved_field.jpg"}
    finally:
        solve_completed_count += 1

@app.route('/solve', methods=['POST'])
def solve():
    """Triggers the background plate solver."""
    global solver_status
    if is_paused:
        return jsonify({"status": "paused"})
    
    # Just signal the worker thread
    if solver_status != "solving":
        solver_status = "solving"
        solve_request_event.set()
    
    return jsonify({"status": "solving"})

@app.route('/solve_status')
def get_solve_status():
    """Returns the current status and result of the plate solver."""
    if solver_status in ("solved", "failed"):
        return jsonify({"status": solver_status, **solver_result})
    return jsonify({"status": solver_status})

@app.route('/set_test_mode', methods=['POST'])
def set_test_mode():
    """Sets the application test mode."""
    global test_mode
    data = request.json
    test_mode = data.get('test_mode', False)
    return "", 204

def safe_set_controls(controls):
    """Safely sets camera controls if they are supported."""
    available = camera.camera_controls
    safe = {k: v for k, v in controls.items() if k in available}
    if safe:
        camera.set_controls(safe)

def main():
    """Main application entry point."""
    # Configure camera
    cam_config = camera.create_still_configuration(
        main={"size": (1456, 1088), "format": "RGB888"},
        lores={"size": (640, 480), "format": "YUV420"}
    )
    camera.configure(cam_config)
    camera.start()

    # Initial controls
    safe_set_controls({
        "AnalogueGain": 1.0,
        "ExposureTime": 10000,
        "Brightness": 0.0,
        "Contrast": 1.0,
        "Sharpness": 1.0,
        "ExposureValue": 0.0
    })

    # Start threads
    threading.Thread(target=capture_and_process_frames, daemon=True).start()
    threading.Thread(target=calculate_solve_fps, daemon=True).start()
    threading.Thread(target=solve_worker, daemon=True).start()

    # Start Flask
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host='0.0.0.0', port=8080, threaded=True)

if __name__ == '__main__':
    main()
