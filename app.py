import logging
from flask import Flask, render_template, Response, request, jsonify
import sys
import io
import math
from math import radians
import os
import random
import numpy as np
# This is needed for some reason...
np.math = math
import datetime
import threading
import time
from PIL import Image, ImageDraw, ImageFont
import ephem
import configparser
import csv
import requests
from solver import get_solver

# Create a new ephem observer
observer = ephem.Observer()

# Load configuration from location.ini
config = configparser.ConfigParser()
config.read('location.ini')

# Set observer's location from the configuration file
observer.lat = config.get('location', 'lat', fallback='0')
observer.lon = config.get('location', 'lon', fallback='0')


# libraries needed to solve for a "proper" WCS coordinate system

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points  


font_path = "/usr/share/fonts/truetype/noto/NotoSansDisplay-Regular.ttf"
font_size = 12
font = ImageFont.truetype(font_path, font_size)




# go ahead and load the ids...

ids = { }
with open("ids.csv", "r") as f:
    rdr = csv.reader(f)
    for a, b, c in rdr:
        if b == '':
            ids[int(a)] = c
        else:
            ids[int(a)] = b

# not the most efficient, but... 

def decode_simbad_greek(text):
    greek_map = { 'alf': 'α',  # alpha
                  'bet': 'β',  # beta
                  'gam': 'γ',  # gamma 
                  'del': 'δ',  # delta
                  'eps': 'ε',  # epsilon
                  'zet': 'ζ',  # zeta
                  'eta': 'η',  # eta 
                  'tet': 'θ',  # theta 
                  'iot': 'ι',  # iota 
                  'kap': 'κ',  # kappa 
                  'lam': 'λ',  # lambda 
                  'mu.': 'μ',  # mu 
                  'nu.': 'ν',  # nu 
                  'ksi': 'ξ',  # xi 
                  'omi': 'ο',  # omicron 
                  'pi.': 'π',  # pi 
                  'rho': 'ρ',  # rho 
                  'sig': 'σ',  # sigma 
                  'tau': 'τ',  # tau 
                  'ups': 'υ',  # upsilon 
                  'phi': 'φ',  # phi 
                  'chi': 'χ',  # chi 
                  'psi': 'ψ',  # psi 
                  'ome': 'ω',  # omega 
          } 
    result = text
    for code, greek in greek_map.items(): 
        result = result.replace(code, greek) 
    return result

constellation_boundaries = {}

def load_constellation_boundaries():
    """Load constellation boundaries from bound_20.dat."""
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
        print("Warning: bound_20.dat not found. Constellation boundaries will not be drawn.")

load_constellation_boundaries()

def point_stellarium(ra_radians, dec_radians, stellarium_url="http://192.168.1.139:8090"):
    endpoint = f"{stellarium_url}/api/main/view" 
    x = math.cos(dec_radians) * math.cos(ra_radians)
    y = math.cos(dec_radians) * math.sin(ra_radians)
    z = math.sin(dec_radians)
    params = { 'j2000' : str([ x, y, z ]) }
    response = requests.post(endpoint, data=params)
    return response.status_code == 200

def format_radec_fixed_width(angle_obj, is_ra=True, total_width=10, decimal_places=1):
    """
    Formats an ephem.Angle object to a fixed-width string.
    RA: HH:MM:SS.S (total_width=10)
    Dec: sDD:MM:SS.S (total_width=11, s is sign)
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
    """Close the camera on exit."""
    camera.close()
    print("Camera closed.")

atexit.register(cleanup)

# Solver status
solver_status = "idle"
solver_result = {}
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
    with solved_image_lock:
        if solved_image_bytes:
            return Response(solved_image_bytes, mimetype='image/jpeg')
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
last_solve_time = time.time()
solve_count = 0
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

# In-memory storage for the solved image bytes to avoid writing to disk.
# Access guarded by solved_image_lock.
solved_image_bytes = None
solved_image_lock = threading.Lock()

def solve_plate():
    """Capture an image and solve for RA/Dec/Roll."""
    global solver_status, solver_result, test_mode, solved_image_bytes, is_paused
    if is_paused:
        solver_status = "paused"
        return
    
    img = None
    try:
        if test_mode:
            # For testing, load from a local file instead of capturing from camera
            test_images_dir = "test-images"
            image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
            if not image_files:
                solver_status = "failed"
                solver_result = {"error": "No test images found."}
                return
            random_image_file = random.choice(image_files)
            image_path = os.path.join(test_images_dir, random_image_file)
            img = Image.open(image_path)
        else:
            # Capture from Picamera
            buffer = io.BytesIO()
            camera.capture_file(buffer, name='lores', format='jpeg')
            buffer.seek(0)
            img = Image.open(buffer)

        # Use our new PlateSolver
        solver = get_solver()
        result = solver.solve(img)

        if result:
            ra_val = result['ra']
            dec_val = result['dec']
            roll_val = result['roll']
            fov_val = result['fov']

            ra_hms = ephem.hours(radians(ra_val))
            dec_dms = ephem.degrees(radians(dec_val))

            # compute the alt/az for the solved center.
            observer.date = ephem.now()
            target = ephem.FixedBody()
            target._ra = ra_hms
            target._dec = dec_dms
            target.compute(observer)
            
            solver_result = {
                "ra": f"{ra_val:.4f}",
                "dec": f"{dec_val:.4f}",
                "roll": f"{roll_val:.4f}",
                "ra_hms": format_radec_fixed_width(ra_hms, is_ra=True, total_width=10, decimal_places=1),
                "dec_dms": format_radec_fixed_width(dec_dms, is_ra=False, total_width=11, decimal_places=1),
                "alt": f"{math.degrees(target.alt):.1f}",
                "az": f"{math.degrees(target.az):.1f}",
                "solved_image_url": "/solved_field.jpg",
                "constellation": ephem.constellation((radians(ra_val), radians(dec_val)))[0],
                "matched_stars_count": result.get('matched_stars_count', 0),
            }

            # Draw annotations
            draw = ImageDraw.Draw(img)
            matched_catID = result.get('matched_catID', [])
            matched_centroids = result.get('matched_centroids', [])
            
            for star_id, p in zip(matched_catID, matched_centroids):
                try:
                    # Coordinate adjustment based on previous implementation
                    # p is usually (y, x) from tetra3
                    pos = (int(p[1]) + 8, int(p[0]) - 8)
                    id_str = decode_simbad_greek(ids.get(star_id, str(star_id)))
                    id_fields = id_str.split()
                    if id_fields and id_fields[0] == "*":
                        id_str = ' '.join(id_fields[1:])
                    draw.text(pos, f"{id_str}", fill=(255,255,255), font=font)
                except Exception as e:
                    print(f"Error drawing annotation for star {star_id}: {e}")

            # Restore Constellation Boundaries logic
            try:
                matched_stars = np.array(result.get("matched_stars", []))
                matched_centroids_arr = np.array(matched_centroids)

                if len(matched_stars) > 0 and len(matched_centroids_arr) > 0:
                    # x is returned as the second column by tetra3
                    # y is the first column
                    star_x = matched_centroids_arr[:,1]
                    star_y = matched_centroids_arr[:,0]
                    star_xy = (star_x, star_y)

                    star_ra = np.array(matched_stars[:,0]) * u.deg
                    star_dec = np.array(matched_stars[:,1]) * u.deg

                    world_coords = SkyCoord(ra = star_ra, dec = star_dec, frame = 'icrs')

                    # fit the WCS model
                    wcs = fit_wcs_from_points(
                        star_xy,
                        world_coords, 
                        projection='TAN',
                        sip_degree=2)

                    # Draw constellation boundaries
                    constellation_name = solver_result.get("constellation").upper()
                    if constellation_name and constellation_name in constellation_boundaries:
                        points = constellation_boundaries[constellation_name]
                        pixel_points = []
                        for ra, dec in points:
                            try:
                                px, py = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
                                pixel_points.append((px, py))
                            except Exception as e:
                                pixel_points.append(None)
                        
                        for i in range(len(pixel_points) - 1):
                            p1 = pixel_points[i]
                            p2 = pixel_points[i+1]
                            if p1 and p2:
                                draw.line([p1, p2], fill="yellow", width=1)
            except Exception as e:
                print(f"EXCEPTION DURING WCS/CONSTELLATION HANDLING: {e}")

            # Save the annotated image into memory (JPEG)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            buf.seek(0)
            with solved_image_lock:
                solved_image_bytes = buf.getvalue()

            solver_status = "solved"
        else:
            # Save the original input image into memory so the UI can still display something
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            buf.seek(0)
            with solved_image_lock:
                solved_image_bytes = buf.getvalue()

            solver_status = "failed"
            solver_result = {"solved_image_url": "/solved_field.jpg"}

    except Exception as e:
        print(f"Error in solve_plate: {e}")
        if img:
            try:
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                buf.seek(0)
                with solved_image_lock:
                    solved_image_bytes = buf.getvalue()
            except Exception:
                pass
        solver_status = "failed"
        solver_result = {"solved_image_url": "/solved_field.jpg"}
    finally:
        global solve_completed_count
        solve_completed_count += 1

@app.route('/solve', methods=['POST'])
def solve():
    """Initiate plate solving in a background thread."""
    global solver_status
    solver_status = "solving"
    solver_thread = threading.Thread(target=solve_plate)
    solver_thread.start()
    return jsonify({"status": "solving"})

@app.route('/solve_status')
def get_solve_status():
    """Return the status of the plate solver."""
    if solver_status == "solved" or solver_status == "failed":
        return jsonify({"status": solver_status, **solver_result})
    else:
        return jsonify({"status": solver_status})

@app.route('/set_test_mode', methods=['POST'])
def set_test_mode():
    """Set the test mode state."""
    global test_mode
    data = request.json
    test_mode = data.get('test_mode', False)
    return "", 204


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
app.run(host='0.0.0.0', port=8080, threaded=True)
