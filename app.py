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
from astropy.io import fits
import datetime
import threading
import time
import tetra3
from PIL import Image, ImageDraw, ImageFont
import ephem
import csv
import requests

tetra = tetra3.Tetra3()

font_path = "/usr/share/fonts/truetype/noto/NotoSansDisplay-Regular.ttf"
font_size = 12
font = ImageFont.truetype(font_path, font_size)

for k, v in tetra.database_properties.items():
        print(k, v)


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

def point_stellarium(ra_radians, dec_radians, stellarium_url="http://192.168.1.139:8090"):
    endpoint = f"{stellarium_url}/api/main/view" 
    x = math.cos(dec_radians) * math.cos(ra_radians)
    y = math.cos(dec_radians) * math.sin(ra_radians)
    z = math.sin(dec_radians)
    params = { 'j2000' : str([ x, y, z ]) }
    response = requests.post(endpoint, data=params)
    return response.status_code == 200

from picamera2 import Picamera2


app = Flask(__name__)

import atexit

def close_camera():
    """Close the camera on exit."""
    camera.close()

atexit.register(close_camera)

# Solver status
solver_status = "idle"
solver_result = {}
test_mode = False # Global variable for test mode
# Global variables for video feed and FPS
latest_frame_bytes = None
current_fps = 0
last_frame_time = time.time()
frame_count = 0
solve_fps = 0
last_solve_time = time.time()
solve_count = 0

# Global variables for video feed and FPS
latest_frame_bytes = None
current_fps = 0
last_frame_time = time.time()
frame_count = 0

def capture_and_process_frames():
    """Continuously captures frames, calculates FPS, and stores the latest frame."""
    global latest_frame_bytes, current_fps, last_frame_time, frame_count
    while True:
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

# Start the frame capture and processing in a separate thread
frame_capture_thread = threading.Thread(target=capture_and_process_frames)
frame_capture_thread.daemon = True
frame_capture_thread.start()

# In-memory storage for the solved image bytes to avoid writing to disk.
# Access guarded by solved_image_lock.
solved_image_bytes = None
solved_image_lock = threading.Lock()

def solve_plate():
    """Capture an image and solve for RA/Dec/Roll."""
    global solver_status, solver_result, test_mode, solved_image_bytes
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

        solution = tetra.solve_from_image(img,
                return_visual=True, return_matches=True,
                distortion=-0.003857906866170312)
        if solution and 'RA' in solution and 'Dec' in solution and 'Roll' in solution:
            # Get the visual solution
            visual_solution = solution['visual']

            # Convert both images to numpy arrays
            img_array = np.array(img.convert('RGB'))
            visual_array = np.array(visual_solution.convert('RGB'))

            # Resize visual solution to match input image dimensions
            if img_array.shape != visual_array.shape:
                visual_solution_resized = visual_solution.resize(img.size)
                visual_array = np.array(visual_solution_resized.convert('RGB'))
                visual_solution = visual_solution_resized

            # Combine the images by taking the maximum pixel value
            combined_array = np.maximum(img_array, visual_array)

            # Create a new image from the combined array
            combined_image = Image.fromarray(combined_array)

            # okay, MTV - draw annotations
            draw = ImageDraw.Draw(combined_image)
            for id, p in zip(solution.get("matched_catID", []), solution.get("matched_centroids", [])):
                try:
                    # not sure why x and y are swapped here...
                    p = (int(p[1]) + 8, int(p[0]) - 8)
                    id_str = decode_simbad_greek(ids.get(id, str(id)))
                    id_fields = id_str.split()
                    if id_fields and id_fields[0] == "*":
                        id_str = ' '.join(id_fields[1:])
                    draw.text(p, f"{id_str}", fill=(255,255,255), font=font)
                except Exception:
                    pass

            # Save the combined image into memory (JPEG)
            buf = io.BytesIO()
            combined_image.save(buf, format='JPEG')
            buf.seek(0)
            with solved_image_lock:
                solved_image_bytes = buf.getvalue()

            # Build solver_result
            solution_time_val = solution.get("T_solve", 0.0)
            solver_result = {
                "ra": f"{solution['RA']:.4f}",
                "dec": f"{solution['Dec']:.4f}",
                "roll": f"{solution['Roll']:.4f}",
                "solved_image_url": "/solved_field.jpg",
                "solution_time": f"{solution_time_val:.2f}ms",
                "constellation": ephem.constellation((radians(solution['RA']), radians(solution['Dec'])))[1],
            }

            solver_status = "solved"

            # send the center to stellarium...
            print(point_stellarium(radians(solution['RA']), radians(solution['Dec'])))

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
        global solve_fps, last_solve_time, solve_count
        solve_count += 1
        current_time = time.time()
        elapsed_time = current_time - last_solve_time
        if elapsed_time >= 5.0:
            solve_fps = solve_count / elapsed_time
            solve_count = 0
            last_solve_time = current_time

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

    return jsonify(cpu_temp=f"{temp:.1f}", cpu_load=load)

@app.route('/set_test_mode', methods=['POST'])
def set_test_mode():
    """Set the test mode state."""
    global test_mode
    data = request.json
    test_mode = data.get('test_mode', False)
    print(f"Test mode set to: {test_mode}")
    return "", 204

camera = Picamera2()

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
print("STILL IMAGE CONFIGURATION")
for k, v in config.items():
	print(k, v)
camera.configure(config)
camera.start()

# Print available controls for debugging
print("\nAvailable Camera Controls:")
print("--------------------------------------------------------------------------------")
print(f"{'Control Name':<25} {'Min':<15} {'Max':<15} {'Default':<15}")
print("--------------------------------------------------------------------------------")
for control_name in sorted(camera.camera_controls.keys()):
    min_val, max_val, default_val = camera.camera_controls[control_name]
    print(f"{control_name:<25} {str(min_val):<15} {str(max_val):<15} {str(default_val):<15}")
print("--------------------------------------------------------------------------------")

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
        print(f"Attempting to set controls: {safe_controls}")
        camera.set_controls(safe_controls)

safe_set_controls(initial_controls)



@app.route('/get_fps')
def get_fps():
    """Return the current FPS."""
    return jsonify(fps=f"{current_fps:.1f}")

@app.route('/get_solve_fps')
def get_solve_fps():
    """Return the current solve FPS."""
    return jsonify(fps=f"{solve_fps:.1f}")

@app.route('/')
def index():
    """Return the main page with initial slider values."""
    global test_mode
    # Get camera properties
    camera_properties = camera.camera_properties
    model = camera_properties.get("Model", "Unknown")
    pixel_array_size = camera_properties.get("PixelArraySize", "Unknown")

    # Get current camera controls
    current_controls = camera.controls

    # Map current camera values to slider values (0-100)
    slider_values = {}

    # AnalogueGain
    current_gain = getattr(current_controls, "AnalogueGain", 1.0)
    slider_values['gain'] = int(current_gain)

    # ExposureTime
    exposure_times = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 125000, 250000, 500000, 1000000]
    current_exposure = getattr(current_controls, "ExposureTime", 10000)
    # Find the index of the closest exposure time
    exposure_index = min(range(len(exposure_times)), key=lambda i: abs(exposure_times[i] - current_exposure))
    slider_values['exposure_index'] = exposure_index
    slider_values['exposure_times'] = exposure_times

    # Brightness
    min_bright, max_bright, _ = camera.camera_controls.get("Brightness", (-1.0, 1.0, 0.0))
    current_brightness = getattr(current_controls, "Brightness", 0.0)
    slider_values['brightness'] = int(((current_brightness - min_bright) / (max_bright - min_bright)) * 100) if (max_bright - min_bright) != 0 else 0

    # Contrast
    min_contrast, max_contrast, _ = camera.camera_controls.get("Contrast", (0.0, 32.0, 1.0))
    current_contrast = getattr(current_controls, "Contrast", 1.0)
    slider_values['contrast'] = int(((current_contrast - min_contrast) / (max_contrast - min_contrast)) * 100) if (max_contrast - min_contrast) != 0 else 0

    # Sharpness
    min_sharp, max_sharp, _ = camera.camera_controls.get("Sharpness", (0.0, 16.0, 1.0))
    current_sharpness = getattr(current_controls, "Sharpness", 1.0)
    slider_values['sharpness'] = int(((current_sharpness - min_sharp) / (max_sharp - min_sharp)) * 100) if (max_sharp - min_sharp) != 0 else 0



    return render_template('index.html', model=model, pixel_array_size=pixel_array_size, test_mode=test_mode, **slider_values)

@app.route('/video_feed')
def video_feed():
    """Return the latest video frame."""
    global latest_frame_bytes
    if latest_frame_bytes:
        return Response(latest_frame_bytes, mimetype='image/jpeg')
    return "", 204 # No content if no frame is available yet

@app.route('/capture_lores_jpeg')
def capture_lores_jpeg():
    """Capture a lores (640x480) JPEG image."""
    buffer = io.BytesIO()
    camera.capture_file(buffer, name='lores', format='jpeg')
    frame = buffer.getvalue()
    return Response(frame, mimetype='image/jpeg')

@app.route('/snapshot')
def snapshot():
    """Capture a full resolution (1456x1088) JPEG image."""
    buffer = io.BytesIO()
    camera.capture_file(buffer, name='main', format='jpeg')
    frame = buffer.getvalue()
    return Response(frame, mimetype='image/jpeg')

@app.route('/set_controls', methods=['POST'])
def set_controls():
    """Set camera controls."""
    data = request.json
    
    controls_to_set = {"AeEnable": False} # Disable auto exposure when setting manual controls

    if 'gain' in data:
        gain = float(data.get('gain'))
        controls_to_set["AnalogueGain"] = gain

    if 'exposure_index' in data:
        exposure_times = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 125000, 250000, 500000, 1000000]
        exposure_index = int(data.get('exposure_index', 0))
        exposure_time = exposure_times[exposure_index]
        controls_to_set["ExposureTime"] = exposure_time

    if 'brightness' in data:
        brightness = float(data.get('brightness'))
        min_bright, max_bright, _ = camera.camera_controls.get("Brightness", (-1.0, 1.0, 0.0))
        brightness_val = min_bright + (brightness / 100.0) * (max_bright - min_bright)
        controls_to_set["Brightness"] = brightness_val

    if 'contrast' in data:
        contrast = float(data.get('contrast'))
        min_contrast, max_contrast, _ = camera.camera_controls.get("Contrast", (0.0, 32.0, 1.0))
        contrast_val = min_contrast + (contrast / 100.0) * (max_contrast - min_contrast)
        controls_to_set["Contrast"] = contrast_val

    if 'sharpness' in data:
        sharpness = float(data.get('sharpness'))
        min_sharp, max_sharp, _ = camera.camera_controls.get("Sharpness", (0.0, 16.0, 1.0))
        sharpness_val = min_sharp + (sharpness / 100.0) * (max_sharp - min_sharp)
        controls_to_set["Sharpness"] = sharpness_val

    if 'ScalerCrop' in data:
        scaler_crop = data.get('ScalerCrop')
        # Ensure scaler_crop is a tuple/list of 4 integers
        if isinstance(scaler_crop, list) and len(scaler_crop) == 4 and all(isinstance(x, int) for x in scaler_crop):
            controls_to_set["ScalerCrop"] = tuple(scaler_crop)
        else:
            print(f"Warning: Invalid ScalerCrop value received: {scaler_crop}")

    safe_set_controls(controls_to_set)
    return "", 204

@app.route('/solved_field.jpg')
def serve_solved_image():
    """Serve the most recent solved image from memory (avoids writing to disk)."""
    with solved_image_lock:
        if solved_image_bytes:
            return Response(solved_image_bytes, mimetype='image/jpeg')
        else:
            return "", 404

if __name__ == '__main__':
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host='0.0.0.0', port=8080, threaded=True)
