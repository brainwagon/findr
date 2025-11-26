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
import tetra3
from PIL import Image, ImageDraw, ImageFont
import ephem
import configparser
import csv
import requests
import i2c
try:
    from smbus2 import SMBus
except ImportError:
    print("The smbus2 library is not installed. Please install it.")
    # exit(1) # exit is not needed here...

# --- INA219 Configuration ---
# Default I2C address
INA219_ADDRESS = 0x43

# Register Addresses
INA219_REG_CONFIG = 0x00
INA219_REG_SHUNTVOLTAGE = 0x01
INA219_REG_BUSVOLTAGE = 0x02
INA219_REG_POWER = 0x03
INA219_REG_CURRENT = 0x04
INA219_REG_CALIBRATION = 0x05

# --- Configuration Settings ---
# These settings are for a 32V, 2A range.
# Adjust them according to your specific needs.
# Bus Voltage Range: 0-32V
# Shunt ADC Resolution: 12-bit, 4 samples (532us conversion time)
# Bus ADC Resolution: 12-bit, 4 samples (532us conversion time)
# Mode: Shunt and Bus, Continuous
CONFIG = 0x199F

# --- Calibration ---
# This value is calculated for a 0.1-ohm shunt resistor and a max expected current of 2A.
# See the INA219 datasheet for the calibration calculation.
# For a 0.1 ohm shunt, and 2A max current:
# current_lsb = 2A / 32768 = 61.035uA/bit -> round to 60uA/bit for calculation
# cal = 0.04096 / (current_lsb * 0.1) = 0.04096 / (0.00006 * 0.1) = 6826
# We will use a more standard value of 4096 which is for 3.2A max current and 0.1 ohm shunt
CALIBRATION_VALUE = 4096
# With CALIBRATION_VALUE = 4096:
# current_lsb = 0.04096 / (4096 * 0.1) = 0.0001 A/bit (100uA/bit)
CURRENT_LSB = 0.1  # mA per bit
POWER_LSB = 2  # mW per bit (20 * current_lsb)


class INA219:
    """
    A class to interact with the INA219 sensor directly over I2C.
    """
    def __init__(self, bus, address=INA219_ADDRESS):
        self.bus = bus
        self.address = address
        try:
            self.configure()
            self.calibrate()
        except OSError as e:
            print(f"Error configuring or calibrating INA219: {e}")
            self.address = None # Mark this instance as invalid

    def _write_register(self, register, value):
        """Write a 16-bit value to a register."""
        # The INA219 expects the data in big-endian format.
        data = [(value >> 8) & 0xFF, value & 0xFF]
        self.bus.write_i2c_block_data(self.address, register, data)

    def _read_register(self, register):
        """Read a 16-bit value from a register."""
        data = self.bus.read_i2c_block_data(self.address, register, 2)
        return (data[0] << 8) | data[1]

    def configure(self):
        """Configure the INA219 with the default settings."""
        self._write_register(INA219_REG_CONFIG, CONFIG)

    def calibrate(self):
        """Set the calibration register."""
        self._write_register(INA219_REG_CALIBRATION, CALIBRATION_VALUE)

    def get_bus_voltage(self):
        """
        Reads the bus voltage.
        The LSB is 4mV. The result is shifted right by 3 bits.
        """
        raw_value = self._read_register(INA219_REG_BUSVOLTAGE)
        # Check for conversion ready bit
        if (raw_value & 0x0002) == 0:
            return 0.0
        # Shift right 3 to remove status bits and get the voltage reading
        voltage_reading = (raw_value >> 3) * 4
        return voltage_reading / 1000.0  # Convert mV to V

    def get_shunt_voltage(self):
        """
        Reads the shunt voltage.
        The LSB is 10uV. Result is in mV.
        """
        raw_value = self._read_register(INA219_REG_SHUNTVOLTAGE)
        # Convert to signed value if necessary
        if raw_value > 32767:
            raw_value -= 65536
        return raw_value * 0.01  # Convert 10uV steps to mV

    def get_current(self):
        """Reads the current from the sensor in mA."""
        raw_current = self._read_register(INA219_REG_CURRENT)
        # Convert to signed value if necessary
        if raw_current > 32767:
            raw_current -= 65536
        return raw_current * CURRENT_LSB

# --- End of INA219 Class ---

# Try to initialize the INA219 sensor
try:
    i2c_bus = SMBus(1)
    ina219 = INA219(i2c_bus)
    print("INA219 sensor initialized.")
except (FileNotFoundError, NameError):
    ina219 = None
    print("I2C bus not found or smbus2 not installed. INA219 sensor disabled.")

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

tetra = tetra3.Tetra3()

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

from picamera2 import Picamera2


app = Flask(__name__)

import atexit

def cleanup():
    """Close the camera and I2C bus on exit."""
    camera.close()
    if ina219:
        i2c_bus.close()
    print("Camera and I2C bus closed.")

atexit.register(cleanup)

# Solver status
solver_status = "idle"
solver_result = {}
test_mode = False # Global variable for test mode
is_paused = False # Global variable for pause state

@app.route('/')
def index():
    return render_template('index.html')

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
    safe_set_controls(data)
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

            # Build solver_result
            solution_time_val = solution.get("T_solve", 0.0)
            ra_hms = ephem.hours(radians(solution['RA']))
            dec_dms = ephem.degrees(radians(solution['Dec']))

            # now we can compute the alt/az for the solved center.
            observer.date = ephem.now()
            target = ephem.FixedBody()
            target._ra = ra_hms
            target._dec = dec_dms
            target.compute(observer)
            
            solver_result = {
                "ra": f"{solution['RA']:.4f}",
                "dec": f"{solution['Dec']:.4f}",
                "roll": f"{solution['Roll']:.4f}",
                "ra_hms": format_radec_fixed_width(ra_hms, is_ra=True, total_width=10, decimal_places=1),
                "dec_dms": format_radec_fixed_width(dec_dms, is_ra=False, total_width=11, decimal_places=1),
                "alt": f"{math.degrees(target.alt):.1f}",
                "az": f"{math.degrees(target.az):.1f}",
                "solved_image_url": "/solved_field.jpg",
                "solution_time": f"{solution_time_val:.2f}ms",
                "constellation": ephem.constellation((radians(solution['RA']), radians(solution['Dec'])))[0],
                "matched_stars_count": len(solution.get("matched_catID", [])),
            }

            solver_status = "solved"

            # MTV here is some magic... 
            # We asked for the solver to return the list of matched stars.
            # The data will include both the RA/DEC (in degrees) for each 
            # of the stars, as well as the RA/DEC.   Using the two, we 
            # can ask astropy to compute an appropriate conversion object,
            # including the possibility of adding the distortion parameter.

            try:

                matched_stars = np.array(solution["matched_stars"])
                matched_centroids = np.array(solution["matched_centroids"])

                # x is returned as the second column by tetra3
                # y is the first column

                star_x = matched_centroids[:,1]
                star_y = matched_centroids[:,0]
                star_xy = (star_x, star_y)

                star_ra = np.array(matched_stars[:,0]) * u.deg
                star_dec = np.array(matched_stars[:,1]) * u.deg

                world_coords = SkyCoord(ra = star_ra, dec = star_dec, frame = 'icrs')

                # now, fit the model.. 
                wcs = fit_wcs_from_points(
                    star_xy,
                    world_coords, 
                    projection='TAN',
                    sip_degree=2)

                # Draw constellation boundaries
                constellation_name = solver_result.get("constellation").upper()
                if constellation_name and constellation_name in constellation_boundaries:
                    points = constellation_boundaries[constellation_name]
                    # we need to transform the points to pixel coordinates
                    pixel_points = []
                    for ra, dec in points:
                        try:
                            px, py = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
                            pixel_points.append((px, py))
                        except Exception as e:
                            print(f"Error converting point to pixel: {e}")
                            pixel_points.append(None)
                    
                    # Draw lines between consecutive points
                    for i in range(len(pixel_points) - 1):
                        p1 = pixel_points[i]
                        p2 = pixel_points[i+1]
                        # relies on clipping from the ImageDraw library...
                        draw.line([p1, p2], fill="yellow", width=1)

            except Exception as e:
                print(f"EXCEPTION DURING WCS HANDLING: {e}")

            # now that we have modified the image, we must re-save it
            # before we return.
            #
            # Save the combined image into memory (JPEG)
            buf = io.BytesIO()
            combined_image.save(buf, format='JPEG')
            buf.seek(0)
            with solved_image_lock:
                solved_image_bytes = buf.getvalue()

            # send the center to stellarium...
            if False:
                point_stellarium(radians(solution['RA']), radians(solution['Dec']))

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
    
    voltage = "N/A"
    current = "N/A"
    if ina219 and ina219.address is not None:
        try:
            voltage = f"{ina219.get_bus_voltage():.2f}"
            current = f"{ina219.get_current():.2f}"
        except Exception as e:
            # On the first pass, this might fail if the sensor is not
            # yet ready.   We can ignore it.
            pass


    return jsonify(cpu_temp=f"{temp:.1f}", cpu_load=load, voltage=voltage, current=current)

@app.route('/set_test_mode', methods=['POST'])
def set_test_mode():
    """Set the test mode state."""
    global test_mode
    data = request.json
    test_mode = data.get('test_mode', False)
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
