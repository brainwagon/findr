from flask import Flask, render_template, Response, request
import io
import numpy as np
from astropy.io import fits
import datetime

from picamera2 import Picamera2

app = Flask(__name__)

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
    raw = {
        "size" : (1456, 1088),
        "format" : "R10_CSI2P",
        }
        )
print("STILL IMAGE CONFIGURATION")
for k, v in config.items():
	print(k, v)
camera.configure(config)
camera.start()

# Print available controls for debugging
print("\nAvailable Camera Controls:")
print("--------------------------------------------------------------------------------")
print(f"{"Control Name":<25} {"Min":<15} {"Max":<15} {"Default":<15}")
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

def gen_frames():
    """Generate frames for the video stream."""
    while True:
        buffer = io.BytesIO()
        camera.capture_file(buffer, name='lores', format='jpeg')
        frame = buffer.getvalue()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Return the main page with initial slider values."""
    # Get camera properties
    camera_properties = camera.camera_properties
    model = camera_properties.get("Model", "Unknown")
    pixel_array_size = camera_properties.get("PixelArraySize", "Unknown")

    # Get current camera controls
    current_controls = camera.controls

    # Map current camera values to slider values (0-100)
    slider_values = {}

    # AnalogueGain
    min_gain, max_gain, _ = camera.camera_controls.get("AnalogueGain", (1.0, 251.1886444091797, 1.0))
    current_gain = getattr(current_controls, "AnalogueGain", 1.0)
    slider_values['gain'] = int(((current_gain - min_gain) / (max_gain - min_gain)) * 9) + 1 if (max_gain - min_gain) != 0 else 1

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



    return render_template('index.html', model=model, pixel_array_size=pixel_array_size, **slider_values)

@app.route('/video_feed')
def video_feed():
    """Return the video feed."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    gain = float(request.json.get('gain'))

    brightness = float(request.json.get('brightness'))
    contrast = float(request.json.get('contrast'))
    sharpness = float(request.json.get('sharpness'))


    # Map slider values (0-100) to camera control values
    min_gain, max_gain, _ = camera.camera_controls.get("AnalogueGain", (1.0, 251.1886444091797, 1.0))
    analogue_gain = min_gain + ((gain - 1) / 9.0) * (max_gain - min_gain)
    exposure_times = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 125000, 250000, 500000, 1000000]
    exposure_index = int(request.json.get('exposure_index', 0))
    exposure_time = exposure_times[exposure_index]
    min_bright, max_bright, _ = camera.camera_controls.get("Brightness", (-1.0, 1.0, 0.0))
    brightness_val = min_bright + (brightness / 100.0) * (max_bright - min_bright)
    min_contrast, max_contrast, _ = camera.camera_controls.get("Contrast", (0.0, 32.0, 1.0))
    contrast_val = min_contrast + (contrast / 100.0) * (max_contrast - min_contrast)
    min_sharp, max_sharp, _ = camera.camera_controls.get("Sharpness", (0.0, 16.0, 1.0))
    sharpness_val = min_sharp + (sharpness / 100.0) * (max_sharp - min_sharp)

    controls_to_set = {
        "AeEnable": False,
        "AnalogueGain": analogue_gain,
        "ExposureTime": exposure_time,
        "Brightness": brightness_val,
        "Contrast": contrast_val,
        "Sharpness": sharpness_val
    }
    safe_set_controls(controls_to_set)
    return "", 204

@app.route('/capture_full_fits', methods=['POST'])
def capture_full_fits():
    """Capture a full resolution raw image and save it as a FITS file."""
    try:
        request = camera.capture_request() 
        raw_array = request.make_array("raw")
        metadata = request.get_metadata() 
        print("Got a capture request.")
        print(metadata)

        # Capture the raw data from the 'raw' stream
        stream = io.BytesIO()
        camera.capture_file(stream, name='raw', format='raw')
        stream.seek(0)

        # The raw data is 10-bit, so we need to unpack it.
        # This is a simplified example; real unpacking might be more complex
        # and depend on the specific camera's output format.
        # Assuming 10-bit packed into 16-bit for simplicity in dummy camera
        data = np.frombuffer(stream.read(), dtype=np.uint16)
        
        # Reshape the data based on the sensor's resolution
        # Ensure the dimensions match the full resolution (1456x1088)
        sensor_resolution = (1456, 1088) # Hardcode for clarity, but could get from camera_properties
        data = data.reshape(sensor_resolution[1], sensor_resolution[0])

        # Create an HDU
        hdu = fits.PrimaryHDU(data)

        # Add some metadata
        hdu.header['EXPOSURE'] = camera.controls.get('ExposureTime')
        hdu.header['GAIN'] = camera.controls.get('AnalogueGain')
        hdu.header['DATE'] = datetime.datetime.now().isoformat()

        # Generate a filename
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = f"capture-{timestamp}.fits"
        
        # Save the FITS file
        hdu.writeto(filename, overwrite=True)

        return f"Successfully captured raw image to {filename}", 200
    except Exception as e:
        return str(e), 500
    finally: 
        request.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
