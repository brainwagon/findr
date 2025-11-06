from flask import Flask, render_template, Response, request
import io
import numpy as np
from astropy.io import fits
import datetime

try:
    from picamera2 import Picamera2
except ImportError:
    from camera_dummy import Picamera2

app = Flask(__name__)

camera = Picamera2()

# Initialize camera and set initial controls once
camera.configure(camera.create_still_configuration())
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
    "AeEnable": True,
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
        camera.capture_file(buffer, format='jpeg')
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
    slider_values['gain'] = int(((current_gain - min_gain) / (max_gain - min_gain)) * 100) if (max_gain - min_gain) != 0 else 0

    # ExposureTime
    min_exp, max_exp = 10000, 500000
    current_exposure = max(min_exp, min(getattr(current_controls, "ExposureTime", 10000), max_exp))
    slider_values['exposure'] = int(((current_exposure - min_exp) / (max_exp - min_exp)) * 100) if (max_exp - min_exp) != 0 else 0
    slider_values['exposure_ms'] = f"{current_exposure / 1000:.2f} ms"

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

    # AeEnable
    current_ae_enable = getattr(current_controls, "AeEnable", False)
    slider_values['ae_enable'] = current_ae_enable

    # ExposureValue
    min_ev, max_ev, _ = camera.camera_controls.get("ExposureValue", (-8.0, 8.0, 0.0))
    current_exposure_value = getattr(current_controls, "ExposureValue", 0.0)
    slider_values['exposure_value'] = int(((current_exposure_value - min_ev) / (max_ev - min_ev)) * 100) if (max_ev - min_ev) != 0 else 0
    slider_values['exposure_value_display'] = f"{current_exposure_value:.2f}"

    # AeExposureMode
    current_ae_exposure_mode = getattr(current_controls, "AeExposureMode", 0)
    slider_values['ae_exposure_mode'] = current_ae_exposure_mode

    return render_template('index.html', model=model, pixel_array_size=pixel_array_size, **slider_values)

@app.route('/video_feed')
def video_feed():
    """Return the video feed."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_controls', methods=['POST'])
def set_controls():
    """Set camera controls."""
    gain = float(request.json.get('gain'))
    exposure = float(request.json.get('exposure'))
    brightness = float(request.json.get('brightness'))
    contrast = float(request.json.get('contrast'))
    sharpness = float(request.json.get('sharpness'))
    ae_enable = request.json.get('ae_enable') # This will be a boolean
    exposure_value = float(request.json.get('exposure_value', 50.0))
    ae_exposure_mode = int(request.json.get('ae_exposure_mode', 0))

    # Map slider values (0-100) to camera control values
    min_gain, max_gain, _ = camera.camera_controls.get("AnalogueGain", (1.0, 251.1886444091797, 1.0))
    analogue_gain = min_gain + (gain / 100.0) * (max_gain - min_gain)
    min_exp, max_exp = 10000, 500000
    exposure_time = int(min_exp + (exposure / 100.0) * (max_exp - min_exp))
    min_bright, max_bright, _ = camera.camera_controls.get("Brightness", (-1.0, 1.0, 0.0))
    brightness_val = min_bright + (brightness / 100.0) * (max_bright - min_bright)
    min_contrast, max_contrast, _ = camera.camera_controls.get("Contrast", (0.0, 32.0, 1.0))
    contrast_val = min_contrast + (contrast / 100.0) * (max_contrast - min_contrast)
    min_sharp, max_sharp, _ = camera.camera_controls.get("Sharpness", (0.0, 16.0, 1.0))
    sharpness_val = min_sharp + (sharpness / 100.0) * (max_sharp - min_sharp)

    # Map ExposureValue slider (0-100) to -8.0 to 8.0
    min_ev, max_ev, _ = camera.camera_controls.get("ExposureValue", (-8.0, 8.0, 0.0))
    exposure_value_val = min_ev + (exposure_value / 100.0) * (max_ev - min_ev)
    
    controls_to_set = {
        "AeEnable": ae_enable,
        "AnalogueGain": analogue_gain,
        "ExposureTime": exposure_time,
        "Brightness": brightness_val,
        "Contrast": contrast_val,
        "Sharpness": sharpness_val,
        "ExposureValue": exposure_value_val,
        "AeExposureMode": ae_exposure_mode
    }
    safe_set_controls(controls_to_set)
    return "", 204

@app.route('/capture_raw', methods=['POST'])
def capture_raw():
    """Capture a full resolution raw image and save it as a FITS file."""
    try:
        camera.stop()
        # Configure for raw capture
        raw_config = camera.create_still_configuration(raw={'format': 'R10_CSI2P'})
        camera.configure(raw_config)
        camera.start()

        # Capture the raw data
        stream = io.BytesIO()
        camera.capture_file(stream, format='raw')
        stream.seek(0)

        # Create a FITS file
        # The raw data is 10-bit, so we need to unpack it.
        # This is a simplified example; real unpacking might be more complex
        # and depend on the specific camera's output format.
        data = np.frombuffer(stream.read(), dtype=np.uint16)
        
        # Reshape the data based on the sensor's resolution
        # This is an example, you might need to adjust the dimensions
        sensor_resolution = camera.camera_properties['PixelArraySize']
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
        # Reconfigure back to the default configuration
        camera.stop()
        camera.configure(camera.create_still_configuration())
        camera.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
