"""The findr web application.

`create_app` builds a Flask app from injected hardware and reference data;
`main` assembles the real adapters and runs the server. Importing this module
has no side effects: it does not open a camera or I2C bus, start a thread, or
serve anything.
"""

import atexit
import configparser
import io
import logging
import os
import threading
import time

import ephem
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image

import camera as camera_module
import power
from catalog import load_catalog
from solve import (
    CameraImageSource,
    ImageSourceError,
    SolveOutcome,
    SolveStore,
    TestImageSource,
    run_solve,
)
from solver import get_solver

logger = logging.getLogger(__name__)

EXPOSURE_TIMES = [1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]


class AppState:
    """The adapters and mutable state behind one running findr app."""

    def __init__(self, camera, sensor, observer, catalog, solver):
        self.camera = camera
        self.sensor = sensor
        self.observer = observer
        self.catalog = catalog
        self.solver = solver
        self.solve_store = SolveStore()
        self.test_mode = False
        self.is_paused = False
        self.latest_frame_bytes = None
        self.current_fps = 0.0
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.solve_fps = 0.0
        self.solve_completed_count = 0

    def set_controls(self, controls):
        """Set only the controls the camera actually exposes."""
        available = self.camera.camera_controls
        safe = {k: v for k, v in controls.items() if k in available}
        if safe:
            self.camera.set_controls(safe)

    def capture_and_process_frames(self):
        """Continuously capture frames, track FPS, and keep the latest frame."""
        while True:
            if self.is_paused:
                time.sleep(0.1)
                continue
            try:
                buffer = io.BytesIO()
                self.camera.capture_file(buffer, name='lores', format='jpeg')
                self.latest_frame_bytes = buffer.getvalue()

                self.frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.last_frame_time
                if elapsed_time >= 1.0:
                    self.current_fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.last_frame_time = current_time
                time.sleep(0.01)
            except Exception as e:
                print(f"Error capturing frame: {e}")
                time.sleep(1)

    def calculate_solve_fps(self):
        """Continuously recalculate the solve FPS."""
        while True:
            time.sleep(5)
            self.solve_fps = self.solve_completed_count / 5.0
            self.solve_completed_count = 0

    def solve_plate(self):
        """Acquire an image, solve it, and store the outcome."""
        try:
            if self.is_paused:
                self.solve_store.set_status("paused")
                return

            source = (
                TestImageSource() if self.test_mode
                else CameraImageSource(self.camera)
            )
            try:
                image = source.acquire()
            except ImageSourceError as e:
                self.solve_store.finish(SolveOutcome(error=str(e)))
                return

            try:
                outcome = run_solve(
                    image, self.solver, self.observer, self.catalog
                )
            except Exception as e:
                logger.error("Error in solve_plate: %s", e)
                outcome = SolveOutcome(error=str(e), image=image)

            self.solve_store.finish(outcome, _encode_jpeg(outcome.image))
        finally:
            self.solve_completed_count += 1

    def start_solve(self):
        """Begin a solve in a background thread."""
        self.solve_store.begin()
        threading.Thread(target=self.solve_plate).start()

    def start_background(self):
        """Start the capture and solve-FPS daemon threads."""
        for target in (self.capture_and_process_frames, self.calculate_solve_fps):
            threading.Thread(target=target, daemon=True).start()


def _encode_jpeg(image):
    """Encode a PIL image to JPEG bytes."""
    if image is None:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _gen_frames(state):
    """Yield the latest frame as an MJPEG stream."""
    while True:
        if state.latest_frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + state.latest_frame_bytes + b'\r\n')
        time.sleep(0.05)


def create_app(camera, sensor, observer, catalog, solver):
    """Build the Flask app for the given hardware and reference data."""
    state = AppState(camera, sensor, observer, catalog, solver)
    app = Flask(__name__)
    app.config['STATE'] = state

    @app.route('/')
    def index():
        properties = camera.camera_properties
        sensor_width, sensor_height = camera_module.STILL_CONFIGURATION["main"]["size"]
        try:
            exposure_index = EXPOSURE_TIMES.index(10000)
        except ValueError:
            exposure_index = 2
        return render_template(
            'index.html',
            model=properties.get('Model', 'N/A'),
            pixel_array_size=str(properties.get('PixelArraySize', 'N/A')),
            gain=1,
            exposure_index=exposure_index,
            exposure_times=EXPOSURE_TIMES,
            brightness=50,
            contrast=50,
            sharpness=50,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            test_mode=state.test_mode,
        )

    @app.route('/video_feed')
    def video_feed():
        return Response(
            _gen_frames(state),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )

    @app.route('/toggle_pause', methods=['POST'])
    def toggle_pause():
        state.is_paused = not state.is_paused
        return jsonify({"is_paused": state.is_paused})

    @app.route('/get_pause_state')
    def get_pause_state():
        return jsonify({"is_paused": state.is_paused})

    @app.route('/get_fps')
    def get_fps():
        return jsonify({"fps": f"{state.current_fps:.2f}"})

    @app.route('/get_solve_fps')
    def get_solve_fps():
        return jsonify({"fps": f"{state.solve_fps:.2f}"})

    @app.route('/set_controls', methods=['POST'])
    def set_controls():
        data = request.json
        controls_to_set = {}

        if 'gain' in data:
            controls_to_set['AnalogueGain'] = float(data['gain'])

        if 'exposure_index' in data and data['exposure_index'] != '':
            try:
                exposure_idx = int(data['exposure_index'])
                if 0 <= exposure_idx < len(EXPOSURE_TIMES):
                    controls_to_set['ExposureTime'] = EXPOSURE_TIMES[exposure_idx]
            except (ValueError, TypeError):
                pass  # Ignore if not a valid index

        if 'brightness' in data:
            # scale from 0-100 to -1.0 to 1.0
            controls_to_set['Brightness'] = float(data['brightness']) / 50.0 - 1.0

        if 'contrast' in data:
            # scale from 0-100 to 0.0 to 2.0
            controls_to_set['Contrast'] = float(data['contrast']) / 50.0

        if 'ScalerCrop' in data:
            controls_to_set['ScalerCrop'] = data['ScalerCrop']

        state.set_controls(controls_to_set)
        return "", 204

    @app.route('/capture_lores_jpeg')
    def capture_lores_jpeg():
        if state.latest_frame_bytes:
            return Response(state.latest_frame_bytes, mimetype='image/jpeg')
        return "No frame available", 404

    @app.route('/snapshot')
    def snapshot():
        buffer = io.BytesIO()
        camera.capture_file(buffer, name='main', format='jpeg')
        return Response(buffer.getvalue(), mimetype='image/jpeg')

    @app.route('/solved_field.jpg')
    def solved_field():
        image_bytes = state.solve_store.get_image_bytes()
        if image_bytes:
            return Response(image_bytes, mimetype='image/jpeg')
        # Return a black image if no solved image is available
        image = Image.new('RGB', (640, 480), color='black')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return Response(buffer.getvalue(), mimetype='image/jpeg')

    @app.route('/solve', methods=['POST'])
    def solve():
        state.start_solve()
        return jsonify({"status": "solving"})

    @app.route('/solve_status')
    def get_solve_status():
        status, outcome = state.solve_store.snapshot()
        if status in ("solved", "failed") and outcome is not None:
            payload = {"status": status, "solved_image_url": "/solved_field.jpg"}
            payload.update(outcome.to_json())
            return jsonify(payload)
        return jsonify({"status": status})

    @app.route('/system-stats')
    def system_stats():
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

        reading = power.read_power(state.sensor)

        return jsonify(
            cpu_temp=f"{temp:.1f}" if isinstance(temp, float) else temp,
            cpu_load=load,
            voltage=f"{reading.voltage:.2f}" if reading else "N/A",
            current=f"{reading.current:.2f}" if reading else "N/A",
            low_voltage_warning=reading.low_voltage if reading else False,
            power_source=reading.source if reading else "N/A",
            battery_time_remaining=reading.time_remaining if reading else "N/A",
        )

    @app.route('/set_test_mode', methods=['POST'])
    def set_test_mode():
        state.test_mode = request.json.get('test_mode', False)
        return "", 204

    @app.route('/get_solver')
    def get_solver_info():
        return jsonify({
            'current': solver.get_current_solver_type(),
            'available': solver.available_solvers(),
        })

    @app.route('/set_solver', methods=['POST'])
    def set_solver():
        data = request.get_json()
        solver_type = data.get('solver')
        if not solver_type:
            return jsonify({'error': 'No solver specified'}), 400
        try:
            solver.set_solver(solver_type)
            return jsonify({
                'status': 'success',
                'current': solver.get_current_solver_type(),
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


def _make_observer(config_path='location.ini'):
    """Build an ephem Observer from the site in location.ini."""
    config = configparser.ConfigParser()
    config.read(config_path)
    observer = ephem.Observer()
    observer.lat = config.get('location', 'lat', fallback='0')
    observer.lon = config.get('location', 'lon', fallback='0')
    return observer


def main():
    """Assemble the real adapters and run the server."""
    camera = camera_module.open_camera()

    sensor = power.open_sensor(1)
    if sensor:
        print("INA219 sensor initialized.")
    else:
        print("I2C bus not found or smbus2 not installed. INA219 sensor disabled.")

    app = create_app(
        camera, sensor, _make_observer(), load_catalog(), get_solver()
    )
    app.config['STATE'].start_background()

    def cleanup():
        camera.close()
        if sensor:
            sensor.bus.close()
        print("Camera and I2C bus closed.")

    atexit.register(cleanup)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)


if __name__ == "__main__":
    main()
