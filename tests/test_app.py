import unittest

import ephem
from PIL import Image

import app as app_module
from app import create_app
from catalog import Catalog


class FakeCamera:
    """A Camera adapter with no hardware behind it."""

    def __init__(self):
        self._controls = {}

    @property
    def camera_properties(self):
        return {"Model": "fake", "PixelArraySize": (640, 480)}

    @property
    def camera_controls(self):
        return {
            "AnalogueGain": (1.0, 100.0, 1.0),
            "ExposureTime": (1, 1000000, 10000),
        }

    def create_still_configuration(self, **kwargs):
        return {}

    def configure(self, config):
        pass

    def start(self):
        pass

    def close(self):
        pass

    def capture_file(self, buffer, name=None, format="jpeg"):
        Image.new("RGB", (8, 8), "grey").save(buffer, format=format)

    def set_controls(self, controls):
        self._controls.update(controls)


class FakeSolverManager:
    """A Solver manager with no star database behind it."""

    def __init__(self):
        self.current = 'tetra3'

    def available_solvers(self):
        return ['tetra3', 'cedar-solve']

    def get_current_solver_type(self):
        return self.current

    def set_solver(self, solver_type):
        if solver_type not in self.available_solvers():
            raise ValueError(f"Unknown solver type: {solver_type}")
        self.current = solver_type

    def solve(self, image):
        return None


def make_app():
    observer = ephem.Observer()
    observer.lat = "0"
    observer.lon = "0"
    app = create_app(FakeCamera(), None, observer, Catalog(), FakeSolverManager())
    app.config['TESTING'] = True
    return app


class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        self.app = make_app()
        self.client = self.app.test_client()

    def test_index(self):
        self.assertEqual(self.client.get('/').status_code, 200)

    def test_solve_status_idle(self):
        self.assertEqual(
            self.client.get('/solve_status').get_json(), {"status": "idle"}
        )

    def test_system_stats_without_sensor(self):
        data = self.client.get('/system-stats').get_json()
        self.assertEqual(data['power_source'], "N/A")
        self.assertEqual(data['voltage'], "N/A")
        self.assertEqual(data['battery_time_remaining'], "N/A")
        self.assertFalse(data['low_voltage_warning'])

    def test_get_solver(self):
        data = self.client.get('/get_solver').get_json()
        self.assertEqual(data['available'], ['tetra3', 'cedar-solve'])
        self.assertEqual(data['current'], 'tetra3')

    def test_set_solver(self):
        response = self.client.post('/set_solver', json={'solver': 'cedar-solve'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['current'], 'cedar-solve')

    def test_set_solver_rejects_unknown(self):
        response = self.client.post('/set_solver', json={'solver': 'nope'})
        self.assertEqual(response.status_code, 500)

    def test_set_test_mode(self):
        response = self.client.post('/set_test_mode', json={'test_mode': True})
        self.assertEqual(response.status_code, 204)
        self.assertTrue(self.app.config['STATE'].test_mode)

    def test_toggle_pause(self):
        response = self.client.post('/toggle_pause')
        self.assertTrue(response.get_json()['is_paused'])

    def test_snapshot_is_an_image(self):
        response = self.client.get('/snapshot')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/jpeg')

    def test_solved_field_is_an_image(self):
        response = self.client.get('/solved_field.jpg')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/jpeg')

    def test_state_is_per_app(self):
        other = make_app()
        other.config['STATE'].test_mode = True
        self.assertFalse(self.app.config['STATE'].test_mode)


class TestNoImportSideEffects(unittest.TestCase):
    def test_import_does_not_build_hardware_or_app(self):
        self.assertFalse(hasattr(app_module, 'camera'))
        self.assertFalse(hasattr(app_module, 'ina219'))
        self.assertFalse(hasattr(app_module, 'app'))
        self.assertTrue(hasattr(app_module, 'main'))


if __name__ == "__main__":
    unittest.main()
