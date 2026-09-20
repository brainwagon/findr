import unittest

import power


class FakeSensor:
    """An INA219-shaped sensor with fixed readings."""

    def __init__(self, voltage, current):
        self.address = 0x43
        self._voltage = voltage
        self._current = current

    def get_bus_voltage(self):
        return self._voltage

    def get_current(self):
        return self._current


class BrokenSensor(FakeSensor):
    def get_bus_voltage(self):
        raise OSError("bus error")


class TestEstimateSoc(unittest.TestCase):
    def test_full(self):
        self.assertEqual(power.estimate_soc(4.2), 100.0)

    def test_midpoint(self):
        self.assertEqual(power.estimate_soc(3.8), 50.0)

    def test_empty(self):
        self.assertEqual(power.estimate_soc(2.9), 0.0)


class TestReadPower(unittest.TestCase):
    def test_no_sensor_returns_none(self):
        self.assertIsNone(power.read_power(None))

    def test_invalid_sensor_returns_none(self):
        sensor = FakeSensor(5.0, 50)
        sensor.address = None
        self.assertIsNone(power.read_power(sensor))

    def test_unreadable_sensor_returns_none(self):
        self.assertIsNone(power.read_power(BrokenSensor(5.0, 50)))

    def test_ac_source(self):
        reading = power.read_power(FakeSensor(5.0, 50))
        self.assertEqual(reading.source, "AC")
        self.assertIsNone(reading.time_remaining)
        self.assertIsNone(reading.soc)
        self.assertFalse(reading.low_voltage)
        self.assertAlmostEqual(reading.voltage, 5.0)
        self.assertAlmostEqual(reading.current, 50)

    def test_battery_source_with_time_remaining(self):
        # 3.8 V -> 50% of 10000 mAh = 5000 mAh; at 500 mA -> 10h 0m.
        reading = power.read_power(FakeSensor(3.8, 500))
        self.assertEqual(reading.source, "BATTERY")
        self.assertEqual(reading.soc, 50.0)
        self.assertEqual(reading.time_remaining, "10h 0m")

    def test_low_voltage_warning(self):
        reading = power.read_power(FakeSensor(3.0, 500))
        self.assertTrue(reading.low_voltage)

    def test_capacity_argument(self):
        reading = power.read_power(FakeSensor(3.8, 500), total_capacity_mah=2000)
        self.assertEqual(reading.time_remaining, "2h 0m")


if __name__ == "__main__":
    unittest.main()
