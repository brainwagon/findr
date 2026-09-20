#!/usr/bin/env python3
"""Standalone INA219 power monitor.

Reads the sensor in a loop and prints its values. The INA219 driver lives in
power.py, so there is only one implementation of it.
"""

import time

import power


def main():
    """Initialize the sensor and print its values in a loop."""
    sensor = power.open_sensor(1)
    if sensor is None:
        print("INA219 not available. Check that I2C is enabled and smbus2 "
              "is installed.")
        return

    print("INA219 sensor reader initialized.")
    print("---------------------------------")
    try:
        while True:
            print(f"Bus Voltage:    {sensor.get_bus_voltage():.2f} V")
            print(f"Shunt Voltage:  {sensor.get_shunt_voltage():.2f} mV")
            print(f"Current:        {sensor.get_current():.2f} mA")
            print(f"Power:          {sensor.get_power():.2f} mW")
            print("---------------------------------")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        if sensor.bus:
            sensor.bus.close()
            print("I2C bus closed.")


if __name__ == "__main__":
    main()
