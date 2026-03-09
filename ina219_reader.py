#!/usr/bin/env python3
import time

# Try to import smbus2, if not found, provide installation instructions
try:
    from smbus2 import SMBus
except ImportError:
    print("The smbus2 library is not installed.")
    print("Please install it by running: sudo pip3 install smbus2")
    exit(1)

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
        self.configure()
        self.calibrate()

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
        print(f"INA219 configured with value: {CONFIG:#06x}")

    def calibrate(self):
        """Set the calibration register."""
        self._write_register(INA219_REG_CALIBRATION, CALIBRATION_VALUE)
        print(f"INA219 calibrated with value: {CALIBRATION_VALUE}")

    def get_bus_voltage(self):
        """
        Reads the bus voltage.
        The LSB is 4mV. The result is shifted right by 3 bits.
        """
        raw_value = self._read_register(INA219_REG_BUSVOLTAGE)
        # Check for conversion ready bit
        if (raw_value & 0x0002) == 0:
            print("Warning: Conversion not ready for bus voltage.")
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

    def get_power(self):
        """Reads the power from the sensor in mW."""
        raw_power = self._read_register(INA219_REG_POWER)
        return raw_power * POWER_LSB


def main():
    """
    Main function to initialize the sensor and read values in a loop.
    """
    try:
        # Initialize I2C bus. Assumes bus 1.
        bus = SMBus(1)
        ina = INA219(bus)
        print("INA219 sensor reader initialized.")
        print("---------------------------------")
    except FileNotFoundError:
        print("I2C bus not found. Make sure I2C is enabled on your Raspberry Pi.")
        print("You can enable it using 'sudo raspi-config'.")
        return
    except Exception as e:
        print(f"An error occurred during initialization: {e}")
        return

    try:
        while True:
            bus_voltage = ina.get_bus_voltage()
            shunt_voltage_mv = ina.get_shunt_voltage()
            current_ma = ina.get_current()
            power_mw = ina.get_power()

            print(f"Bus Voltage:    {bus_voltage:.2f} V")
            print(f"Shunt Voltage:  {shunt_voltage_mv:.2f} mV")
            print(f"Current:        {current_ma:.2f} mA")
            print(f"Power:          {power_mw:.2f} mW")
            print("---------------------------------")

            time.sleep(2)
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        bus.close()
        print("I2C bus closed.")


if __name__ == "__main__":
    main()
