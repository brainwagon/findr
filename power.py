"""Power monitoring.

Holds the register-level INA219 driver, the state-of-charge model, and
`read_power`, which turns a sensor reading into a `PowerReading` carrying the
bus voltage and current, whether the finder is on AC or battery, the estimated
state of charge and time remaining, and the low-voltage warning.

`read_power` accepts any object with `get_bus_voltage()` and `get_current()`,
so tests can pass a fake and production passes the INA219 driver.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# --- INA219 Configuration ---
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
# Bus Voltage Range: 0-32V
# Shunt ADC Resolution: 12-bit, 4 samples (532us conversion time)
# Bus ADC Resolution: 12-bit, 4 samples (532us conversion time)
# Mode: Shunt and Bus, Continuous
CONFIG = 0x199F

# --- Calibration ---
# This value is calculated for a 0.1-ohm shunt resistor and a max expected
# current of 2A. See the INA219 datasheet for the calibration calculation.
# With CALIBRATION_VALUE = 4096:
# current_lsb = 0.04096 / (4096 * 0.1) = 0.0001 A/bit (100uA/bit)
CALIBRATION_VALUE = 4096
CURRENT_LSB = 0.1  # mA per bit
POWER_LSB = 2  # mW per bit (20 * current_lsb)

LOW_VOLTAGE_THRESHOLD = 3.1
DEFAULT_CAPACITY_MAH = 10000


class INA219:
    """A register-level driver for the INA219 voltage/current sensor."""

    def __init__(self, bus, address=INA219_ADDRESS):
        self.bus = bus
        self.address = address
        try:
            self.configure()
            self.calibrate()
        except OSError as e:
            logger.warning("Error configuring or calibrating INA219: %s", e)
            self.address = None  # Mark this instance as invalid

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
        """Read the bus voltage in volts."""
        raw_value = self._read_register(INA219_REG_BUSVOLTAGE)
        # Check for conversion ready bit
        if (raw_value & 0x0002) == 0:
            return 0.0
        # Shift right 3 to remove status bits, LSB is 4mV
        return ((raw_value >> 3) * 4) / 1000.0

    def get_shunt_voltage(self):
        """Read the shunt voltage in mV. The LSB is 10uV."""
        raw_value = self._read_register(INA219_REG_SHUNTVOLTAGE)
        if raw_value > 32767:
            raw_value -= 65536
        return raw_value * 0.01

    def get_current(self):
        """Read the current in mA."""
        raw_current = self._read_register(INA219_REG_CURRENT)
        if raw_current > 32767:
            raw_current -= 65536
        return raw_current * CURRENT_LSB

    def get_power(self):
        """Read the power in mW."""
        return self._read_register(INA219_REG_POWER) * POWER_LSB


@dataclass(frozen=True)
class PowerReading:
    """A snapshot of the finder's power supply."""

    voltage: float
    current: float
    source: str
    low_voltage: bool
    soc: Optional[float] = None
    time_remaining: Optional[str] = None


def open_sensor(bus_number=1):
    """Open the INA219 on the given I2C bus, or return None if unavailable."""
    try:
        from smbus2 import SMBus
    except ImportError:
        logger.warning("smbus2 is not installed; INA219 disabled.")
        return None
    try:
        return INA219(SMBus(bus_number))
    except OSError as e:
        logger.warning("I2C bus unavailable; INA219 disabled: %s", e)
        return None


def estimate_soc(voltage):
    """Estimate the state of charge of a LiPo battery from its voltage."""
    if voltage >= 4.2:
        return 100.0
    elif voltage >= 4.1:
        return 90.0 + (voltage - 4.1) * 100.0
    elif voltage >= 4.0:
        return 80.0 + (voltage - 4.0) * 10.0
    elif voltage >= 3.9:
        return 70.0 + (voltage - 3.9) * 10.0
    elif voltage >= 3.8:
        return 50.0 + (voltage - 3.8) * 20.0
    elif voltage >= 3.7:
        return 30.0 + (voltage - 3.7) * 20.0
    elif voltage >= 3.6:
        return 10.0 + (voltage - 3.6) * 20.0
    elif voltage >= 3.5:
        return 5.0 + (voltage - 3.5) * 10.0
    elif voltage >= 3.0:
        return (voltage - 3.0) * 5.0 / 0.5
    else:
        return 0.0


def read_power(sensor, total_capacity_mah=DEFAULT_CAPACITY_MAH):
    """Read the current power state, or None if no sensor is available.

    Args:
        sensor: an object with get_bus_voltage() and get_current(), or None.
        total_capacity_mah: battery capacity, for the time-remaining estimate.

    Returns:
        A PowerReading, or None when there is no usable sensor.
    """
    if sensor is None or getattr(sensor, "address", None) is None:
        return None

    try:
        bus_voltage = sensor.get_bus_voltage()
        bus_current = sensor.get_current()
    except Exception as e:
        logger.warning("Could not read the power sensor: %s", e)
        return None

    source = "AC"
    soc = None
    time_remaining = None

    # Below the charger's output and drawing real current: running on battery.
    if bus_voltage < 4.1 and bus_current > 100:
        source = "BATTERY"
        soc = estimate_soc(bus_voltage)
        remaining_capacity_mah = total_capacity_mah * (soc / 100.0)
        remaining_time_hours = remaining_capacity_mah / bus_current
        hours = int(remaining_time_hours)
        minutes = int((remaining_time_hours * 60) % 60)
        time_remaining = f"{hours}h {minutes}m"

    return PowerReading(
        voltage=bus_voltage,
        current=bus_current,
        source=source,
        low_voltage=bus_voltage < LOW_VOLTAGE_THRESHOLD,
        soc=soc,
        time_remaining=time_remaining,
    )
