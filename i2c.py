# I2C peripheral support module

import smbus2
import time
import os

def get_default_bus():
    """Get the default I2C bus number."""
    for busnum in range(10, -1, -1):
        if os.path.exists(f'/dev/i2c-{busnum}'):
            return busnum
    return 1 # Default to 1 if no bus is found

# I2C bus number to use
I2C_BUS = get_default_bus()

class I2CPeripheral:
    """Base class for an I2C peripheral."""
    def __init__(self, addr):
        self.addr = addr
        try:
            self.bus = smbus2.SMBus(I2C_BUS)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error opening I2C bus: {e}")
            self.bus = None

    def is_present(self):
        """Check if the peripheral is present on the bus."""
        if not self.bus:
            return False
        try:
            self.bus.write_quick(self.addr)
            return True
        except Exception:
            return False

    def get_value(self, name):
        """Get a named value from the peripheral."""
        raise NotImplementedError

    def get_value_names(self):
        """Get a list of value names for the peripheral."""
        raise NotImplementedError

class INA219(I2CPeripheral):
    """INA219 Voltage/Current Sensor"""
    INA219_REG_CONFIG = 0x00
    INA219_REG_SHUNTVOLTAGE = 0x01
    INA219_REG_BUSVOLTAGE = 0x02
    INA219_REG_POWER = 0x03
    INA219_REG_CURRENT = 0x04
    INA219_REG_CALIBRATION = 0x05

    def __init__(self, addr=0x43):
        super().__init__(addr)
        self.shunt_ohms = 0.1
        self.max_expected_amps = 2.0
        self._calibrate()

    def _calibrate(self):
        # Set configuration to default values
        # 16V, GAIN_AUTO, ADC_128S, ADC_128S
        config = 0x019F
        self.bus.write_i2c_block_data(self.addr, self.INA219_REG_CONFIG, [config >> 8, config & 0xFF])

        # Calibrate for 32V, 2A
        cal = 4096
        self.bus.write_i2c_block_data(self.addr, self.INA219_REG_CALIBRATION, [cal >> 8, cal & 0xFF])


    def get_value(self, name):
        if name == "voltage":
            value = self.bus.read_word_data(self.addr, self.INA219_REG_BUSVOLTAGE)
            return ((value >> 8) & 0xFF | (value << 8) & 0xFF00) >> 3 * 4 / 1000.0
        elif name == "current":
            value = self.bus.read_word_data(self.addr, self.INA219_REG_CURRENT)
            return ((value >> 8) & 0xFF | (value << 8) & 0xFF00) / 1000.0
        elif name == "power":
            value = self.bus.read_word_data(self.addr, self.INA219_REG_POWER)
            return ((value >> 8) & 0xFF | (value << 8) & 0xFF00) * 20 / 1000.0
        else:
            return None

    def get_value_names(self):
        return ["voltage", "current", "power"]

class DS3231(I2CPeripheral):
    """DS3231 Real-Time Clock"""
    def __init__(self, addr=0x68):
        super().__init__(addr)

    def _bcd_to_dec(self, bcd):
        return (bcd // 16 * 10) + (bcd % 16)

    def get_value(self, name):
        if name == "datetime":
            try:
                data = self.bus.read_i2c_block_data(self.addr, 0x00, 7)
                sec = self._bcd_to_dec(data[0])
                min = self._bcd_to_dec(data[1])
                hour = self._bcd_to_dec(data[2] & 0x3F)
                day = self._bcd_to_dec(data[4])
                month = self._bcd_to_dec(data[5])
                year = self._bcd_to_dec(data[6]) + 2000
                return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{min:02d}:{sec:02d}"
            except Exception:
                return None
        elif name == "temperature":
            try:
                data = self.bus.read_i2c_block_data(self.addr, 0x11, 2)
                temp = (data[0] << 8 | data[1]) >> 6
                if data[0] & 0x80:
                    temp -= 256
                return temp * 0.25
            except Exception:
                return None
        else:
            return None

    def get_value_names(self):
        return ["datetime", "temperature"]

class BME280(I2CPeripheral):
    """BME280 Temperature/Humidity/Pressure Sensor"""
    def __init__(self, addr=0x76):
        super().__init__(addr)

    def init(self):
        self._load_calibration_data()

    def _load_calibration_data(self):
        # Read calibration data from the sensor
        calib = self.bus.read_i2c_block_data(self.addr, 0x88, 24)
        calib += self.bus.read_i2c_block_data(self.addr, 0xA1, 1)
        calib += self.bus.read_i2c_block_data(self.addr, 0xE1, 7)

        self.dig_T1 = (calib[1] << 8) | calib[0]
        self.dig_T2 = (calib[3] << 8) | calib[2]
        if self.dig_T2 & 0x8000: self.dig_T2 = (-self.dig_T2 ^ 0xFFFF) + 1
        self.dig_T3 = (calib[5] << 8) | calib[4]
        if self.dig_T3 & 0x8000: self.dig_T3 = (-self.dig_T3 ^ 0xFFFF) + 1

        self.dig_P1 = (calib[7] << 8) | calib[6]
        self.dig_P2 = (calib[9] << 8) | calib[8]
        if self.dig_P2 & 0x8000: self.dig_P2 = (-self.dig_P2 ^ 0xFFFF) + 1
        self.dig_P3 = (calib[11] << 8) | calib[10]
        if self.dig_P3 & 0x8000: self.dig_P3 = (-self.dig_P3 ^ 0xFFFF) + 1
        self.dig_P4 = (calib[13] << 8) | calib[12]
        if self.dig_P4 & 0x8000: self.dig_P4 = (-self.dig_P4 ^ 0xFFFF) + 1
        self.dig_P5 = (calib[15] << 8) | calib[14]
        if self.dig_P5 & 0x8000: self.dig_P5 = (-self.dig_P5 ^ 0xFFFF) + 1
        self.dig_P6 = (calib[17] << 8) | calib[16]
        if self.dig_P6 & 0x8000: self.dig_P6 = (-self.dig_P6 ^ 0xFFFF) + 1
        self.dig_P7 = (calib[19] << 8) | calib[18]
        if self.dig_P7 & 0x8000: self.dig_P7 = (-self.dig_P7 ^ 0xFFFF) + 1
        self.dig_P8 = (calib[21] << 8) | calib[20]
        if self.dig_P8 & 0x8000: self.dig_P8 = (-self.dig_P8 ^ 0xFFFF) + 1
        self.dig_P9 = (calib[23] << 8) | calib[22]
        if self.dig_P9 & 0x8000: self.dig_P9 = (-self.dig_P9 ^ 0xFFFF) + 1

        self.dig_H1 = calib[24]
        self.dig_H2 = (calib[26] << 8) | calib[25]
        if self.dig_H2 & 0x8000: self.dig_H2 = (-self.dig_H2 ^ 0xFFFF) + 1
        self.dig_H3 = calib[27]
        self.dig_H4 = (calib[28] << 4) | (calib[29] & 0x0F)
        if self.dig_H4 & 0x800: self.dig_H4 = (-self.dig_H4 ^ 0xFFF) + 1
        self.dig_H5 = (calib[30] << 4) | ((calib[29] >> 4) & 0x0F)
        if self.dig_H5 & 0x800: self.dig_H5 = (-self.dig_H5 ^ 0xFFF) + 1
        self.dig_H6 = calib[31]
        if self.dig_H6 & 0x80: self.dig_H6 = (-self.dig_H6 ^ 0xFF) + 1

    def get_value(self, name):
        """Get a named value from the peripheral."""
        # Set oversampling and mode
        self.bus.write_byte_data(self.addr, 0xF2, 1)  # Humidity oversampling x1
        self.bus.write_byte_data(self.addr, 0xF4, 0x27) # Temp/Pressure oversampling x1, normal mode
        self.bus.write_byte_data(self.addr, 0xF5, 0xA0) # Standby 1000ms, filter off

        # Read raw data
        data = self.bus.read_i2c_block_data(self.addr, 0xF7, 8)
        raw_press = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
        raw_temp = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4)
        raw_hum = (data[6] << 8) | data[7]

        # Compensate temperature
        var1 = (raw_temp / 16384.0 - self.dig_T1 / 1024.0) * self.dig_T2
        var2 = ((raw_temp / 131072.0 - self.dig_T1 / 8192.0) * (raw_temp / 131072.0 - self.dig_T1 / 8192.0)) * self.dig_T3
        t_fine = var1 + var2
        temp = t_fine / 5120.0

        # Compensate pressure
        var1 = (t_fine / 2.0) - 64000.0
        var2 = var1 * var1 * self.dig_P6 / 32768.0
        var2 = var2 + var1 * self.dig_P5 * 2.0
        var2 = (var2 / 4.0) + (self.dig_P4 * 65536.0)
        var1 = (self.dig_P3 * var1 * var1 / 524288.0 + self.dig_P2 * var1) / 524288.0
        var1 = (1.0 + var1 / 32768.0) * self.dig_P1
        if var1 == 0:
            pressure = 0
        else:
            pressure = 1048576.0 - raw_press
            pressure = (pressure - (var2 / 4096.0)) * 6250.0 / var1
            var1 = self.dig_P9 * pressure * pressure / 2147483648.0
            var2 = pressure * self.dig_P8 / 32768.0
            pressure = pressure + (var1 + var2 + self.dig_P7) / 16.0

        # Compensate humidity
        h = t_fine - 76800.0
        h = (raw_hum - (self.dig_H4 * 64.0 + self.dig_H5 / 16384.0 * h)) * \
            (self.dig_H2 / 65536.0 * (1.0 + self.dig_H6 / 67108864.0 * h * \
            (1.0 + self.dig_H3 / 67108864.0 * h)))
        humidity = h * (1.0 - self.dig_H1 * h / 524288.0)
        if humidity > 100:
            humidity = 100
        elif humidity < 0:
            humidity = 0

        if name == "temperature":
            return temp
        elif name == "pressure":
            return pressure
        elif name == "humidity":
            return humidity
        else:
            return None

    def get_value_names(self):
        return ["temperature", "humidity", "pressure"]

# Dictionary to hold detected peripherals
peripherals = {}

def init_peripherals():
    """Detect and initialize I2C peripherals."""
    # INA219
    ina219 = INA219()
    if ina219.is_present():
        ina219.init()
        peripherals["ina219"] = ina219

    # DS3231
    ds3231 = DS3231()
    if ds3231.is_present():
        peripherals["ds3231"] = ds3231

    # BME280
    bme280 = BME280()
    if bme280.is_present():
        bme280.init()
        peripherals["bme280"] = bme280

def get_peripheral_value(peripheral_name, value_name):
    """Get a value from a peripheral."""
    if peripheral_name in peripherals:
        return peripherals[peripheral_name].get_value(value_name)
    return None

def get_peripheral_value_names(peripheral_name):
    """Get the value names for a peripheral."""
    if peripheral_name in peripherals:
        return peripherals[peripheral_name].get_value_names()
    return []

def get_detected_peripherals():
    """Get a list of detected peripherals."""
    return list(peripherals.keys())