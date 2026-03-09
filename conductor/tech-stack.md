# Technology Stack

## Language
- **Python:** The primary programming language used for both core logic and the web application backend.

## Backend Framework
- **Flask:** A lightweight and flexible WSGI web application framework for Python, ideal for building simple yet powerful control interfaces.

## Core Libraries & APIs
- **picamera:** Provides a pure Python interface to the Raspberry Pi camera module, allowing direct control over various camera settings.
- **numpy:** Used for efficient numerical computations and array manipulations, likely involved in image processing or plate solving.
- **pillow (PIL Fork):** Provides image processing capabilities, such as resizing, cropping, and format conversion.
- **cedar-solve:** A high-performance, lost-in-space star tracker plate solver (forked from tetra3).
- **scipy:** Required by cedar-solve for scientific computing and optimization.

## Hardware Platform
- **Raspberry Pi:** The target hardware platform for running the findr application and controlling the telescope camera.

## Change Log
- **2026-03-08:** Added `tetra3` and `scipy` to the tech stack for plate solving functionality.
- **2026-03-09:** Replaced `tetra3` with `cedar-solve` and consolidated plate solving architecture.
- **2026-03-09:** Removed `smbus2` and legacy I2C power monitoring functionality.
