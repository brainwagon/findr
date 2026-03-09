# Initial Concept
A prototype telescope finder providing a web-based interface to control a Raspberry Pi camera remotely.

# Product Guide

## Vision
The findr platform aims to empower amateur astronomers and DIY telescope makers by providing a simple, robust, and accessible web-based interface for remote control and monitoring of Raspberry Pi-based telescope cameras.

## Target Audience
- **Amateur Astronomers:** Using Raspberry Pi telescopes for observations and celestial monitoring.
- **DIY Telescope Makers:** Hobbyists building custom telescope finder scopes and looking for integrated control solutions.

## Core Goals
- **Live Monitoring:** Deliver a stable, low-latency live MJPEG video stream from the telescope camera to any web browser.
- **Remote Control:** Enable precise remote adjustment of camera settings (gain, exposure, white balance, etc.) through an intuitive web interface.
- **Web Accessibility:** Ensure the control interface is easily accessible from various devices on the same network without complex setup.
- **Plate Solving:** Integrate plate solving capabilities using libraries like cedar-solve to provide real-time information on the camera's pointing direction.

## Core Features
- **MJPEG Streaming:** Real-time video stream displayed in the web interface for immediate feedback.
- **Camera Parameter Adjustment:** Comprehensive controls for gain, exposure, white balance, and other camera-specific parameters.
- **Integrated Plate Solving:** Leverage libraries like cedar-solve to analyze captured imagery and determine current celestial coordinates.
- **Development Mode:** A dedicated dummy camera interface to facilitate software development on machines without a Raspberry Pi camera.
