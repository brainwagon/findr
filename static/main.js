document.addEventListener('DOMContentLoaded', (event) => {
    const gainSlider = document.getElementById('gain');
    const exposureSlider = document.getElementById('exposure');
    const brightnessSlider = document.getElementById('brightness');
    const contrastSlider = document.getElementById('contrast');
    const sharpnessSlider = document.getElementById('sharpness');
    const aeEnableCheckbox = document.getElementById('ae_enable');
    const exposureValueSlider = document.getElementById('exposure_value');
    const aeExposureModeSelect = document.getElementById('ae_exposure_mode');

    const gainValueSpan = document.getElementById('gain_value');
    const exposureTimeValueSpan = document.getElementById('exposure_time_value');
    const brightnessValueSpan = document.getElementById('brightness_value');
    const contrastValueSpan = document.getElementById('contrast_value');
    const sharpnessValueSpan = document.getElementById('sharpness_value');
    const exposureValueDisplaySpan = document.getElementById('exposure_value_display');

    function toggleExposureValueSlider() {
        if (aeEnableCheckbox.checked) {
            exposureValueSlider.removeAttribute('disabled');
        } else {
            exposureValueSlider.setAttribute('disabled', 'disabled');
        }
    }

    function updateControlValueDisplay() {
        gainValueSpan.innerText = gainSlider.value;

        // Convert exposure slider value (0-100) to microseconds, then to milliseconds
        const exposureSliderValue = parseFloat(exposureSlider.value);
        const minExposure = 29; // From camera_dummy.py
        const maxExposure = 15534385; // From camera_dummy.py
        const exposureMicroseconds = minExposure + (exposureSliderValue / 100.0) * (maxExposure - minExposure);
        exposureTimeValueSpan.innerText = `${(exposureMicroseconds / 1000).toFixed(2)} ms`;

        brightnessValueSpan.innerText = brightnessSlider.value;
        contrastValueSpan.innerText = contrastSlider.value;
        sharpnessValueSpan.innerText = sharpnessSlider.value;

        // Convert ExposureValue slider (0-100) to -8.0 to 8.0
        let actualEv = 0.0;
        if (exposureValueSlider) {
            const evSliderValue = parseFloat(exposureValueSlider.value || '0'); // Default to '0' if value is empty
            actualEv = ((evSliderValue / 100.0) * 16.0) - 8.0;
            exposureValueDisplaySpan.innerText = `${actualEv.toFixed(2)}`;
        } else {
            exposureValueDisplaySpan.innerText = "N/A";
        }
    }

    function sendControls() {
        const controls = {
            gain: gainSlider.value,
            exposure: exposureSlider.value,
            brightness: brightnessSlider.value,
            contrast: contrastSlider.value,
            sharpness: sharpnessSlider.value,
            ae_enable: aeEnableCheckbox.checked,
            exposure_value: exposureValueSlider.value,
            ae_exposure_mode: aeExposureModeSelect.value
        };

        fetch('/set_controls', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(controls)
        });
        updateControlValueDisplay();
    }

    gainSlider.addEventListener('input', sendControls);
    exposureSlider.addEventListener('input', sendControls);
    brightnessSlider.addEventListener('input', sendControls);
    contrastSlider.addEventListener('input', sendControls);
    sharpnessSlider.addEventListener('input', sendControls);
    aeEnableCheckbox.addEventListener('change', () => {
        toggleExposureValueSlider();
        sendControls();
    });
    exposureValueSlider.addEventListener('input', () => {
        sendControls();
    });
    aeExposureModeSelect.addEventListener('change', sendControls);

    const captureRawButton = document.getElementById('capture_raw_button');
    captureRawButton.addEventListener('click', () => {
        fetch('/capture_raw', {
            method: 'POST'
        })
        .then(response => response.text())
        .then(data => {
            alert(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error capturing raw image.');
        });
    });

    // Send initial control values to the backend and update display when the page loads
    toggleExposureValueSlider(); // Set initial state
    sendControls();
});
