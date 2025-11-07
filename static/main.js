document.addEventListener('DOMContentLoaded', (event) => {
    const gainSlider = document.getElementById('gain');
    const exposureSelect = document.getElementById('exposure_select');
    const brightnessSlider = document.getElementById('brightness');
    const contrastSlider = document.getElementById('contrast');
    const sharpnessSlider = document.getElementById('sharpness');



    const brightnessValueSpan = document.getElementById('brightness_value');
    const contrastValueSpan = document.getElementById('contrast_value');
    const sharpnessValueSpan = document.getElementById('sharpness_value');




    function updateControlValueDisplay() {

        brightnessValueSpan.innerText = brightnessSlider.value;
        contrastValueSpan.innerText = contrastSlider.value;
        sharpnessValueSpan.innerText = sharpnessSlider.value;


    }

    function sendControls() {
        const controls = {
            gain: gainSlider.value,
            exposure_index: exposureSelect.value,
            brightness: brightnessSlider.value,
            contrast: contrastSlider.value,
            sharpness: sharpnessSlider.value,

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
    exposureSelect.addEventListener('change', sendControls);
    brightnessSlider.addEventListener('input', sendControls);
    contrastSlider.addEventListener('input', sendControls);
    sharpnessSlider.addEventListener('input', sendControls);

    const gainDownButton = document.getElementById('gain_down_button');
    const gainUpButton = document.getElementById('gain_up_button');

    gainDownButton.addEventListener('click', () => {
        gainSlider.value = Math.max(parseInt(gainSlider.value) - 1, gainSlider.min);
        sendControls();
    });

    gainUpButton.addEventListener('click', () => {
        gainSlider.value = Math.min(parseInt(gainSlider.value) + 1, gainSlider.max);
        sendControls();
    });


    const captureLoresJpegButton = document.getElementById('capture_lores_jpeg_button');
    const captureFullJpegButton = document.getElementById('capture_full_jpeg_button');
    const captureFullFitsButton = document.getElementById('capture_full_fits_button');

    captureLoresJpegButton.addEventListener('click', () => {
        fetch('/capture_lores_jpeg')
        .then(response => {
            if (response.ok) {
                return response.blob();
            } else {
                throw new Error('Failed to capture lores JPEG.');
            }
        })
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `lores_jpeg_${new Date().toISOString()}.jpg`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error capturing lores JPEG.');
        });
    });

    captureFullJpegButton.addEventListener('click', () => {
        fetch('/snapshot')
        .then(response => {
            if (response.ok) {
                return response.blob();
            } else {
                throw new Error('Failed to capture full JPEG.');
            }
        })
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `full_jpeg_${new Date().toISOString()}.jpg`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error capturing full JPEG.');
        });
    });

    captureFullFitsButton.addEventListener('click', () => {
        fetch('/capture_full_fits', {
            method: 'POST'
        })
        .then(response => response.text())
        .then(data => {
            alert(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error capturing full FITS image.');
        });
    });

    // Send initial control values to the backend and update display when the page loads

    sendControls();
});