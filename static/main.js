document.addEventListener('DOMContentLoaded', (event) => {
    const videoFeedImg = document.getElementById('video_feed_img');
    const fpsDisplay = document.getElementById('fps_display');
    const videoModeSelect = document.getElementById('video_mode_select');

    let currentVideoMode = 'live'; // Default to live mode

    videoModeSelect.addEventListener('change', () => {
        currentVideoMode = videoModeSelect.value;
        updateVideoFeedAndFPS(); // Update the feed immediately
    });

    // Function to update the video feed and fetch FPS
    function updateVideoFeedAndFPS() {
        if (videoFeedImg && currentVideoMode === 'live') {
            // Add a timestamp to the URL to prevent caching and force reload
            videoFeedImg.src = '/video_feed?t=' + new Date().getTime();
        }

        if (fpsDisplay) {
            fetch('/get_fps')
                .then(response => response.json())
                .then(data => {
                    fpsDisplay.innerText = `FPS: ${data.fps}`;
                })
                .catch(error => console.error('Error fetching FPS:', error));
        }
    }

    // Update video feed and FPS every 100ms (adjust as needed)
    setInterval(updateVideoFeedAndFPS, 100);

    const gainSelect = document.getElementById('gain_select');
    const exposureSelect = document.getElementById('exposure_select');
    const brightnessSlider = document.getElementById('brightness');
    const contrastSlider = document.getElementById('contrast');
    const sharpnessSlider = document.getElementById('sharpness');



    const brightnessValueSpan = document.getElementById('brightness_value');
    const contrastValueSpan = document.getElementById('contrast_value');
    const sharpnessValueSpan = document.getElementById('sharpness_value');

    const saveSettingsButton = document.getElementById('save_settings_button');
    const zoomSelect = document.getElementById('zoom_select');
    const testModeCheckbox = document.getElementById('test_mode_checkbox');

    // Hardcoded sensor dimensions
    const sensorWidth = 1456;
    const sensorHeight = 1088;

    function saveSettings() {
        const settings = {
            gain: gainSelect.value,
            exposure_index: exposureSelect.value,
            zoom_setting: zoomSelect.value,
            test_mode: testModeCheckbox.checked
        };
        localStorage.setItem('cameraSettings', JSON.stringify(settings));
        alert('Camera settings saved!');
    }

    function sendScalerCrop(crop) {
        fetch('/set_controls', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"ScalerCrop": crop})
        });
    }

    zoomSelect.addEventListener('change', () => {
        const selectedZoom = zoomSelect.value;
        let crop = [0, 0, sensorWidth, sensorHeight]; // Default to full sensor

        if (selectedZoom === '640x480') {
            crop = [408, 304, 640, 480];
        } else if (selectedZoom === '320x240') {
            crop = [568, 424, 320, 240];
        }
        sendScalerCrop(crop);
    });

    function loadSettings() {
        const savedSettings = localStorage.getItem('cameraSettings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            gainSelect.value = settings.gain;
            exposureSelect.value = settings.exposure_index;
            if (settings.zoom_setting !== undefined) {
                zoomSelect.value = settings.zoom_setting;
                // Manually trigger the change event to apply the crop
                zoomSelect.dispatchEvent(new Event('change'));
            }
            if (settings.test_mode !== undefined) {
                testModeCheckbox.checked = settings.test_mode;
                sendTestMode(); // Send the loaded state to the backend
            }
            // sendControls will be called later in DOMContentLoaded, so no need to call it here
        }
    }

    loadSettings(); // Load settings as soon as the DOM is ready

    function updateControlValueDisplay() {

        brightnessValueSpan.innerText = brightnessSlider.value;
        contrastValueSpan.innerText = contrastSlider.value;
        sharpnessValueSpan.innerText = sharpnessSlider.value;


    }

    function sendControls() {
        const controls = {
            gain: gainSelect.value,
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

    gainSelect.addEventListener('change', sendControls);
    exposureSelect.addEventListener('change', sendControls);
    brightnessSlider.addEventListener('input', sendControls);
    contrastSlider.addEventListener('input', sendControls);
    sharpnessSlider.addEventListener('input', sendControls);


    const captureLoresJpegButton = document.getElementById('capture_lores_jpeg_button');
    const captureFullJpegButton = document.getElementById('capture_full_jpeg_button');
    const solveFieldButton = document.getElementById('solve_field_button');

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

    let solveStatusPollInterval;

    function pollSolveStatus() {
        fetch('/solve_status')
            .then(response => response.json())
            .then(data => {
                const solverStatusEl = document.getElementById('solver-status');
                const solverResultEl = document.getElementById('solver-result');

                if (data.status === 'solved') {
                    solverStatusEl.innerText = 'Solved';
                    solverResultEl.innerText = `RA: ${data.ra}, Dec: ${data.dec}, Roll: ${data.roll}, Solution Time: ${data.solution_time} Constellation: ${data.constellation}`;
                    videoFeedImg.src = data.solved_image_url + '?t=' + new Date().getTime(); // Add timestamp to avoid caching
                    videoModeSelect.value = 'solved';
                    currentVideoMode = 'solved';
                    clearInterval(solveStatusPollInterval);
                    solveFieldButton.disabled = false;
                } else if (data.status === 'failed') {
                    solverStatusEl.innerText = 'Solver failed.';
                    solverResultEl.innerText = '';
                    if (data.solved_image_url) {
                        videoFeedImg.src = data.solved_image_url + '?t=' + new Date().getTime();
                    } else {
                        videoFeedImg.src = '/static/black_640x480.jpg'; // Display black image on failure
                    }
                    videoModeSelect.value = 'solved';
                    currentVideoMode = 'solved';
                    clearInterval(solveStatusPollInterval);
                    solveFieldButton.disabled = false;
                } else {
                    solverStatusEl.innerText = `Solver status: ${data.status}`;
                }
            })
            .catch(error => {
                console.error('Error fetching solver status:', error);
                clearInterval(solveStatusPollInterval);
                solveFieldButton.disabled = false;
            });
    }

    function solveField() {
        const solverStatusEl = document.getElementById('solver-status');
        solverStatusEl.innerText = 'Starting solver...';
        solveFieldButton.disabled = true;

        fetch('/solve', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'solving') {
                solverStatusEl.innerText = 'Solving...';
                // Start polling for status
                solveStatusPollInterval = setInterval(pollSolveStatus, 200);
            } else {
                solverStatusEl.innerText = 'Failed to start solver.';
                solveFieldButton.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            solverStatusEl.innerText = 'Error starting solver.';
            solveFieldButton.disabled = false;
        });
    }

    solveFieldButton.addEventListener('click', solveField);

    saveSettingsButton.addEventListener('click', saveSettings);

    function sendTestMode() {
        fetch('/set_test_mode', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ test_mode: testModeCheckbox.checked })
        });
    }

    testModeCheckbox.addEventListener('change', sendTestMode);

    // Send initial control values to the backend and update display when the page loads
    sendControls();


    function updateSystemStats() {
        fetch('/system-stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('cpu-temp').innerText = data.cpu_temp;
                document.getElementById('cpu-load').innerText = data.cpu_load;
            })
            .catch(error => console.error('Error fetching system stats:', error));
    }

    // Fetch system stats every 5 seconds
    setInterval(updateSystemStats, 15000);
    // Initial call to populate system stats
    updateSystemStats();

    const darkModeToggle = document.getElementById('dark_mode_toggle');

    function applyDarkMode(darkMode) {
        if (darkMode === 'enabled') {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    }

    darkModeToggle.addEventListener('click', () => {
        let darkMode = localStorage.getItem('darkMode');
        if (darkMode === 'enabled') {
            localStorage.setItem('darkMode', 'disabled');
        } else {
            localStorage.setItem('darkMode', 'enabled');
        }
        applyDarkMode(localStorage.getItem('darkMode'));
    });

    // Apply dark mode on page load
    applyDarkMode(localStorage.getItem('darkMode'));
});
