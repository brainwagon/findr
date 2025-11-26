document.addEventListener('DOMContentLoaded', (event) => {
    const videoFeedImg = document.getElementById('video_feed_img');
    const fpsDisplay = document.getElementById('fps_display');
    const videoModeSelect = document.getElementById('video_mode_select');
    const videoModeOverlay = document.getElementById('video_mode_overlay');
    const radecContainer = document.getElementById('radec-container');
    const raDisplay = document.getElementById('ra-display');
    const decDisplay = document.getElementById('dec-display');
    const altDisplay = document.getElementById('alt-display');
    const azDisplay = document.getElementById('az-display');
    const matchedStarsOverlay = document.getElementById('matched_stars_overlay');
    matchedStarsOverlay.style.display = 'none'; // Initialize as hidden

    let currentVideoMode = 'live'; // Default to live mode
    let isSolving = false; // Flag to prevent multiple simultaneous solves

    function updateVideoModeOverlay() {
        if (videoModeOverlay) {
            videoModeOverlay.innerText = currentVideoMode.toUpperCase();
        }
    }

    videoModeSelect.addEventListener('change', () => {
        currentVideoMode = videoModeSelect.value;
        updateVideoModeOverlay();
        updateVideoFeed(); // Update the feed immediately
        if (currentVideoMode === 'live') {
            videoModeOverlay.classList.remove('solve-success', 'solve-fail');
            radecContainer.style.display = 'none';
            matchedStarsOverlay.innerText = '';
            matchedStarsOverlay.style.display = 'none'; // Hide the overlay
        } else if (currentVideoMode === 'solved') {
            radecContainer.style.display = 'block';
            if (!isSolving) {
                solveField();
            }
        }
    });

    function updateVideoFeed() {
        if (videoFeedImg) {
            if (currentVideoMode === 'live') {
                videoFeedImg.src = '/video_feed?t=' + new Date().getTime();
            } else {
                videoFeedImg.src = '/solved_field.jpg?t=' + new Date().getTime();
            }
        }
    }

    // Update video feed and FPS every 100ms (adjust as needed)
    setInterval(() => {
        updateVideoFeed();
        fetch('/get_pause_state') // New endpoint to get pause state
            .then(response => response.json())
            .then(data => {
                if (data.is_paused) {
                    fpsDisplay.innerText = 'FPS: Paused';
                } else if (fpsDisplay) {
                    const fps_url = currentVideoMode === 'live' ? '/get_fps' : '/get_solve_fps';
                    fetch(fps_url)
                        .then(response => response.json())
                        .then(data => {
                            fpsDisplay.innerText = `FPS: ${data.fps}`;
                        })
                        .catch(error => console.error('Error fetching FPS:', error));
                }
            })
            .catch(error => console.error('Error fetching pause state:', error));
    }, 100);

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
    const pauseButton = document.getElementById('pause_button');

    pauseButton.addEventListener('click', () => {
        fetch('/toggle_pause', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.is_paused) {
                pauseButton.innerText = 'Resume';
            } else {
                pauseButton.innerText = 'Pause';
                if (currentVideoMode === 'solved' && !isSolving) {
                    solveField();
                }
            }
        });
    });


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
                if (data.status === 'solved') {
                    raDisplay.innerText = data.ra_hms;
                    decDisplay.innerText = data.dec_dms;
                    altDisplay.innerText = data.alt;
                    azDisplay.innerText = data.az;
                    videoModeOverlay.innerText = 'SOLVE';
                    videoModeOverlay.classList.remove('solve-fail');
                    videoModeOverlay.classList.add('solve-success');
                    matchedStarsOverlay.innerText = data.matched_stars_count + ' stars';
                    matchedStarsOverlay.style.display = 'block'; // Show the overlay
                    clearInterval(solveStatusPollInterval);
                    isSolving = false; // Reset flag
                    if (currentVideoMode === 'solved') {
                        solveField();
                    }
                } else if (data.status === 'failed') {
                    raDisplay.innerText = '--:--:--.-';
                    decDisplay.innerText = '--:--:--.-';
                    altDisplay.innerText = '--.-';
                    azDisplay.innerText = '--.-';
                    videoModeOverlay.innerText = 'FAIL';
                    videoModeOverlay.classList.remove('solve-success');
                    videoModeOverlay.classList.add('solve-fail');
                    matchedStarsOverlay.innerText = '';
                    matchedStarsOverlay.style.display = 'none'; // Hide the overlay
                    clearInterval(solveStatusPollInterval);
                    isSolving = false; // Reset flag
                    if (currentVideoMode === 'solved') {
                        solveField();
                    }
                } else if (data.status === 'paused') {
                    clearInterval(solveStatusPollInterval);
                    isSolving = false; // Reset flag
                } else {
                }
            })
            .catch(error => {
                console.error('Error fetching solver status:', error);
                clearInterval(solveStatusPollInterval);
            });
    }

    function solveField() {
        if (isSolving) return; // Prevent multiple solves

        isSolving = true; // Set flag

        fetch('/solve', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'solving') {
                // Start polling for status
                solveStatusPollInterval = setInterval(pollSolveStatus, 200);
            } else {
                isSolving = false; // Reset flag on failure to start
            }
        })
        .catch(error => {
            console.error('Error:', error);
            isSolving = false; // Reset flag on error
        });
    }



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
    updateVideoModeOverlay(); // Set initial overlay text



    function updateSystemStats() {
        fetch('/system-stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('cpu-temp').innerText = data.cpu_temp;
                document.getElementById('cpu-load').innerText = data.cpu_load;
                document.getElementById('voltage').innerText = data.voltage;
                document.getElementById('current').innerText = data.current;
                const lowVoltageWarningElement = document.getElementById('low-voltage-warning');
                if (data.low_voltage_warning) {
                    lowVoltageWarningElement.style.display = 'inline';
                } else {
                    lowVoltageWarningElement.style.display = 'none';
                }
                document.getElementById('power-source-display').innerText = data.power_source;
                const batteryTimeContainer = document.getElementById('battery-time-remaining-container');
                if (data.power_source === 'BATTERY') {
                    batteryTimeContainer.style.display = 'inline';
                    document.getElementById('battery-time-remaining').innerText = data.battery_time_remaining;
                } else {
                    batteryTimeContainer.style.display = 'none';
                }
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
