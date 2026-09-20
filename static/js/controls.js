import { get, set } from './state.js';
import { solveField } from './solve.js';

const DEFAULT_SENSOR = { sensorWidth: 1456, sensorHeight: 1088 };

export function initControls() {
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
  const overlayCheckbox = document.getElementById('overlay_checkbox');
  const boundariesCheckbox = document.getElementById('boundaries_checkbox');

  function updateControlValueDisplay() {
    brightnessValueSpan.innerText = brightnessSlider.value;
    contrastValueSpan.innerText = contrastSlider.value;
    sharpnessValueSpan.innerText = sharpnessSlider.value;
  }

  // Send only the control that changed. Sending the whole set on load would
  // push the UI's defaults at the camera, which on a USB camera means taking
  // it off auto exposure into a dark manual exposure before the user has asked.
  function sendControl(control) {
    fetch('/set_controls', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(control),
    });
    updateControlValueDisplay();
  }

  function sendScalerCrop(crop) {
    fetch('/set_controls', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ScalerCrop: crop }),
    });
  }

  function sendTestMode() {
    fetch('/set_test_mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test_mode: testModeCheckbox.checked }),
    });
  }

  function saveSettings() {
    const settings = {
      gain: gainSelect.value,
      exposure_index: exposureSelect.value,
      zoom_setting: zoomSelect.value,
      test_mode: testModeCheckbox.checked,
    };
    localStorage.setItem('cameraSettings', JSON.stringify(settings));
    alert('Camera settings saved!');
  }

  function loadSettings() {
    const savedSettings = localStorage.getItem('cameraSettings');
    if (!savedSettings) return;
    const settings = JSON.parse(savedSettings);
    gainSelect.value = settings.gain;
    exposureSelect.value = settings.exposure_index;
    if (settings.zoom_setting !== undefined) {
      zoomSelect.value = settings.zoom_setting;
      // Trigger the change event to apply the crop.
      zoomSelect.dispatchEvent(new Event('change'));
    }
    if (settings.test_mode !== undefined) {
      testModeCheckbox.checked = settings.test_mode;
      sendTestMode();
    }
    if (settings.gain !== undefined || settings.exposure_index !== undefined) {
      sendControl({
        gain: settings.gain,
        exposure_index: settings.exposure_index,
      });
    }
  }

  zoomSelect.addEventListener('change', () => {
    const { sensorWidth, sensorHeight } = window.findrConfig || DEFAULT_SENSOR;
    let crop = [0, 0, sensorWidth, sensorHeight];
    if (zoomSelect.value === '640x480') {
      crop = [408, 304, 640, 480];
    } else if (zoomSelect.value === '320x240') {
      crop = [568, 424, 320, 240];
    }
    sendScalerCrop(crop);
  });

  gainSelect.addEventListener('change', () => {
    sendControl({ gain: gainSelect.value });
  });
  exposureSelect.addEventListener('change', () => {
    sendControl({ exposure_index: exposureSelect.value });
  });
  brightnessSlider.addEventListener('input', () => {
    sendControl({ brightness: brightnessSlider.value });
  });
  contrastSlider.addEventListener('input', () => {
    sendControl({ contrast: contrastSlider.value });
  });
  sharpnessSlider.addEventListener('input', () => {
    sendControl({ sharpness: sharpnessSlider.value });
  });
  testModeCheckbox.addEventListener('change', sendTestMode);
  overlayCheckbox.addEventListener('change', () => {
    set('showOverlay', overlayCheckbox.checked);
  });
  boundariesCheckbox.addEventListener('change', () => {
    fetch('/set_boundaries', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ boundaries: boundariesCheckbox.checked }),
    });
    // The server skips the boundary work at solve time, so refresh the
    // displayed image with a new solve.
    if (get('currentVideoMode') !== 'live' && !get('isSolving')) {
      solveField();
    }
  });
  saveSettingsButton.addEventListener('click', saveSettings);

  loadSettings();
}
