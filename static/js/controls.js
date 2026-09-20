import { set } from './state.js';

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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(controls),
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

  gainSelect.addEventListener('change', sendControls);
  exposureSelect.addEventListener('change', sendControls);
  brightnessSlider.addEventListener('input', sendControls);
  contrastSlider.addEventListener('input', sendControls);
  sharpnessSlider.addEventListener('input', sendControls);
  testModeCheckbox.addEventListener('change', sendTestMode);
  overlayCheckbox.addEventListener('change', () => {
    set('showOverlay', overlayCheckbox.checked);
  });
  saveSettingsButton.addEventListener('click', saveSettings);

  loadSettings();
  sendControls();
}
