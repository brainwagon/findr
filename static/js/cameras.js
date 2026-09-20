import { get, set } from './state.js';

export function initCameras() {
  const cameraSelect = document.getElementById('camera_select');

  function refresh() {
    fetch('/cameras')
      .then((response) => response.json())
      .then((data) => {
        cameraSelect.innerHTML = '';
        data.available.forEach((camera) => {
          const option = document.createElement('option');
          option.value = camera.id;
          option.text = camera.label;
          if (camera.id === data.current) {
            option.selected = true;
          }
          cameraSelect.appendChild(option);
        });
      })
      .catch((error) => console.error('Error fetching cameras:', error));
  }

  cameraSelect.addEventListener('change', () => {
    fetch('/set_camera', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera: cameraSelect.value }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          alert('Error changing camera: ' + data.error);
          refresh();
        } else {
          // A different camera is a different view, so drop back to live and
          // reconnect the preview stream to be sure it is showing this camera.
          set('currentVideoMode', 'live');
          set('feedTick', get('feedTick') + 1);
        }
      })
      .catch((error) => {
        console.error('Error:', error);
        alert('Failed to change camera.');
        refresh();
      });
  });

  refresh();
}
