import { get, set, subscribe } from './state.js';
import { solveField } from './solve.js';

const MODE_LIVE = 'live';
const MODE_SOLVED = 'solved';

export function initVideo() {
  const videoFeedImg = document.getElementById('video_feed_img');
  const fpsDisplay = document.getElementById('fps_display');
  const videoModeSelect = document.getElementById('video_mode_select');
  const videoModeOverlay = document.getElementById('video_mode_overlay');
  const radecContainer = document.getElementById('radec-container');
  const matchedStarsOverlay = document.getElementById('matched_stars_overlay');
  const pauseButton = document.getElementById('pause_button');

  matchedStarsOverlay.style.display = 'none';

  function updateFeed() {
    const solved = get('currentVideoMode') !== MODE_LIVE;
    const base = solved ? '/solved_field.jpg' : '/video_feed';
    let url = base + '?t=' + new Date().getTime();
    if (solved && !get('showOverlay')) {
      url += '&overlay=0';
    }
    videoFeedImg.src = url;
  }

  function updateOverlay() {
    videoModeOverlay.innerText = get('currentVideoMode').toUpperCase();
  }

  videoModeSelect.addEventListener('change', () => {
    set('currentVideoMode', videoModeSelect.value);
  });

  subscribe((name, value) => {
    if (name === 'showOverlay') {
      updateFeed();
      return;
    }
    if (name !== 'currentVideoMode') return;
    updateOverlay();
    updateFeed();
    if (value === MODE_LIVE) {
      videoModeOverlay.classList.remove('solve-success', 'solve-fail');
      radecContainer.style.display = 'none';
      matchedStarsOverlay.innerText = '';
      matchedStarsOverlay.style.display = 'none';
    } else {
      radecContainer.style.display = 'block';
      if (!get('isSolving')) {
        solveField();
      }
    }
  });

  // Update the feed and FPS display every 100ms.
  setInterval(() => {
    updateFeed();
    fetch('/get_pause_state')
      .then((response) => response.json())
      .then((data) => {
        if (data.is_paused) {
          fpsDisplay.innerText = 'FPS: Paused';
          return;
        }
        const url = get('currentVideoMode') === MODE_LIVE
          ? '/get_fps'
          : '/get_solve_fps';
        fetch(url)
          .then((response) => response.json())
          .then((fps) => {
            fpsDisplay.innerText = `FPS: ${fps.fps}`;
          })
          .catch((error) => console.error('Error fetching FPS:', error));
      })
      .catch((error) => console.error('Error fetching pause state:', error));
  }, 100);

  pauseButton.addEventListener('click', () => {
    fetch('/toggle_pause', { method: 'POST' })
      .then((response) => response.json())
      .then((data) => {
        if (data.is_paused) {
          pauseButton.innerText = 'Resume';
        } else {
          pauseButton.innerText = 'Pause';
          if (get('currentVideoMode') === MODE_SOLVED && !get('isSolving')) {
            solveField();
          }
        }
      });
  });

  updateOverlay();
  updateFeed();
}
