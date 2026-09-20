import { get, set, subscribe } from './state.js';
import { solveField } from './solve.js';

const MODE_LIVE = 'live';
const MODE_SOLVED = 'solved';
const FEED_REFRESH_MS = 200;
const STATUS_POLL_MS = 1000;

export function initVideo() {
  const videoFeedImg = document.getElementById('video_feed_img');
  const fpsDisplay = document.getElementById('fps_display');
  const videoModeSelect = document.getElementById('video_mode_select');
  const videoModeOverlay = document.getElementById('video_mode_overlay');
  const radecContainer = document.getElementById('radec-container');
  const matchedStarsOverlay = document.getElementById('matched_stars_overlay');
  const pauseButton = document.getElementById('pause_button');

  matchedStarsOverlay.style.display = 'none';

  let currentUrl = null;

  function updateFeed() {
    const solved = get('currentVideoMode') !== MODE_LIVE;
    const base = solved ? '/solved_field.jpg' : '/video_feed';
    let url = base + '?t=' + new Date().getTime();
    if (solved && !get('showOverlay')) {
      url += '&overlay=0';
    }
    currentUrl = url;
    videoFeedImg.src = url;
  }

  // Refresh the solved image when a new solve lands, but no faster than
  // FEED_REFRESH_MS, so a fast solve loop cannot flood the browser.
  let feedPending = false;
  function scheduleFeed() {
    if (feedPending) return;
    feedPending = true;
    setTimeout(() => {
      feedPending = false;
      if (get('currentVideoMode') !== MODE_LIVE) {
        updateFeed();
      }
    }, FEED_REFRESH_MS);
  }

  function updateOverlay() {
    videoModeOverlay.innerText = get('currentVideoMode').toUpperCase();
  }

  videoModeSelect.addEventListener('change', () => {
    set('currentVideoMode', videoModeSelect.value);
  });

  // If the stream or image drops, retry once, but only while that URL is still
  // the one we want (changing src fires an error on the abandoned request).
  videoFeedImg.addEventListener('error', () => {
    const droppedUrl = currentUrl;
    setTimeout(() => {
      if (currentUrl === droppedUrl && droppedUrl) {
        videoFeedImg.src = droppedUrl + '&r=' + Date.now();
      }
    }, 1000);
  });

  subscribe((name, value) => {
    if (name === 'showOverlay') {
      updateFeed();
      return;
    }
    if (name === 'solveTick') {
      scheduleFeed();
      return;
    }
    if (name === 'feedTick') {
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

  // Update the pause/FPS readout; the image itself is refreshed on demand.
  setInterval(() => {
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
  }, STATUS_POLL_MS);

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
