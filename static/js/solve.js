import { get, set } from './state.js';

const MODE_SOLVED = 'solved';
const POLL_INTERVAL_MS = 50;

// Guards against two solve loops running at once. Without it, overlapping
// status polls can each start a new loop, which piles up requests until the
// browser runs out of connections (net::ERR_INSUFFICIENT_RESOURCES).
let running = false;

function showSolved(data) {
  document.getElementById('ra-display').innerText = data.ra_hms;
  document.getElementById('dec-display').innerText = data.dec_dms;
  document.getElementById('alt-display').innerText = data.alt;
  document.getElementById('az-display').innerText = data.az;

  const overlay = document.getElementById('video_mode_overlay');
  overlay.innerText = 'SOLVE';
  overlay.classList.remove('solve-fail');
  overlay.classList.add('solve-success');

  const stars = document.getElementById('matched_stars_overlay');
  stars.innerText = data.matched_stars_count + ' stars';
  stars.style.display = 'block';
}

function showFailed() {
  document.getElementById('ra-display').innerText = '--:--:--.-';
  document.getElementById('dec-display').innerText = '--:--:--.-';
  document.getElementById('alt-display').innerText = '--.-';
  document.getElementById('az-display').innerText = '--.-';

  const overlay = document.getElementById('video_mode_overlay');
  overlay.innerText = 'FAIL';
  overlay.classList.remove('solve-success');
  overlay.classList.add('solve-fail');

  const stars = document.getElementById('matched_stars_overlay');
  stars.innerText = '';
  stars.style.display = 'none';
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function solveLoop() {
  if (running) return;
  running = true;
  set('isSolving', true);
  try {
    // Keep solving for as long as Solved mode is selected. Each iteration
    // waits for the previous solve to finish, so requests never overlap and
    // the loop paces itself at whatever rate the backend can manage.
    while (get('currentVideoMode') === MODE_SOLVED) {
      const started = await fetch('/solve', { method: 'POST' });
      const startData = await started.json();
      if (startData.status !== 'solving') break;

      let data;
      do {
        await delay(POLL_INTERVAL_MS);
        const response = await fetch('/solve_status');
        data = await response.json();
      } while (data.status === 'solving');

      if (data.status === 'solved') {
        showSolved(data);
      } else if (data.status === 'failed') {
        showFailed();
      } else {
        // "paused", or an unexpected status: stop the loop.
        break;
      }
      set('solveTick', get('solveTick') + 1);
    }
  } catch (error) {
    console.error('Solve loop stopped:', error);
  } finally {
    running = false;
    set('isSolving', false);
  }
}

export function solveField() {
  solveLoop();
}
