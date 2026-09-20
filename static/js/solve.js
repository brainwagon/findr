import { get, set } from './state.js';

const MODE_SOLVED = 'solved';
let solveStatusPollInterval;

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

function pollSolveStatus() {
  fetch('/solve_status')
    .then((response) => response.json())
    .then((data) => {
      if (data.status === 'solved' || data.status === 'failed') {
        if (data.status === 'solved') {
          showSolved(data);
        } else {
          showFailed();
        }
        clearInterval(solveStatusPollInterval);
        set('isSolving', false);
        if (get('currentVideoMode') === MODE_SOLVED) {
          setTimeout(solveField, 500);
        }
      } else if (data.status === 'paused') {
        clearInterval(solveStatusPollInterval);
        set('isSolving', false);
      }
    })
    .catch((error) => {
      console.error('Error fetching solver status:', error);
      clearInterval(solveStatusPollInterval);
    });
}

export function solveField() {
  if (get('isSolving')) return;
  set('isSolving', true);

  fetch('/solve', { method: 'POST' })
    .then((response) => response.json())
    .then((data) => {
      if (data.status === 'solving') {
        solveStatusPollInterval = setInterval(pollSolveStatus, 200);
      } else {
        set('isSolving', false);
      }
    })
    .catch((error) => {
      console.error('Error:', error);
      set('isSolving', false);
    });
}
