import { initVideo } from './video.js';
import { initControls } from './controls.js';
import { initStats } from './stats.js';
import { initSolver } from './solver.js';
import { initCameras } from './cameras.js';
import { initDarkMode } from './darkmode.js';

document.addEventListener('DOMContentLoaded', () => {
  initDarkMode();
  initVideo();
  initControls();
  initStats();
  initSolver();
  initCameras();
});
