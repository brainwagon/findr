export function initDarkMode() {
  const darkModeToggle = document.getElementById('dark_mode_toggle');

  function applyDarkMode(mode) {
    if (mode === 'enabled') {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  }

  darkModeToggle.addEventListener('click', () => {
    const current = localStorage.getItem('darkMode');
    localStorage.setItem('darkMode', current === 'enabled' ? 'disabled' : 'enabled');
    applyDarkMode(localStorage.getItem('darkMode'));
  });

  // Dark mode is the default.
  if (!localStorage.getItem('darkMode')) {
    localStorage.setItem('darkMode', 'enabled');
  }
  applyDarkMode(localStorage.getItem('darkMode'));
}
