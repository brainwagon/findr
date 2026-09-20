export function initSolver() {
  const solverSelect = document.getElementById('solver_select');

  function refresh() {
    fetch('/get_solver')
      .then((response) => response.json())
      .then((data) => {
        solverSelect.innerHTML = '';
        data.available.forEach((solver) => {
          const option = document.createElement('option');
          option.value = solver;
          option.text = solver;
          if (solver === data.current) {
            option.selected = true;
          }
          solverSelect.appendChild(option);
        });
      })
      .catch((error) => console.error('Error fetching solver info:', error));
  }

  solverSelect.addEventListener('change', () => {
    fetch('/set_solver', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ solver: solverSelect.value }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          alert('Error changing solver: ' + data.error);
          refresh();
        } else {
          console.log('Solver changed to:', data.current);
        }
      })
      .catch((error) => {
        console.error('Error:', error);
        alert('Failed to change solver.');
        refresh();
      });
  });

  refresh();
}
