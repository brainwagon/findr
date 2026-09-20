export function initStats() {
  function updateSystemStats() {
    fetch('/system-stats')
      .then((response) => response.json())
      .then((data) => {
        document.getElementById('cpu-temp').innerText = data.cpu_temp;
        document.getElementById('cpu-load').innerText = data.cpu_load;
        document.getElementById('voltage').innerText = data.voltage;
        document.getElementById('current').innerText = data.current;

        const lowVoltageWarning = document.getElementById('low-voltage-warning');
        lowVoltageWarning.style.display = data.low_voltage_warning ? 'inline' : 'none';

        document.getElementById('power-source-display').innerText = data.power_source;

        const batteryTimeContainer = document.getElementById('battery-time-remaining-container');
        if (data.power_source === 'BATTERY') {
          batteryTimeContainer.style.display = 'inline';
          document.getElementById('battery-time-remaining').innerText = data.battery_time_remaining;
        } else {
          batteryTimeContainer.style.display = 'none';
        }
      })
      .catch((error) => console.error('Error fetching system stats:', error));
  }

  setInterval(updateSystemStats, 15000);
  updateSystemStats();
}
