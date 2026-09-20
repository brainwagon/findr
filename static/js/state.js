// The few pieces of UI state shared across concerns.
const values = {
  currentVideoMode: 'live',
  isSolving: false,
  showOverlay: true,
  solveTick: 0,
  feedTick: 0,
};

const listeners = new Set();

export function get(name) {
  return values[name];
}

export function set(name, value) {
  if (values[name] === value) return;
  values[name] = value;
  listeners.forEach((listener) => listener(name, value));
}

export function subscribe(listener) {
  listeners.add(listener);
}
