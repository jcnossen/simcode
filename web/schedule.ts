export type Schedule = { axes: number; phases: number; interleaved: boolean };
export const defaultSchedule: Schedule = { axes: 2, phases: 3, interleaved: false };
export const axisName = (i: number) => String.fromCharCode(65 + i);
export function axisAt(t: number, s: Schedule) {
  return s.interleaved ? t % s.axes : Math.floor(t / s.phases) % s.axes;
}
export function phaseAt(t: number, s: Schedule) {
  return (s.interleaved ? Math.floor(t / s.axes) : t) % s.phases;
}
export const phaseDegrees = (t: number, s: Schedule) => phaseAt(t, s) * 360 / s.phases;
export const phaseRadians = (t: number, s: Schedule) => phaseAt(t, s) * 2 * Math.PI / s.phases;
export const cycle = (s: Schedule) => Array.from({ length: s.phases * s.axes }, (_, t) => axisAt(t, s)).join('');

export function scheduleControls() {
  const controls = document.createElement('fieldset');
  controls.className = 'pattern-order';
  controls.innerHTML = '<legend>Illumination sequence</legend><label>Number of axes <select></select></label> <label>Number of phases per axis <select class="phases"></select></label> <label><input type="checkbox"> Interleaved axes</label><p>Angle indices <output></output></p><p>Phase steps per axis (°) <output class="steps"></output></p>';
  const select = controls.querySelector('select')!;
  const phases = controls.querySelector<HTMLSelectElement>('.phases')!;
  for (let i = 3; i <= 10; i++) phases.add(new Option(String(i), String(i)));
  phases.value = String(defaultSchedule.phases);
  const checkbox = controls.querySelector('input')!;
  for (let i = 1; i <= 9; i++) select.add(new Option(String(i), String(i)));
  select.value = String(defaultSchedule.axes);
  try {
    const cookie = document.cookie.split('; ').find(c => c.startsWith('simcode_sequence='));
    if (cookie) {
      const [count, interleave, steps] = cookie.split('=')[1].split('-');
      if ([...select.options].some(o => o.value === count)) select.value = count;
      if ([...phases.options].some(o => o.value === steps)) phases.value = steps;
      checkbox.checked = interleave === '1';
    }
  } catch {}
  const read = () => ({ axes: Number(select.value), phases: Number(phases.value), interleaved: checkbox.checked });
  const update = () => {
    const s = read();
    controls.querySelector('output')!.textContent = cycle(s);
    controls.querySelector('.steps')!.textContent = Array.from({length: s.phases}, (_, i) => Number((360*i/s.phases).toFixed(2))).join(', ');
  };
  controls.onchange = () => {
    update();
    try { document.cookie = 'simcode_sequence=' + select.value + '-' + Number(checkbox.checked) + '-' + phases.value + '; Max-Age=31536000; Path=/; SameSite=Lax'; } catch {}
  };
  update();
  document.querySelector('#inputs')!.after(controls);
  return read;
}
