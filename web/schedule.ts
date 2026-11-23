export type Schedule = { axes: number; phases: number; interleaved: boolean; estimatePhases?: boolean };
export const defaultSchedule: Schedule = { axes: 2, phases: 3, interleaved: false };
export const axisName = (i: number) => String.fromCharCode(65 + i);
export function axisAt(t: number, s: Schedule) {
  return s.interleaved ? t % s.axes : Math.floor(t / s.phases) % s.axes;
}
export function phaseAt(t: number, s: Schedule) {
  return (s.interleaved ? Math.floor(t / s.axes) : t) % s.phases;
}
export const phaseDegrees = (t: number, s: Schedule) => (phaseAt(t, s) * 360) / s.phases;
export const phaseRadians = (t: number, s: Schedule) => (phaseAt(t, s) * 2 * Math.PI) / s.phases;
export const cycle = (s: Schedule) =>
  Array.from({ length: s.phases * s.axes }, (_, t) => axisAt(t, s)).join('');

export function scheduleControls() {
  const controls = document.createElement('fieldset');
  controls.className = 'pattern-order';
  controls.innerHTML =
    '<legend>Illumination sequence</legend><label>Number of axes <select></select></label> <label>Number of phases per axis <select class="phases"></select></label> <label><input type="checkbox"> Interleaved axes</label> <label><input class="free-phases" type="checkbox"> Estimate phases individually</label><p>Angle indices <output></output></p><p>Nominal phase steps per axis (°) <output class="steps"></output></p><small class="phase-note"></small>';
  const select = controls.querySelector('select')!;
  const phases = controls.querySelector<HTMLSelectElement>('.phases')!;
  for (let i = 1; i <= 10; i++) phases.add(new Option(String(i), String(i)));
  phases.value = String(defaultSchedule.phases);
  const free = controls.querySelector<HTMLInputElement>('.free-phases')!;
  const checkbox = controls.querySelector('input')!;
  for (let i = 1; i <= 9; i++) select.add(new Option(String(i), String(i)));
  select.value = String(defaultSchedule.axes);
  try {
    const cookie = document.cookie.split('; ').find((c) => c.startsWith('simcode_sequence='));
    if (cookie) {
      const [count, interleave, steps, fitSteps] = cookie.split('=')[1].split('-');
      if ([...select.options].some((o) => o.value === count)) select.value = count;
      if ([...phases.options].some((o) => o.value === steps)) phases.value = steps;
      checkbox.checked = interleave === '1';
      free.checked = fitSteps === '1';
    }
  } catch {}
  const read = () => ({
    axes: Number(select.value),
    phases: Number(phases.value),
    interleaved: checkbox.checked,
    estimatePhases: free.checked && Number(phases.value) >= 3,
  });
  const update = () => {
    const s = read();
    free.disabled = s.phases < 3;
    controls.querySelector('.phase-note')!.textContent = free.disabled
      ? 'PSF, drift and bead intensities are available. The harmonic pattern fit requires at least three phase steps per axis.'
      : s.estimatePhases
        ? 'Each axis uses its first exposure as the phase reference (0°).'
        : '';
    controls.querySelector('output')!.textContent = cycle(s);
    controls.querySelector('.steps')!.textContent = Array.from({ length: s.phases }, (_, i) =>
      Number(((360 * i) / s.phases).toFixed(2)),
    ).join(', ');
  };
  controls.onchange = () => {
    update();
    try {
      document.cookie =
        'simcode_sequence=' +
        select.value +
        '-' +
        Number(checkbox.checked) +
        '-' +
        phases.value +
        '-' +
        Number(free.checked) +
        '; Max-Age=31536000; Path=/; SameSite=Lax';
    } catch {}
  };
  update();
  document.querySelector('#inputs')!.after(controls);
  return read;
}
