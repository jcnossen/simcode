export type Schedule = { axes: number; interleaved: boolean };
export const defaultSchedule: Schedule = { axes: 2, interleaved: false };
export const axisName = (i: number) => String.fromCharCode(65 + i);
export function axisAt(t: number, s: Schedule) {
  return s.interleaved ? t % s.axes : Math.floor(t / 3) % s.axes;
}
export function phaseAt(t: number, s: Schedule) {
  return (s.interleaved ? Math.floor(t / s.axes) : t) % 3;
}
export const cycle = (s: Schedule) => Array.from({ length: 3 * s.axes }, (_, t) => axisAt(t, s)).join('');

export function scheduleControls() {
  const controls = document.createElement('fieldset');
  controls.className = 'pattern-order';
  controls.innerHTML = '<legend>Illumination sequence</legend><label>Number of axes <select></select></label> <label><input type="checkbox"> Interleaved axes</label><p>Angle indices <output></output></p>';
  const select = controls.querySelector('select')!;
  const checkbox = controls.querySelector('input')!;
  for (let i = 1; i <= 9; i++) select.add(new Option(String(i), String(i)));
  select.value = String(defaultSchedule.axes);
  try {
    const cookie = document.cookie.split('; ').find(c => c.startsWith('simcode_sequence='));
    if (cookie) {
      const [count, interleave] = cookie.split('=')[1].split('-');
      if ([...select.options].some(o => o.value === count)) select.value = count;
      checkbox.checked = interleave === '1';
    }
  } catch {}
  const read = () => ({ axes: Number(select.value), interleaved: checkbox.checked });
  const update = () => { controls.querySelector('output')!.textContent = cycle(read()); };
  controls.onchange = () => {
    update();
    try { document.cookie = 'simcode_sequence=' + select.value + '-' + Number(checkbox.checked) + '; Max-Age=31536000; Path=/; SameSite=Lax'; } catch {}
  };
  update();
  document.querySelector('#inputs')!.after(controls);
  return read;
}
