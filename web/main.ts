import { scheduleControls, axisName } from './schedule';
import { liveEstimator } from './live';
import { installMovieExample } from './example';
import { acquisitionSettings } from './settings';
import { mount, download } from 'torch-dense/app';
import { modulation } from './modulation';
const estimator = liveEstimator(() => example(), () => order());
mount(
  'SIMCODE bead calibration',
  async (r, out) => {
    const schedule = order();
    if (schedule.phases < 3) {
      const message = document.createElement('p');
      message.textContent = 'PSF and bead photometry complete. This phase schedule provides fewer than three steps for the harmonic pattern fit.';
      out.append(message);
      return;
    }
    const result = modulation(r, schedule);
    estimator.pattern(result);
    download(out, 'modulation.json', JSON.stringify(result), 'application/json');
    const table = document.createElement('table');
    table.innerHTML = '<tr><th>Pattern</th><th>Pitch (nm)</th><th>Wavevector angle</th></tr>';
    result.patterns.forEach((p, i) => {
      const row = table.insertRow();
      for (const v of [axisName(i), p.pitch_nm.toFixed(1), p.orientation_deg.toFixed(1) + '°'])
        row.insertCell().textContent = v;
    });
    out.append(table);
    if (schedule.estimatePhases) {
      const phases = document.createElement('table');
      phases.innerHTML = '<tr><th>Axis</th><th>Estimated phase steps (° relative to step 0)</th></tr>';
      result.patterns.forEach((p,i) => {
        const row = phases.insertRow();
        row.insertCell().textContent = axisName(i);
        row.insertCell().textContent = p.phase_steps_rad.map(v => (v * 180 / Math.PI).toFixed(1)).join(', ');
      });
      out.append(phases);
    }
    const c = document.createElement('canvas');
    const fields = [...result.patterns.map((_,i) => ['depth_' + axisName(i), 'Modulation depth ' + axisName(i)]),
      ...result.ratios.map((_,i) => ['ratio_A_over_' + axisName(i+1), 'Intensity ratio A / ' + axisName(i+1)])];
    const cols = Math.min(3, fields.length);
    c.width = 6 + 328 * cols;
    c.height = 370 * Math.ceil(fields.length / cols);
    const g = c.getContext('2d')!;
    g.fillStyle = 'white';
    g.fillRect(0, 0, c.width, c.height);
    fields.forEach(([key, label], p) => {
      g.save();
      g.translate(0, Math.floor(p / cols) * 370);
      const ox = 20 + (p % cols) * 328,
        oy = 54,
        vals = result.maps.map((v) => v[key]),
        lo = Math.min(...vals),
        hi = Math.max(...vals);
      g.fillStyle = '#193136';
      g.font = '20px sans-serif';
      g.fillText(label, ox, 33);
      const color = (f: number) =>
        'rgb(' +
        Math.round(26 + 209 * f) +
        ',' +
        Math.round(58 + 150 * f) +
        ',' +
        Math.round(100 - 38 * f) +
        ')';
      vals.forEach((v, i) => {
        g.fillStyle = color((v - lo) / Math.max(1e-9, hi - lo));
        g.fillRect(ox + (i % 32) * 8, oy + Math.floor(i / 32) * 8, 8, 8);
      });
      for (let j = 0; j < 256; j++) {
        g.fillStyle = color(j / 255);
        g.fillRect(ox + j, 321, 1, 9);
      }
      g.fillStyle = '#193136';
      g.font = '15px sans-serif';
      g.fillText(lo.toFixed(3), ox, 354);
      g.textAlign = 'right';
      g.fillText(hi.toFixed(3), ox + 256, 354);
      g.textAlign = 'left';
      g.restore();
    });
    out.append(c);
    const blob = await new Promise<Blob | null>((resolve) => c.toBlob(resolve));
    if (blob) download(out, 'fov-overview.png', blob, 'image/png');
  },
  estimator,
);

document.querySelector('#purpose')!.textContent = 'Fit a bead movie to recover the PSF, illumination patterns and field maps.';
const order = scheduleControls();
const example = installMovieExample(acquisitionSettings(), order);
