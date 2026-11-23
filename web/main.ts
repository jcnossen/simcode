import { liveEstimator } from './live';
import { installMovieExample } from './example';
import { acquisitionSettings } from './settings';
import { mount, download } from 'torch-dense/app';
import { modulation } from './modulation';
const estimator = liveEstimator();
mount(
  'SIMCODE bead calibration',
  async (r, out) => {
    const result = modulation(r);
    estimator.pattern(result);
    download(out, 'modulation.json', JSON.stringify(result), 'application/json');
    const table = document.createElement('table');
    table.innerHTML = '<tr><th>Pattern</th><th>Pitch (nm)</th><th>Wavevector angle</th></tr>';
    result.patterns.forEach((p, i) => {
      const row = table.insertRow();
      for (const v of [i ? 'B' : 'A', p.pitch_nm.toFixed(1), p.orientation_deg.toFixed(1) + '°'])
        row.insertCell().textContent = v;
    });
    out.append(table);
    const c = document.createElement('canvas');
    c.width = 990;
    c.height = 370;
    const g = c.getContext('2d')!;
    g.fillStyle = 'white';
    g.fillRect(0, 0, c.width, c.height);
    const fields = ['depth_A', 'depth_B', 'ratio_A_over_B'] as const;
    fields.forEach((key, p) => {
      const ox = 20 + p * 328,
        oy = 54,
        vals = result.maps.map((v) => v[key]),
        lo = Math.min(...vals),
        hi = Math.max(...vals);
      g.fillStyle = '#193136';
      g.font = '20px sans-serif';
      g.fillText(['Modulation depth A', 'Modulation depth B', 'Intensity ratio A / B'][p], ox, 33);
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
    });
    out.append(c);
    const blob = await new Promise<Blob | null>((resolve) => c.toBlob(resolve));
    if (blob) download(out, 'fov-overview.png', blob, 'image/png');
  },
  estimator,
);

installMovieExample(acquisitionSettings());
