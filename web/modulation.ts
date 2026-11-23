import type { PhotometrySnapshot } from 'torch-dense/movie';
import { axisAt, phaseDegrees, phaseRadians, cycle, axisName, defaultSchedule } from './schedule';
const tau = 2 * Math.PI;
function fft(a: Float64Array, b: Float64Array) {
  const n = a.length;
  for (let i = 1, j = 0; i < n; i++) {
    let m = n >> 1;
    for (; j & m; m >>= 1) j ^= m;
    j ^= m;
    if (i < j) {
      [a[i], a[j]] = [a[j], a[i]];
      [b[i], b[j]] = [b[j], b[i]];
    }
  }
  for (let len = 2; len <= n; len *= 2) {
    const angle = -tau / len,
      cr = Math.cos(angle),
      ci = Math.sin(angle);
    for (let start = 0; start < n; start += len) {
      let wr = 1,
        wi = 0;
      for (let j = 0; j < len / 2; j++) {
        const l = start + j,
          h = l + len / 2,
          r = a[h] * wr - b[h] * wi,
          s = a[h] * wi + b[h] * wr;
        a[h] = a[l] - r;
        b[h] = b[l] - s;
        a[l] += r;
        b[l] += s;
        [wr, wi] = [wr * cr - wi * ci, wi * cr + wr * ci];
      }
    }
  }
}
function solve(a: number[][], v: number[]) {
  const m = a.map((r, i) => [...r, v[i]]),
    n = v.length;
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let j = k + 1; j < n; j++) if (Math.abs(m[j][k]) > Math.abs(m[p][k])) p = j;
    [m[p], m[k]] = [m[k], m[p]];
    if (Math.abs(m[k][k]) < 1e-10) throw Error('Insufficient independent bead or phase samples.');
    const d = m[k][k];
    m[k] = m[k].map((x) => x / d);
    for (let i = 0; i < n; i++)
      if (i !== k) {
        const f = m[i][k];
        for (let j = k; j <= n; j++) m[i][j] -= f * m[k][j];
      }
  }
  return m.map((r) => r[n]);
}
function regression(x: number[][], y: number[], weights?: number[]) {
  const n = x[0].length,
    a = Array.from({ length: n }, () => Array(n).fill(0)),
    v = Array(n).fill(0);
  x.forEach((r, i) => {
    const w = weights?.[i] ?? 1;
    for (let j = 0; j < n; j++) {
      v[j] += w * r[j] * y[i];
      for (let k = 0; k < n; k++) a[j][k] += w * r[j] * r[k];
    }
  });
  return solve(a, v);
}
const dot = (a: number[], b: number[]) => a.reduce((s, x, j) => s + x * b[j], 0);
function robust(x: number[][], y: number[]) {
  let fit = regression(x, y);
  for (let j = 0; j < 3; j++) {
    const e = y.map((v, i) => Math.abs(v - dot(x[i], fit))),
      s = [...e].sort((a, b) => a - b)[Math.floor(e.length / 2)] * 2 + 0.0001;
    fit = regression(
      x,
      y,
      e.map((v) => Math.min(1, s / v)),
    );
  }
  return fit;
}
function carrier(r: PhotometrySnapshot, cs: number[][]) {
  const amplitudes = cs.map((a) => Math.hypot(a[1], a[2])).sort((a, b) => a - b);
  const cap = Math.max(1, amplitudes[Math.floor(0.85 * amplitudes.length)]);
  const scale = 2,
    n = 2 ** Math.ceil(Math.log2(scale * Math.max(r.width, r.height)));
  if (n > 4096) throw Error('Carrier FFT exceeds the supported field size (2048 px).');
  const re = new Float64Array(n * n),
    im = re.slice();
  r.xy.forEach(([x, y], i) => {
    const j = Math.round(scale * y) * n + Math.round(scale * x),
      a = cs[i];
    const norm = Math.max(cap, Math.hypot(a[1], a[2]));
    re[j] += a[1] / norm;
    im[j] += a[2] / norm;
  });
  for (let y = 0; y < n; y++) fft(re.subarray(y * n, (y + 1) * n), im.subarray(y * n, (y + 1) * n));
  const col = new Float64Array(n),
    ci = col.slice();
  let best = 0,
    k = [0, 0];
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) {
      col[y] = re[y * n + x];
      ci[y] = im[y * n + x];
    }
    fft(col, ci);
    for (let y = 0; y < n; y++) {
      const fx = ((x < n / 2 ? x : x - n) * scale) / n,
        fy = ((y < n / 2 ? y : y - n) * scale) / n;
      if (Math.hypot(fx, fy) < 2 / Math.max(r.width, r.height)) continue;
      const power = col[y] ** 2 + ci[y] ** 2;
      if (power > best) {
        best = power;
        k = [tau * fx, tau * fy];
      }
    }
  }
  // Refine off-bin using the continuous bead coordinates.
  const score = (kx: number, ky: number) => {
    let cr = 0,
      si = 0;
    r.xy.forEach(([x, y], i) => {
      const phase = kx * x + ky * y,
        co = Math.cos(phase),
        sn = Math.sin(phase),
        a = cs[i],
        norm = Math.max(cap, Math.hypot(a[1], a[2]));
      cr += (a[1] * co + a[2] * sn) / norm;
      si += (a[2] * co - a[1] * sn) / norm;
    });
    return cr * cr + si * si;
  };
  for (let step = (tau * scale) / n / 2; step > (tau * scale) / n / 64; step /= 2) {
    const prev = k;
    let top = -1;
    for (const dx of [-step, 0, step])
      for (const dy of [-step, 0, step]) {
        const tryK = [prev[0] + dx, prev[1] + dy],
          s = score(...(tryK as [number, number]));
        if (s > top) {
          top = s;
          k = tryK;
        }
      }
  }
  return k;
}
export function modulation(r: PhotometrySnapshot, schedule = defaultSchedule) {
  if (r.drift.length < schedule.phases * schedule.axes || r.xy.length < 6)
    throw Error('Modulation maps need a full illumination cycle and at least six beads across the field.');
  const basis = (x: number, y: number) => {
    x = (2 * x) / (r.width - 1) - 1;
    y = (2 * y) / (r.height - 1) - 1;
    return [1, x, y, x * x, x * y, y * y];
  };
  const X = r.xy.map(([x, y]) => basis(x, y)),
    means: number[][] = [];
  const patterns = Array.from({ length: schedule.axes }, (_, dir) => {
    const frames = r.drift.map((_, t) => t).filter((t) => axisAt(t, schedule) === dir);
    const harmonic = (k: number[]) => {
      const A = frames.map((t) => {
        const p = phaseRadians(t, schedule) + k[0] * r.drift[t][0] + k[1] * r.drift[t][1];
        return [1, Math.cos(p), -Math.sin(p)];
      });
      return r.xy.map((_, i) =>
        regression(
          A,
          frames.map((t) => r.intensity[t][i]),
        ),
      );
    };
    let cs = harmonic([0, 0]),
      k = carrier(r, cs);
    cs = harmonic(k);
    k = carrier(r, cs);
    cs = harmonic(k);
    const valid = cs
      .map((a, i) => (a[0] > 1 && Math.hypot(a[1], a[2]) > 0.01 * a[0] ? i : -1))
      .filter((i) => i >= 0);
    if (valid.length < 6) throw Error('Too few modulated beads for a quadratic field fit.');
    const q = (Math.hypot(...k) * r.settings.diameterNm) / (2000 * r.settings.optics.pixel_um);
    const transfer = q < 0.001 ? 1 : (3 * (Math.sin(q) - q * Math.cos(q))) / q ** 3;
    if (transfer < 0.1) throw Error('Bead diameter attenuates the measured carrier too strongly.');
    const residual = valid.map((i) => Math.atan2(cs[i][2], cs[i][1]) - dot(k, r.xy[i]));
    const phase0 = Math.atan2(
      residual.reduce((a, v) => a + Math.sin(v), 0),
      residual.reduce((a, v) => a + Math.cos(v), 0),
    );
    const design = valid.map((i) => X[i]),
      phase = robust(
        design,
        residual.map((v) => phase0 + Math.atan2(Math.sin(v - phase0), Math.cos(v - phase0))),
      );
    const depth = robust(
      design,
      valid.map((i) => Math.hypot(cs[i][1], cs[i][2]) / cs[i][0] / transfer),
    );
    means.push(cs.map((a) => a[0]));
    return {
      kx_rad_px: k[0],
      ky_rad_px: k[1],
      pitch_nm: (tau / Math.hypot(...k)) * r.settings.optics.pixel_um * 1000,
      orientation_deg: ((Math.atan2(k[1], k[0]) * 180) / Math.PI + 360) % 360,
      phase,
      depth,
      beads: valid.length,
    };
  });
  const ratios = means.slice(1).map(denom => {
    const valid = r.xy.map((_, i) => means[0][i] > 1 && denom[i] > 1 ? i : -1).filter(i => i >= 0);
    if (valid.length < 6) throw Error('Too few shared beads for an intensity ratio map.');
    return robust(valid.map(i => X[i]), valid.map(i => means[0][i] / denom[i]));
  });
  const maps = Array.from({ length: 32 * 32 }, (_, i) => {
    const x = ((i % 32) * (r.width - 1)) / 31,
      y = (Math.floor(i / 32) * (r.height - 1)) / 31,
      b = basis(x, y);
    const angles = patterns.map((p) => {
      const c = p.phase,
        u = b[1],
        v = b[2];
      const kx = p.kx_rad_px + (2 * (c[1] + 2 * c[3] * u + c[4] * v)) / (r.width - 1);
      const ky = p.ky_rad_px + (2 * (c[2] + c[4] * u + 2 * c[5] * v)) / (r.height - 1);
      return ((Math.atan2(ky, kx) * 180) / Math.PI + 360) % 360;
    });
    const row: Record<string, number> = { x_px: x, y_px: y };
    patterns.forEach((p, i) => {
      const label = axisName(i);
      row['orientation_' + label + '_deg'] = angles[i];
      row['depth_' + label] = dot(p.depth, b);
      row['phase_' + label + '_rad'] = p.kx_rad_px*x + p.ky_rad_px*y + dot(p.phase,b);
      if (i) row['ratio_A_over_' + label] = dot(ratios[i-1], b);
    });
    return row;
  });
  return {
    schedule: Array.from({length: schedule.phases * schedule.axes}, (_,t) => axisName(axisAt(t,schedule)) + Number(phaseDegrees(t,schedule).toFixed(4))).join(' '),
    phases_per_axis: schedule.phases,
    angle_indices: cycle(schedule),
    coordinate_system: 'x right, y down. Orientation is the wavevector angle. Ratios divide mean axis 0 intensity by each other axis.',
    basis: '[1,u,v,u²,uv,v²], u,v normalized to [-1,1]',
    patterns,
    ratio: ratios[0],
    ratios,
    maps,
  };
}
