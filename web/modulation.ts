import { fieldMaps } from './pattern-field';
import type { MovieDepth } from './movie-depth';
import type { PhotometrySnapshot } from 'torch-dense/movie';
import { axisAt, phaseDegrees, phaseAt, cycle, axisName, defaultSchedule } from './schedule';
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
export function regression(x: number[][], y: number[], weights?: number[]) {
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
  // Remove spatially uniform phase-dependent gain before searching for a carrier.
  const phasors = cs.map((a) => {
    const denom = Math.max(cap, Math.hypot(a[1], a[2]));
    return [a[1] / denom, a[2] / denom];
  });
  const offset = [0, 1].map((j) => phasors.reduce((s, p) => s + p[j], 0) / phasors.length);
  phasors.forEach((p) => {
    p[0] -= offset[0];
    p[1] -= offset[1];
  });
  r.xy.forEach(([x, y], i) => {
    const j = Math.round(scale * y) * n + Math.round(scale * x);
    re[j] += phasors[i][0];
    im[j] += phasors[i][1];
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
        a = phasors[i];
      cr += a[0] * co + a[1] * sn;
      si += a[1] * co - a[0] * sn;
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
export function modulation(r: PhotometrySnapshot, schedule = defaultSchedule, image?: MovieDepth) {
  if (schedule.phases < 3) throw Error('The harmonic pattern fit requires three or more phase steps.');
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
    let steps = Array.from({ length: schedule.phases }, (_, i) => (tau * i) / schedule.phases);
    const frames = r.drift.map((_, t) => t).filter((t) => axisAt(t, schedule) === dir);
    const harmonic = (k: number[]) => {
      const A = frames.map((t) => {
        const p = steps[phaseAt(t, schedule)] + k[0] * r.drift[t][0] + k[1] * r.drift[t][1];
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
    if (schedule.estimatePhases) {
      // At a given slot the bead intensity is a sinusoid in spatial position.
      // Its complex amplitude gives the phase; the common spatial origin drops
      // out when referencing slot zero. Refit bead DC for non-uniform phase steps.
      const means = r.xy.map((_, i) => frames.reduce((v, t) => v + r.intensity[t][i], 0) / frames.length);
      const bright = [...means].sort((a, b) => a - b)[
        Math.min(Math.floor(0.7 * means.length), means.length - 6)
      ];
      const selected = means.map((v, i) => (v >= bright && v > 1 ? i : -1)).filter((i) => i >= 0);
      if (selected.length < 6)
        throw Error('Too few bright beads to estimate individual illumination phases.');
      let baseline = means;
      const positions = selected.map((i) => dot(k, r.xy[i]));
      for (let round = 0; round < 4; round++) {
        const sums = steps.map(() => [0, 0]);
        const normal = steps.map(() => Array.from({ length: 3 }, () => [0, 0, 0]));
        const rhs = steps.map(() => [0, 0, 0]);
        for (const t of frames) {
          const slot = phaseAt(t, schedule),
            shift = k[0] * r.drift[t][0] + k[1] * r.drift[t][1];
          selected.forEach((i, j) => {
            const dc = baseline[i];
            if (dc <= 1) return;
            const angle = positions[j] + shift,
              x = [1, Math.cos(angle), -Math.sin(angle)];
            const y = (r.intensity[t][i] - dc) / Math.max(1, means[i]);
            for (let a = 0; a < 3; a++) {
              rhs[slot][a] += x[a] * y;
              for (let b = 0; b < 3; b++) normal[slot][a][b] += x[a] * x[b];
            }
          });
        }
        for (let j = 0; j < steps.length; j++) {
          const [, c, s] = solve(normal[j], rhs[j]);
          sums[j] = [c, s];
        }
        // With unknown bead brightness the first projection has an unknown
        // constant complex offset. Equal modulation amplitude places the
        // slot phasors on a circle; its centre estimates that offset.
        const center =
          round === 0
            ? regression(
                sums.map(([c, s]) => [2 * c, 2 * s, 1]),
                sums.map(([c, s]) => c * c + s * s),
              )
            : [0, 0];
        const angles = sums.map(([c, s]) => Math.atan2(s - center[1], c - center[0]));
        const next = angles.map((a) => (a - angles[0] + tau) % tau);
        const change = Math.max(
          ...next.map((a, j) => Math.abs(Math.atan2(Math.sin(a - steps[j]), Math.cos(a - steps[j])))),
        );
        steps = next;
        cs = harmonic(k);
        baseline = cs.map((p) => p[0]);
        if (change < 1e-4) break;
      }
      k = carrier(r, cs);
      cs = harmonic(k);
    }
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
    const beadDepth = robust(
      design,
      valid.map((i) => Math.hypot(cs[i][1], cs[i][2]) / cs[i][0] / transfer),
    );
    let depth = beadDepth;
    let depthTiles = 0;
    if (image && r.xy.length >= 6 * image.nx * image.ny) {
      const phaseDesign = steps.map(a => [1, Math.cos(a), -Math.sin(a)]);
      const columns = steps.map((_,j) => regression(phaseDesign,steps.map((_,i) => Number(i===j))));
      const weights = [0,1,2].map(j => columns.map(c => c[j]));
      const tiles = image.samples(dir,weights,k[0],k[1],transfer);
      if (tiles.length >= 9) {
        try {
          depth = robust(tiles.map(t => basis(t.x,t.y)),tiles.map(t => t.depth));
          depthTiles = tiles.length;
        } catch (e) {
          // Signal confined to one row or column cannot support a quadratic surface.
          if (!(e instanceof Error && e.message === 'Insufficient independent bead or phase samples.')) throw e;
        }
      }
    }
    means.push(cs.map((a) => a[0]));
    return {
      phase_steps_rad: steps,
      kx_rad_px: k[0],
      ky_rad_px: k[1],
      pitch_nm: (tau / Math.hypot(...k)) * r.settings.optics.pixel_um * 1000,
      orientation_deg: ((Math.atan2(k[1], k[0]) * 180) / Math.PI + 360) % 360,
      phase,
      depth,
      bead_depth: beadDepth,
      depth_method: depthTiles ? 'random-bead image power' : 'bead photometry',
      depth_tiles: depthTiles,
      beads: valid.length,
    };
  });
  const ratios = means.slice(1).map((denom) => {
    const valid = r.xy.map((_, i) => (means[0][i] > 1 && denom[i] > 1 ? i : -1)).filter((i) => i >= 0);
    if (valid.length < 6) throw Error('Too few shared beads for an intensity ratio map.');
    return robust(
      valid.map((i) => X[i]),
      valid.map((i) => means[0][i] / denom[i]),
    );
  });
  const maps = fieldMaps({patterns,ratios},r.width,r.height);
  return {
    schedule: Array.from(
      { length: schedule.phases * schedule.axes },
      (_, t) =>
        axisName(axisAt(t, schedule)) +
        Number(
          ((patterns[axisAt(t, schedule)].phase_steps_rad[phaseAt(t, schedule)] * 360) / tau).toFixed(4),
        ),
    ).join(' '),
    nominal_schedule: Array.from(
      { length: schedule.phases * schedule.axes },
      (_, t) => axisName(axisAt(t, schedule)) + Number(phaseDegrees(t, schedule).toFixed(4)),
    ).join(' '),
    phase_step_mode: schedule.estimatePhases ? 'estimated relative to first exposure per axis' : 'nominal',
    phases_per_axis: schedule.phases,
    angle_indices: cycle(schedule),
    coordinate_system:
      'x right, y down. Orientation is the wavevector angle. Ratios divide mean axis 0 intensity by each other axis.',
    basis: '[1,u,v,u²,uv,v²], u,v normalized to [-1,1]',
    patterns,
    ratio: ratios[0],
    ratios,
    maps,
  };
}
