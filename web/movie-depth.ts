import { axisAt, phaseAt, type Schedule } from './schedule';

// Band-limit to bead texture below the illumination carrier. For randomly
// positioned beads the DC and shifted illumination sidebands have equal object
// power before modulation. Their power ratio cancels the common optical blur.
function gaussian(sigma: number) {
  const radius = Math.ceil(3 * sigma);
  const values = Array.from({ length: 2 * radius + 1 }, (_, i) =>
    Math.exp(-0.5 * ((i - radius) / sigma) ** 2),
  );
  const sum = values.reduce((a, b) => a + b, 0);
  return values.map((x) => x / sum);
}
const narrow = gaussian(1.15),
  broad = gaussian(2.6),
  margin = broad.length >> 1;
const noiseGain = broad.reduce(
  (sum, by, y) =>
    sum +
    broad.reduce((v, bx, x) => {
      const offset = (broad.length - narrow.length) / 2;
      const a = (narrow[y - offset] ?? 0) * (narrow[x - offset] ?? 0);
      return v + (a - by * bx) ** 2;
    }, 0),
  0,
);

// Each filtered pixel uses the same camera footprint at every phase. Decimating
// after filtering reduces memory while preserving the local variance estimate.
export class MovieDepth {
  private slots: Float32Array[][] = [];
  private noise: Float64Array[][] = [];
  private cov: Float64Array[];
  private cycles: number[];
  private xs: number[] = [];
  private ys: number[] = [];
  private tileOf: Uint16Array;
  readonly nx: number;
  readonly ny: number;
  constructor(
    readonly w: number,
    readonly h: number,
    private schedule: Schedule,
  ) {
    this.nx = Math.max(0, Math.min(16, Math.floor((w - 2 * margin) / 40)));
    this.ny = Math.max(0, Math.min(16, Math.floor((h - 2 * margin) / 40)));
    const stride = Math.max(2, Math.ceil(Math.max(w, h) / 300));
    for (let x = margin; x < w - margin; x += stride) this.xs.push(x);
    for (let y = margin; y < h - margin; y += stride) this.ys.push(y);
    this.tileOf = new Uint16Array(this.xs.length * this.ys.length);
    this.ys.forEach((y, j) =>
      this.xs.forEach((x, i) => {
        this.tileOf[j * this.xs.length + i] =
          Math.floor(((y - margin) * this.ny) / (h - 2 * margin)) * this.nx +
          Math.floor(((x - margin) * this.nx) / (w - 2 * margin));
      }),
    );
    this.cycles = Array(schedule.axes).fill(0);
    this.cov = Array.from(
      { length: schedule.axes },
      () => new Float64Array(Math.max(0, this.nx * this.ny) * schedule.phases ** 2),
    );
  }
  private blurred(im: Float32Array, kernel: number[]) {
    const nx = this.xs.length,
      rad = kernel.length >> 1;
    const temp = new Float32Array(this.h * nx);
    for (let y = 0; y < this.h; y++)
      for (let i = 0; i < nx; i++) {
        const start = y * this.w + this.xs[i] - rad;
        let sum = 0;
        for (let k = 0; k < kernel.length; k++) sum += kernel[k] * im[start + k];
        temp[y * nx + i] = sum;
      }
    const out = new Float32Array(this.tileOf.length);
    for (let j = 0; j < this.ys.length; j++)
      for (let i = 0; i < nx; i++) {
        const start = (this.ys[j] - rad) * nx + i;
        let sum = 0;
        for (let k = 0; k < kernel.length; k++) sum += kernel[k] * temp[start + k * nx];
        out[j * nx + i] = sum;
      }
    return out;
  }
  add(im: Float32Array, t: number) {
    const { phases } = this.schedule;
    if (this.nx < 3 || this.ny < 3 || phases < 3) return;
    const axis = axisAt(t, this.schedule),
      slot = phaseAt(t, this.schedule);
    const signal = this.blurred(im, narrow),
      slow = this.blurred(im, broad);
    const shot = new Float64Array(this.nx * this.ny);
    for (let i = 0; i < signal.length; i++) {
      signal[i] -= slow[i];
      shot[this.tileOf[i]] += noiseGain * slow[i];
    }
    (this.slots[axis] ??= [])[slot] = signal;
    (this.noise[axis] ??= [])[slot] = shot;
    if (slot !== phases - 1) return;
    const cov = this.cov[axis],
      planes = this.slots[axis];
    for (let a = 0; a < phases; a++)
      for (let b = 0; b <= a; b++) {
        const left = planes[a],
          right = planes[b],
          idx = a * phases + b;
        for (let i = 0; i < left.length; i++)
          cov[this.tileOf[i] * phases * phases + idx] += left[i] * right[i];
        if (a === b)
          for (let k = 0; k < shot.length; k++) cov[k * phases * phases + idx] -= this.noise[axis][a][k];
      }
    this.cycles[axis]++;
    this.slots[axis] = [];
    this.noise[axis] = [];
  }
  samples(axis: number, weights: number[][], kx: number, ky: number, transfer: number) {
    const fx = kx / (2 * Math.PI),
      fy = ky / (2 * Math.PI);
    // A resolved carrier inside the texture band biases the AC power.
    if (!this.cycles[axis] || Math.hypot(fx - Math.round(fx), fy - Math.round(fy)) < 0.35) return [];
    const n = this.schedule.phases,
      cov = this.cov[axis];
    const rows: { x: number; y: number; depth: number }[] = [];
    for (let tile = 0; tile < this.nx * this.ny; tile++) {
      const power = weights.map((row) => {
        let sum = 0;
        for (let a = 0; a < n; a++)
          for (let b = 0; b <= a; b++)
            sum += (a === b ? 1 : 2) * row[a] * row[b] * cov[tile * n * n + a * n + b];
        return sum;
      });
      if (power[0] <= 0) continue;
      const ac = power[1] + power[2];
      rows.push({
        x: margin + (((tile % this.nx) + 0.5) * (this.w - 2 * margin)) / this.nx,
        y: margin + ((Math.floor(tile / this.nx) + 0.5) * (this.h - 2 * margin)) / this.ny,
        depth: (Math.sign(ac) * Math.sqrt(Math.abs(ac) / power[0])) / transfer,
      });
    }
    return rows;
  }
}
