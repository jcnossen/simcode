import { getDevice } from 'torch-web/gpu/device';
import { VecEngine } from 'torch-web/gpu/ops';
import { tiffStackBlob } from 'torch-web/psfgrab/tiffstack';
import { axisAt, phaseAt, type Schedule } from './schedule';
import type { Acquisition } from './settings';

export const size = 400,
  frameCount = 100;
export const pitchNm = 200;
export const angleAt = (i: number, axes: number) => axes === 2 ? [13, 104][i] : 13 + 180 * i / axes;
export function depthAt(dir: number, u: number, v: number) {
  return dir ? 0.88 + 0.05 * v : 0.84 + 0.06 * u;
}
export const ratioAt = (u: number) => 1.12 - 0.16 * u;
export interface ExampleTruth {
  coefficients: number[];
  psfs: Float32Array[];
  drift: number[][];
  meanFocusNm: number;
  schedule: Schedule;
}
export const zDriftPeakToPeakNm = 1000;
export function illumination(pixelNm: number, x: number, y: number, t: number, schedule: Schedule) {
  const dir = axisAt(t, schedule),
    angle = (angleAt(dir, schedule.axes) * Math.PI) / 180,
    k = (tau * pixelNm) / pitchNm;
  const u = (2 * x) / size - 1,
    v = (2 * y) / size - 1;
  return (
    depthAt(dir, u, v) *
    Math.cos(k * (x * Math.cos(angle) + y * Math.sin(angle)) + 0.25 * u * v + (tau * phaseAt(t, schedule)) / 3)
  );
}
const beadCount = 12000,
  // Keep the broad, defocused bead footprints in the rendered image.
  roi = 33,
  shifts = 3,
  tau = 2 * Math.PI;

// Subpixel stamps are shared across beads. Each stamp averages six sphere nodes
// whose second moments equal those of a uniformly fluorescent solid bead.
export async function generateMovie(
  s: Acquisition,
  schedule: Schedule,
  signal: AbortSignal,
  show: (
    frame: Float32Array,
    index: number,
    psf: Float32Array,
    driftNm: number[],
    coefficients: number[],
    ideal: Float32Array,
  ) => void,
) {
  const noll = [5, 6, 7, 8, 9, 10, 11];
  // Include appreciable astigmatism and coma in every realization.
  const coefficients = noll.map(j => {
    const magnitude = j === 6 ? 0.10 + 0.04 * Math.random()
      : j === 7 || j === 8 ? 0.05 + 0.04 * Math.random() : 0.015 + 0.04 * Math.random();
    return (Math.random() < 0.5 ? -1 : 1) * magnitude;
  });
  const flat = noll.map(() => 0);
  const { device } = await getDevice();
  signal.throwIfAborted();
  const engine = new VecEngine(
    device,
    {
      pixel_um: s.pixelNm / 1000,
      NA: s.na,
      wavelength_um: s.wavelengthNm / 1000,
      n_imm: s.immersionIndex,
      n_med: s.sampleIndex,
      roisize: roi,
      npupil: 32,
      oversample: 2,
      zernike_noll: noll,
    },
    [],
  );
  const frames: Float32Array[] = [];
  try {
    engine.setZernike(coefficients);
    const beads = Array.from({ length: beadCount }, () => ({
      x: Math.random() * size,
      y: Math.random() * size,
      I: 2500 + 4000 * Math.random(),
    }));
    const radius = s.diameterNm / 2000,
      node = radius * Math.sqrt(3 / 5);
    const nodes = node
      ? [
          [node, 0, 0],
          [-node, 0, 0],
          [0, node, 0],
          [0, -node, 0],
          [0, 0, node],
          [0, 0, -node],
        ]
      : [[0, 0, 0]];
    const q = (Math.PI * s.diameterNm) / pitchNm;
    const transfer = q < 0.001 ? 1 : (3 * (Math.sin(q) - q * Math.cos(q))) / q ** 3;
    for (let t = 0; t < frameCount; t++) {
      signal.throwIfAborted();
      const time = (tau * t) / frameCount,
        dx = 0.28 * Math.sin(time),
        dy = 0.21 * Math.sin(2 * time + 0.5),
        dz = (zDriftPeakToPeakNm / 2000) * Math.sin(time),
        focus = radius + 0.08 + dz;
      const images = [];
      for (let sy = 0; sy < shifts; sy++)
        for (let sx = 0; sx < shifts; sx++)
          for (const [nx, ny, nz] of nodes)
            images.push({
              x: sx / shifts + (nx * 1000) / s.pixelNm,
              y: sy / shifts + (ny * 1000) / s.pixelNm,
              ze: radius + nz,
              f: focus,
              I: 1,
              bg: 0,
            });
      // A centered point emitter shares the optical model and focus with this exposure.
      images.push({ x: 0, y: 0, ze: radius, f: focus, I: 1, bg: 0 });
      engine.setImages(images);
      await engine.run(images.length, 'psf');
      const bank = await engine.readPsf(images.length);
      signal.throwIfAborted();
      const stamps = new Float32Array(shifts * shifts * roi * roi);
      for (let i = 0; i < stamps.length; i++) {
        const base = Math.floor(i / (roi * roi)) * nodes.length * roi * roi + (i % (roi * roi));
        for (let j = 0; j < nodes.length; j++) stamps[i] += bank[base + j * roi * roi] / nodes.length;
      }
      const frame = new Float32Array(size * size).fill(25);
      const dir = axisAt(t, schedule);
      for (const bead of beads) {
        const x = bead.x + dx,
          y = bead.y + dy,
          u = (2 * x) / size - 1;
        const flux = bead.I * (dir ? 1 : ratioAt(u)) * (1 + transfer * illumination(s.pixelNm, x, y, t, schedule));
        const xx = Math.round(x * shifts),
          yy = Math.round(y * shifts);
        const left = Math.floor(xx / shifts) - (roi >> 1),
          top = Math.floor(yy / shifts) - (roi >> 1);
        const start =
          ((((yy % shifts) + shifts) % shifts) * shifts + (((xx % shifts) + shifts) % shifts)) * roi * roi;
        for (let py = Math.max(0, -top); py < Math.min(roi, size - top); py++) {
          const row = (top + py) * size + left,
            stamp = start + py * roi;
          for (let px = Math.max(0, -left); px < Math.min(roi, size - left); px++)
            frame[row + px] += flux * stamps[stamp + px];
        }
      }
      // Normal approximation to photon shot noise (background >= 25 photons).
      for (let i = 0; i < frame.length; i += 2) {
        const a = Math.sqrt(-2 * Math.log(1 - Math.random())),
          b = tau * Math.random();
        frame[i] =
          s.offset + Math.max(0, Math.round(frame[i] + Math.sqrt(frame[i]) * a * Math.cos(b))) / s.gain;
        frame[i + 1] =
          s.offset +
          Math.max(0, Math.round(frame[i + 1] + Math.sqrt(frame[i + 1]) * a * Math.sin(b))) / s.gain;
      }
      // Render the flat-pupil reference separately, at identical focus and sampling.
      engine.setZernike(flat);
      engine.setImages([images[images.length - 1]]);
      await engine.run(1, 'psf');
      const ideal = await engine.readPsf(1);
      engine.setZernike(coefficients);
      frames.push(frame);
      show(
        frame,
        t,
        bank.subarray(bank.length - roi * roi),
        [dx * s.pixelNm, dy * s.pixelNm, dz * 1000],
        coefficients,
        ideal,
      );
      await new Promise<void>((resolve) => setTimeout(resolve, 40));
    }
    signal.throwIfAborted();
    return new File(
      [tiffStackBlob(frames, frameCount, 1, size, size, { axis: 'frames' })],
      'random-bead-movie.tif',
      { type: 'image/tiff' },
    );
  } finally {
    engine.destroy();
  }
}
