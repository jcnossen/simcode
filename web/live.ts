import { movieView } from './movie-view';
import { modulation } from './modulation';
import type { PhotometrySnapshot } from 'torch-dense/movie';
import { VecEngine } from 'torch-web/gpu/ops';
import type { MovieIteration } from 'torch-dense/movie';

export function drawPsf(
  canvas: HTMLCanvasElement,
  pixels: Float32Array,
  exposure = Math.max(...pixels, 1e-12),
) {
  const side = Math.sqrt(pixels.length);
  canvas.width = canvas.height = side;
  const ctx = canvas.getContext('2d')!,
    image = ctx.createImageData(side, side);

  pixels.forEach((p, i) => {
    const value = (255 * Math.max(0, p)) / exposure;
    image.data.set([value, value, value, 255], 4 * i);
  });
  ctx.putImageData(image, 0, 0);
}

export function plot(
  canvas: HTMLCanvasElement,
  rows: number[][],
  total: number,
  names: string[],
  xlabel: string,
) {
  const w = 410,
    h = 142,
    left = 55,
    right = w - 12,
    top = 27,
    bottom = h - 31;
  canvas.width = w * 2;
  canvas.height = h * 2;
  const c = canvas.getContext('2d')!;
  c.scale(2, 2);
  const colors = ['#196c76', '#aa4a20', '#704b99'];

  let lo = Infinity,
    hi = -Infinity;
  for (const row of rows)
    for (const value of row) {
      lo = Math.min(lo, value);
      hi = Math.max(hi, value);
    }
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const y = (v: number) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);
  c.font = '12px system-ui';
  c.fillStyle = '#33454a';
  c.textAlign = 'right';
  for (const tick of [lo, (lo + hi) / 2, hi]) {
    c.fillText(Math.abs(tick) >= 1e4 ? tick.toExponential(1) : tick.toFixed(1), left - 5, y(tick) + 4);
    c.strokeStyle = '#dee6e5';
    c.beginPath();
    c.moveTo(left, y(tick));
    c.lineTo(right, y(tick));
    c.stroke();
  }
  c.textAlign = 'left';
  c.fillText('1', left, bottom + 18);
  c.fillText(xlabel, (left + right) / 2 - 18, bottom + 18);
  c.textAlign = 'right';
  c.fillText(String(total), right, bottom + 18);
  c.textAlign = 'left';
  names.forEach((name, j) => {
    c.fillStyle = c.strokeStyle = colors[j];
    c.fillText(name, left + j * 66, 15);
    c.lineWidth = 1.8;
    c.beginPath();
    rows.forEach((row, i) => {
      const x = left + (i * (right - left)) / Math.max(1, total - 1);
      if (i) c.lineTo(x, y(row[j]));
      else c.moveTo(x, y(row[j]));
    });
    c.stroke();
    if (rows.length === 1) c.fillRect(left - 1, y(rows[0][j]) - 1, 3, 3);
  });
}

export function liveEstimator() {
  let panel: HTMLElement,
    engine: VecEngine | null = null;
  let viewer: ReturnType<typeof movieView>;
  let width = 0,
    height = 0;
  function overlay(result: ReturnType<typeof modulation>, name: string) {
    viewer.setPattern((x, y, t) => {
      const p = result.patterns[Math.floor((t % 6) / 3)],
        u = (2 * x) / (width - 1) - 1,
        v = (2 * y) / (height - 1) - 1;
      const poly = (c: number[]) => c[0] + c[1] * u + c[2] * v + c[3] * u * u + c[4] * u * v + c[5] * v * v;
      return (
        (1 +
          poly(p.depth) *
            Math.cos(p.kx_rad_px * x + p.ky_rad_px * y + poly(p.phase) + ((t % 3) * 2 * Math.PI) / 3)) /
        2
      );
    }, name);
  }
  let scoreStage = '',
    scores: number[][] = [],
    motion: number[][] = [],
    origin: number[] = [];
  return {
    start() {
      panel = document.querySelector<HTMLElement>('.estimator-live') ?? document.createElement('section');
      panel.className = 'estimator-live';
      document.querySelector('article')!.before(panel);
      const form = document.querySelector<HTMLFormElement>('main form')!;
      const purpose = document.querySelector<HTMLElement>('#purpose')!;
      const status = document.querySelector<HTMLElement>('main [role=status]')!;
      const example = form.querySelector<HTMLDetailsElement>('.movie-example');
      if (example) example.open = false;
      form.hidden = true;
      purpose.hidden = true;
      panel.hidden = false;
      panel.tabIndex = -1;
      panel.innerHTML = `
        <header class="estimator-heading"><h2>Estimating PSF and pattern</h2><a href="#psf-report" hidden>View pupil and aberrations</a><button type="button" hidden>Back to inputs</button></header>
        <div class="estimate-grid">
          <figure class="processing-movie"><figcaption>Input movie</figcaption>
            <div class="fit-movie"></div>
          </figure>
          <div><figure class="pattern-zoom"></figure>
            <figure><figcaption>Current PSF</figcaption><canvas class="psf" aria-label="Current fitted point PSF"></canvas><small class="parameters">Preparing optics…</small></figure>
            <div class="live-pupil"></div><figure class="objective" hidden><figcaption>Optical search score</figcaption><canvas aria-label="Optical search score by trial"></canvas></figure>
            <figure class="motion" hidden><figcaption>Estimated drift from frame 1 (nm)</figcaption><canvas aria-label="Estimated XYZ drift by frame"></canvas></figure>
          </div>
        </div>`;
      panel.querySelector('header')!.after(status);
      panel.querySelector('button')!.onclick = () => {
        panel.before(status);
        status.textContent = '';
        panel.hidden = true;
        form.hidden = false;
        purpose.hidden = false;
        form.querySelector<HTMLInputElement>('input')?.focus();
      };
      viewer = movieView(
        panel.querySelector('.fit-movie')!,
        panel.querySelector('.pattern-zoom')!,
        'Movie frame being processed',
      );
      panel.focus({ preventScroll: true });
      panel.scrollIntoView({ block: 'start' });
      scoreStage = '';
      scores = [];
      motion = [];
      origin = [];
    },
    frame(
      pixels: Float32Array,
      widthArg: number,
      heightArg: number,
      index: number,
      total: number,
      stage: 'read' | 'fit',
    ) {
      width = widthArg;
      height = heightArg;
      viewer.frame(pixels, width, height, index - 1, total, stage === 'fit' ? 'Fitting' : 'Reading');
    },
    photometry(snapshot: PhotometrySnapshot) {
      try {
        overlay(modulation(snapshot), 'Preliminary fitted pattern');
      } catch {
        /* A later exposure group can resolve an underdetermined pattern. */
      }
    },
    pattern(result: ReturnType<typeof modulation>) {
      overlay(result, 'Fitted pattern');
    },
    async update(dev: GPUDevice, u: MovieIteration) {
      const report = document.getElementById('psf-report');
      const destination =
        u.stage === 'final' ? document.querySelector('article') : panel.querySelector('.live-pupil');
      if (report && destination && report.parentElement !== destination) destination.prepend(report);
      const pupilLink = panel.querySelector<HTMLAnchorElement>('a[href="#psf-report"]')!;
      pupilLink.hidden = !document.querySelector('#psf-report:not([hidden])');
      const m = u.model;
      engine ??= new VecEngine(
        dev,
        { ...m.optics, roisize: m.roisize, npupil: m.npupil, oversample: m.oversample, zernike_noll: m.noll },
        [],
      );
      engine.setZernike(m.zcoef);
      engine.setImages([{ x: 0, y: 0, ze: u.emitterHeightUm, f: u.focusUm, I: 1, bg: 0 }]);
      await engine.run(1, 'psf');
      drawPsf(panel.querySelector('.psf')!, await engine.readPsf(1));
      const z6 = m.noll.indexOf(6);
      const stage =
        u.stage === 'calibration'
          ? 'Stack iteration ' + u.index + '/' + u.total
          : u.stage === 'profile'
            ? 'Optical trial ' + u.index + '/' + u.total
            : u.stage === 'frame'
              ? 'Frame ' + u.index + '/' + u.total
              : 'Final mean focus';
      panel.querySelector('.parameters')!.textContent =
        stage +
        ' · focus relative to bead ' +
        Math.round((u.focusUm - u.emitterHeightUm) * 1000) +
        ' nm' +
        (z6 < 0 ? '' : ' · Z6 ' + m.zcoef[z6].toFixed(3) + ' waves RMS');
      if (u.score !== undefined) {
        if (scoreStage !== u.stage) {
          scores = [];
          scoreStage = u.stage;
        }
        scores.push([u.score]);
        const fig = panel.querySelector<HTMLElement>('.objective')!;
        fig.hidden = false;
        fig.querySelector('figcaption')!.textContent =
          u.stage === 'calibration'
            ? 'Stack fit · negative log likelihood per pixel'
            : 'Optical search score';
        plot(
          fig.querySelector('canvas')!,
          scores,
          u.total,
          [u.stage === 'calibration' ? 'NLL / px' : 'Score'],
          u.stage === 'calibration' ? 'Iteration' : 'Trial',
        );
      }
      if (u.driftNm) {
        if (!origin.length) origin = u.driftNm;
        motion.push(u.driftNm.map((v, i) => v - origin[i]));
        const fig = panel.querySelector<HTMLElement>('.motion')!;
        fig.hidden = false;
        plot(fig.querySelector('canvas')!, motion, u.total, ['X', 'Y', 'Z'], 'Frame');
      }
    },
    end() {
      panel.querySelector('button')!.hidden = false;

      const partial = panel.querySelector('.live-pupil #psf-report');
      if (partial) document.querySelector('article')!.prepend(partial);
      engine?.destroy();
      engine = null;
    },
  };
}
