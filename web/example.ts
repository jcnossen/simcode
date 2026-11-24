import { defaultPhotons } from './example-photons';
import { defaultAberration } from './example-aberration';
import { axisName, type Schedule } from './schedule';
import { movieView } from './movie-view';
import { drawPsf, plot } from './live';
import { defaultFrameCount, exampleModulation, focusMidpointNm, generateMovie, illumination, pitchNm, size, zDriftPeakToPeakNm } from './generate';
import type { ExampleTruth } from './generate';
import type { Acquisition } from './settings';
import './example.css';

export function installMovieExample(settings: () => Acquisition, getSchedule: () => Schedule) {
  const inputs = document.querySelector<HTMLElement>('#inputs')!;
  const movie = inputs.querySelector<HTMLInputElement>('input[type=file]')!;
  const submit = inputs.closest('form')!.querySelector<HTMLButtonElement>('button')!;
  const view = document.createElement('details');
  view.className = 'movie-example';
  view.innerHTML = `<summary>Example: randomly generated bead movie</summary><label class="example-length">Example frames <input type="number" min="2" step="1" value="${defaultFrameCount}" required></label><label class="example-aberration">Pupil aberration <input type="range" min="0" max="500" step="10" value="${defaultAberration}" form="generated-movie-options"><output></output></label><span class="example-photons"><label>Mean signal (photons/bead/exposure) <input class="signal" type="number" min="0" step="any" value="${defaultPhotons.signal}" required form="generated-movie-options"></label><label>Background (photons/pixel/exposure) <input class="background" type="number" min="0" step="any" value="${defaultPhotons.background}" required form="generated-movie-options"></label></span><div class="example-content"><figure><figcaption class="movie-caption"><span>Movie frame</span></figcaption><div class="generated-movie"></div></figure><figure class="pattern-zoom"></figure><figure><figcaption>True PSF · aberrated pupil</figcaption><canvas class="psf" aria-label="True point PSF at the current frame"></canvas><small class="true-modes"></small></figure><figure class="example-drift"><figcaption>True drift from frame 1 (nm)</figcaption><canvas class="trace" aria-label="True XYZ drift by frame"></canvas></figure><div class="example-actions"><p>400 × 400 pixels · 12,000 beads · ${pitchNm} nm illumination pitch · −${zDriftPeakToPeakNm/2} to +${zDriftPeakToPeakNm/2} nm Z excursion · XYZ random walk</p><details class="example-pattern"><summary>True illumination parameters</summary><table></table></details><p class="frame-status" aria-live="polite"></p><progress aria-label="Movie generation progress"></progress><p><input type="button" class="again" value="Generate again" hidden> <a download="random-bead-movie.tif" hidden>Download TIFF</a></p></div></div>`;
  inputs.after(view);
  const countInput = view.querySelector<HTMLInputElement>('.example-length input')!;
  const aberrationControl = view.querySelector<HTMLLabelElement>('.example-aberration')!;
  const strength = aberrationControl.querySelector('input')!;
  const photonControls = view.querySelector<HTMLElement>('.example-photons')!;
  const readPhotons = () => {
    const fields = Array.from(photonControls.querySelectorAll('input'));
    if (!fields.every(field=>field.reportValidity())) throw Error('Enter nonnegative signal and background photon counts.');
    return {signal:fields[0].valueAsNumber,background:fields[1].valueAsNumber};
  };
  const readAberration = () => strength.valueAsNumber;
  const updateStrength = () => { aberrationControl.querySelector('output')!.value = strength.value + ' mλ RMS'; };
  strength.oninput = updateStrength;
  updateStrength();
  // Keep example validation independent of the real-movie upload form.
  const options = document.createElement('form');
  options.id = 'generated-movie-options';
  options.hidden = true;
  options.onsubmit = event => event.preventDefault();
  view.closest('form')!.after(options);
  countInput.setAttribute('form', options.id);
  // Classic TIFF offsets are unsigned 32-bit; each float32 frame has a 126-byte directory.
  countInput.max = String(Math.floor((2**32 - 1024)/(size*size*4 + 126)));
  let frameCount = defaultFrameCount;
  const psfs: Float32Array[] = [];
  const drift: number[][] = [];
  let truth: ExampleTruth | undefined;
  const viewer = movieView(
    view.querySelector('.generated-movie')!,
    view.querySelector('.pattern-zoom')!,
    'Generated bead movie',
    (i) => {
      if (psfs[i]) drawPsf(view.querySelector('.psf')!, psfs[i]);
      if (drift[i])
        plot(view.querySelector('.trace')!, drift.slice(0, i + 1), frameCount, ['X', 'Y', 'Z'], 'Frame');
    },
  );
  const progress = view.querySelector('progress')!,
    status = view.querySelector<HTMLElement>('.frame-status')!;
  const again = view.querySelector<HTMLInputElement>('.again')!,
    download = view.querySelector('a')!;
  let generated: File | null = null;
  let selectedMovie: File | undefined;
  let fitting = false;
  const form = submit.form!;
  const fit = form.onsubmit!;
  form.onsubmit = async function(event) {
    event.preventDefault();
    if (submit.disabled || !selectedMovie || fitting) return;
    fitting = true;
    try {
      await fit.call(form,event);
    } finally {
      fitting = false;
      refresh();
    }
  };
  function refresh() {
    if (!form.closest('.gel-selected')) submit.disabled = !selectedMovie || !!active || fitting;
  }
  progress.max = frameCount;
  let active: AbortController | null = null;
  let installing = false;
  function cancel() {
    if (!active) return;
    active.abort();
    active = null;
    refresh();
    again.hidden = false;
    status.textContent = 'Generation stopped.';
  }
  async function start() {
    if (active || document.querySelector('main.gel-selected,main.stack-selected')) return;
    if (fitting) {
      status.textContent = 'A movie is being fitted. Generate the example when the fit finishes.';
      again.hidden = false;
      return;
    }
    if (!countInput.reportValidity()) return;
    frameCount = countInput.valueAsNumber;
    progress.max = frameCount;
    const controller = new AbortController();
    active = controller;
    refresh();
    if (download.href) URL.revokeObjectURL(download.href);
    download.removeAttribute('href');
    download.hidden = true;
    again.hidden = true;
    generated = null;
    viewer.reset();
    truth = undefined;
    psfs.length = 0;
    drift.length = 0;
    progress.value = 0;
    status.textContent = 'Preparing vectorial PSF…';

    for (const c of view.querySelectorAll<HTMLCanvasElement>('.psf, .trace'))
      c.getContext('2d')!.clearRect(0, 0, c.width, c.height);
    try {
      const s = settings();
      const schedule = getSchedule();
      viewer.setPattern((x, y, t) => (1 + illumination(s.pixelNm, x, y, t, schedule)) / 2, 'Simulated pattern', schedule);
      let first: number[] = [];
      const file = await generateMovie(s, schedule, controller.signal, (frame, t, psf, motion, coefficients, beads) => {
        if (!t) {
          first = motion;
          const firstFrameXY = new Float32Array(2 * beads.length);
          beads.forEach(({x,y}, i) => {
            firstFrameXY[2*i] = x + motion[0]/s.pixelNm;
            firstFrameXY[2*i+1] = y + motion[1]/s.pixelNm;
          });
          truth = { firstFrameXY, pixelNm: s.pixelNm, modulation:exampleModulation(s.pixelNm,schedule), coefficients, psfs, drift, firstFocusNm: focusMidpointNm + motion[2], schedule };
          const table=view.querySelector<HTMLTableElement>('.example-pattern table')!;
          table.innerHTML='<tr><th>Axis</th><th>Pitch (nm)</th><th>Angle (°)</th><th>Steps (°)</th><th>Center depth</th><th>Center A/axis ratio</th></tr>';
          truth.modulation.patterns.forEach((p,i)=>{
            const row=table.insertRow();
            for(const value of [axisName(i),p.pitch_nm,p.orientation_deg,p.phase_steps_rad.map(v=>(v*180/Math.PI).toFixed(1)).join(', '),p.depth[0].toFixed(4),i?truth!.modulation.ratios[i-1][0].toFixed(4):'1'])
              row.insertCell().textContent=String(value);
          });
          view.querySelector('.true-modes')!.textContent =
            coefficients.map((v, i) => 'Z' + (i + 5) + ' ' + (1000*v).toFixed(1)).join(' · ') + ' mλ RMS';
        }
        psfs.push(psf.slice());
        drift.push(motion.map((v, i) => v - first[i]));
        viewer.frame(frame, size, size, t, frameCount, 'Generating', s.offset);
        progress.value = t + 1;
        status.textContent = 'Generating frame ' + (t + 1) + ' of ' + frameCount;
      }, frameCount, readAberration(), readPhotons());
      controller.signal.throwIfAborted();
      const selection = new DataTransfer();
      selection.items.add(file);
      installing = true;
      try {
        movie.files = selection.files;
        movie.dispatchEvent(new Event('change', { bubbles: true }));
      } finally {
        installing = false;
      }
      generated = file;
      download.href = URL.createObjectURL(file);
      download.hidden = false;
      status.textContent = frameCount + ' frames generated. Movie selected for fitting.';
    } catch (e) {
      if (!controller.signal.aborted) status.textContent = e instanceof Error ? e.message : String(e);
    } finally {
      if (active === controller) {
        active = null;
        refresh();
        again.hidden = false;
      }
    }
  }
  view.ontoggle = () => {
    if (!view.open) cancel();
    else if (download.hidden) void start();
  };
  countInput.onchange = () => {
    cancel();
    if (countInput.reportValidity() && view.open) void start();
  };
  strength.onchange = () => {
    if (!view.closest('.gel-selected')) {
      cancel();
      if (view.open) void start();
    }
  };
  photonControls.addEventListener('change',()=>{
    if(!view.closest('.gel-selected') && view.open){cancel();void start();}
  });
  again.onclick = () => void start();
  // Selecting a real file or changing its calibration takes priority over an in-flight example.
  function replaced(e: Event) {
    if ((e.target as Element).closest('[data-kind=slide]')) {
      generated = null;
    }
    cancel();
  }
  document.querySelector('.pattern-order')!.addEventListener('change', (event) => {
    if ((event.target as HTMLElement).matches('.free-phases')) return;
    cancel();
    if (generated && !view.open) view.open = true;
    if (view.open) void start();
  });
  inputs.addEventListener('input', e => { if ((e.target as HTMLInputElement).type === 'number') cancel(); });
  inputs.addEventListener('drop', replaced, true);
  inputs.addEventListener('change', (e) => {
    if (!installing) replaced(e);
  });
  document.querySelector('#settings')!.addEventListener('input', cancel);
  return {
    aberrationControl, readAberration, photonControls, readPhotons,
    truth: () => generated ? truth : undefined,
    select(file?: File) { selectedMovie = file; refresh(); },
    refresh,
  };
}
