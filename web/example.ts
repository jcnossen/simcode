import { defaultPhotons } from './example-photons';
import { defaultAberration } from './example-aberration';
import { axisName, type Schedule } from './schedule';
import { movieView } from './movie-view';
import { drawPsf, plot } from './live';
import { defaultBeadDensity, defaultFrameCount, exampleModulation, focusMidpointNm, generateMovie, illumination, defaultPitchNm, size } from './generate';
import type { ExampleTruth } from './generate';
import type { Acquisition } from './settings';
import './example.css';

export function installMovieExample(settings: () => Acquisition, getSchedule: () => Schedule, getRepeats:()=>number) {
  const inputs = document.querySelector<HTMLElement>('#inputs')!;
  let mode:'slide'|'stack'='slide';
  const driftByMode={slide:400,stack:0};
  const catalogs=new WeakMap<File,ExampleTruth>();
  const submit = inputs.closest('form')!.querySelector<HTMLButtonElement>('button')!;
  const view = document.createElement('details');
  view.className = 'movie-example';
  view.innerHTML = `<summary>Example: randomly generated bead movie</summary><label class="example-length">Example frames <input type="number" min="2" step="1" value="${defaultFrameCount}" required></label><label class="example-pitch example-density">True illumination pitch (nm) <input type="number" min="0" step="any" value="${defaultPitchNm}" required form="generated-movie-options"></label><label class="example-aberration">Pupil aberration <input type="range" min="0" max="500" step="10" value="${defaultAberration}" form="generated-movie-options"><output></output></label><span class="example-photons"><label>Mean signal (photons/bead/exposure) <input class="signal" type="number" min="0" step="any" value="${defaultPhotons.signal}" required form="generated-movie-options"></label><label>Background (photons/pixel/exposure) <input class="background" type="number" min="0" step="any" value="${defaultPhotons.background}" required form="generated-movie-options"></label></span><div class="example-content"><figure><figcaption class="movie-caption"><span>Movie frame</span></figcaption><div class="generated-movie"></div></figure><figure class="pattern-zoom"></figure><figure><figcaption>True PSF · aberrated pupil</figcaption><canvas class="psf" aria-label="True point PSF at the current frame"></canvas><small class="true-modes"></small></figure><figure class="example-drift"><figcaption>True drift from frame 1 (nm)</figcaption><canvas class="trace" aria-label="True XYZ drift by frame"></canvas></figure><div class="example-actions"><p>400 × 400 pixels</p><details class="example-pattern"><summary>True illumination parameters</summary><table></table></details><p class="frame-status" aria-live="polite"></p><progress aria-label="Movie generation progress"></progress><p><input type="button" class="again" value="Generate again" hidden> <a download="random-bead-movie.tif" hidden>Download TIFF</a></p></div></div>`;
  inputs.after(view);
  const countInput = view.querySelector<HTMLInputElement>('.example-length input')!;
  const pitchInput = view.querySelector<HTMLInputElement>('.example-pitch input')!;
  pitchInput.oninput = () => pitchInput.setCustomValidity('');
  const aberrationControl = view.querySelector<HTMLLabelElement>('.example-aberration')!;
  const strength = aberrationControl.querySelector('input')!;
  const photonControls = view.querySelector<HTMLElement>('.example-photons')!;
  const readPhotons = () => {
    const fields = Array.from(photonControls.querySelectorAll('input'));
    if (!fields.every(field=>field.reportValidity())) throw Error('Enter nonnegative signal and background photon counts.');
    return {signal:fields[0].valueAsNumber,background:fields[1].valueAsNumber};
  };
  const driftControl=document.createElement('label');
  driftControl.className='example-aberration example-drift-scale';
  driftControl.innerHTML='XYZ drift (peak-to-peak per axis) <input type="range" min="0" max="1200" step="10" value="400" form="generated-movie-options"><output></output>';
  aberrationControl.after(driftControl);
  const driftSlider=driftControl.querySelector('input')!;
  const readDriftNm=()=>driftSlider.valueAsNumber;
  const showScale=()=>{driftControl.querySelector('output')!.value=readDriftNm()+' nm';};
  driftSlider.oninput=showScale;
  showScale();
  const densityControl=document.createElement('label');
  densityControl.className='example-density';
  densityControl.innerHTML=`Bead density (beads/µm²) <input type="number" min="0.0001" step="any" value="${defaultBeadDensity}" required form="generated-movie-options"><output></output>`;
  const density=densityControl.querySelector('input')!;
  const countAt=(side:number)=>Math.round(density.valueAsNumber*(side*settings().pixelNm/1000)**2);
  const readCount=(side:number)=>{
    if(!density.reportValidity())throw Error('Enter a positive bead density.');
    const n=countAt(side);
    if(n<1 || !Number.isSafeInteger(n))throw Error('The selected density gives no representable bead count in this field.');
    return n;
  };
  const showCount=()=>{
    const out=densityControl.querySelector('output')!;
    try{
      const n=countAt(size);
      out.value=Number.isSafeInteger(n)&&n>=0?n.toLocaleString()+' beads':'';
    }catch{out.value='';}
  };
  density.oninput=showCount;
  document.querySelector('#settings')!.addEventListener('input',showCount);
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
  const description=view.querySelector<HTMLElement>('.example-actions > p')!;
  const slideDescription=description.textContent;
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
    submit.disabled = !selectedMovie || !!active || fitting;
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
    if (active) return;
    if (fitting) {
      status.textContent = 'A movie is being fitted. Generate the example when the fit finishes.';
      again.hidden = false;
      return;
    }
    const pitchNm = pitchInput.valueAsNumber;
    pitchInput.setCustomValidity(Number.isFinite(pitchNm) && pitchNm > 0 ? '' : 'Enter a positive illumination pitch in nm.');
    if (!countInput.reportValidity() || !pitchInput.reportValidity()) return;
    const repeats=mode==='stack' ? getRepeats() : 1;
    if(!Number.isSafeInteger(repeats)||repeats<1){status.textContent='Enter a positive whole number of frames per step.';return;}
    frameCount=countInput.valueAsNumber*repeats;
    if(frameCount>Number(countInput.max)){status.textContent='Reduce the example length or frames per step.';return;}
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
      // Span two axial diffraction scales about the bead plane at the current optics.
      const planes=countInput.valueAsNumber;
      const autoStep=2*s.immersionIndex*s.wavelengthNm/(s.na*s.na*(planes-1));
      const scan=mode==='stack' ? {stepNm:s.stepNm||autoStep,repeats} : undefined;
      const count=readCount(size);
      description.textContent=size+' × '+size+' pixels · '+count.toLocaleString()+' beads, '+pitchNm+' nm pitch · '+readDriftNm()+' nm peak-to-peak drift per axis'+(scan?' · '+countInput.value+' Z steps of '+Number(scan.stepNm.toFixed(2))+' nm ('+Math.round(Math.abs(scan.stepNm)*(planes-1))+' nm span) · '+repeats+(repeats===1?' frame per step':' frames per step'):'');
      viewer.setPattern((x, y, t) => (1 + illumination(s.pixelNm, x, y, t, schedule, pitchNm)) / 2, 'Simulated pattern', schedule);
      let first: number[] = [];
      const file = await generateMovie(s, schedule, controller.signal, (frame, t, psf, motion, coefficients, beads) => {
        if (!t) {
          first = motion;
          const firstFrameXY = new Float32Array(2 * beads.length);
          beads.forEach(({x,y}, i) => {
            firstFrameXY[2*i] = x + motion[0]/s.pixelNm;
            firstFrameXY[2*i+1] = y + motion[1]/s.pixelNm;
          });
          truth = { firstFrameXY, pixelNm: s.pixelNm, modulation:exampleModulation(s.pixelNm,schedule,pitchNm), coefficients, psfs, drift, firstFocusNm: (scan ? 0 : focusMidpointNm) + motion[2], schedule };
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
      }, frameCount, readAberration(), readPhotons(),scan,readDriftNm(),count,pitchNm);
      controller.signal.throwIfAborted();
      const selection = new DataTransfer();
      selection.items.add(file);
      installing = true;
      try {
        const movie=inputs.querySelector<HTMLInputElement>('[data-kind='+mode+'] input[type=file]')!;
        movie.files = selection.files;
        movie.dispatchEvent(new Event('change', { bubbles: true }));
      } finally {
        installing = false;
      }
      generated = file;
      if(truth)catalogs.set(file,{...(truth as ExampleTruth),psfs:psfs.slice(),drift:drift.slice()});
      download.download=file.name;
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
  pitchInput.onchange = countInput.onchange = () => {
    cancel();
    if (countInput.reportValidity() && view.open) void start();
  };
  strength.onchange = () => {
    if (!view.closest('.gel-selected')) {
      cancel();
      if (view.open) void start();
    }
  };
  driftSlider.onchange=density.onchange=strength.onchange;
  photonControls.addEventListener('change',()=>{
    if(view.open){cancel();void start();}
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
    aberrationControl, readAberration, driftControl, readDriftNm, densityControl,readCount, photonControls, readPhotons,
    setMode(next:typeof mode){
      if(mode!==next){
        driftByMode[mode]=readDriftNm();
        driftSlider.value=String(driftByMode[next]);showScale();
        cancel(); mode=next; generated=null; download.hidden=true;viewer.reset();
        progress.value=0; status.textContent=''; description.textContent=slideDescription;
        countInput.min=mode==='stack'?'3':'2';
        view.querySelector('.example-length')!.firstChild!.textContent=mode==='stack'?'Example Z steps ':'Example frames ';
      }
      showCount();
      view.querySelector('summary')!.textContent='Example: randomly generated '+({slide:'2D bead slide movie',stack:'2D bead slide Z stack'}[next]);
    },
    truth: () => selectedMovie ? catalogs.get(selectedMovie) : undefined,
    select(file?: File) { selectedMovie = file; refresh(); },
    refresh,
  };
}
