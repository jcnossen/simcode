import { getDevice } from 'torch-web/gpu/device';
import { scheduleControls, axisName } from './schedule';
import { liveEstimator } from './live';
import { installMovieExample } from './example';
import { acquisitionSettings } from './settings';
import { mount, download } from 'torch-dense/app';
import { movieModulation } from './dense-slide';
import {validationPanel} from './cross-validation';
type InputKind='slide'|'stack';
let kind:InputKind='slide';
const selected:Partial<Record<InputKind,File>>={};
const framesPerStep=document.createElement('input');
const simulation=()=>example.truth();
const estimator = liveEstimator(simulation, () => order());
mount(
  'Joint PSF and modulation pattern estimator',
  async (r, out) => {
    estimator.positions(r);
    const simulated = simulation()?.modulation;
    if(simulated) download(out,'true-modulation.json',JSON.stringify(simulated),'application/json');
    const schedule = order();
    if(kind==='slide' && selected.slide) validationPanel(out,selected.slide,r.settings,schedule,document.querySelector<HTMLInputElement>('.random-beads')!.checked);
    if (schedule.phases < 3) {
      const message = document.createElement('p');
      message.textContent = 'PSF and bead photometry complete. This phase schedule provides fewer than three steps for the harmonic pattern fit.';
      out.append(message);
      return;
    }
    if (r.drift.length < schedule.axes * schedule.phases) {
      const message = document.createElement('p');
      message.textContent = 'PSF and bead intensities are available. Estimating a pattern needs one complete illumination cycle.';
      out.append(message);
      return;
    }
    const result = await movieModulation((await getDevice()).device,selected[kind],r, schedule, document.querySelector<HTMLInputElement>('.random-beads')!.checked ? estimator.depthData() : undefined, true, message=>{document.querySelector('[role=status]')!.textContent=message;});
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
    const overview = fields.flatMap(([key,label]) => simulated
      ? [{key,label:label+' · true',maps:simulated.maps},{key,label:label+' · fit',maps:result.maps}]
      : [{key,label,maps:result.maps}]);
    const cols = simulated ? 2 : Math.min(3, fields.length);
    c.width = 6 + 328 * cols;
    c.height = 370 * Math.ceil(overview.length / cols);
    const g = c.getContext('2d')!;
    g.fillStyle = 'white';
    g.fillRect(0, 0, c.width, c.height);
    overview.forEach(({key, label,maps}, p) => {
      g.save();
      g.translate(0, Math.floor(p / cols) * 370);
      const ox = 20 + (p % cols) * 328,
        oy = 54,
        vals = maps.map(v => v[key]),
        scale = [...result.maps.map(v=>v[key]), ...(simulated?.maps.map(v=>v[key]) ?? [])],
        lo = Math.min(...scale),
        hi = Math.max(...scale);
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
    // Keep the downloadable overview at publication resolution.
    const method=document.createElement('p');
    method.textContent='Depth estimate · '+result.patterns.map((p,i)=>axisName(i)+' — '+p.depth_method).join(' / ');
    out.append(method);
    const blob = await new Promise<Blob | null>((resolve) => c.toBlob(resolve));
    if (blob) download(out, 'fov-overview.png', blob, 'image/png');
  },
  estimator,
  ()=>({file:selected[kind],framesPerStep:kind==='stack' ? framesPerStep.valueAsNumber : undefined}),
);

document.querySelector('#purpose')!.textContent = 'Fit a bead movie to recover the PSF, illumination patterns and field maps.';
const repoLink = document.createElement('a');
repoLink.href = 'https://github.com/jcnossen/simcode';
repoLink.textContent = repoLink.href;
const exampleLink = document.createElement('a');
exampleLink.href = repoLink.href + '/blob/master/examples/estimate_bead_slide.py';
exampleLink.textContent = 'Python example: simulate and fit a 2D bead slide';
const resources = document.createElement('nav');
resources.setAttribute('aria-label', 'Source and examples');
resources.append(repoLink, ' · ', exampleLink);
document.querySelector('main h1')!.before(resources);
const gpuSubtitle = document.createElement('p');
gpuSubtitle.className = 'gpu-subtitle';
document.querySelector('#purpose')!.after(gpuSubtitle);
gpuSubtitle.textContent = 'Checking WebGPU adapter…';
getDevice().then(({info}) => {
  const name = info.description.trim() || [info.vendor, info.architecture, info.device].filter(Boolean).join(' ');
  gpuSubtitle.textContent = name
    ? 'Uses WebGPU, currently using: ' + name
    : 'Uses WebGPU · adapter name unavailable';
  if (info.isFallback) gpuSubtitle.textContent += ' (software adapter)';
}).catch(() => { gpuSubtitle.textContent = 'WebGPU unavailable'; });
const order = scheduleControls();
const settings = acquisitionSettings();
const population = document.createElement('label');
population.innerHTML = '<input class="random-beads" type="checkbox" checked> Uniformly random bead positions (enable image-power depth estimation)';
document.querySelector('.pattern-order')!.append(population);
const randomBeads = population.querySelector('input')!;
try { randomBeads.checked = !document.cookie.split('; ').includes('simcode_random_beads=0'); } catch {}
randomBeads.onchange = () => {
  try { document.cookie = 'simcode_random_beads='+Number(randomBeads.checked)+'; Max-Age=31536000; Path=/; SameSite=Lax'; } catch {}
};
const example = installMovieExample(settings, order,()=>framesPerStep.valueAsNumber);
installSampleChoice();

function installSampleChoice() {
  const form = document.querySelector<HTMLFormElement>('#inputs')!.closest('form')!;
  const inputs = form.querySelector<HTMLElement>('#inputs')!;
  const submit = inputs.querySelector<HTMLButtonElement>('button')!;
  const exampleBox = form.querySelector<HTMLDetailsElement>('.movie-example')!;
  const main = form.closest('main')!;
  const cards=document.createElement('div');
  cards.className='input-cards';
  cards.setAttribute('role','radiogroup');
  cards.setAttribute('aria-label','Acquisition input');
  const existing=inputs.querySelectorAll<HTMLLabelElement>('label.drop');
  const controls=inputs.querySelector('.sample-controls')!;
  const labels=[existing[0],existing[1]];
  const names=['2D bead slide movie','2D bead slide Z stack'];
  const modes:InputKind[]=['slide','stack'];
  const step=document.querySelector<HTMLInputElement>('#settings input[placeholder]')!;
  const stepLabel=step.closest('label')!;
  stepLabel.replaceChildren('Z stack step (nm)',step);
  const repeatLabel=document.createElement('label');
  framesPerStep.type='number'; framesPerStep.min='1'; framesPerStep.step='1'; framesPerStep.value='1'; framesPerStep.required=true;
  repeatLabel.append('Frames per step',framesPerStep);
  const scanFields=document.createElement('div');
  scanFields.className='scan-fields';
  scanFields.append(stepLabel,repeatLabel);
  const tiles=labels.map((label,i)=>{
    const mode=modes[i], tile=document.createElement('div');
    tile.className='input-card'; tile.dataset.kind=mode;
    tile.setAttribute('role','radio'); tile.setAttribute('aria-label',names[i]);
    const title=label.querySelector('strong')!, filename=label.querySelector('span')!;
    title.textContent=names[i];filename.textContent='Drop a TIFF here';
    const input=label.querySelector<HTMLInputElement>('input')!;
    input.setAttribute('aria-label','Browse '+names[i]);input.onchange=null;
    label.className='browse-tiff';label.replaceChildren('Browse TIFF',input);
    tile.append(title,filename,label);
    if(mode==='stack') tile.append(scanFields);
    cards.append(tile);
    const select=()=>choose(mode);
    tile.onclick=select;
    tile.onkeydown=e=>{
      if(e.target!==tile)return;
      if(e.key===' '||e.key==='Enter'){e.preventDefault();select();}
      const delta=['ArrowLeft','ArrowUp'].includes(e.key)?-1:['ArrowRight','ArrowDown'].includes(e.key)?1:0;
      if(delta){e.preventDefault();const j=(i+delta+modes.length)%modes.length;choose(modes[j]);tiles[j].focus();}
    };
    const set=(file?:File)=>{
      selected[mode]=file;
      filename.textContent=file?.name ?? 'Drop a TIFF here';
      choose(mode,true);
    };
    input.onchange=()=>set(input.files?.[0]);
    tile.ondragover=e=>e.preventDefault();
    label.ondrop=null;
    tile.ondrop=e=>{e.preventDefault();if(e.dataTransfer?.files[0])set(e.dataTransfer.files[0]);};
    return tile;
  });
  inputs.replaceChildren(cards,controls);
  function choose(next:InputKind,refresh=false){
    if(next===kind && !refresh)return;
    const changed=kind!==next;
    kind=next;
    tiles.forEach((tile,i)=>{
      const current=modes[i]===kind;
      tile.setAttribute('aria-checked',String(current));tile.tabIndex=current?0:-1;
    });
    if(changed)exampleBox.open=false;
    const scan=kind==='stack';
    main.classList.toggle('stack-selected',scan);
    framesPerStep.disabled=step.disabled=!scan;
    example.setMode(kind);
    const target=exampleBox.querySelector('.example-pitch')!;
    target.after(example.aberrationControl,example.driftControl,example.densityControl,example.photonControls);
    example.select(selected[kind]);
  }
  choose('slide',true);
}
