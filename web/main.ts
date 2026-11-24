import { getDevice } from 'torch-web/gpu/device';
import { scheduleControls, axisName } from './schedule';
import { liveEstimator } from './live';
import { installMovieExample } from './example';
import { acquisitionSettings } from './settings';
import { mount, download } from 'torch-dense/app';
import { modulation } from './modulation';
import { gelView } from './gel-view';
const estimator = liveEstimator(() => example.truth(), () => order());
mount(
  'Joint PSF and modulation pattern estimator',
  async (r, out) => {
    estimator.positions(r);
    const simulated = example.truth()?.modulation;
    if(simulated) download(out,'true-modulation.json',JSON.stringify(simulated),'application/json');
    const schedule = order();
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
    const result = modulation(r, schedule, document.querySelector<HTMLInputElement>('.random-beads')!.checked ? estimator.depthData() : undefined);
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
);

document.querySelector('#purpose')!.textContent = 'Fit a bead movie to recover the PSF, illumination patterns and field maps.';
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
const example = installMovieExample(settings, order);
installSampleChoice();

function installSampleChoice() {
  const form = document.querySelector<HTMLFormElement>('#inputs')!.closest('form')!;
  const choice = document.createElement('fieldset');
  choice.className = 'sample-choice';
  choice.innerHTML = '<legend>Sample geometry</legend>' +
    '<label><input type="radio" name="sampleGeometry" value="slide" checked> 2D bead slide</label>' +
    '<label><input type="radio" name="sampleGeometry" value="gel"> 3D bead gel</label>';
  const inputs = form.querySelector<HTMLElement>('#inputs')!;
  const submit = inputs.querySelector<HTMLButtonElement>('button')!;
  const exampleBox = form.querySelector<HTMLDetailsElement>('.movie-example')!;
  const api = gelView(settings, order, submit, example.readAberration, example.readPhotons);
  const main = form.closest('main')!;
  const radios = choice.querySelectorAll<HTMLInputElement>('input');
  form.prepend(choice);
  exampleBox.append(api.options);
  exampleBox.addEventListener('toggle',()=>{ if(exampleBox.open) api.openExample(); });
  form.after(api.panel);
  const selected: (File | undefined)[] = [];
  inputs.querySelectorAll<HTMLLabelElement>('.drop').forEach((drop,i) => {
    const input = drop.querySelector('input')!;
    const set = (file?:File) => { selected[i]=file; if (i===0) example.select(file); api.select(selected[0],selected[1]); };
    input.addEventListener('change', () => set(input.files?.[0]));
    drop.addEventListener('drop', e => set(e.dataTransfer?.files[0]));
  });
  const update = () => {
    const isGel = radios[1].checked;
    exampleBox.open = false;
    main.classList.toggle('gel-selected', isGel);
    const lengthLabel = isGel ? api.options.querySelector('[data-key=planes]')!.closest('label')! : exampleBox.querySelector('.example-length')!;
    api.activate(isGel);
    lengthLabel.after(example.aberrationControl, example.photonControls);
    example.refresh();
    api.panel.hidden = !isGel;
    api.options.hidden = !isGel;
  };
  try {
    if (document.cookie.split('; ').includes('simcode_sample=gel')) radios[1].checked = true;
  } catch {}
  const requested = new URLSearchParams(location.search).get('sample');
  if (requested === 'slide' || requested === 'gel') radios[requested === 'gel' ? 1 : 0].checked = true;
  update();
  choice.addEventListener('change', () => {
    update();
    try {
      document.cookie = 'simcode_sample=' + (radios[1].checked ? 'gel' : 'slide') +
        '; Max-Age=31536000; Path=/; SameSite=Lax';
    } catch {}
  });
  // A gel scan must never enter the planar movie fitter, including via keyboard submission.
  form.addEventListener('submit', e => {
    if (radios[1].checked) {
      e.preventDefault();
      e.stopImmediatePropagation();
      api.fit();
    }
  }, true);
}
