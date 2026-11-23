import { fieldInspector } from './field-inspector';
import { MovieDepth } from './movie-depth';
import { PointCloudView } from 'torch-web/ui/pointcloud';
import { NCOL } from 'torch-web/smlm/movie';
import { drawZernikeSets } from 'torch-web/ui/zernike-bars';
import { axisAt, axisName, phaseAt, type Schedule } from './schedule';
import { illumination, type ExampleTruth } from './generate';
import { movieView } from './movie-view';
import { modulation } from './modulation';
import type { PhotometrySnapshot } from 'torch-dense/movie';
import { VecEngine } from 'torch-web/gpu/ops';
import type { MovieIteration } from 'torch-dense/movie';

export function drawPsf(
  canvas: HTMLCanvasElement,
  pixels: Float32Array,
  exposure = Math.max(...pixels, 1e-12),
  displaySide = Math.sqrt(pixels.length),
) {
  const side = Math.sqrt(pixels.length);
  canvas.width = canvas.height = displaySide;
  const ctx = canvas.getContext('2d')!,
    image = ctx.createImageData(side, side);

  pixels.forEach((p, i) => {
    const value = (255 * Math.max(0, p)) / exposure;
    image.data.set([value, value, value, 255], 4 * i);
  });
  const offset = Math.floor((displaySide - side) / 2);
  ctx.putImageData(image, offset, offset);
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

export function liveEstimator(getTruth: () => ExampleTruth | undefined, getOrder: () => Schedule) {
  let schedule: Schedule;
  let truth: ExampleTruth | undefined;
  function compare(selector: string, rows: [string, number, number | undefined][], digits: number) {
    const table = panel.querySelector<HTMLTableElement>(selector)!;
    table.innerHTML = '<tr><th>Parameter</th><th>True</th><th>Estimated</th></tr>';
    for (const [label, actual, estimate] of rows) {
      const row = table.insertRow();
      for (const value of [label, actual.toFixed(digits), estimate?.toFixed(digits) ?? '—']) row.insertCell().textContent = value;
    }
  }
  let panel: HTMLElement,
    engine: VecEngine | null = null;
  let viewer: ReturnType<typeof movieView>;
  const cloudHost = document.createElement('div');
  cloudHost.className = 'bead-cloud';
  const cloudCanvas = document.createElement('canvas');
  cloudCanvas.setAttribute('aria-label','Fitted bead positions in the camera plane');
  const cameraLayer = document.createElement('canvas');
  cameraLayer.className = 'registered-frame';
  const truthLayer = document.createElement('canvas');
  truthLayer.className = 'true-bead-positions';
  truthLayer.setAttribute('aria-label', 'Ground-truth bead centers');
  cloudHost.append(cloudCanvas, cameraLayer, truthLayer);
  // This overlay is a camera-plane view. Keep pan and pinch, suppress 3D orbit.
  cloudCanvas.addEventListener('pointerdown', e => {
    if (e.shiftKey || e.button === 2) e.stopImmediatePropagation();
  }, true);
  let selectedFrame = 0, pixelNm = 1;
  let fittedDrift: number[][] = [];
  let marker: HTMLDivElement;
  let lastPupilGrid: Element | null = null;
  function registerFrame() {
    if (!cloud || !renderedPositions || !viewer || !marker || !cloud.scale) return;
    const d = fittedDrift[selectedFrame];
    marker.hidden = cameraLayer.hidden = !d;
    truthLayer.hidden = !d || !truth?.drift[selectedFrame];
    if (!d) return;
    const W = cloudCanvas.width, H = cloudCanvas.height;
    if (cameraLayer.width !== W) cameraLayer.width = W;
    if (cameraLayer.height !== H) cameraLayer.height = H;
    const scale = cloud.scale / pixelNm; // camera pixels per cloud display pixel
    // The fitted spot at (x,y) is observed at (x+dx,y+dy) in the raw frame.
    const left = width/2 + cloud.cx/pixelNm - W*scale/2 + d[0];
    const top = height/2 + cloud.cy/pixelNm - H*scale/2 + d[1];
    const ctx = cameraLayer.getContext('2d')!;
    ctx.clearRect(0,0,W,H);
    const {canvas, bin} = viewer.display();
    // Each preview sample is taken at the first camera pixel of its bin.
    const edge = (1-bin)/2;
    ctx.drawImage(canvas,(edge-left)/scale,(edge-top)/scale,canvas.width*bin/scale,canvas.height*bin/scale);
    if (truth && !truthLayer.hidden) {
      if (truthLayer.width !== W) truthLayer.width = W;
      if (truthLayer.height !== H) truthLayer.height = H;
      const gt = truthLayer.getContext('2d')!;
      gt.clearRect(0, 0, W, H);
      const [dx,dy] = truth.drift[selectedFrame];
      // First-frame centers plus simulated motion, in the same raw-frame ROI as the image.
      const ox = dx/truth.pixelNm + 0.5 - left, oy = dy/truth.pixelNm + 0.5 - top;
      const dpr = window.devicePixelRatio || 1;
      const arm = Math.min(2.5*dpr, Math.max(dpr, 0.8/scale));
      gt.strokeStyle = '#267bff';
      gt.lineWidth = dpr;
      gt.beginPath();
      const xy = truth.firstFrameXY;
      for (let i = 0; i < xy.length; i += 2) {
        const x = (xy[i]+ox)/scale, y = (xy[i+1]+oy)/scale;
        if (x < -arm || y < -arm || x > W+arm || y > H+arm) continue;
        gt.moveTo(x-arm,y); gt.lineTo(x+arm,y);
        gt.moveTo(x,y-arm); gt.lineTo(x,y+arm);
      }
      gt.stroke();
    }
    const clip = (v:number, limit:number) => Math.min(Math.max(v,0),limit);
    const x0=clip(left, width), x1=clip(left+W*scale, width);
    const y0=clip(top, height), y1=clip(top+H*scale, height);
    marker.hidden = x0>=x1 || y0>=y1;
    Object.assign(marker.style,{left:100*(x0-edge)/(canvas.width*bin)+'%',top:100*(y0-edge)/(canvas.height*bin)+'%',
      width:100*(x1-x0)/(canvas.width*bin)+'%',height:100*(y1-y0)/(canvas.height*bin)+'%'});
  }
  let imageDepth: MovieDepth | undefined;
  let device: GPUDevice;
  let cloud: PointCloudView | undefined;
  let renderedPositions: number[][] | undefined;
  function positions(snapshot: PhotometrySnapshot) {
    fittedDrift = snapshot.drift.slice();
    if (!device || snapshot.xy === renderedPositions) { registerFrame(); return; }
    pixelNm = snapshot.settings.optics.pixel_um * 1000;
    cloud ??= new PointCloudView(device,cloudCanvas,()=>({mode:'fixed',blurNm:pixelNm/4,capNm:pixelNm,
      maxCrlbNm:0,colorZ:false,zlo:0,zhi:1,zExag:1}));
    cloudHost.hidden = false;
    cloud.start(snapshot.width,snapshot.height,pixelNm);
    cloud.onChange = registerFrame;
    const displayShortSide = Math.min(cloudCanvas.clientWidth, cloudCanvas.clientHeight)*(window.devicePixelRatio || 1);
    if(displayShortSide) cloud.scale = Math.min(96,snapshot.width,snapshot.height)*pixelNm/displayShortSide;
    const rows = new Float32Array(snapshot.xy.length * NCOL);
    snapshot.xy.forEach(([x,y],i)=>{rows[i*NCOL+1]=x;rows[i*NCOL+2]=y;});
    cloud.append(rows);
    renderedPositions = snapshot.xy;
    panel.querySelector('.cloud-status')!.textContent = snapshot.xy.length.toLocaleString() + ' fitted spots on XY drift-corrected frames' + (truth ? ' · blue crosses: true bead centers' : '') + ' · scroll to zoom, drag to pan';
  }
  let width = 0,
    height = 0;
  type Fit = ReturnType<typeof modulation>;
  let inspectField: ReturnType<typeof fieldInspector>;
  function displayPattern(fit?:Fit) {
    inspectField(fit,truth?.modulation);
    if(!truth) return;
    const rows:[string, number,number|undefined][]=[];
    truth.modulation.patterns.forEach((p,i)=>{
      const q=fit?.patterns[i], axis=axisName(i);
      for(const [key,label] of [['pitch_nm','pitch (nm)'],['orientation_deg','angle (°)'],['kx_rad_px','kx (rad/px)'],['ky_rad_px','ky (rad/px)']] as const)
        rows.push([axis+' '+label,p[key],q?.[key]]);
      p.phase_steps_rad.forEach((v,j)=>rows.push([axis+' step '+(j+1)+' (°, '+(schedule.estimatePhases?'fitted':'fixed')+')',v*180/Math.PI,q && q.phase_steps_rad[j]*180/Math.PI]));
    });
    compare('.pattern-values',rows,4);
    const coefficients:[string,number,number|undefined][]=[];
    function terms(label:string,c:number[],estimate?:number[]) {
      ['1','u','v','u²','uv','v²'].forEach((term,j)=>coefficients.push([label+' · '+term,c[j],estimate?.[j]]));
    }
    truth.modulation.patterns.forEach((p,i)=>{
      terms(axisName(i)+' depth',p.depth,fit?.patterns[i]?.depth);
      terms(axisName(i)+' phase offset (rad)',p.phase,fit?.patterns[i]?.phase);
    });
    truth.modulation.ratios.forEach((c,i)=>terms('A/'+axisName(i+1)+' intensity ratio',c,fit?.ratios[i]));
    compare('.field-coefficients',coefficients,6);

  }
  function overlay(result: ReturnType<typeof modulation>, name: string) {
    displayPattern(result);
    viewer.setPattern((x, y, t) => {
      const p = result.patterns[axisAt(t, schedule)],
        u = (2 * x) / (width - 1) - 1,
        v = (2 * y) / (height - 1) - 1;
      const poly = (c: number[]) => c[0] + c[1] * u + c[2] * v + c[3] * u * u + c[4] * u * v + c[5] * v * v;
      return (
        (1 +
          poly(p.depth) *
            Math.cos(p.kx_rad_px * x + p.ky_rad_px * y + poly(p.phase) + p.phase_steps_rad[phaseAt(t, schedule)])) /
        2
      );
    }, name, schedule);
  }
  let scoreStage = '',
    scores: number[][] = [],
    motion: number[][] = [],
    origin: number[] = [];
  return {
    positions,
    depthData: () => imageDepth,
    period: () => schedule.axes * schedule.phases,
    start() {
      imageDepth = undefined;
      schedule = getOrder();
      truth = getTruth();
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
        <header class="estimator-heading"><h2>Joint PSF and modulation pattern estimator</h2><a href="#psf-report" hidden>View pupil and aberrations</a><button type="button" hidden>Back to inputs</button></header>
        <div class="estimate-grid">
          <figure class="processing-movie"><figcaption>Input movie</figcaption>
            <div class="fit-movie"></div>
          </figure>

            <div class="psf-column"><div class="compared-psfs"><figure><figcaption>Estimated PSF · XY</figcaption><canvas class="psf" aria-label="Estimated lateral PSF at the current focus"></canvas></figure><figure><figcaption>Estimated PSF · XZ</figcaption><canvas class="psf axial" aria-label="Estimated through-focus PSF at the central Y coordinate"></canvas></figure>
            ${truth ? '<figure><figcaption class="true-caption">True PSF</figcaption><canvas class="psf reference"></canvas></figure>' : ''}</div><small class="parameters">Preparing optics…</small>
            <div class="reconstruction-row"><figure class="reconstruction"><figcaption>Reconstruction</figcaption><small class="cloud-status">Waiting for fitted positions…</small></figure><figure class="pattern-zoom"></figure></div></div>
            <div class="optics-row"><figure class="zernike-chart"><figcaption>Zernike coefficients (mλ RMS)</figcaption><canvas aria-label="Live Zernike coefficients by Noll index"></canvas></figure><div class="pupil-preview"></div></div>
            <section class="modulation-field" hidden></section>
            ${truth ? '<section class="truth-comparison"><table class="modes"></table><table class="drift-values"></table><div><h3>Illumination pattern</h3><table class="pattern-values"></table></div><div><details><summary>Field polynomial coefficients</summary><small>u and v run from −1 to +1 across the camera.</small><table class="field-coefficients"></table></details></div></section>' : ''}
            <div class="live-pupil"></div><figure class="objective" hidden><figcaption>Optical search score</figcaption><canvas aria-label="Optical search score by trial"></canvas></figure>
            <figure class="motion" hidden><figcaption>Estimated drift from frame 1 (nm)</figcaption><canvas aria-label="Estimated XYZ drift by frame"></canvas></figure>
        </div>`;
      renderedPositions = undefined;
      fittedDrift = [];
      cameraLayer.hidden = truthLayer.hidden = true;
      cloudHost.hidden = true;
      panel.querySelector('.cloud-status')!.before(cloudHost);
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
        i => { selectedFrame = i; registerFrame(); },
      );
      const frame = viewer.display().canvas;
      const wrap = document.createElement('div');
      wrap.className = 'movie-with-roi';
      frame.before(wrap);
      marker = document.createElement('div');
      marker.className = 'reconstruction-roi';
      marker.hidden = true;
      wrap.append(frame,marker);
      let drag: {id:number; x:number; y:number; cx:number; cy:number} | undefined;
      const pointerPixels = (event:PointerEvent) => {
        const box = frame.getBoundingClientRect(), {bin} = viewer.display();
        return {x:(event.clientX-box.left)*frame.width*bin/box.width+(1-bin)/2,
          y:(event.clientY-box.top)*frame.height*bin/box.height+(1-bin)/2};
      };
      wrap.onpointerdown = event => {
        if (!cloud || !renderedPositions || event.button !== 0 || drag) return;
        const drift = fittedDrift[selectedFrame];
        if (!drift) return;
        const {x,y} = pointerPixels(event), box = marker.getBoundingClientRect();
        if (marker.hidden || event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) {
          cloud.cx = (x - drift[0] - width/2)*pixelNm;
          cloud.cy = (y - drift[1] - height/2)*pixelNm;
          cloud.request();
        }
        drag = {id:event.pointerId,x,y,cx:cloud.cx,cy:cloud.cy};
        wrap.setPointerCapture(event.pointerId);
        event.preventDefault();
      };
      wrap.onpointermove = event => {
        if (!cloud || !drag || drag.id !== event.pointerId) return;
        const {x,y} = pointerPixels(event);
        cloud.cx = drag.cx + (x-drag.x)*pixelNm;
        cloud.cy = drag.cy + (y-drag.y)*pixelNm;
        cloud.request();
      };
      wrap.onlostpointercapture = () => { drag = undefined; };
      inspectField = fieldInspector(panel.querySelector('.modulation-field')!);
      displayPattern();
      if (truth) {
        const simulation=truth;
        const k=simulation.modulation.patterns[0];
        const pixelNm=Math.hypot(k.kx_rad_px,k.ky_rad_px)*k.pitch_nm/(2*Math.PI);
        viewer.setPattern((x,y,t)=>(1+illumination(pixelNm,x,y,t,simulation.schedule))/2,'True simulated pattern',simulation.schedule);
      }
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
      if (stage==='read' && schedule.phases>=3) {
        imageDepth ??= new MovieDepth(width,height,schedule);
        imageDepth.add(pixels,index-1);
      }
      viewer.frame(pixels, width, height, index - 1, total, stage === 'fit' ? 'Fitting' : 'Reading');
    },
    photometry(snapshot: PhotometrySnapshot) {
      positions(snapshot);
      if (schedule.phases < 3) return;
      try {
        overlay(modulation(snapshot, schedule, imageDepth), 'Preliminary fitted pattern');
      } catch {
        /* A later exposure group can resolve an underdetermined pattern. */
      }
    },
    pattern(result: ReturnType<typeof modulation>) {
      overlay(result, 'Fitted pattern');
    },
    async update(dev: GPUDevice, u: MovieIteration) {
      device = dev;
      const report = document.getElementById('psf-report');
      const destination =
        u.stage === 'final' ? document.querySelector('article') : panel.querySelector('.live-pupil');
      if (report && destination && report.parentElement !== destination) destination.prepend(report);
      const pupilGrid = report?.querySelector('.pupil-grid');
      if (pupilGrid && pupilGrid !== lastPupilGrid) {
        const preview = pupilGrid.cloneNode(true) as HTMLElement;
        const source = pupilGrid.querySelectorAll('canvas');
        preview.querySelectorAll('canvas').forEach((canvas,i) => canvas.getContext('2d')!.drawImage(source[i],0,0));
        panel.querySelector('.pupil-preview')!.replaceChildren(preview);
        lastPupilGrid = pupilGrid;
      }
      const pupilLink = panel.querySelector<HTMLAnchorElement>('a[href="#psf-report"]')!;
      pupilLink.hidden = !document.querySelector('#psf-report:not([hidden])');
      if (!truth && schedule.phases < 3) panel.querySelector('.pattern-zoom small')!.textContent = 'Movie ROI · PSF and photometry only';
      const m = u.model;
      const sets = [{ z: Array.from(m.zcoef), color: '#2f6feb', label: 'Estimated' }];
      if (truth) sets.push({ z: m.noll.map(n => truth!.coefficients[n-5] ?? 0), color: '#c25d25', label: 'True (simulation)' });
      drawZernikeSets(panel.querySelector('.zernike-chart canvas')!, m.noll, sets, 'mλ');
      engine ??= new VecEngine(
        dev,
        { ...m.optics, roisize: m.roisize, npupil: m.npupil, oversample: m.oversample, zernike_noll: m.noll },
        [],
      );
      engine.setZernike(m.zcoef);
      const side = m.roisize, center = Math.floor(side / 2), area = side * side;
      const dz = m.optics.pixel_um;
      // Each plane uses the same emitter and pupil. The center plane is the live XY view.
      engine.setImages(Array.from({length:side}, (_, row) => ({
        x:0, y:0, ze:u.emitterHeightUm, f:u.focusUm + (center-row)*dz, I:1, bg:0,
      })));
      await engine.run(side, 'psf');
      const stack = await engine.readPsf(side);
      const estimated = stack.subarray(center*area,(center+1)*area);
      const axial = new Float32Array(area);
      for(let row=0; row<side; row++) {
        const start = row*area + center*side;
        axial.set(stack.subarray(start,start+side),row*side);
      }
      const frameIndex = u.stage === 'frame' ? u.index - 1 : 0;
      const reference = truth?.psfs[frameIndex];
      const exposure = Math.max(...estimated,...axial,...(reference ?? []),1e-12);
      drawPsf(panel.querySelector('.psf')!, estimated, exposure);
      drawPsf(panel.querySelector('.axial')!, axial, exposure);
      if (truth && reference) {
        drawPsf(panel.querySelector('.reference')!, reference, exposure, Math.sqrt(estimated.length));
        compare('.modes', m.noll.flatMap((n, i): [string, number, number][] =>
          n >= 5 && n <= 11 ? [['Z' + n + ' (mλ RMS)', 1000 * truth!.coefficients[n - 5], 1000 * m.zcoef[i]]] : []), 1);
        const rows: [string, number, number][] = [['Focus relative to bead (nm)',
          truth.firstFocusNm + truth.drift[frameIndex][2], (u.focusUm - u.emitterHeightUm) * 1000]];
        if (u.driftNm) {
          const start = origin.length ? origin : u.driftNm;
          u.driftNm.forEach((v, i) => rows.push(['Drift ' + ['X','Y','Z'][i] + ' (nm)',
            truth!.drift[frameIndex][i], v - start[i]]));
        } else if (u.stage === 'final' && motion.length) {
          ['X','Y','Z'].forEach((axis,i) => {
            const span = (values: number[][]) => Math.max(...values.map(v=>v[i])) - Math.min(...values.map(v=>v[i]));
            rows.push([axis + ' peak-to-peak drift (nm)', span(truth!.drift), span(motion)]);
          });
        }
        compare('.drift-values', rows, 1);
      }
      const z6 = m.noll.indexOf(6);
      const stage =
        u.stage === 'calibration'
          ? 'Stack iteration ' + u.index + '/' + u.total
          : u.stage === 'profile'
            ? 'Optical trial ' + u.index + '/' + u.total
            : u.stage === 'diversity' ? 'Phase diversity '+u.index+'/'+u.total
            : u.stage === 'frame'
              ? 'Frame ' + u.index + '/' + u.total
              : 'Final mean focus';
      panel.querySelector('.parameters')!.textContent =
        stage + ' · XZ: focus increases upward, ' + (dz*1000).toFixed(1) + ' nm/row' +
        ' · focus relative to bead ' +
        Math.round((u.focusUm - u.emitterHeightUm) * 1000) +
        ' nm' +
        (z6 < 0 ? '' : ' · Z6 ' + (1000 * m.zcoef[z6]).toFixed(1) + ' mλ RMS');
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
        const nm = m.optics.pixel_um * 1000;
        fittedDrift[u.index-1] = [u.driftNm[0]/nm,u.driftNm[1]/nm];
        registerFrame();
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
