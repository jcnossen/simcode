import {fitProgress} from './fit-progress';
import type { Optics as AcquisitionOptics } from 'torch-web/psf/vectorial';
import { TiffMovie } from 'torch-web/io/tiff';
import { plot } from './live';
import type { gelPattern } from './gel-pattern';
import type { Acquisition } from './settings';
import { axisAt, phaseAt, phaseRadians, type Schedule } from './schedule';
import { movieView } from './movie-view';
import { moviePupilReport } from 'torch-dense/report';
export function gelView(acquisition: () => Acquisition, schedule: () => Schedule, estimate: HTMLButtonElement, readAberration: () => number, readPhotons: () => {signal:number;background:number}) {
    const options = document.createElement('div');
    options.innerHTML = `<div class="gel-fields">
    <label>Beads in the volume<input data-key="beads" type="number" value="160" min="8" max="4000"></label>
    <label>Focus cycles over 10 &micro;m<input data-key="planes" type="number" value="101" min="10" max="201"></label>
    <button class="generate-gel" type="button">Regenerate</button><progress class="gel-generation-progress" aria-label="Movie generation progress" hidden></progress><span class="example-files"></span></div>
    <div class="example-content gel-example-preview" hidden>
      <figure><figcaption>Movie frame</figcaption><div class="gel-example-movie"></div></figure>
      <figure class="pattern-zoom"></figure>
      <section class="pupil-preview"><h3>True pupil</h3><div class="pupil-grid"></div></section>
      <figure class="example-drift gel-focus"><figcaption>True focal position (stage position + Z drift)</figcaption>
      <canvas class="trace" aria-label="Focal position by movie frame"></canvas><output></output></figure>
    </div>`;
    const panel = document.createElement('section');
    panel.className = 'gel-workflow';
    panel.innerHTML = `<p class="gel-status" role="status"></p><div class="gel-results"></div>`;
    const el = <T extends HTMLElement>(s: string) => panel.querySelector<T>(s)!;
    const val = (key:string) => options.querySelector<HTMLInputElement>('[data-key="'+key+'"]')!.valueAsNumber;
    const generate = options.querySelector<HTMLButtonElement>('.generate-gel')!;
    const results = el('.gel-results'), files = options.querySelector<HTMLElement>('.example-files')!, status = el('.gel-status');
    for (const control of options.querySelectorAll('.gel-fields input')) control.setAttribute('form', 'generated-movie-options');
    let movie:File|undefined, stack:File|undefined, active=false;
    let worker: Worker | undefined, busy = false, generated = false, reader: TiffMovie | undefined, revision = 0;
    let truth: {
        beads: {x:number; y:number; z:number}[];
        phase_rad: number[];
        k_cycles_um: number[][];
        depths: number[];
        coefficients: number[];
        intensity_ratio: number;
    } | undefined;
    const urls = new Map<HTMLElement, string[]>();
    function clear(host: HTMLElement) {
        for (const u of urls.get(host) ?? [])
            URL.revokeObjectURL(u);
        urls.delete(host);
        host.replaceChildren();
    }
    function link(host: HTMLElement, name: string, blob: Blob) {
        const a = document.createElement('a'), url = URL.createObjectURL(blob);
        urls.set(host, [...(urls.get(host) ?? []), url]);
        a.href = url;
        a.download = name;
        a.textContent = name;
        host.append(a);
    }
    function controls() {
        generate.disabled = busy;
        if (active) estimate.disabled = busy || !(generated || movie);
        for (const i of options.querySelectorAll<HTMLInputElement>('.gel-fields input'))
            i.disabled = busy;
    }
    const generationBar=options.querySelector<HTMLProgressElement>('.gel-generation-progress')!;
    const progress=fitProgress("gel");
    progress.host.hidden=true;
    status.before(progress.host);
    let totalFitFrames=0;
    const preview = options.querySelector<HTMLElement>('.gel-example-preview')!;
    const focusFigure = preview.querySelector<HTMLElement>('.gel-focus')!;
    const viewer = movieView(preview.querySelector('.gel-example-movie')!,preview.querySelector('.pattern-zoom')!,
        'Gel movie',showFocus,true);
    let displayOffset = 0;
    let focusRows:number[][]=[], trueFocus:number[]=[], fittedFocus:number[]=[], selected=0;
    let labels:string[]=[];
    const heading=focusFigure.querySelector('figcaption')!;
    function updateFocus() {
        labels = [...(trueFocus.length?['True']:[]),...(fittedFocus.length?['Estimated']:[])];
        const length=Math.max(trueFocus.length,fittedFocus.length);
        focusRows=Array.from({length},(_,i)=>[...(trueFocus.length?[trueFocus[i]]:[]),...(fittedFocus.length?[fittedFocus[i]]:[])]);
        heading.textContent='Focal position (µm) · stage position + Z drift';
        focusFigure.hidden=!length;
        showFocus(selected);
    }
    function showFocus(index:number) {
        selected=index;
        if(!focusRows[index]) return;
        plot(focusFigure.querySelector('canvas')!,focusRows,focusRows.length,labels,'Frame',index);
        focusFigure.querySelector('output')!.textContent='Frame '+(index+1)+' · '+focusRows[index].map((v,i)=>labels[i]+' '+v.toFixed(2)+' µm').join(' · ');
    }
    let pupilKey='';
    function pupil(optics:AcquisitionOptics,modes:number[],coefficients:number[],label:string) {
        preview.querySelector('h3')!.textContent=label;
        const key=JSON.stringify([optics,modes,coefficients]);
        if(key===pupilKey)return;
        pupilKey=key;
        const report=document.createElement('div');
        moviePupilReport(report)({optics,noll:modes,zcoef:coefficients,npupil:64,roisize:25,oversample:2,focus_um:0,z_range_um:[-1,1]});
        preview.querySelector('.pupil-grid')!.replaceChildren(...report.querySelectorAll('.pupil-grid figure'));
    }
    function table(headers: string[], rows: string[][]) {
        const t = document.createElement('table');
        const head = t.createTHead().insertRow();
        headers.forEach(s => { const th = document.createElement('th'); th.textContent = s; head.append(th); });
        const body = t.createTBody();
        rows.forEach(row => { const tr = body.insertRow(); row.forEach(c => { tr.insertCell().textContent = c; }); });
        results.append(t);
    }
    const getWorker = () => {
        if (worker)
            return worker;
        worker = new Worker(new URL('./gel-worker.ts', import.meta.url), { type: 'module' });
        worker.onerror = e => { progress.stop(); status.textContent = e.message; worker?.terminate(); worker = undefined; generated = false; busy = false; controls(); };
        worker.onmessage = async ({ data: d }) => {
            if (d.type === 'status') {
                status.textContent = d.message;
                if (/Calibrating the initial pupil|Estimating relative focus|Calibrating the initial pupil from the optional/.test(d.message)) progress.update(1);
                const round=d.message.match(/Refining .*XYZ drift .*?(\d+)\/(\d+)/);
                if(round) progress.update(2,(Number(round[1])-1)/Number(round[2]));
                if(d.message.startsWith('Solving per-exposure')) progress.update(3);
                if(d.message.startsWith('Estimating 3D illumination')) progress.update(4);
                return;
            }
            if (d.type === 'example-optics') {
                const order: Schedule = d.schedule, period = order.axes * order.phases;
                trueFocus = d.focus.flatMap((z:number) => Array(period).fill(z));
                updateFocus();
                displayOffset = d.offset;
                viewer.setPattern((x,y,t)=>{
                    const axis=axisAt(t,order), k=d.k[axis], z=focusRows[t][0];
                    const angle=2*Math.PI*((k[0]*x+k[1]*y)*d.optics.pixel_um+k[2]*z)+phaseRadians(t,order)+d.phase[axis];
                    return (1+d.depth[axis]*Math.cos(angle))/2;
                },'Simulated pattern at focus',order);
                pupil(d.optics,d.modes,d.coefficients,'True pupil');
                preview.hidden = false;
                return;
            }
            if (d.type === 'example-frames') {
                generationBar.max=d.total; generationBar.value=d.first+d.planes.length;
                d.planes.forEach((pixels:Float32Array,j:number)=>viewer.frame(pixels,d.w,d.h,d.first+j,d.total,
                    'Generated',displayOffset,j===d.planes.length-1));
                return;
            }
            if (d.type === 'fit-frames') {
                const n=d.w*d.h, count=d.chunk.length/n;
                totalFitFrames=d.total; progress.update(0,(d.first+count)/d.total);
                for(let j=0;j<count;j++) viewer.frame(d.chunk.subarray(j*n,(j+1)*n),d.w,d.h,d.first+j,d.total,'Reading',displayOffset,j===count-1);
                return;
            }
            if(d.type==='fit-optics') {
                fittedFocus=d.focus.flatMap((z:number)=>Array(d.period).fill(z));
                updateFocus();
                pupil(d.optics,d.modes,d.coefficients,'Estimated pupil');
                return;
            }
            if(d.type==='fit-plane') { progress.update(3,(d.index+1)/totalFitFrames); viewer.processing(d.index,'Fitting'); return; }
            if(d.type==='fit-pattern') {
                const order:Schedule=d.sequence;
                viewer.setPattern((x,y,t)=>{
                    const p=d.patterns[axisAt(t,order)],k=p.k_cycles_um;
                    const angle=2*Math.PI*((k[0]*x+k[1]*y)*d.pixel+k[2]*fittedFocus[t])+p.phase_rad+p.phase_steps_rad[phaseAt(t,order)];
                    return (1+(p.depth??0)*Math.cos(angle))/2;
                },'Estimated pattern at focus',order);
                return;
            }
            try {
                if (d.type === 'error')
                    throw Error(d.message);
                if (d.type === 'generated') {
                    const file = new File([d.movie], 'gel-movie.tif');
                    const picker = document.querySelector<HTMLInputElement>('#inputs .drop input')!;
                    const transfer = new DataTransfer();
                    transfer.items.add(file);
                    movie = file;
                    picker.files = transfer.files;
                    picker.dispatchEvent(new Event('change',{bubbles:true}));
                    const version = revision;
                    truth = d.truth;
                    const source = await TiffMovie.open([file]);
                    if (version !== revision) return;
                    reader = source;
                    generated = true;
                    link(files, 'gel-movie.tif', file);

                    status.textContent = `Generated ${reader.n} exposures at ${d.bursts} focus stops.`;
                }
                else if (d.type === 'fit') {
                    const r = d.result, m = d.modulation;
                    status.textContent = `Recovered ${r.xyz_px_px_um.length} beads from ${r.estimated_focus_um.length} focus stops.`;
                    const sections = document.createElement('div');
                    sections.className = 'gel-slices';
                    results.append(sections);
                    const cloud = document.createElement('figure');
                    const projection = document.createElement('canvas'), caption = document.createElement('figcaption');
                    const stops = r.estimated_focus_um as number[];
                    const lo = Math.min(...stops), hi = Math.max(...stops), width = reader!.W;
                    projection.width = 240; projection.height = 180;
                    const pen = projection.getContext('2d')!;
                    const align = document.createElement('input');
                    align.type='checkbox';
                    const errors = trueFocus.length===fittedFocus.length
                        ? fittedFocus.map((z,t)=>z-trueFocus[t]).sort((a,b)=>a-b) : [];
                    const mid=Math.floor(errors.length/2);
                    const offset=errors.length ? (errors[mid]+errors[Math.floor((errors.length-1)/2)])/2 : 0;
                    function drawProjection() {
                        const correction=align.checked ? offset : 0;
                        const fit=r.xyz_px_px_um as number[][];
                        let zmin=lo-correction, zmax=hi-correction;
                        for(const b of truth?.beads??[]) { zmin=Math.min(zmin,b.z); zmax=Math.max(zmax,b.z); }
                        for(const b of fit) { zmin=Math.min(zmin,b[2]-correction); zmax=Math.max(zmax,b[2]-correction); }
                        const py=(z:number)=>179*(zmax-z)/Math.max(.001,zmax-zmin);
                        pen.fillStyle='#102125'; pen.fillRect(0,0,240,180);
                        pen.strokeStyle='#6baaff'; pen.beginPath();
                        for(const b of truth?.beads??[]) {
                            const u=239*b.x/(width-1),v=py(b.z);
                            pen.moveTo(u-2,v); pen.lineTo(u+2,v); pen.moveTo(u,v-2); pen.lineTo(u,v+2);
                        }
                        pen.stroke(); pen.fillStyle='#e4f3e3';
                        for(const [x,,z] of fit) pen.fillRect(239*x/(width-1)-.8,py(z-correction)-.8,1.6,1.6);
                        caption.textContent='Bead reconstruction · XZ ('+zmin.toFixed(1)+'–'+zmax.toFixed(1)+' µm in Z)'+(truth?' · blue crosses show truth':'');
                    }
                    projection.style.width='240px'; cloud.append(projection,caption); sections.append(cloud);
                    if(truth && errors.length) {
                        const label=document.createElement('label');
                        label.append(align,' Align Z origins using focus traces (fit minus truth '+offset.toFixed(2)+' µm)');
                        cloud.append(label);
                        align.onchange=drawProjection;
                    }
                    drawProjection();
                    for (const [label, slice] of [['Estimated PSF · XY', d.sections.xy[3]], ['Estimated PSF · XZ', d.sections.xz]] as const) {
                        const f = document.createElement('figure'), c = document.createElement('canvas'), caption = document.createElement('figcaption');
                        caption.textContent = label;
                        c.width = slice.width;
                        c.height = slice.height;
                        const g = c.getContext('2d')!, im = g.createImageData(c.width, c.height);
                        const hi = Math.max(...slice.data);
                        for (let p = 0; p < slice.data.length; p++) {
                            const intensity = 255 * Math.log1p(99 * slice.data[p] / hi) / Math.log(100), i = label.endsWith('XZ') ? (c.height - 1 - Math.floor(p / c.width)) * c.width + p % c.width : p;
                            im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = intensity;
                            im.data[4 * i + 3] = 255;
                        }
                        g.putImageData(im, 0, 0);
                        f.append(c, caption);
                        sections.append(f);
                    }
                    if (m) {
                        const row=(label:string,k:number[],depth:number|null,phase:number,coherence?:number)=>[
                            label,(1000/Math.hypot(k[0],k[1])).toFixed(2),(Math.atan2(k[1],k[0])*180/Math.PI).toFixed(2),
                            k[2].toFixed(3),depth?.toFixed(3)??'Unresolved',phase.toFixed(3),coherence?.toFixed(3)??'—'];
                        const rows:string[][]=[];
                        (m as ReturnType<typeof gelPattern>).patterns.forEach((p,i)=>{
                            const axis=String.fromCharCode(65+i);
                            rows.push(row(axis+' fit',p.k_cycles_um,p.depth,p.phase_rad,p.coherence));
                            if(truth?.k_cycles_um[i])rows.push(row(axis+' true',truth.k_cycles_um[i],truth.depths[i],truth.phase_rad[i]));
                        });
                        table(['Axis','Pitch (nm)','Angle (°)','kz (cycles/µm)','Depth','Phase (rad)','Coherence'],rows);
                        table(['Axis','Depth method','Depth from individual bead intensities'],
                            (m as ReturnType<typeof gelPattern>).patterns.map((p,i)=>[String.fromCharCode(65+i),p.depth_method,p.bead_depth?.toFixed(3)??'Unresolved']));
                        const ratio = document.createElement('p');
                        ratio.textContent = 'Intensity ratios (A / each other axis) · ' + (m.intensity_ratios ?? [m.intensity_ratio]).map((v:number|null)=>v?.toFixed(3)??'Unresolved').join(', ') + (truth ? ' · true ' + truth.intensity_ratio : '');
                        if (m.intensity_ratios?.length) results.append(ratio);
                        link(results, 'modulation.json', new Blob([JSON.stringify(m)], { type: 'application/json' }));
                    }
                    else {
                        status.textContent += ' Pattern fit unavailable. ' + d.patternError;
                    }
                    table(['Pupil (mλ RMS)', ...r.noll.map((n: number) => 'Z' + n)], [['Fit', ...r.pupil_waves.map((v: number) => (v * 1000).toFixed(1))],
                        ...(truth ? [['True', ...truth.coefficients.map((v: number) => (v * 1000).toFixed(1))]] : [])]);
                    link(results, 'psf.zip', new Blob([d.archive], { type: 'application/zip' }));
                    link(results, 'gel-fit.json', new Blob([JSON.stringify(r)], { type: 'application/json' }));
                    progress.finish();
                }
            }
            catch (e) {
                progress.stop();
                status.textContent = e instanceof Error ? e.message : String(e);
            }
            finally {
                busy = false;
                controls();
            }
        };
        return worker;
    };
    function reset() {
        progress.stop(); progress.host.hidden=true; generationBar.hidden=true;
        worker?.terminate();
        worker = undefined;
        busy = false;
        generated = false;
        truth = undefined;
        reader = undefined;

        revision++;
        clear(files);
        clear(results);
        options.append(preview);
        viewer.reset();
        preview.hidden = true;
        trueFocus=[]; fittedFocus=[]; selected=0; updateFocus();
        status.textContent = '';
        controls();
    }
    async function start(action: 'generate' | 'fit') {
        try {
            if (action === 'generate') {
                reset();
                generationBar.hidden=false; generationBar.removeAttribute('value');

            }
            const version = revision;
            const uploaded = action === 'fit' ? movie : undefined;
            busy = true;
            controls();
            clear(results);
            status.textContent = action === 'generate' ? 'Generating vectorial gel scan…' : 'Opening gel scan…';
            if (uploaded) {
                progress.start();
                const source = await TiffMovie.open([uploaded]);
                if (version !== revision)
                    return;
                reader = source;
                viewer.reset();
                viewer.setPattern(null,'',schedule());
                fittedFocus=[]; selected=0; updateFocus();
                preview.querySelector('h3')!.textContent='Estimating pupil…';
                preview.querySelector('.pupil-grid')!.replaceChildren();
                pupilKey='';
                status.after(preview);
                preview.hidden=false;
                options.closest('details')!.open=false;
            }
            const a = acquisition();
            displayOffset=a.offset;
            const optics = {pixel_um:a.pixelNm/1000, NA:a.na, wavelength_um:a.wavelengthNm/1000, n_imm:a.immersionIndex, n_med:a.sampleIndex};
            getWorker().postMessage({ action, settings:{optics,diameterUm:a.diameterNm/1000}, acquisition:a, schedule:schedule(),
                ...(action === 'generate' ? {aberration:readAberration(),photons:readPhotons()} : {}), count:val('beads'), planes:val('planes'), movie:uploaded, stack, gain:a.gain, offset:a.offset, randomGel:document.querySelector<HTMLInputElement>('.random-beads')!.checked });
        }
        catch (e) {
            busy = false;
            controls();
            progress.stop();
            status.textContent = String(e);
        }
    }
    generate.onclick = () => void start('generate');

    return { panel, options, openExample:()=>{if(active && !busy && !generated) void start('generate');}, fit:()=>{if(active && !estimate.disabled) void start('fit');},
        select:(file?:File, calibration?:File)=>{ if(file!==movie) reset(); movie=file; stack=calibration; controls(); },
        activate:(enabled:boolean)=>{ if(active && !enabled){ if(busy) reset(); estimate.disabled=false; }
            active=enabled; controls(); } };
}
