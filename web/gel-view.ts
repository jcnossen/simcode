import {fitProgress} from './fit-progress';
import type { Optics as AcquisitionOptics } from 'torch-web/psf/vectorial';
import { TiffMovie } from 'torch-web/io/tiff';
import { plot, type liveEstimator } from './live';
import type { VolumeSlices } from './gel-psf-view';
import { getDevice } from 'torch-web/gpu/device';
import { fieldMaps } from './pattern-field';
import type { gelPattern } from './gel-pattern';
import type { Acquisition } from './settings';
import { axisAt, phaseAt, phaseRadians, type Schedule } from './schedule';
import { movieView } from './movie-view';
import { moviePupilReport } from 'torch-dense/report';
export function gelView(estimator:ReturnType<typeof liveEstimator>, acquisition: () => Acquisition, schedule: () => Schedule, estimate: HTMLButtonElement, readAberration: () => number, readPhotons: () => {signal:number;background:number}, readDriftNm:()=>number,readCount:(side:number)=>number) {
    const options = document.createElement('div');
    options.innerHTML = `<div class="gel-fields">

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
        drift_um:number[][];
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
    let reference:VolumeSlices|undefined;
    let generatedSchedule:Schedule|undefined;
    let presenting=false;
    function endFit(){
        if(presenting) estimator.end();
    }
    function field(k:number[][],depth:(number|null)[],phase:number[],steps:number[][],ratios:(number|null)[],pixel:number) {
        const constant=(v:number|null)=>[v??NaN,0,0,0,0,0];
        const patterns=k.map(([kx,ky,kz],i)=>({
            pitch_nm:1000/Math.hypot(kx,ky), kz_cycles_um:kz, orientation_deg:Math.atan2(ky,kx)*180/Math.PI,
            kx_rad_px:2*Math.PI*kx*pixel, ky_rad_px:2*Math.PI*ky*pixel,
            depth:constant(depth[i]),phase:constant(phase[i]),phase_steps_rad:steps[i],
        }));
        const model={patterns,ratios:ratios.map(constant)};
        return {...model,maps:fieldMaps(model,reader!.W,reader!.H)};
    }
    const getWorker = () => {
        if (worker)
            return worker;
        worker = new Worker(new URL('./gel-worker.ts', import.meta.url), { type: 'module' });
        worker.onerror = e => { endFit(); progress.stop(); status.textContent = e.message; worker?.terminate(); worker = undefined; generated = false; busy = false; controls(); };
        // Keep asynchronous GPU previews in worker message order.
        let pending=Promise.resolve();
        const current=worker;
        worker.onmessage = ({data}) => {
            pending=pending.then(()=>worker===current ? handle(data) : undefined).catch(e=>{
                endFit(); progress.stop(); status.textContent=String(e); busy=false; controls();
            });
        };
        async function handle(d:MessageEvent['data']) {
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
                reference=d.reference; generatedSchedule=order;
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
                estimator.setDimensions(d.w,d.h);
                for(let j=0;j<count;j++) estimator.movie().frame(d.chunk.subarray(j*n,(j+1)*n),d.w,d.h,d.first+j,d.total,'Reading',displayOffset,j===count-1);
                return;
            }
            if(d.type==='fit-optics') {
                fittedFocus=d.focus.flatMap((z:number)=>Array(d.period).fill(z));
                const {device}=await getDevice();
                if(d.slices) {
                    await estimator.update(device,{stage:'profile',index:1,total:1,emitterHeightUm:5,focusUm:5,
                        model:{optics:d.optics,noll:d.modes,zcoef:d.coefficients,npupil:64,roisize:27,oversample:2,focus_um:5,z_range_um:[4,6]}},d.slices);
                }
                if(d.xyz && d.driftPx){
                    const nm=d.optics.pixel_um*1000;
                    const drift:number[][]=d.driftPx.flatMap(([x,y]:number[],i:number)=>Array(d.period).fill([x,y,1000*(d.focus[i]-d.focus[0])]));
                    estimator.positions({xy:d.xyz,drift,width:reader!.W,height:reader!.H,settings:{optics:d.optics}});
                    const motion=drift.map(([x,y,z])=>[(x-drift[0][0])*nm,(y-drift[0][1])*nm,z]);
                    const actual=truth?.drift_um.flatMap(([x,y],i)=>Array(trueFocus.length/truth!.drift_um.length).fill([
                        (x-truth!.drift_um[0][0])*1000,(y-truth!.drift_um[0][1])*1000,(trueFocus[i*trueFocus.length/truth!.drift_um.length]-trueFocus[0])*1000]));
                    estimator.drift(motion,actual?.length===motion.length ? actual : undefined);
                }
                return;
            }
            if(d.type==='fit-plane') { progress.update(3,(d.index+1)/totalFitFrames); estimator.movie().processing(d.index,'Fitting'); return; }
            if(d.type==='fit-pattern') {
                const order:Schedule=d.sequence;
                estimator.movie().setPattern((x,y,t)=>{
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
                    const picker = document.querySelector<HTMLInputElement>('#inputs [data-kind=gel] input[type=file]')!;
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
                    if (m) {
                        const fit=m as ReturnType<typeof gelPattern>, ps=fit.patterns;
                        estimator.displayPattern(field(ps.map(p=>p.k_cycles_um),ps.map(p=>p.depth),ps.map(p=>p.phase_rad),
                            ps.map(p=>p.phase_steps_rad),fit.intensity_ratios,acquisition().pixelNm/1000));
                        link(results, 'modulation.json', new Blob([JSON.stringify(m)], { type: 'application/json' }));
                    }
                    else {
                        status.textContent += ' Pattern fit unavailable. ' + d.patternError;
                    }
                    link(results, 'psf.zip', new Blob([d.archive], { type: 'application/zip' }));
                    link(results, 'gel-fit.json', new Blob([JSON.stringify(r)], { type: 'application/json' }));
                    progress.finish(); endFit();
                }
            }
            catch (e) {
                endFit(); progress.stop();
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
        truth = undefined; reference=undefined;
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
                const source = await TiffMovie.open([uploaded]);
                if (version !== revision) return;
                reader=source;
                fittedFocus=[]; selected=0;
                const seq=generatedSchedule ?? schedule(), px=acquisition().pixelNm/1000;
                const actual=truth && field(truth.k_cycles_um,truth.depths,truth.phase_rad,
                    truth.depths.map(()=>Array.from({length:seq.phases},(_,j)=>2*Math.PI*j/seq.phases)),
                    truth.depths.slice(1).map(()=>truth!.intensity_ratio),px);
                presenting=true;
                estimator.start({status,progress,coefficients:truth?.coefficients,reference,
                    firstFrameXY:truth && Float32Array.from(truth.beads.flatMap(b=>[b.x,b.y])),field:actual,
                    onBack:()=>{presenting=false;panel.append(status,results);clear(results);}});
                estimator.output().append(results);
            }
            const a = acquisition();
            displayOffset=a.offset;
            const optics = {pixel_um:a.pixelNm/1000, NA:a.na, wavelength_um:a.wavelengthNm/1000, n_imm:a.immersionIndex, n_med:a.sampleIndex};
            getWorker().postMessage({ action, settings:{optics,diameterUm:a.diameterNm/1000}, acquisition:a, schedule:schedule(),
                ...(action === 'generate' ? {aberration:readAberration(),photons:readPhotons(),driftNm:readDriftNm()} : {}), count:action==='generate'?readCount(64):undefined, planes:val('planes'), movie:uploaded, stack, gain:a.gain, offset:a.offset, randomGel:document.querySelector<HTMLInputElement>('.random-beads')!.checked });
        }
        catch (e) {
            busy = false;
            controls();
            endFit(); progress.stop();
            status.textContent = String(e);
        }
    }
    generate.onclick = () => void start('generate');
    options.addEventListener('change',event=>{
        if(active && options.closest('details')?.open && (event.target as Element).closest('.example-drift-scale,.example-density'))
            void start('generate');
    });

    return { panel, options, openExample:()=>{if(active && !busy && !generated) void start('generate');}, fit:()=>{if(active && !estimate.disabled) void start('fit');},
        select:(file?:File, calibration?:File)=>{ if(file!==movie) reset(); movie=file; stack=calibration; controls(); },
        activate:(enabled:boolean)=>{ if(active && !enabled){ if(busy) reset(); estimate.disabled=false; }
            active=enabled; controls(); } };
}
