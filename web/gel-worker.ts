import {volumePsfSlices} from './gel-psf-view';
import {defaultPhotons, photonCount} from './example-photons';
import { getDevice } from 'torch-web/gpu/device';
import { TiffMovie } from 'torch-web/io/tiff';
import { tiffStackBlob } from 'torch-web/psfgrab/tiffstack';
import { buildPsfZip } from 'torch-web/psf/makezip';
import { makeFitReport, pupilMaps, volumeSlices } from 'torch-web/report/fromapp';
import { fitGel, opticalBank, footprint, modes, type GelScan, type GelOptics } from 'torch-dense/gel';
import { gelPattern, transfer } from './gel-pattern';
import { illuminationSeeds } from './gel-seeds';
const status = (message: string) => postMessage({ type: 'status', message });
import { defaultAberration, exampleCoefficients } from './example-aberration';
import { inferFocus } from './gel-focus';
import { axisAt, phaseRadians, defaultSchedule, type Schedule } from './schedule';
import { calibrate } from 'torch-dense/movie';
const compactSampling = 'SIMCODE PSF sampling: compact-v1';
const noise = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
async function generate(device: GPUDevice, settings: GelOptics, count: number, planes: number, gain: number, offset: number, schedule: Schedule, aberration = defaultAberration, photons = defaultPhotons) {
    if (!Number.isInteger(count) || count < 8 || count > 4000 || !Number.isInteger(planes) || planes < 10 || planes > 201)
        throw Error('Use 8–4,000 beads and 10–201 scan planes.');
    const w = 64, h = 64, n = w * h;
    const stage = Array.from({length: planes}, (_, i) => 10 * i / (planes - 1));
    const period=schedule.axes*schedule.phases;
    if (stage.length*period*n>100000000) throw Error('Reduce the example length or illumination cycle to fit in browser memory.');
    const coeff = exampleCoefficients(modes, aberration);
    const bank = await opticalBank(device, settings, coeff);
    const beads = Array.from({ length: count }, () => ({ x: Math.random() * 63, y: Math.random() * 63, z: .05 + 9.9 * Math.random(), photons: photons.signal * (.75 + .5 * Math.random()) }));
    const k = Array.from({length:schedule.axes},(_,i)=>{
        const angle=(13+91*i)*Math.PI/180, pitch=.4+.05*i;
        return [Math.cos(angle)/pitch,Math.sin(angle)/pitch,i%2?-.9:.7];
    });
    const depth=k.map((_,i)=>i%2?.86:.78), ratio=1.16, phase=k.map((_,i)=>i%2?-.73:.52);
    const motion = stage.map(() => [0, 0, 0]);
    for (let d = 0; d < 3; d++) {
        const knots = Array.from({ length: 6 }, (_, i) => i ? noise() : 0);
        for (let i = 1; i < knots.length; i++)
            knots[i] += knots[i - 1];
        const span = Math.max(...knots) - Math.min(...knots) || 1;
        for (let t = 0; t < stage.length; t++) {
            const v = 5 * t / (stage.length - 1), lo = Math.min(4, Math.floor(v)), f = v - lo;
            motion[t][d] = ((1 - f) * knots[lo] + f * knots[lo + 1]) * (d === 2 ? .4 : .18) / span;
        }
    }
    const focus = stage.map((z,t) => z + motion[t][2]);
    const reference=volumePsfSlices(bank);
    postMessage({type:'example-optics',reference, focus, schedule, optics:settings.optics, modes, coefficients:coeff,
        k, phase, depth, offset});
    const frames = new Float32Array(stage.length * period * n), px = settings.optics.pixel_um;
    const planesOut: Float32Array[] = [];
    for (let t = 0; t < stage.length; t++) {
        const [dx, dy] = motion[t];
        for (let c = 0; c < period; c++) {
            const axis = axisAt(c,schedule), frame = frames.subarray((period * t + c) * n, (period * t + c + 1) * n);
            for (let i = 0; i < n; i++)
                frame[i] = photons.background;
            for (const b of beads) {
                const a = k[axis], theta = 2 * Math.PI * (a[0] * (b.x * px + dx) + a[1] * (b.y * px + dy) + a[2] * b.z ) + phaseRadians(c,schedule) + phase[axis];
                const I = b.photons * (axis ? 1 : ratio) * (1 + depth[axis] * transfer(a, settings.diameterUm) * Math.cos(theta));
                footprint(w, h, bank, b.x + dx / px, b.y + dy / px, focus[t] - b.z, (p, v) => { frame[p] += v * I; });
            }
            for (let i = 0; i < n; i++)
                frame[i] = photonCount(frame[i], noise());
            planesOut.push(gain === 1 && offset === 0 ? frame : frame.map(p => p / gain + offset));

        }
        postMessage({type:'example-frames', planes:planesOut.slice(-period), w, h, first:t*period,
            total:stage.length*period});
    }

    postMessage({ type: 'generated', movie: tiffStackBlob(planesOut, planesOut.length, 1, h, w, {info:compactSampling}),
        truth: { beads, coefficients: coeff, drift_um: motion, phase_rad: phase, depths: depth, k_cycles_um: k, intensity_ratio: ratio }, bursts: stage.length });
}
self.onmessage = async ({ data }) => {
    try {
        const { device } = await getDevice(), settings: GelOptics = data.settings;
        if (!Object.values(settings.optics).every(v => Number.isFinite(v) && v > 0) || settings.optics.NA > settings.optics.n_imm || !Number.isFinite(settings.diameterUm) || settings.diameterUm < 0)
            throw Error('Use positive optics with NA at or below the immersion index, and a nonnegative bead diameter.');
        const sequence:Schedule = data.schedule ?? defaultSchedule;
        const period=sequence.axes*sequence.phases;
        if (data.action === 'generate') {
            await generate(device, settings, data.count, data.planes, data.gain, data.offset, sequence, data.aberration, data.photons);
            return;
        }
        if (!data.movie) throw Error('Select a movie TIFF.');
        const movie = await TiffMovie.open([data.movie]);
        if (movie.W*movie.H*movie.n>100000000) throw Error('Crop the gel scan before fitting this field in the browser.');
        const pixels=movie.W*movie.H;
        const raw=new Float32Array(movie.n*pixels);
        for(let first=0;first<movie.n;first+=period) {
            const end=Math.min(movie.n,first+period), chunk=await movie.read(first,end);
            raw.set(chunk,first*pixels);
            postMessage({type:'fit-frames',chunk,first,w:movie.W,h:movie.H,total:movie.n},[chunk.buffer]);
        }
        for (let i=0;i<raw.length;i++) raw[i]=Math.max(0,(raw[i]-data.offset)*data.gain);
        let initial:number[]|undefined;
        if (data.stack) {
            status('Calibrating the initial pupil from the optional Z-stack');
            const a=data.acquisition;
            const model={optics:settings.optics,noll:modes.slice(),zcoef:modes.map(()=>0),npupil:32,oversample:2,roisize:25,focus_um:0,z_range_um:[-.4,.4] as [number,number]};
            await calibrate(device,data.stack,{optics:settings.optics,diameterNm:a.diameterNm,stepNm:a.stepNm,offset:a.offset,gain:a.gain},model);
            initial=model.zcoef;
        }
        status('Estimating relative focus from the bead images and vectorial PSF');
        const compact=movie.description.split(/\r?\n/).includes('Info='+compactSampling);
        const stage=await inferFocus(device,raw,movie.W,movie.H,period,settings,initial,compact);
        const scan:GelScan={frames:raw,width:movie.W,height:movie.H,stage,scanIndex:stage.map((_,i)=>i),estimateFocus:true,sampling:compact?'compact':'wide'};
        let previewBank:Float32Array|undefined;
        const snapshot=(coefficients:number[],focus:number[],beads?:{x:number;y:number;z:number}[],driftPx?:number[][],bank?:Float32Array)=>{
            const slices=bank && bank!==previewBank ? volumePsfSlices(bank) : undefined;
            previewBank=bank;
            postMessage({slices,
            xyz:beads?.map(b=>[b.x,b.y,b.z]),driftPx,type:'fit-optics',coefficients,focus,optics:settings.optics,modes,period});
        };
        snapshot(initial??modes.map(()=>0),stage);
        const canonical=sequence.axes===2 && sequence.phases===3 && !sequence.interleaved;
        const fit=await fitGel(device,scan,settings,status,canonical?illuminationSeeds:undefined,initial,snapshot,
            t=>postMessage({type:'fit-plane',index:(t+1)*period-1}));
        status('Estimating 3D illumination wavevectors');
        let modulation, patternError;
        try {
            modulation = gelPattern(fit, scan, settings, data.randomGel,sequence);
        }
        catch (e) {
            patternError = e instanceof Error ? e.message : String(e);
        }
        snapshot(fit.coefficients,stage.map((z,i)=>z+fit.drift[i][2]),fit.beads,fit.drift.map(d=>[d[0]/settings.optics.pixel_um,d[1]/settings.optics.pixel_um]));
        if(modulation) postMessage({type:"fit-pattern",patterns:modulation.patterns,sequence,pixel:settings.optics.pixel_um});
        const vectorial = { optics: settings.optics, noll: modes, zcoef: fit.coefficients, focus_um: 0, npupil: 96, oversample: 2, roisize: 25, z_range_um: [-1, 1] as [
                number,
                number
            ] };
        let sections;
        const archive = await buildPsfZip(device, { vectorial, report: { estimator: '3D bead volume (experimental)' }, pdf: volume => {
                sections = volumeSlices(volume.stack, { ...vectorial, nxy: volume.W, nz: volume.Z, dxy_um: settings.optics.pixel_um,
                    zmin: volume.z_um[0], zmax: volume.z_um[volume.Z - 1], ze_um: 0 }, 'z');
                return makeFitReport({
                    title: '3D bead volume (experimental) PSF', settings: { 'bead diameter (nm)': 1000 * settings.diameterUm },
                    results: { beads: fit.beads.length, bursts: scan.stage.length }, zernike: { noll: modes, values: fit.coefficients },
                    pupil: pupilMaps(settings.optics, modes, fit.coefficients, vectorial.npupil), psf: sections,
                });
            } });
        postMessage({ type: 'fit', modulation, patternError, sections, archive: archive.bytes, result: { xyz_px_px_um: fit.beads.map(b => [b.x, b.y, b.z]), xy_drift_um: fit.drift.map(d=>d.slice(0,2)),
                estimated_focus_um: stage.map((z,i)=>z+fit.drift[i][2]), flux: fit.flux.map(x => Array.from(x)), noll: modes, pupil_waves: fit.coefficients, objective: fit.history } });
    }
    catch (e) {
        postMessage({ type: 'error', message: e instanceof Error ? e.message : String(e) });
    }
};
