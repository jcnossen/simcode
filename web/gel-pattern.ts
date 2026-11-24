import { fft1d } from 'torch-web/psfgrab/fft';
import { median, type fitGel, type GelScan, type GelOptics } from 'torch-dense/gel';
import { gelImageDepth, overlapFraction } from './gel-depth';
import { regression } from './modulation';
import { axisAt, phaseAt, defaultSchedule } from './schedule';
const tau = 2 * Math.PI;
export function transfer(k: number[], diameter: number) {
    const q = Math.PI * Math.hypot(...k) * diameter;
    return q < .01 ? 1 - q * q / 10 : 3 * (Math.sin(q) - q * Math.cos(q)) / q ** 3;
}
export interface Sample {
    xyz: number[];
    re: number;
    im: number;
    weight: number;
}
export function carrier(samples: Sample[]) {
    const spacing = [.08, .08, .16], low = [Infinity, Infinity, Infinity], high = low.map(() => -Infinity);
    for (const s of samples)
        for (let d = 0; d < 3; d++) {
            low[d] = Math.min(low[d], s.xyz[d]);
            high[d] = Math.max(high[d], s.xyz[d]);
        }
    if (high[2] - low[2] < .1)
        throw new RangeError('Insufficient axial coverage for a 3D pattern fit.');
    const [nx, ny, nz] = spacing.map((s, i) => 2 ** Math.ceil(Math.log2(Math.max(64, (high[i] - low[i]) / s + 3))));
    if (nx * ny * nz > 2 ** 24)
        throw new RangeError('This field exceeds the browser carrier grid. Crop a smaller ROI.');
    const re = new Float64Array(nx * ny * nz), im = new Float64Array(re.length);
    for (const a of samples) {
        const grid = a.xyz.map((x, i) => (x - low[i]) / spacing[i]), cell = grid.map(Math.floor), f = grid.map((x, i) => x - cell[i]);
        for (let z = 0; z < 2; z++)
            for (let y = 0; y < 2; y++)
                for (let x = 0; x < 2; x++) {
                    const i = ((cell[2] + z) * ny + cell[1] + y) * nx + cell[0] + x;
                    const w = a.weight * (z ? f[2] : 1 - f[2]) * (y ? f[1] : 1 - f[1]) * (x ? f[0] : 1 - f[0]);
                    re[i] += w * a.re;
                    im[i] += w * a.im;
                }
    }
    for (let z = 0; z < nz; z++)
        for (let y = 0; y < ny; y++)
            fft1d(re, im, nx, (z * ny + y) * nx);
    function lines(n: number, stride: number, bases: (run: (base: number) => void) => void) {
        const r = new Float64Array(n), u = new Float64Array(n);
        bases(base => {
            for (let i = 0; i < n; i++) {
                r[i] = re[base + i * stride];
                u[i] = im[base + i * stride];
            }
            fft1d(r, u, n);
            for (let i = 0; i < n; i++) {
                re[base + i * stride] = r[i];
                im[base + i * stride] = u[i];
            }
        });
    }
    lines(ny, nx, run => { for (let z = 0; z < nz; z++)
        for (let x = 0; x < nx; x++)
            run(z * ny * nx + x); });
    lines(nz, nx * ny, run => { for (let p = 0; p < nx * ny; p++)
        run(p); });
    const freq = (i: number, n: number, s: number) => (i < n / 2 ? i : i - n) / (s * n);
    let k = [0, 0, 0], max = -1;
    for (let z = 0; z < nz; z++)
        for (let y = 0; y < ny; y++)
            for (let x = 0; x < nx; x++) {
                const kz = freq(z, nz, spacing[2]), kx = freq(x, nx, spacing[0]), ky = freq(y, ny, spacing[1]);
                if (Math.abs(kz) > 2 || Math.hypot(kx, ky) < .2 || Math.hypot(kx, ky) > 1 / .18)
                    continue;
                const i = (z * ny + y) * nx + x, v = re[i] ** 2 + im[i] ** 2;
                if (v > max) {
                    max = v;
                    k = [kx, ky, kz];
                }
            }
    const score = (v: number[]) => {
        let r = 0, u = 0, w = 0;
        for (const a of samples) {
            const angle = -tau * (a.xyz[0] * v[0] + a.xyz[1] * v[1] + a.xyz[2] * v[2]), c = Math.cos(angle), s = Math.sin(angle);
            r += a.weight * (a.re * c - a.im * s);
            u += a.weight * (a.im * c + a.re * s);
            w += a.weight;
        }
        return [Math.hypot(r, u) / w, Math.atan2(u, r)];
    };
    let best = score(k);
    for (let step = .08; step > .0001; step /= 2)
        for (let repeat = 0; repeat < 3; repeat++)
            for (let i = 0; i < 3; i++) {
                const start = k.slice();
                for (const sign of [-1, 1]) {
                    const next = start.slice();
                    next[i] += step * sign;
                    const val = score(next);
                    if (val[0] > best[0]) {
                        best = val;
                        k = next;
                    }
                }
            }
    return { k_cycles_um: k, phase_rad: best[1], coherence: best[0], lateral_pitch_nm: 1000 / Math.hypot(k[0], k[1]) };
}
export function gelPattern(fit: Awaited<ReturnType<typeof fitGel>>, scan: GelScan, settings: GelOptics, randomGel: boolean, schedule = defaultSchedule) {
    if(schedule.phases<3) throw Error('The harmonic pattern fit requires at least three phases per axis.');
    const pixel=settings.optics.pixel_um, period=schedule.phases*schedule.axes;
    const overlap=overlapFraction(fit.beads,settings);
    const ratios=Array.from({length:schedule.axes-1},()=>[] as number[]);
    const slots=Array.from({length:schedule.axes},(_,axis)=>Array.from({length:period},(_,c)=>c)
        .filter(c=>axisAt(c,schedule)===axis).sort((a,b)=>phaseAt(a,schedule)-phaseAt(b,schedule)));
    const patterns=slots.map((indices,axis)=>{
        const rows:{xyz:number[];values:number[];bead:number}[]=[];
        fit.beads.forEach((b,j)=>{
            const localRatios=ratios.map(()=>[] as number[]);
            for(let t=0;t<scan.stage.length;t++) {
                if(Math.abs(scan.stage[t]+fit.drift[t][2]-b.z)>.55) continue;
                const values=indices.map(c=>fit.flux[t*period+c][j]);
                if(axis===0) {
                    const means=slots.map(ix=>ix.reduce((v,c)=>v+fit.flux[t*period+c][j],0)/ix.length);
                    localRatios.forEach((r,i)=>{if(means[i+1]>0) r.push(means[0]/means[i+1]);});
                }
                if(values.some(v=>v>0)) rows.push({bead:j,xyz:[b.x*pixel+fit.drift[t][0],b.y*pixel+fit.drift[t][1],b.z],values});
            }
            localRatios.forEach((r,i)=>{if(r.length) ratios[i].push(median(r));});
        });
        let steps=indices.map((_,i)=>tau*i/indices.length);
        function harmonics() {
            const design=steps.map(p=>[1,Math.cos(p),-Math.sin(p)]);
            const cols=steps.map((_,j)=>regression(design,steps.map((_,i)=>Number(i===j))));
            const samples:Sample[]=[], depths=fit.beads.map(()=>[] as number[]);
            for(const {xyz,values,bead} of rows){
                const [mean,re,im]=[0,1,2].map(d=>values.reduce((sum,v,i)=>sum+v*cols[i][d],0));
                const amp=Math.hypot(re,im);
                if(mean<=0 || amp<.05*mean) continue;
                depths[bead].push(amp/mean);
                samples.push({xyz,re:re/amp,im:im/amp,weight:bead});
            }
            for(const a of samples) a.weight=1/depths[a.weight].length;
            const usable=depths.filter(a=>a.length).map(median);
            if(usable.length<8) throw Error('Fewer than eight usable bead traces for the illumination fit.');
            return {result:carrier(samples),depth:median(usable),beads:usable.length};
        }
        let h=harmonics();
        if(schedule.estimatePhases){
            const design=rows.map(({xyz})=>{
                const p=tau*xyz.reduce((v,x,i)=>v+x*h.result.k_cycles_um[i],0);
                return [1,Math.cos(p),-Math.sin(p)];
            });
            const phases=steps.map((_,i)=>{
                const [,re,im]=regression(design,rows.map(({values})=>values[i]/Math.max(1,values.reduce((a,b)=>a+b)/values.length)));
                return Math.atan2(im,re);
            });
            steps=phases.map(p=>(p-phases[0]+tau)%tau);
            h=harmonics();
        }
        const attenuation=transfer(h.result.k_cycles_um,settings.diameterUm);
        const beadDepth=Math.abs(attenuation)>.001?h.depth/Math.abs(attenuation):null;
        const imageDepth=randomGel && overlap>.5 && !schedule.estimatePhases && schedule.phases===3
            ? gelImageDepth(scan,settings,axis,h.result.k_cycles_um,attenuation,period,indices) : null;
        return {...h.result,depth:imageDepth??beadDepth,bead_depth:beadDepth,image_power_depth:imageDepth,
            depth_method:imageDepth===null?'bead photometry':'random-gel image power',beads:h.beads,phase_steps_rad:steps};
    });
    const intensity_ratios=ratios.map(r=>r.length?median(r):null);
    return {patterns,intensity_ratio:intensity_ratios[0]??null,intensity_ratios,overlapping_fraction:overlap};
}
