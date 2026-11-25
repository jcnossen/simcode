import { FourierDevice } from './dense-fourier';
import { solve } from './modulation';
export type Bin = {
    idx: number;
    x: number;
    y: number;
    weight: number;
};
const count = 14;
function basis({ x, y }: Bin) {
    x /= .18;
    y /= .18;
    return [x * x, y * y, x * y, x ** 4, y ** 4, x * x * y * y, x ** 3 * y, x * y ** 3, x, y, x ** 3, x * x * y, x * y * y, y ** 3];
}
// Real harmonic basis [1,cos,sin]; both quadratures together have twice the
// combined power in the +1 and -1 complex orders used by the PyTorch code.
function orderShader(nt: number, np: number, calibration: boolean) {
    return `@group(0) @binding(0) var<storage,read> data:array<vec2f>;
@group(0) @binding(1) var<storage,read> modes:array<f32>;
@group(0) @binding(2) var<storage,read> params:array<f32>;
@group(0) @binding(3) var<storage,read> trig:array<vec2f>;
@group(0) @binding(4) var<storage,read_write> out:array<${calibration ? 'vec2f' : 'vec4f'}>;
fn mul(a:vec2f,b:vec2f)->vec2f{return a*b.x+vec2f(-a.y,a.x)*b.y;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let f=id.x;if(f>=arrayLength(&modes)/14u){return;}
 var hh:array<vec2f,${nt}>;var proj:array<vec3f,${nt}>;
 var grad:array<vec2f,${nt}>;var gram=mat3x3f();
 for(var t=0u;t<${nt}u;t++){
 var logh=0.;var phase=0.;
 for(var j=0u;j<14u;j++){let v=modes[14u*f+j]*params[14u*t+j];if(j<8u){logh+=v;}else{phase+=v;}}
 let h=exp(clamp(logh,-3.,3.))*vec2f(cos(phase),sin(phase));hh[t]=h;
 let row=vec3f(1.,trig[t]);let weighted=row*dot(h,h);
 gram+=mat3x3f(weighted*row.x,weighted*row.y,weighted*row.z);
 }
 gram+=mat3x3f(vec3f(1,0,0),vec3f(0,2,0),vec3f(0,0,2))*${calibration ? '0.00001' : '0.0000001'};
 let cof=mat3x3f(cross(gram[1],gram[2]),cross(gram[2],gram[0]),cross(gram[0],gram[1]));
 let inv=cof*(1./max(1e-15,dot(gram[0],cof[0])));
 for(var t=0u;t<${nt}u;t++){proj[t]=inv*vec3f(1.,trig[t]);}
 for(var p=0u;p<${np}u;p++){
 var re=vec3f(0);var im=vec3f(0);
 for(var t=0u;t<${nt}u;t++){
 let h=hh[t];let v=mul(data[(f*${np}u+p)*${nt}u+t],vec2f(h.x,-h.y));
 re+=proj[t]*v.x; im+=proj[t]*v.y;}
 var power=re*re+im*im;
 for(var t=0u;t<${nt}u;t++){
 let row=vec3f(1.,trig[t]);let pred=mul(hh[t],vec2f(dot(row,re),dot(row,im)));
 let residual=pred-data[(f*${np}u+p)*${nt}u+t];
 ${calibration ? 'grad[t]+=vec2f(dot(residual,pred),dot(residual,vec2f(-pred.y,pred.x)));' :
        'let h2=dot(hh[t],hh[t]);let leverage=h2*dot(proj[t],row);power-=dot(residual,residual)/max(.05,1.-leverage)*h2*proj[t]*proj[t];'}
 }
 ${calibration ? '' : 'out[f*' + np + 'u+p]=vec4f(power,re.x*re.x+im.x*im.x);'}
 }
 ${calibration ? 'for(var t=0u;t<' + nt + 'u;t++){out[f*' + nt + 'u+t]=grad[t];}' : ''}
}`;
}
// Each record contains the three noise-corrected powers and the uncorrected
// DC power. Estimate one optical envelope across all patches, then compare
// increasing frequency bands with the low-frequency reference.
function fieldPowers(records: Float32Array, bins: Bin[], spatial: number[][], side: number) {
    const n = spatial.length;
    const sums = new Float64Array(6 * side), counts = new Uint32Array(sums.length);
    const sectors = bins.map(({x, y}, f) => {
        const angle = Math.atan2(y, x) / Math.PI;
        const sector = 6 * Math.round(Math.hypot(x, y) * side) +
            Math.min(5, Math.floor(6 * (angle - Math.floor(angle))));
        for (let p = 0; p < n; p++) sums[sector] += records[4 * (f * n + p) + 3] / n;
        counts[sector]++;
        return sector;
    });
    const cuts = [.11, .155, .20, .245, Infinity];
    const bands = cuts.map(() => Array.from({length:n}, () => [0, 0]));
    const occupied = cuts.map(() => false);
    bins.forEach((bin, f) => {
        const band = cuts.findIndex(limit => Math.hypot(bin.x, bin.y) <= limit);
        occupied[band] = true;
        const sector = sectors[f], weight = bin.weight / Math.max(1, sums[sector] / counts[sector]);
        bands[band].forEach((pair, p) => {
            const i = 4 * (f * n + p);
            pair[0] += weight * records[i];
            pair[1] += weight * (records[i+1] + records[i+2]);
        });
    });
    const block = (v: number) => Math.max(0, Math.min(3, Math.floor((v + 1) * 2)));
    function ratios(powers: number[][]) {
        const blocks = Array.from({length:16}, () => [0, 0]);
        let dc = 0, ac = 0;
        powers.forEach(([base, harmonic], p) => {
            const cell = blocks[block(spatial[p][1]) + 4 * block(spatial[p][2])];
            cell[0] += base; cell[1] += harmonic;
            dc += base; ac += harmonic;
        });
        return [ac / Math.max(1e-12, dc), ...blocks.map(([d, a]) => (ac-a) / Math.max(1e-12, dc-d))];
    }
    const reference = ratios(bands[0]);
    const result = Array.from({length:n}, () => [0, 0]);
    for (let b = 0; b < bands.length; b++) {
        if (!occupied[b]) continue;
        if (b > 0 && occupied[0]) {
            const diff = ratios(bands[b]).map((v,i) => v-reference[i]);
            const jackknife = diff.slice(1);
            const mean = jackknife.reduce((a,v) => a+v, 0) / 16;
            const se = Math.sqrt(15 / 16 * jackknife.reduce((a,v) => a+(v-mean)**2, 0));
            if (Math.abs(diff[0]) > 3 * se) break;
        }
        bands[b].forEach(([dc, ac], p) => {result[p][0] += dc; result[p][1] += ac;});
    }
    return result;
}

export async function fitPowers(gpu: FourierDevice, spectra: Float32Array[][], angles: number[], train: Bin[], band: Bin[], spatial: number[][], side: number) {
    const nt = angles.length, np = spectra[0].length;
    const params = new Float32Array(nt * count), initial = new Float32Array(params.length);
    const first = new Float32Array(params.length), second = first.slice();
    const trig = gpu.buffer(Float32Array.from(angles.flatMap(a => [Math.cos(a), Math.sin(a)])));
    const par = gpu.buffer(params);
    async function run(bins: Bin[], cal: boolean) {
        const patches = Array.from({ length: np }, (_, i) => i).filter(p => !cal || p % 3 === 0);
        const n = patches.length, nf = bins.length;
        const data = new Float32Array(2 * nf * n * nt), modes = new Float32Array(nf * count);
        const normal = Array.from({ length: 8 }, () => Array(8).fill(0)), rhs = Array.from({ length: nt }, () => Array(8).fill(0));
        for (let f = 0; f < nf; f++) {
            const b = basis(bins[f]);
            modes.set(b, count * f);
            const power = new Float64Array(nt);
            let total = 0;
            let off = 2 * f * n * nt;
            for (const p of patches)
                for (let t = 0; t < nt; t++) {
                    const spec = spectra[t][p], i = 2 * bins[f].idx;
                    data[off++] = spec[i];
                    data[off++] = spec[i + 1];
                    const v = spec[i] ** 2 + spec[i + 1] ** 2;
                    power[t] += v / n;
                    total += v / (nt * n);
                }
            if (cal) {
                const scale = Math.sqrt(Math.max(4, total));
                for (let i = 2 * f * n * nt; i < off; i++)
                    data[i] /= scale;
                const log = power.map(p => .5 * Math.log(Math.max(.001, p / (scale * scale))));
                const avg = log.reduce((a, v) => a + v, 0) / nt;
                for (let j = 0; j < 8; j++) {
                    for (let k = 0; k < 8; k++)
                        normal[j][k] += b[j] * b[k];
                    for (let t = 0; t < nt; t++)
                        rhs[t][j] += b[j] * (log[t] - avg);
                }
            }
        }
        if (cal)
            rhs.forEach((r, t) => initial.set(solve(normal, r), count * t));
        if (cal)
            params.set(initial);
        const storage = gpu.buffer(data), features = gpu.buffer(modes);
        const out = gpu.buffer(cal ? 8 * nf * nt : 16 * nf * n);
        const reduce = gpu.buffer(params.byteLength);
        const gradShader = `@group(0) @binding(0) var<storage,read> src:array<vec2f>;
@group(0) @binding(1) var<storage,read> modes:array<f32>;
@group(0) @binding(2) var<storage,read_write> dest:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
let i=id.x;if(i>=arrayLength(&dest)){return;}let j=i%14u;var sum=0.;
for(var f=0u;f<${nf}u;f++){let g=src[f*${nt}u+i/14u];sum+=select(g.y,g.x,j<8u)*modes[f*14u+j];}
dest[i]=sum*${2 / (nf * nt * n)};}`;
        try {
            for (let it = 1; it <= (cal ? 95 : 1); it++) {
                const centered = params.slice();
                for (let j = 0; j < count; j++) {
                    let mean = 0;
                    for (let t = 0; t < nt; t++)
                        mean += params[t * count + j] / nt;
                    for (let t = 0; t < nt; t++)
                        centered[t * count + j] -= mean;
                }
                gpu.device.queue.writeBuffer(par, 0, centered);
                const enc = gpu.device.createCommandEncoder();
                gpu.dispatch(enc, orderShader(nt, n, cal), [storage, features, par, trig, out], nf);
                if (cal)
                    gpu.dispatch(enc, gradShader, [out, features, reduce], params.length);
                gpu.device.queue.submit([enc.finish()]);
                if (!cal) {
                    return fieldPowers(await gpu.read(out), bins, spatial, side);
                }
                const grad = await gpu.read(reduce);
                for (let j = 0; j < count; j++) {
                    let mean = 0;
                    for (let t = 0; t < nt; t++)
                        mean += grad[t * count + j] / nt;
                    for (let t = 0; t < nt; t++) {
                        const i = t * count + j;
                        const g = grad[i] - mean + .0003 * (params[i] - initial[i]) / (nt * (j < 8 ? 8 : 6));
                        first[i] = .9 * first[i] + .1 * g;
                        second[i] = .999 * second[i] + .001 * g * g;
                        params[i] -= .022 * (it > 60 ? .35 : 1) * (first[i] / (1 - .9 ** it)) / (Math.sqrt(second[i] / (1 - .999 ** it)) + 1e-8);
                    }
                }
            }
        }
        finally {
            [storage, features, out, reduce].forEach(b => b.destroy());
        }
    }
    try {
        await run(train, true);
        return (await run(band, false))!;
    }
    finally {
        trig.destroy();
        par.destroy();
    }
}
