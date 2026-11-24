import { footprint, modes, opticalBank, type GelOptics } from 'torch-dense/gel';

// The same spatial contrast filter is applied to the movie and to the through-focus PSF.
function texture(image: Float32Array, w: number, h: number, stride: number) {
    const values: number[] = [];
    let noise = 0;
    for (let y = 2; y < h - 2; y += stride) {
        for (let x = 2; x < w - 2; x += stride) {
            const p = y*w+x, around = image[p-2]+image[p+2]+image[p-2*w]+image[p+2*w];
            values.push(image[p] - around/4);
            noise += image[p] + around/16;
        }
    }
    return { values: Float32Array.from(values), noise };
}
function dot(a: Float32Array, b: Float32Array) {
    let sum = 0;
    for (let p = 0; p < a.length; p++) sum += a[p]*b[p];
    return sum;
}

export async function inferFocus(device: GPUDevice, frames: Float32Array, w: number, h: number,
        period: number, settings: GelOptics, coefficients = modes.map(() => 0)) {
    const n = w*h, T = frames.length/(period*n);
    if (!Number.isInteger(T) || T < 5) throw Error('Gel focus estimation needs five or more complete illumination cycles.');
    if (T > 1024) throw Error('Select a gel movie with at most 1,024 focus stops.');
    const bank = await opticalBank(device, settings, coefficients);
    const dz = .06, size = 31;
    const slices = Array.from({length: 101}, (_, i) => {
        const img = new Float32Array(size*size);
        footprint(size,size,bank,15,15,(i-50)*dz,(p,v) => { img[p] += v; });
        return texture(img,size,size,1).values;
    });
    // Axial autocorrelation of an incoherent random volume is the autocorrelation
    // of its intensity PSF integrated over emitter depth.
    const lut = Array.from({length: 32}, (_, lag) =>
        slices.slice(lag).reduce((sum, a, j) => sum + dot(a,slices[j]), 0));
    const norm = lut[0];
    if (!(norm>0) || !Number.isFinite(norm)) throw Error('Optical settings produced no usable axial PSF contrast.');
    for (let i=0;i<lut.length;i++) lut[i]=Math.min(i?lut[i-1]:1,lut[i]/norm);
    function separation(c: number) {
        const j = lut.findIndex(v => v <= c);
        if (j === 0) return 0;
        if (j < 0) return Infinity;
        return dz*(j-1+(lut[j-1]-c)/Math.max(1e-9,lut[j-1]-lut[j]));
    }
    const fingerprints = Array.from({length:T}, (_, t) => {
        const img = new Float32Array(n);
        for(let c=0;c<period;c++) for(let p=0;p<n;p++) img[p]+=frames[(t*period+c)*n+p]/period;
        const f = texture(img,w,h,Math.max(1,Math.ceil(Math.sqrt(n/1536))));
        const scale = Math.sqrt(Math.max(0,dot(f.values,f.values)-f.noise/period));
        if (!scale) throw Error('Too little bead contrast to estimate relative focus.');
        return f.values.map(v=>v/scale);
    });
    // Short reliable image-to-image distances connect repeated and reversed sweeps
    // without prescribing a scan direction or total travel.
    const edges = fingerprints.map((a,i) => {
        const neighbors = fingerprints.map((b,j)=>({ j, corr: i===j ? -1 : dot(a,b) }))
            .filter(v=>v.corr>.45).sort((x,y)=>y.corr-x.corr).slice(0,8);
        return neighbors.map(({j,corr})=>({j,d:Math.max(.005,separation(Math.min(1,corr)))}));
    });
    for(let i=0;i<T;i++) for(const {j,d} of edges[i])
        if(!edges[j].some(e=>e.j===i)) edges[j].push({j:i,d});
    function distances(start:number) {
        const d = Array<number>(T).fill(Infinity), seen = new Uint8Array(T);
        d[start]=0;
        for(let k=0;k<T;k++){
            let v=-1;
            for(let i=0;i<T;i++) if(!seen[i] && (v<0 || d[i]<d[v])) v=i;
            if(!Number.isFinite(d[v])) throw Error('The movie has disconnected focus planes. Adjacent planes need overlapping bead images.');
            seen[v]=1;
            for(const e of edges[v]) d[e.j]=Math.min(d[e.j],d[v]+e.d);
        }
        return d;
    }
    const far=(d:number[])=>d.indexOf(Math.max(...d));
    const left=distances(far(distances(0))), right=distances(far(left)), span=Math.max(...left);
    if(span<3*dz) throw Error('Insufficient axial diversity to resolve gel focus.');
    const z=left.map((v,i)=>(span+v-right[i])/2);
    const changes=z.slice(1).map((v,i)=>v-z[i]).filter(v=>Math.abs(v)>.005).sort((a,b)=>a-b);
    if(changes[Math.floor(changes.length/2)]<0) for(let i=0;i<T;i++) z[i]=span-z[i];
    return z;
}
