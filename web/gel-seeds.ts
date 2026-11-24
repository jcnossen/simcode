import { footprint, median, type Bead, type GelOptics, type GelScan } from 'torch-dense/gel';
import { fft1d } from 'torch-web/psfgrab/fft';
import { carrier, type Sample } from './gel-pattern';
type Field = {
    re: Float64Array;
    im: Float64Array;
};
const tau = 2 * Math.PI;
const field = (n: number): Field => ({ re: new Float64Array(n), im: new Float64Array(n) });
function fft3(a: Field, dims: number[], inverse = false) {
    let stride = 1;
    for (const n of dims) {
        const r = new Float64Array(n), u = new Float64Array(n);
        for (let block = 0; block < a.re.length; block += stride * n)
            for (let j = 0; j < stride; j++) {
                const base = block + j;
                for (let i = 0; i < n; i++) {
                    r[i] = a.re[base + i * stride];
                    u[i] = a.im[base + i * stride];
                }
                fft1d(r, u, n, 0, inverse);
                for (let i = 0; i < n; i++) {
                    a.re[base + i * stride] = r[i];
                    a.im[base + i * stride] = u[i];
                }
            }
        stride *= n;
    }
    if (inverse)
        for (let i = 0; i < a.re.length; i++) {
            a.re[i] /= stride;
            a.im[i] /= stride;
        }
}
// A high-pass aperture provides an initial carrier without having to assign
// each shared pixel to an emitter. Only near-focus triplets contribute.
function coarseCarrier(scan: GelScan, beads: Bead[], drift: number[][], pixel: number, axis: number) {
    const { width: w, height: h } = scan, samples: Sample[] = [];
    for (const b of beads) {
        const near: Sample[] = [];
        for (let t = 0; t < scan.stage.length; t++) {
            const d = drift[t];
            if (Math.abs(scan.stage[t] + d[2] - b.z) > .5)
                continue;
            const x = Math.round(b.x + d[0]), y = Math.round(b.y + d[1]), v = [0, 0, 0];
            for (let c = 0; c < 3; c++) {
                const start = (t * 6 + 3 * axis + c) * w * h;
                let core = 0, ring = 0;
                for (let j = -3; j <= 3; j++)
                    for (let i = -3; i <= 3; i++) {
                        const a = scan.frames[start + Math.min(h - 1, Math.max(0, y + j)) * w + Math.min(w - 1, Math.max(0, x + i))];
                        if (Math.abs(j) <= 1 && Math.abs(i) <= 1)
                            core += a;
                        if (Math.abs(j) === 3 || Math.abs(i) === 3)
                            ring += a;
                    }
                v[c] = Math.max(1, core / 9 - ring / 24);
            }
            const r = (2 * v[0] - v[1] - v[2]) / 3, u = (v[2] - v[1]) / Math.sqrt(3), amp = Math.hypot(r, u);
            if (amp < .05 * (v[0] + v[1] + v[2]) / 3)
                continue;
            near.push({ xyz: [(b.x + d[0]) * pixel, (b.y + d[1]) * pixel, b.z], re: r / amp, im: u / amp, weight: 1 });
        }
        for (const s of near) {
            s.weight = 1 / near.length;
            samples.push(s);
        }
    }
    if (samples.length < 8)
        return null;
    try {
        const result = carrier(samples);
        return result.coherence > 4 / Math.sqrt(beads.length) ? result.k_cycles_um : null;
    }
    catch (error) {
        // The conventional overlap fit can proceed without a resolved carrier.
        if (error instanceof RangeError)
            return null;
        throw error;
    }
}
/** SIM-specific candidate proposal: bring the two complex illumination
 * sidebands back to object coordinates and combine them with the scan DC
 * using the fitted vectorial transfer function. Only the proposed XYZ peaks
 * enter TorchPSF. All subsequent fitting uses the original photon counts. */
export function illuminationSeeds(scan: GelScan, settings: GelOptics, beads: Bead[], drift: number[][], bank: Float32Array): Bead[] | null {
    const { width: w, height: h } = scan, pixel = settings.optics.pixel_um;
    if (Math.min(w, h) < 8)
        return null;
    const times = scan.stage.map((_, i) => i).filter(i => scan.scanIndex[i] >= 0)
        .sort((a, b) => scan.stage[a] + drift[a][2] - scan.stage[b] - drift[b][2]);
    const nz = times.length, n = w * h;
    const dims = [w, h, nz].map(x => 2 ** Math.ceil(Math.log2(x + 16)));
    const [nx, ny, sz] = dims, N = nx * ny * sz;
    // Each transform needs two doubles per voxel; use the ordinary detector
    // if the padded sideband workspace would exceed 384 MiB.
    if (N * 8 * 16 > 384 * 1024 ** 2)
        return null;
    const ks = [0, 1].map(axis => coarseCarrier(scan, beads, drift, pixel, axis));
    if (ks.some(k => !k))
        return null;
    const zs = times.map(t => scan.stage[t] + drift[t][2]), z0 = zs[0], step = (zs[nz - 1] - z0) / (nz - 1);
    if (!(step > 0))
        return null;
    const pad = [(nx - w) >> 1, (ny - h) >> 1, (sz - nz) >> 1];
    const mirror = (p: number, size: number) => { const u = ((p % (2 * size - 2)) + 2 * size - 2) % (2 * size - 2); return Math.min(u, 2 * size - 2 - u); };
    const wrap = (p: number, size: number) => (p + size) % size;
    function measured(axis: number, k: number[]) {
        const aligned = field(n * nz);
        for (let iz = 0; iz < nz; iz++) {
            const t = times[iz], d = drift[t], phase = -tau * pixel * (k[0] * d[0] + k[1] * d[1]), c = Math.cos(phase), s = Math.sin(phase);
            for (let y = 0; y < h; y++)
                for (let x = 0; x < w; x++) {
                    const xx = Math.max(0, Math.min(w - 1, x + d[0])), yy = Math.max(0, Math.min(h - 1, y + d[1]));
                    const ix = Math.floor(xx), iy = Math.floor(yy), fx = xx - ix, fy = yy - iy;
                    const sample = (slot: number) => {
                        const p = (6 * t + slot) * n, im = scan.frames, x1 = Math.min(ix + 1, w - 1), y1 = Math.min(iy + 1, h - 1);
                        return (1 - fy) * ((1 - fx) * im[p + iy * w + ix] + fx * im[p + iy * w + x1]) + fy * ((1 - fx) * im[p + y1 * w + ix] + fx * im[p + y1 * w + x1]);
                    };
                    const p = iz * n + y * w + x;
                    if (axis < 0) {
                        for (let j = 0; j < 6; j++)
                            aligned.re[p] += sample(j) / 6;
                    }
                    else {
                        const a = sample(axis * 3), b = sample(axis * 3 + 1), v = sample(axis * 3 + 2);
                        const r = (2 * a - b - v) / 3, u = (v - b) / Math.sqrt(3);
                        aligned.re[p] = r * c - u * s;
                        aligned.im[p] = r * s + u * c;
                    }
                }
        }
        const regular = field(n * nz);
        let prev = 0;
        for (let z = 0; z < nz; z++) {
            const height = z0 + z * step;
            while (prev < nz - 2 && zs[prev + 1] < height)
                prev++;
            const f = Math.max(0, Math.min(1, (height - zs[prev]) / Math.max(1e-8, zs[prev + 1] - zs[prev])));
            for (let y = 0; y < h; y++)
                for (let x = 0; x < w; x++) {
                    const i = y * w + x, p = prev * n + i, j = z * n + i;
                    const r = (1 - f) * aligned.re[p] + f * aligned.re[p + n], u = (1 - f) * aligned.im[p] + f * aligned.im[p + n];
                    const phase = -tau * (pixel * (k[0] * x + k[1] * y) + k[2] * height), c = Math.cos(phase), s = Math.sin(phase);
                    regular.re[j] = c * r - s * u;
                    regular.im[j] = s * r + c * u;
                }
        }
        const a = field(N);
        for (let z = 0; z < sz; z++)
            for (let y = 0; y < ny; y++)
                for (let x = 0; x < nx; x++) {
                    const i = (mirror(z - pad[2], nz) * h + mirror(y - pad[1], h)) * w + mirror(x - pad[0], w), j = (z * ny + y) * nx + x;
                    a.re[j] = regular.re[i];
                    a.im[j] = regular.im[i];
                }
        fft3(a, dims);
        return a;
    }
    function transfer(k: number[]) {
        const a = field(N), cx = nx >> 1, cy = ny >> 1;
        for (let iz = 0; iz < sz; iz++) {
            const z = (iz < sz / 2 ? iz : iz - sz) * step;
            footprint(nx, ny, bank, cx, cy, z, (p, v) => {
                const y = Math.floor(p / nx) - cy, x = p % nx - cx, j = (iz * ny + wrap(y, ny)) * nx + wrap(x, nx);
                const phase = -tau * (pixel * (k[0] * x + k[1] * y) + k[2] * z);
                a.re[j] = v * Math.cos(phase);
                a.im[j] = v * Math.sin(phase);
            });
        }
        fft3(a, dims);
        return a;
    }
    const dc = measured(-1, [0, 0, 0]), H = transfer([0, 0, 0]), out = field(N), power = new Float64Array(N);
    for (let i = 0; i < N; i++) {
        out.re[i] = H.re[i] * dc.re[i] + H.im[i] * dc.im[i];
        out.im[i] = H.re[i] * dc.im[i] - H.im[i] * dc.re[i];
        power[i] = H.re[i] ** 2 + H.im[i] ** 2;
    }
    for (let axis = 0; axis < 2; axis++) {
        const k = ks[axis]!, v = measured(axis, k), q = transfer(k);
        let sr = 0, si = 0, norm = 0;
        for (let i = 0; i < N; i++) {
            const h2 = H.re[i] ** 2 + H.im[i] ** 2, q2 = q.re[i] ** 2 + q.im[i] ** 2;
            if (h2 < .15 ** 2 || h2 > 25 || q2 < .15 ** 2)
                continue;
            // Fit a complex sideband gain in the frequency overlap with DC.
            const ar = dc.re[i] * q.re[i] - dc.im[i] * q.im[i], ai = dc.re[i] * q.im[i] + dc.im[i] * q.re[i];
            const br = v.re[i] * H.re[i] - v.im[i] * H.im[i], bi = v.re[i] * H.im[i] + v.im[i] * H.re[i], wt = 1 / (h2 + q2);
            sr += wt * (ar * br + ai * bi);
            si += wt * (ar * bi - ai * br);
            norm += wt * (ar * ar + ai * ai);
        }
        if (!norm)
            return null;
        sr /= norm;
        si /= norm;
        if (Math.hypot(sr, si) < .03)
            return null;
        for (let z = 0; z < sz; z++)
            for (let y = 0; y < ny; y++)
                for (let x = 0; x < nx; x++) {
                    const i = (z * ny + y) * nx + x, j = (((sz - z) % sz * ny + (ny - y) % ny) * nx + (nx - x) % nx);
                    const qr = q.re[i] * sr - q.im[i] * si, qi = q.re[i] * si + q.im[i] * sr;
                    const r = qr * v.re[i] + qi * v.im[i], u = qr * v.im[i] - qi * v.re[i], den = qr * qr + qi * qi;
                    out.re[i] += r;
                    out.im[i] += u;
                    power[i] += den;
                    out.re[j] += r;
                    out.im[j] -= u;
                    power[j] += den;
                }
    }
    for (let i = 0; i < N; i++) {
        const inv = 1 / (power[i] + .1);
        out.re[i] *= inv;
        out.im[i] *= inv;
    }
    fft3(out, dims, true);
    // Smooth the nonnegative reconstruction at sub-resolution scale for peak finding.
    let smooth = Float32Array.from(out.re, v => Math.max(0, v));
    for (const stride of [1, nx, nx * ny]) {
        const next = smooth.slice();
        for (let i = 2 * stride; i < N - 2 * stride; i++)
            next[i] = .614 * smooth[i] + .188 * (smooth[i - stride] + smooth[i + stride]) + .005 * (smooth[i - 2 * stride] + smooth[i + 2 * stride]);
        smooth = next;
    }
    const values: number[] = [];
    for (let z = 0; z < nz; z += 2)
        for (let y = 0; y < h; y += 2)
            for (let x = 0; x < w; x += 2)
                values.push(smooth[((z + pad[2]) * ny + y + pad[1]) * nx + x + pad[0]]);
    const center = median(values), threshold = center + 3.5 * 1.4826 * median(values.map(v => Math.abs(v - center)));
    const peaks: Bead[] = [];
    for (let z = 0; z < nz; z++)
        for (let y = 0; y < h; y++)
            for (let x = 0; x < w; x++) {
                const i = ((z + pad[2]) * ny + y + pad[1]) * nx + x + pad[0], v = smooth[i];
                if (v <= threshold)
                    continue;
                let peak = true;
                for (let iz = -1; iz <= 1 && peak; iz++)
                    for (let iy = -1; iy <= 1 && peak; iy++)
                        for (let ix = -1; ix <= 1; ix++) {
                            if (smooth[i + iz * nx * ny + iy * nx + ix] > v) {
                                peak = false;
                                break;
                            }
                        }
                if (peak)
                    peaks.push({ x, y, z: z0 + z * step, photons: Math.max(900, v * 12) });
            }
    return peaks.length >= 8 ? peaks : null;
}
