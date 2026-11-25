import { FourierDevice, complex, frequency, tau } from './dense-fourier';
import { fitPowers, type Bin } from './dense-transfer';
import { solve, regression, modulation } from './modulation';
import { VecEngine } from 'torch-web/gpu/ops';
import { TiffMovie } from 'torch-web/io/tiff';
import type { MovieFit } from 'torch-dense/movie';
import { axisAt, type Schedule } from './schedule';
import type { MovieDepth } from './movie-depth';
const side = 64, area = side * side;
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((s, x, i) => s + x * b[i], 0);
function hann(n: number) { return Array.from({ length: n }, (_, i) => .5 - .5 * Math.cos(tau * i / (n - 1))); }
function taper(n: number) { return Array.from({ length: n }, (_, i) => .5 - .5 * Math.cos(Math.PI * Math.min(1, Math.min(i, n - 1 - i) / Math.max(6, Math.floor(n / 20))))); }
function windowed(im: ArrayLike<number>, wx: number[], wy: number[]) {
    const out = new Float32Array(im.length);
    let sum = 0, n = 0;
    for (let y = 0, i = 0; y < wy.length; y++)
        for (let x = 0; x < wx.length; x++, i++) {
            const w = wx[x] * wy[y];
            sum += im[i] * w;
            n += w;
            out[i] = w;
        }
    const mean = sum / n;
    for (let i = 0; i < out.length; i++)
        out[i] *= im[i] - mean;
    return out;
}
function rotate(z: Float32Array, i: number, angle: number) {
    const re = z[i], im = z[i + 1], c = Math.cos(angle), s = Math.sin(angle);
    z[i] = re * c - im * s;
    z[i + 1] = re * s + im * c;
}
// Partial windows taper to zero at the camera edge. Use their actual
// power-weighted spatial moments when fitting a quadratic depth field.
function apertures(length: number) {
    const intervals = Math.min(20, Math.ceil((length - side / 2) / 28));
    const hannWindow = hann(side);
    return Array.from({ length: intervals + 1 }, (_, i) => {
        const start = -side / 4 + i * (length - side / 2) / intervals;
        const origin = start - Math.floor(start) === .5 ? 2 * Math.round(start / 2) : Math.round(start);
        let mass = 0, first = 0, second = 0;
        const window = hannWindow.map((value, j) => {
            const pixel = origin + j, distance = Math.min(pixel, length - 1 - pixel);
            const edge = Math.max(0, Math.min(1, (distance - 3) / 8));
            const weight = value * (.5 - .5 * Math.cos(Math.PI * edge));
            const pos = 2 * pixel / (length - 1) - 1;
            mass += weight ** 2;
            first += weight ** 2 * pos;
            second += weight ** 2 * pos ** 2;
            return weight;
        });
        return { origin, window, first: first / mass, second: second / mass };
    });
}
// The registered low-frequency channels trace an ellipse as the illumination advances.
// Whiten its two principal coordinates, then resolve the sign and phase origin per carrier.
function measuredPhases(samples: Float32Array[]) {
    const nt = samples.length, n = samples[0].length / 2;
    const scales = Array.from({ length: n }, (_, f) => samples.reduce((s, row) => s + row[2 * f] ** 2 + row[2 * f + 1] ** 2, 0) / nt);
    const median = scales.slice().sort((a, b) => a - b)[(n - 1) >> 1];
    for (let j = 0; j < 2 * n; j++) {
        const avg = samples.reduce((s, row) => s + row[j], 0) / nt;
        for (const row of samples)
            row[j] = (row[j] - avg) / Math.sqrt(Math.max(1e-20, scales[j >> 1] + median));
    }
    const covariance = samples.map(a => samples.map(b => dot(a, b)));
    const vectors: number[][] = [];
    for (let axis = 0; axis < 2; axis++) {
        let v = samples.map((_, i) => Math.cos((i + .5) * (axis + 1)));
        for (let it = 0; it < 80; it++) {
            v = covariance.map(row => dot(row, v));
            for (const u of vectors) {
                const proj = dot(u, v);
                v = v.map((a, i) => a - proj * u[i]);
            }
            const norm = Math.sqrt(dot(v, v));
            if (norm < 1e-15)
                throw Error('Insufficient exposure phase diversity.');
            v = v.map(a => a / norm);
        }
        vectors.push(v);
    }
    const xy = samples.map((_, i) => vectors.map(a => a[i] * Math.sqrt(nt)));
    const [a, b, c, d, e] = regression(xy.map(([x, y]) => [x * x, 2 * x * y, y * y, x, y]), Array(nt).fill(1));
    if (a <= 0 || a * c <= b * b)
        return xy.map(([x, y]) => Math.atan2(y, x));
    const center = solve([[a, b], [b, c]], [-d / 2, -e / 2]);
    return xy.map(([x, y]) => {
        x -= center[0];
        y -= center[1];
        return Math.atan2(y * Math.sqrt(c - b * b / a), Math.sqrt(a) * x + b / Math.sqrt(a) * y);
    });
}
async function align(g: FourierDevice, image: Float32Array, w: number, h: number, dx: number, dy: number) {
    const pw = w + 24, ph = h + 24, pad = new Float32Array(pw * ph);
    const reflect = (p: number, n: number) => p < 0 ? -p : p >= n ? 2 * n - 2 - p : p;
    for (let y = 0; y < ph; y++)
        for (let x = 0; x < pw; x++)
            pad[y * pw + x] = image[reflect(y - 12, h) * w + reflect(x - 12, w)];
    const z = await g.fft(complex(pad), pw, ph);
    for (let y = 0; y < ph; y++)
        for (let x = 0; x < pw; x++)
            rotate(z, 2 * (y * pw + x), tau * (dx * frequency(x, pw) + dy * frequency(y, ph)));
    const moved = await g.fft(z, pw, ph, true);
    return image.map((_, i) => moved[2 * ((12 + Math.floor(i / w)) * pw + 12 + i % w)]);
}
async function accumulateCarrier(g: FourierDevice, im: Float32Array, w: number, h: number, acc: Float32Array, t: number) {
    const z = await g.fft(complex(im), w, h), wx = hann(w), wy = hann(h);
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const q = -2 * Math.PI ** 2 * (frequency(x, w) ** 2 + frequency(y, h) ** 2);
            const filter = Math.exp(q * .36) - Math.exp(q * 9), i = 2 * (y * w + x);
            z[i] *= filter;
            z[i + 1] *= filter;
        }
    const blurred = await g.fft(z, w, h, true);
    for (let i = 0; i < w * h; i++) {
        blurred[2 * i] *= wx[i % w] * wy[Math.floor(i / w)];
        blurred[2 * i + 1] = 0;
    }
    const ft = await g.fft(blurred, w, h), up = new Float32Array(8 * w * h);
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const j = 2 * ((y + (y >= Math.ceil(h / 2) ? h : 0)) * 2 * w + x + (x >= Math.ceil(w / 2) ? w : 0)), i = 2 * (y * w + x);
            up[j] = ft[i];
            up[j + 1] = ft[i + 1];
        }
    const fine = await g.fft(up, 2 * w, 2 * h, true);
    let sum = 0;
    for (let i = 0; i < fine.length; i += 2)
        sum += fine[i] ** 2;
    const phase = -tau * (t % 3) / 3, scale = (fine.length / 2) / Math.max(1e-20, sum);
    const c = Math.cos(phase) * scale, s = Math.sin(phase) * scale;
    for (let i = 0; i < fine.length; i += 2) {
        const v = fine[i] ** 2;
        acc[i] += v * c;
        acc[i + 1] += v * s;
    }
}
async function peaks(g: FourierDevice, acc: Float32Array, w: number, h: number) {
    const n = 2 ** Math.ceil(Math.log2(2 * Math.max(w, h))), z = new Float32Array(2 * n * n);
    for (let y = 0; y < 2 * h; y++)
        z.set(acc.subarray(4 * y * w, 4 * (y + 1) * w), 2 * y * n);
    const ft = await g.fft(z, n, n), mag = new Float32Array(n * n), floor = mag.slice(), score = mag.slice();
    for (let i = 0; i < mag.length; i++)
        mag[i] = Math.hypot(ft[2 * i], ft[2 * i + 1]);
    const at = (x: number, y: number) => ((y + n) % n) * n + (x + n) % n;
    for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++)
            for (let j = -7; j <= 7; j++)
                floor[y * n + x] += mag[at(x + j, y)] / 15;
    let max = 0;
    for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++) {
            let avg = 0;
            for (let j = -7; j <= 7; j++)
                avg += floor[at(x, y + j)] / 15;
            score[y * n + x] = avg;
            max = Math.max(max, avg);
        }
    for (let i = 0; i < score.length; i++) {
        const r = 2 * Math.hypot(frequency(i % n, n), frequency(Math.floor(i / n), n));
        score[i] = r >= .05 && r <= .91 ? mag[i] / (score[i] + .001 * max) : 0;
    }
    const local: number[] = [];
    for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++) {
            const i = y * n + x;
            if (!score[i])
                continue;
            let peak = true;
            for (let k = -4; k <= 4 && peak; k++)
                for (let j = -4; j <= 4; j++)
                    if (score[at(x + j, y + k)] > score[i]) {
                        peak = false;
                        break;
                    }
            if (peak)
                local.push(i);
        }
    return local.sort((a, b) => score[b] - score[a]).slice(0, 4).map(i => {
        const x = i % n, y = Math.floor(i / n);
        return [[1, 0], [0, 1]].map(([dx, dy], j) => {
            const l = Math.log(mag[at(x - dx, y - dy)]), r = Math.log(mag[at(x + dx, y + dy)]), c = Math.log(mag[i]);
            const delta = Math.max(-.5, Math.min(.5, .5 * (l - r) / Math.min(-1e-8, l - 2 * c + r)));
            return 2 * tau * (frequency(j ? y : x, n) + delta / n);
        });
    });
}
function resolvePhase(measured: number[], nominal: number[]) {
    return [1, -1].map(sign => {
        const offsets = measured.map((v, i) => nominal[i] - sign * v);
        const c = offsets.reduce((s, v) => s + Math.cos(v), 0), s = offsets.reduce((a, v) => a + Math.sin(v), 0);
        return { angles: measured.map(v => sign * v + Math.atan2(s, c)), consistency: Math.hypot(c, s) / measured.length };
    }).sort((a, b) => b.consistency - a.consistency)[0];
}
// Demodulation is linear, so shifting three unmixed orders is equivalent to shifting every frame.
async function coherence(g: FourierDevice, patches: Float32Array[][], origins: number[][], k: number[], angles: number[], cutoff: number) {
    const bins: number[] = [];
    for (let i = 0; i < area; i++) {
        const x = frequency(i % side, side), y = frequency(Math.floor(i / side), side), a = Math.hypot(x, y), b = Math.hypot(x + k[0] / tau, y + k[1] / tau);
        if (Math.min(a, b) > .08 && Math.max(a, b) < .9 * cutoff)
            bins.push(i);
    }
    if (bins.length < 20 || Math.hypot(...k) < .3)
        return 0;
    const design = angles.map(a => [1, Math.cos(a), Math.sin(a)]);
    const cols = angles.map((_, i) => regression(design, angles.map((_, j) => Number(i === j))));
    const selected = origins.map((_, i) => i).filter(i => i % 2 === 0), sums = [[0, 0], [0, 0]];
    const carrier = Float32Array.from({ length: 2 * area }, (_, i) => {
        const j = i >> 1, a = -(k[0] * (j % side) + k[1] * Math.floor(j / side));
        return i % 2 ? Math.sin(a) : Math.cos(a);
    });
    for (const chunk of [selected.slice(0, Math.ceil(selected.length / 2)), selected.slice(Math.ceil(selected.length / 2))]) {
        if (!chunk.length)
            continue;
        const data = new Float32Array(chunk.length * 4 * 2 * area);
        chunk.forEach((p, j) => {
            const base = j * 8 * area;
            for (let t = 0; t < angles.length; t++) {
                const [a, b, c] = cols[t], patch = patches[t][p];
                for (let i = 0; i < area; i++) {
                    const v = patch[i];
                    data[base + 2 * i] += a * v;
                    data[base + 2 * area + 2 * i] += .5 * b * v;
                    data[base + 2 * area + 2 * i + 1] += .5 * c * v;
                }
            }
            for (let i = 0; i < area; i++) {
                const q = base + 2 * i, c = carrier[2 * i], s = carrier[2 * i + 1];
                data[q + 4 * area] = data[q] * c;
                data[q + 4 * area + 1] = data[q] * s;
                const a = data[q + 2 * area], b = -data[q + 2 * area + 1];
                data[q + 6 * area] = a * c - b * s;
                data[q + 6 * area + 1] = a * s + b * c;
            }
        });
        const ft = await g.fft(data, side, side);
        chunk.forEach((p, j) => {
            const origin = dot(origins[p], k), c = Math.cos(origin), s = Math.sin(origin);
            for (let pair = 0; pair < 2; pair++) {
                const a = (4 * j + pair) * 2 * area, b = (4 * j + (pair ? 2 : 3)) * 2 * area;
                let re = 0, im = 0, p1 = 0, p2 = 0;
                for (const i of bins) {
                    const u = ft[a + 2 * i], v = ft[a + 2 * i + 1], x = ft[b + 2 * i], y = ft[b + 2 * i + 1];
                    re += u * x + v * y;
                    im += v * x - u * y;
                    p1 += u * u + v * v;
                    p2 += x * x + y * y;
                }
                const den = Math.sqrt(Math.max(1e-20, p1 * p2));
                sums[pair][0] += (re * c - im * s) / den;
                sums[pair][1] += (re * s + im * c) / den;
            }
        });
    }
    return sums.reduce((s, a) => s + Math.hypot(...a), 0) / (2 * selected.length);
}
async function opticalPower(g: FourierDevice, r: MovieFit, times: number[]) {
    const m = r.model, rad = r.settings.diameterNm / 2000, off = rad * Math.sqrt(.6);
    const eng = new VecEngine(g.device, { ...m.optics, roisize: 33, npupil: m.npupil, oversample: m.oversample, zernike_noll: m.noll }, [], { maxChunk: 6 });
    const stamps = new Float32Array(2 * times.length * area);
    try {
        eng.setZernike(m.zcoef);
        for (const [i, t] of times.entries()) {
            const f = rad + r.drift[t][2] / 1000;
            eng.setImages([[off, 0, 0], [-off, 0, 0], [0, off, 0], [0, -off, 0], [0, 0, off], [0, 0, -off]].map(([x, y, z]) => ({ x: x / m.optics.pixel_um, y: y / m.optics.pixel_um, ze: rad + z, f, I: 1, bg: 0 })));
            await eng.run(6, 'psf');
            const pixels = await eng.readPsf(6);
            for (let p = 0; p < 33 * 33; p++)
                for (let j = 0; j < 6; j++)
                    stamps[2 * (i * area + Math.floor(p / 33) * side + p % 33)] += pixels[j * 33 * 33 + p] / 6;
        }
    }
    finally {
        eng.destroy();
    }
    const z = await g.fft(stamps, side, side);
    return Float32Array.from({ length: area }, (_, f) => times.reduce((s, _, i) => s + z[2 * (i * area + f)] ** 2 + z[2 * (i * area + f) + 1] ** 2, 0) / times.length);
}
export interface DenseAxis {
    k: number[];
    angles: number[];
    depth: number[];
    tiles: number;
}
export async function denseFields(device: GPUDevice, file: File, r: MovieFit, order: Schedule, initial: number[][], log = (s: string) => { }) {
    const g = new FourierDevice(device), movie = await TiffMovie.open([file]), { W: w, H: h } = movie;
    const xs = apertures(w), ys = apertures(h);
    const windows = ys.flatMap(y => xs.map(x => ({x, y})));
    const origins = windows.map(({x, y}) => [x.origin, y.origin]);
    const spatial = windows.map(({x, y}) => [1, x.first, y.first, x.second, x.first * y.first, y.second]);
    const patches: Float32Array[][] = [], spectra: Float32Array[][] = [], low: Float32Array[] = [];
    const acc = [new Float32Array(8 * w * h), new Float32Array(8 * w * h)];
    const wx = taper(w), wy = taper(h);
    for (let t = 0; t < movie.n; t++) {
        const image = await movie.read(t, t + 1);
        for (let i = 0; i < image.length; i++)
            image[i] = Math.max(0, (image[i] - r.settings.offset) * r.settings.gain);
        const [dx, dy] = r.drift[t];
        const ft = await g.fft(complex(windowed(image, wx, wy)), w, h), samples: number[] = [];
        for (let y = 0; y < h; y++)
            for (let x = 0; x <= w / 2; x++) {
                const fx = x / w, fy = frequency(y, h), q = Math.hypot(fx, fy);
                if (q <= .014 || q >= .047)
                    continue;
                const i = 2 * (y * w + x);
                rotate(ft, i, tau * (dx * fx + dy * fy));
                samples.push(ft[i], ft[i + 1]);
            }
        low.push(Float32Array.from(samples));
        const aligned = await align(g, image, w, h, dx, dy);
        patches[t] = windows.map(({x, y}) => {
            const patch = new Float32Array(area);
            const left = Math.max(0, x.origin), right = Math.min(w, x.origin + side);
            for (let row = Math.max(0, y.origin); row < Math.min(h, y.origin + side); row++) {
                patch.set(aligned.subarray(row * w + left, row * w + right),
                    (row - y.origin) * side + left - x.origin);
            }
            return windowed(patch, x.window, y.window);
        });
        const packed = new Float32Array(origins.length * area * 2);
        patches[t].forEach((p, i) => packed.set(complex(p), 2 * i * area));
        const z = await g.fft(packed, side, side);
        spectra[t] = origins.map((_, i) => z.subarray(2 * i * area, 2 * (i + 1) * area));
        await accumulateCarrier(g, image, w, h, acc[axisAt(t, order)], t);
        if (t % 5 === 0)
            log('Measuring illumination in frame ' + (t + 1) + ' / ' + movie.n);
    }
    const results: DenseAxis[] = [];
    for (let axis = 0; axis < 2; axis++) {
        log('Fitting illumination axis ' + (axis + 1));
        const times = r.drift.map((_, i) => i).filter(t => axisAt(t, order) === axis);
        const measured = measuredPhases(times.map(t => low[t]));
        const proposals = [initial[axis], ...await peaks(g, acc[axis], w, h), initial[axis].map(v => -v)];
        let score = -1, k: number[] = [], angles: number[] = [];
        for (const proposal of proposals) {
            const ph = resolvePhase(measured, times.map(t => tau * (t % 3) / 3 + dot(proposal, r.drift[t])));
            const optics = r.settings.optics;
            const correlation = await coherence(g, times.map(t => patches[t]), origins, proposal, ph.angles, 2 * optics.NA * optics.pixel_um / optics.wavelength_um);
            const value = (correlation + .01) * (.1 + ph.consistency) ** 2;
            if (value > score) {
                score = value;
                k = proposal;
                angles = ph.angles;
            }
        }
        const otf = await opticalPower(g, r, times), train: Bin[] = [], band: Bin[] = [];
        for (let idx = 0; idx < area; idx++) {
            const x = frequency(idx % side, side), y = frequency(Math.floor(idx / side), side), q = Math.hypot(x, y);
            const bin = { idx, x, y, weight: (otf[idx] / (otf[idx] + .015)) ** 2 };
            if (q > .035 && q < .295 && x >= 0)
                train.push(bin);
            if (q > .065 && q < .29 && otf[idx] > .0015)
                band.push(bin);
        }
        if (band.length < 20)
            throw Error('Too little optical bandwidth to estimate modulation depth.');
        const powers = await fitPowers(g, times.map(t => spectra[t]), angles, train, band, spatial, side);
        const q = Math.hypot(...k) * r.settings.diameterNm / (2000 * r.settings.optics.pixel_um);
        const sphere = q < .001 ? 1 : 3 * (Math.sin(q) - q * Math.cos(q)) / q ** 3;
        if (sphere < .1)
            throw Error('The bead diameter suppresses the measured carrier too strongly.');
        const dc = powers.map(p => p[0]), ac = powers.map(p => p[1] / sphere ** 2);
        const total = (v: number[]) => v.reduce((a, b) => a + b, 0);
        if (!(total(dc) > 0 && total(ac) > 0))
            throw Error('Insufficient modulation power above noise.');
        let depth = [Math.sqrt(total(ac) / total(dc)), 0, 0, 0, 0, 0];
        for (let round = 0; round < 4; round++) {
            const d = spatial.map(x => Math.max(.02, dot(x, depth)));
            const lhs = depth.map((_, i) => depth.map((_, j) => total(spatial.map((x, p) => 2 * d[p] * dc[p] * x[i] * x[j]))));
            const rhs = depth.map((_, i) => total(spatial.map((x, p) => (ac[p] - d[p] ** 2 * dc[p]) * x[i])));
            const delta = solve(lhs, rhs);
            depth = depth.map((v, i) => v + delta[i]);
        }
        results.push({ k, angles, depth, tiles: origins.length });
    }
    return results;
}
export async function movieModulation(device: GPUDevice, file: File | undefined, r: MovieFit, schedule: Schedule, image?: MovieDepth, showSpectrum = false, log?: (s: string) => void) {
    const eligible = file && image && schedule.axes === 2 && schedule.phases === 3 && !schedule.interleaved &&
        r.drift.length >= 30 && Math.min(r.width, r.height) >= 224 && r.xy.length >= 6 * Math.floor((r.width - 16) / 40) * Math.floor((r.height - 16) / 40);
    const initial = modulation(r, schedule, eligible ? undefined : image, showSpectrum);
    if (!eligible)
        return initial;
    const fields = await denseFields(device, file, r, schedule, initial.patterns.map(p => [p.kx_rad_px, p.ky_rad_px]), log);
    return modulation(r, schedule, undefined, showSpectrum, fields);
}
