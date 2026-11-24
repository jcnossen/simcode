import { fft1d } from 'torch-web/psfgrab/fft';
import type { Bead, GelOptics, GelScan } from 'torch-dense/gel';
/** Fraction with a neighbour inside the lateral Rayleigh radius and the
 * near-focus photometry interval. Densities are inferred from fitted positions. */
export function overlapFraction(beads: Bead[], settings: GelOptics) {
    const { pixel_um, wavelength_um, NA } = settings.optics;
    const radius2 = (.61 * wavelength_um / (NA * pixel_um)) ** 2;
    let overlapping = 0;
    for (const b of beads) {
        if (beads.some(c => b !== c && Math.abs(b.z - c.z) <= .55 &&
            (b.x - c.x) ** 2 + (b.y - c.y) ** 2 <= radius2))
            overlapping++;
    }
    return overlapping / beads.length;
}
/** For uniformly random bead positions, the DC texture and the first
 * illumination harmonic have the same expected power spectrum apart from
 * modulation and sphere transfer. Compare only frequencies away from DC
 * and the coherent illumination peak. Images are calibrated in photons. */
export function gelImageDepth(scan: GelScan, settings: GelOptics, axis: number, k: number[], sphereTransfer: number, period=6, indices=[axis*3,axis*3+1,axis*3+2]): number | null {
    const { width: w, height: h } = scan;
    if (Math.min(w, h) < 16 || Math.abs(sphereTransfer) < .001)
        return null;
    const nx = 2 ** Math.ceil(Math.log2(w)), ny = 2 ** Math.ceil(Math.log2(h));
    const window = new Float64Array(w * h);
    let windowSum = 0;
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const weight = (1 - Math.cos(2 * Math.PI * x / (w - 1))) * (1 - Math.cos(2 * Math.PI * y / (h - 1))) / 4;
            window[y * w + x] = weight;
            windowSum += weight;
        }
    const { pixel_um, NA, wavelength_um } = settings.optics;
    const fold = (f: number) => f - Math.round(f);
    const kx = fold(k[0] * pixel_um), ky = fold(k[1] * pixel_um);
    const notch = 4 / Math.min(w, h);
    const cutoff = Math.min(.5, .9 * 2 * NA * pixel_um / wavelength_um);
    const bins: number[] = [];
    for (let y = 0; y < ny; y++)
        for (let x = 0; x < nx; x++) {
            const fx = fold(x / nx), fy = fold(y / ny), r = Math.hypot(fx, fy);
            if (r > notch && r < cutoff && Math.hypot(fold(fx - kx), fold(fy - ky)) > notch &&
                Math.hypot(fold(fx + kx), fold(fy + ky)) > notch)
                bins.push(y * nx + x);
        }
    if (bins.length < 16)
        return null;
    const spectra = Array.from({ length: 3 }, () => ({ re: new Float64Array(nx * ny), im: new Float64Array(nx * ny) }));
    const cr = new Float64Array(ny), ci = new Float64Array(ny);
    let dcPower = 0, harmonicPower = 0;
    for (let t = 0; t < scan.stage.length; t++) {
        // Revisited drift references would overrepresent a single slab.
        if (scan.scanIndex[t] < 0)
            continue;
        let shot = 0;
        for (let c = 0; c < 3; c++) {
            const { re, im } = spectra[c], start = (period * t + indices[c]) * w * h;
            re.fill(0);
            im.fill(0);
            let dc = 0;
            for (let p = 0; p < window.length; p++) {
                const value = scan.frames[start + p] * window[p];
                dc += value;
                shot += value * window[p];
            }
            dc /= windowSum;
            for (let y = 0; y < h; y++)
                for (let x = 0; x < w; x++) {
                    const p = y * w + x;
                    re[y * nx + x] = window[p] * (scan.frames[start + p] - dc);
                }
            for (let y = 0; y < ny; y++)
                fft1d(re, im, nx, y * nx);
            for (let x = 0; x < nx; x++) {
                for (let y = 0; y < ny; y++) {
                    cr[y] = re[y * nx + x];
                    ci[y] = im[y * nx + x];
                }
                fft1d(cr, ci, ny);
                for (let y = 0; y < ny; y++) {
                    re[y * nx + x] = cr[y];
                    im[y * nx + x] = ci[y];
                }
            }
        }
        const [a, b, c] = spectra;
        // Three-step DC has weights 1/3. The complex harmonic has weights
        // (2/3) exp(-2πij/3), hence four times as much Poisson noise power.
        for (const p of bins) {
            const r = (a.re[p] + b.re[p] + c.re[p]) / 3, i = (a.im[p] + b.im[p] + c.im[p]) / 3;
            const hr = (2 * a.re[p] - b.re[p] - c.re[p]) / 3 + (b.im[p] - c.im[p]) / Math.sqrt(3);
            const hi = (2 * a.im[p] - b.im[p] - c.im[p]) / 3 + (c.re[p] - b.re[p]) / Math.sqrt(3);
            dcPower += r * r + i * i - shot / 9;
            harmonicPower += hr * hr + hi * hi - 4 * shot / 9;
        }
    }
    return dcPower > 0 && harmonicPower > 0 ? Math.sqrt(harmonicPower / dcPower) / Math.abs(sphereTransfer) : null;
}
