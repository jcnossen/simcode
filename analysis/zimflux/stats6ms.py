"""Quick SMLM-level statistics of the 6 ms ZIMFLUX nanoruler movie + camera gain/offset from the calibration stacks.
Sample only (frames A..A+N of the first file), tifffile, run with ionice -c3."""
import sys, json, glob, tifffile, numpy as np
from scipy import ndimage as ndi
R = '/mnt/data/SIMFLUX Z/'
OUT = '/home/jc/simcode2/out/zimflux/inventory/'
A, N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000, int(sys.argv[2]) if len(sys.argv) > 2 else 3000

def read(fn, a, n):
    with tifffile.TiffFile(fn, is_ome=False) as t:
        n = min(n, len(t.pages) - a)
        return np.stack([t.pages[i].asarray() for i in range(a, a + n)]).astype(np.float32)

res = {}
# --- camera calibration (photon transfer) ---
dark = read(glob.glob(R + '24_12_offset_1/*.tif')[0], 0, 200)
dm, dv = dark.mean(0), dark.var(0)
res['offset_ADU_median'] = float(np.median(dm)); res['dark_var_ADU2_median'] = float(np.median(dv))
for g in ('24_12_gain_1', '24_12_gain_10ms_1', '24_12_gain_20ms_1'):
    b = read(glob.glob(R + g + '/*.tif')[0], 0, 1000)
    # remove slow global intensity fluctuations frame-to-frame before the per-pixel variance
    bm = b.mean(0); fr = b.mean((1, 2)); b2 = b * (fr.mean() / fr)[:, None, None]
    bv = b2.var(0)
    s = (bm - dm) > 20
    gpx = (bm - dm)[s] / np.maximum(bv - dv, 1e-3)[s]          # e-/ADU per pixel (shot-noise limited)
    # pooled regression of var vs mean over illuminated pixels
    slope = np.polyfit((bm - dm)[s].ravel(), (bv - dv)[s].ravel(), 1)[0]
    res[g] = {'frames': len(b), 'mean_signal_ADU': float(np.median((bm - dm)[s])), 'frac_lit': float(s.mean()),
              'gain_e_per_ADU_median_px': float(np.median(gpx)), 'gain_e_per_ADU_regression': float(1 / slope)}
    print(g, res[g], flush=True)
gain = float(np.median([res[g]['gain_e_per_ADU_regression'] for g in ('24_12_gain_1', '24_12_gain_10ms_1', '24_12_gain_20ms_1')]))
off = res['offset_ADU_median']
res['gain_used_e_per_ADU'] = gain

# --- movie sample ---
fn = sorted(glob.glob(R + '24_12_DNA_nanorulers_6ms_1/*_MMStack_Default.ome.tif'))[0]
F = (read(fn, A, N) - off) * gain                                  # photons
T = len(F); res['sample_frames'] = [A, A + T]
res['frame_mean_photons_px'] = float(F.mean())
bgimg = ndi.median_filter(np.median(F[::10], 0), 9)                 # slow background map
res['bg_photons_px_median'] = float(np.median(bgimg)); res['bg_photons_px_p10_p90'] = [float(np.percentile(bgimg, 10)), float(np.percentile(bgimg, 90))]
# per-frame spot detection: 7x7 sum above local background, local max, threshold
S = ndi.uniform_filter(F, (1, 7, 7)) * 49
L = ndi.uniform_filter(F, (1, 21, 21)) * 49
D = S - L
mx = ndi.maximum_filter(D, (1, 9, 9))
thr = 300.0                                                         # photons in 7x7 above local bg
pk = (D == mx) & (D > thr)
pk[:, :8] = pk[:, -8:] = False; pk[:, :, :8] = pk[:, :, -8:] = False
t, y, x = np.nonzero(pk)
ph = D[t, y, x]
H, W = F.shape[1:]
res['spots_per_frame'] = float(len(t) / T)
res['density_per_um2_per_frame'] = float(len(t) / T / (H * W * 0.065 ** 2))
res['spot_photons_7x7_percentiles_10_50_90'] = [float(v) for v in np.percentile(ph, [10, 50, 90])]
print({k: res[k] for k in res if not k.startswith('24_12')}, flush=True)

# --- pattern period: per-frame intensity traces of fixed spot positions (cf. analysis/nanoruler/meta.py) ---
cnt = np.zeros((H, W)); np.add.at(cnt, (y, x), 1)
cnt = ndi.uniform_filter(cnt, 3) * 9
pos = np.argwhere((cnt == ndi.maximum_filter(cnt, 7)) & (cnt >= 15))
Pcands = list(range(2, 9))
spec = []; fold = {P: [] for P in Pcands}
for yy, xx in pos:
    tr = D[:, yy, xx]
    sm = ndi.uniform_filter1d(tr, 12)
    on = sm > 0.6 * thr
    lab, nl = ndi.label(on)
    for l in range(1, nl + 1):
        idx = np.nonzero(lab == l)[0]
        if len(idx) < 24:
            continue
        r = tr[idx] / sm[idx] - 1
        r = r - r.mean()
        f = np.abs(np.fft.rfft(r, n=240)) ** 2 / len(r)
        spec.append(f)
        for P in Pcands:
            g = np.array([r[(idx - A * 0 + A) % P == k].mean() if np.any((idx + A) % P == k) else 0 for k in range(P)])
            fold[P].append(np.ptp(g))
spec = np.array(spec)
res['n_on_segments'] = int(len(spec))
if len(spec):
    ms = spec.mean(0); fq = np.fft.rfftfreq(240)
    top = np.argsort(ms[5:])[::-1][:6] + 5
    res['trace_spectrum_peaks_cycles_per_frame'] = [[float(fq[i]), float(ms[i] / np.median(ms))] for i in top]
    res['fold_ptp_median'] = {P: float(np.median(fold[P])) for P in Pcands}
    np.save(OUT + 'trace_spectrum.npy', np.stack([fq, ms]))
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(OUT + 'stats6ms.json', 'w'), indent=1)
