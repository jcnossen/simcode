"""Pattern cycle of the 6 ms movie: spectrum of the frame-mean trace and of per-spot intensity traces (lower threshold).
gain 0.36 e/ADU (per-pixel photon transfer, = the zimflux runfile value), offset from the dark stack."""
import sys, glob, json, tifffile, numpy as np
from scipy import ndimage as ndi
R = '/mnt/data/SIMFLUX Z/24_12_DNA_nanorulers_6ms_1/'
A, N = int(sys.argv[1]), int(sys.argv[2])
fn = sorted(glob.glob(R + '*_MMStack_Default.ome.tif'))[0]
with tifffile.TiffFile(fn, is_ome=False) as t:
    F = np.stack([t.pages[i].asarray() for i in range(A, A + N)]).astype(np.float32)
    el = [t.pages[i].tags['MicroManagerMetadata'].value.get('ElapsedTime-ms') for i in (A, A + 1, A + 100, A + 1000)]
print('elapsed ms at A, A+1, A+100, A+1000:', el)
G, OFF = 0.36, 100.0
F = (F - OFF) * G
m = F.mean((1, 2)); m = m - ndi.uniform_filter1d(m, 25)
fq = np.fft.rfftfreq(len(m)); P = np.abs(np.fft.rfft(m)) ** 2
top = np.argsort(P[10:])[::-1][:8] + 10
print('frame-mean spectrum peaks (cyc/frame, rel):', [(round(fq[i], 4), round(P[i] / np.median(P), 1)) for i in top])
for P_ in (3, 4, 6, 8):
    print('frame-mean folded P=%d:' % P_, np.round([m[(np.arange(len(m)) + A) % P_ == k].mean() for k in range(P_)], 4))
S = ndi.uniform_filter(F, (1, 7, 7)) * 49; L = ndi.uniform_filter(F, (1, 25, 25)) * 49; D = S - L
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 120
mx = ndi.maximum_filter(D, (1, 9, 9)); pk = (D == mx) & (D > thr)
pk[:, :10] = pk[:, -10:] = False; pk[:, :, :10] = pk[:, :, -10:] = False
t_, y, x = np.nonzero(pk); ph = D[t_, y, x]
print('spots/frame > %g ph: %.2f; photons 7x7 p10/50/90:' % (thr, len(t_) / N), np.percentile(ph, [10, 50, 90]).round(0),
      ' bg ph/px median %.2f' % np.median(L / 49))
cnt = np.zeros(F.shape[1:]); np.add.at(cnt, (y, x), 1); cnt = ndi.uniform_filter(cnt, 3) * 9
pos = np.argwhere((cnt == ndi.maximum_filter(cnt, 7)) & (cnt >= 20))
spec = []; folds = {p: [] for p in (3, 4, 6, 8)}; mods = []
for yy, xx in pos:
    tr = D[:, yy, xx]; sm = ndi.uniform_filter1d(tr, 24)
    lab, nl = ndi.label(sm > 0.5 * thr)
    for l in range(1, nl + 1):
        idx = np.nonzero(lab == l)[0]
        if len(idx) < 48: continue
        r = tr[idx] / np.maximum(sm[idx], 1) - 1; r -= r.mean()
        spec.append(np.abs(np.fft.rfft(r, n=480)) ** 2 / len(r))
        for p in folds:
            g = np.array([r[(idx + A) % p == k].mean() for k in range(p)]); folds[p].append(g)
spec = np.array(spec); print('ON segments used:', len(spec))
if len(spec):
    ms = spec.mean(0); f2 = np.fft.rfftfreq(480); top = np.argsort(ms[8:])[::-1][:8] + 8
    print('spot-trace spectrum peaks:', [(round(f2[i], 4), round(ms[i] / np.median(ms), 1)) for i in top])
    for p in folds:
        g = np.array(folds[p]); amp = np.abs(np.fft.fft(g, axis=1)[:, 1]) * 2 / p
        print('P=%d median |1st harmonic| of folded r: %.3f  (random-phase null ~ %.3f)' % (p, np.median(amp), 0))
    np.save('/home/jc/simcode2/out/zimflux/inventory/spot_spectrum_6ms.npy', np.stack([f2, ms]))
