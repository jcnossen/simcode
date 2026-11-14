"""8-frame cycle of the 6 ms movie: per ON segment, the 8-slot folded profile; split into even / odd slots and compare the
first-harmonic phases (constant difference across emitters -> one orientation; spread -> two orientations)."""
import sys, glob, json, tifffile, numpy as np
from scipy import ndimage as ndi
R = '/mnt/data/SIMFLUX Z/24_12_DNA_nanorulers_6ms_1/'
res = {}
allp, dphi, amp_e, amp_o, ratio_eo, segs = [], [], [], [], [], 0
for A in (20000, 34000):
    fn = sorted(glob.glob(R + '*_MMStack_Default.ome.tif'))[0]
    with tifffile.TiffFile(fn, is_ome=False) as t:
        N = 6000; F = (np.stack([t.pages[i].asarray() for i in range(A, A + N)]).astype(np.float32) - 100) * 0.36
    S = ndi.uniform_filter(F, (1, 7, 7)) * 49; L = ndi.uniform_filter(F, (1, 25, 25)) * 49; D = S - L
    thr = 120; mx = ndi.maximum_filter(D, (1, 9, 9)); pk = (D == mx) & (D > thr)
    pk[:, :10] = pk[:, -10:] = False; pk[:, :, :10] = pk[:, :, -10:] = False
    t_, y, x = np.nonzero(pk)
    cnt = np.zeros(F.shape[1:]); np.add.at(cnt, (y, x), 1); cnt = ndi.uniform_filter(cnt, 3) * 9
    pos = np.argwhere((cnt == ndi.maximum_filter(cnt, 7)) & (cnt >= 20))
    for yy, xx in pos:
        tr = D[:, yy, xx]; sm = ndi.uniform_filter1d(tr, 24)
        lab, nl = ndi.label(sm > 0.5 * thr)
        for l in range(1, nl + 1):
            idx = np.nonzero(lab == l)[0]
            if len(idx) < 48: continue
            idx = idx[:len(idx) // 8 * 8]
            r = tr[idx] / np.maximum(sm[idx], 1)
            g = np.array([r[(idx + A) % 8 == k].mean() for k in range(8)])  # slot = absolute frame % 8
            g = g / g.mean(); allp.append(g); segs += 1
            ge, go = g[0::2], g[1::2]
            fe, fo = np.fft.fft(ge)[1], np.fft.fft(go)[1]
            amp_e.append(abs(fe) * 2 / 4 / ge.mean()); amp_o.append(abs(fo) * 2 / 4 / go.mean())
            dphi.append(np.angle(fo / fe)); ratio_eo.append(go.mean() / ge.mean())
allp = np.array(allp); dphi = np.array(dphi)
print('segments', segs)
print('median 8-slot profile (normalised):', np.median(allp, 0).round(3))
print('modulation depth (1st harmonic of the 4 even / 4 odd slots), median: even %.2f odd %.2f; p25-p75 even %s' %
      (np.median(amp_e), np.median(amp_o), np.percentile(amp_e, [25, 75]).round(2)))
print('odd/even mean intensity ratio median %.3f' % np.median(ratio_eo))
h, _ = np.histogram(dphi, bins=8, range=(-np.pi, np.pi)); print('histogram of phase(odd) - phase(even), 8 bins over 2pi:', h)
print('circular resultant length of dphi: %.2f (1 = constant, 0 = uniform)' % abs(np.mean(np.exp(1j * dphi))))
# alternative hypothesis: one orientation, 8 phase steps: phase of the 8-slot first harmonic vs of slots (even) -> check 2nd/3rd harmonics
H = np.abs(np.fft.fft(allp, axis=1))[:, :5] / 8 * 2
print('median |harmonic| 1..4 of the 8-slot profile:', np.median(H[:, 1:], 0).round(3))
np.save('/home/jc/simcode2/out/zimflux/inventory/cycle6ms_profiles.npy', allp)
