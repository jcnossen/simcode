"""Lateral wave vector of the even-slot and odd-slot patterns of the 6 ms movie: per ON segment, the 4-step phase of the even
and of the odd frames (slot = frame % 8) and a centroid position; grid search of k maximising |sum exp(i(phi - k.r))|."""
import glob, json, tifffile, numpy as np
from scipy import ndimage as ndi
R = '/mnt/data/SIMFLUX Z/24_12_DNA_nanorulers_6ms_1/'
fn = sorted(glob.glob(R + '*_MMStack_Default.ome.tif'))[0]
rows = []
for A in (2000, 12000, 22000, 32000):
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
            img = F[idx, yy - 4:yy + 5, xx - 4:xx + 5].mean(0); img = img - np.median(img); img[img < 0] = 0
            gy, gx = np.mgrid[-4:5, -4:5]; cy = yy + (img * gy).sum() / img.sum(); cx = xx + (img * gx).sum() / img.sum()
            r = tr[idx] / np.maximum(sm[idx], 1)
            g = np.array([r[(idx + A) % 8 == k].mean() for k in range(8)])
            fe, fo = np.fft.fft(g[0::2])[1], np.fft.fft(g[1::2])[1]
            rows.append([cx, cy, np.angle(fe), np.angle(fo), len(idx), A])
rows = np.array(rows); print('segments', len(rows))
np.save('/home/jc/simcode2/out/zimflux/inventory/kvec6ms_rows.npy', rows)
res = {}
for name, col in (('even', 2), ('odd', 3)):
    best = (0, None)
    for pitch in np.arange(4.0, 14.0, 0.05):           # px (65 nm): 260-910 nm
        for ang in np.arange(0, 180, 0.5):
            k = 2 * np.pi / pitch * np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
            v = abs(np.mean(np.exp(1j * (rows[:, col] - rows[:, 0] * k[0] - rows[:, 1] * k[1]))))
            if v > best[0]: best = (v, (pitch, ang))
    # null: shuffled phases
    null = []
    rng = np.random.default_rng(0)
    for _ in range(3):
        ph = rng.permutation(rows[:, col]); b = 0
        for pitch in np.arange(4.0, 14.0, 0.2):
            for ang in np.arange(0, 180, 2):
                k = 2 * np.pi / pitch * np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
                b = max(b, abs(np.mean(np.exp(1j * (ph - rows[:, 0] * k[0] - rows[:, 1] * k[1])))))
        null.append(b)
    res[name] = {'resultant': best[0], 'pitch_px': best[1][0], 'pitch_nm': best[1][0] * 65, 'angle_deg': best[1][1], 'null_max_resultant': null}
    print(name, res[name], flush=True)
json.dump(res, open('/home/jc/simcode2/out/zimflux/inventory/kvec6ms.json', 'w'), indent=1)
