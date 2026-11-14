"""Pattern + SMLM statistics of one ZIMFLUX movie folder (sample chunks of file 1, tifffile, run with ionice -c3).
usage: python pattern_movie.py <folder name> <chunk starts comma-separated> [chunk length=6000] [count_pages=1]
Outputs out/zimflux/inventory/pattern_<folder>.json and _rows.npy.
1. pages per file, frame interval; 2. background, spots/frame, photons (gain 0.36 e-/ADU, offset 100);
3. per ON segment (>= 48 frames) the normalised per-frame intensity r_t; cycle length P from folding at P = 2..12
   (absolute frame index); 4. per slot s of the cycle, the spatial DFT of (r_s - 1) over emitter centroids -> k per slot
   (peak of a 4x upsampled render, refined by Nelder-Mead), slots grouped into orientations by k, phase per slot ->
   phase steps; modulation from the DFT amplitude (2|z|, attenuated by z spread / position noise) and per-segment fits."""
import sys, glob, json, os, tifffile, numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
R = os.environ.get('ZF_ROOT', '/mnt/data/SIMFLUX Z/')   # (jan13 paper movie: ZF_ROOT=.../surfdrive/Zimflux/)
name = sys.argv[1]; starts = [int(a) for a in sys.argv[2].split(',')]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
count = int(sys.argv[4]) if len(sys.argv) > 4 else 1
G, OFF = float(os.environ.get('ZF_GAIN', 0.36)), float(os.environ.get('ZF_OFFSET', 100.0))
OUT = '/home/jc/simcode2/out/zimflux/inventory/pattern_' + os.environ.get('ZF_TAG', name)
files = sorted(glob.glob(R + name + '/*_MMStack_Default*.ome.tif'))
res = {'folder': name, 'files': [os.path.basename(f) for f in files]}
if count:
    res['pages_per_file'] = []
    for f in files:
        with tifffile.TiffFile(f, is_ome=False) as t: res['pages_per_file'].append(len(t.pages))
    print('pages', res['pages_per_file'], flush=True)
segs, stats = [], []
for A in starts:
    with tifffile.TiffFile(files[0], is_ome=False) as t:
        F = np.stack([t.pages[i].asarray() for i in range(A, A + N)]).astype(np.float32)
        try: el = [t.pages[i].tags['MicroManagerMetadata'].value.get('ElapsedTime-ms') for i in (A, A + 1000)]
        except KeyError: el = None   # ImageJ-resaved files (jan13) carry no per-frame metadata
    if el: res.setdefault('ms_per_frame', []).append((el[1] - el[0]) / 1000)
    F = (F - OFF) * G
    S = ndi.uniform_filter(F, (1, 7, 7)) * 49; L = ndi.uniform_filter(F, (1, 25, 25)) * 49; D = S - L
    mx = ndi.maximum_filter(D, (1, 9, 9))
    st = {'A': A, 'bg_ph_px_median': float(np.median(L) / 49)}
    for thr in (60, 120):
        pk = (D == mx) & (D > thr); pk[:, :10] = pk[:, -10:] = False; pk[:, :, :10] = pk[:, :, -10:] = False
        t_, y, x = np.nonzero(pk)
        st[f'spots_per_frame_gt{thr}'] = len(t_) / N
        st[f'box7_photons_p10_50_90_gt{thr}'] = np.percentile(D[t_, y, x], [10, 50, 90]).round(1).tolist()
    stats.append(st); print(st, flush=True)
    thr = 120
    pk = (D == mx) & (D > thr); pk[:, :10] = pk[:, -10:] = False; pk[:, :, :10] = pk[:, :, -10:] = False
    t_, y, x = np.nonzero(pk)
    cnt = np.zeros(F.shape[1:]); np.add.at(cnt, (y, x), 1); cnt = ndi.uniform_filter(cnt, 3) * 9
    pos = np.argwhere((cnt == ndi.maximum_filter(cnt, 7)) & (cnt >= 20))
    gy, gx = np.mgrid[-4:5, -4:5]
    for yy, xx in pos:
        tr = D[:, yy, xx]; sm = ndi.uniform_filter1d(tr, 24)
        lab, nl = ndi.label(sm > 0.5 * thr)
        for l in range(1, nl + 1):
            idx = np.nonzero(lab == l)[0]
            if len(idx) < 48: continue
            img = F[idx, yy - 4:yy + 5, xx - 4:xx + 5].mean(0); img = img - np.median(img); img[img < 0] = 0
            cy = yy + (img * gy).sum() / img.sum(); cx = xx + (img * gx).sum() / img.sum()
            segs.append((cx, cy, idx + A, tr[idx] / np.maximum(sm[idx], 1), float(sm[idx].mean())))
res['stats'] = stats; res['n_segments'] = len(segs)
res['segment_len_p10_50_90'] = np.percentile([len(s[2]) for s in segs], [10, 50, 90]).tolist()
res['segment_box7_photons_per_frame_p10_50_90'] = np.percentile([s[4] for s in segs], [10, 50, 90]).round(1).tolist()
print('segments', len(segs), flush=True)
# cycle length
fold = {}
for P in range(2, 13):
    a = []
    for cx, cy, fr, r, _ in segs:
        g = np.array([r[fr % P == k].mean() for k in range(P)]); a.append(np.std(g))
    fold[P] = float(np.median(a))
res['fold_std_by_P'] = fold; print('fold std by P', {k: round(v, 3) for k, v in fold.items()}, flush=True)
best = max(fold.values()); P = min(p for p in fold if fold[p] > 0.85 * best)   # smallest P reaching the max
res['cycle'] = P
prof = np.array([[r[fr % P == k].mean() for k in range(P)] for cx, cy, fr, r, _ in segs])
prof = prof / prof.mean(1, keepdims=True)
xy = np.array([[s[0], s[1]] for s in segs])
np.save(OUT + '_rows.npy', np.concatenate([xy, prof], 1))
res['slot_mean_intensity_rel'] = (np.median(prof, 0)).round(3).tolist()
H, W = F.shape[1:]; up = 4
def zk(k, w): return np.mean(w * np.exp(-1j * (xy[:, 0] * k[0] + xy[:, 1] * k[1])))
slots = []
for s in range(P):
    w = prof[:, s] - 1
    img = np.zeros((H * up, W * up)); np.add.at(img, ((xy[:, 1] * up).astype(int).clip(0, H * up - 1), (xy[:, 0] * up).astype(int).clip(0, W * up - 1)), w)
    Fi = np.abs(np.fft.fft2(img)); fy = np.fft.fftfreq(H * up, 1 / up) * 2 * np.pi; fx = np.fft.fftfreq(W * up, 1 / up) * 2 * np.pi
    KX, KY = np.meshgrid(fx, fy); kr = np.hypot(KX, KY)
    Fi[(kr < 2 * np.pi / 14) | (kr > 2 * np.pi / 3.7) | (KY < 0)] = 0
    i = np.unravel_index(np.argmax(Fi), Fi.shape); k0 = np.array([KX[i], KY[i]])
    r = minimize(lambda k: -abs(zk(k, w)), k0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-9})
    k = r.x if r.x[1] >= 0 else -r.x
    z = zk(k, w)
    # null: shuffled weights at the same k search (peak of a random permutation)
    slots.append({'slot': s, 'k_rad_px': k.round(5).tolist(), 'pitch_nm': float(2 * np.pi / np.hypot(*k) * 65),
                  'angle_deg': float(np.rad2deg(np.arctan2(k[1], k[0])) % 180), 'm_dft': float(2 * abs(z)), 'phase_deg': float(np.rad2deg(np.angle(z)))})
    print(slots[-1], flush=True)
res['slots'] = slots
# group slots into orientations by angle (within 5 deg) and pitch (within 3 %)
groups = []
for sl in slots:
    for g in groups:
        a = slots[g[0]]
        if min(abs(sl['angle_deg'] - a['angle_deg']), 180 - abs(sl['angle_deg'] - a['angle_deg'])) < 5 and abs(sl['pitch_nm'] / a['pitch_nm'] - 1) < 0.03:
            g.append(sl['slot']); break
    else: groups.append([sl['slot']])
res['orientations'] = []
for g in groups:
    kk = np.mean([slots[s]['k_rad_px'] for s in g], 0)
    ph = [float(np.rad2deg(np.angle(zk(kk, prof[:, s] - 1)))) for s in g]
    steps = [float((ph[i + 1] - ph[i]) % 360) for i in range(len(g) - 1)]
    # per-segment modulation for this orientation (fit 1 + m cos(phi0 + phase_s) with the measured slot phases)
    phs = np.deg2rad(ph); X = np.stack([np.ones(len(g)), np.cos(phs), np.sin(phs)], 1)
    c = np.linalg.lstsq(X, prof[:, g].T, rcond=None)[0]; mseg = np.hypot(c[1], c[2]) / c[0]
    o = {'slots': g, 'k_rad_px': kk.round(5).tolist(), 'pitch_nm': float(2 * np.pi / np.hypot(*kk) * 65),
         'angle_deg': float(np.rad2deg(np.arctan2(kk[1], kk[0])) % 180), 'slot_phases_deg': ph, 'steps_deg': steps,
         'm_dft_mean': float(np.mean([slots[s]['m_dft'] for s in g])), 'm_per_segment_median': float(np.median(mseg)),
         'relint': float(prof[:, g].mean())}
    res['orientations'].append(o); print('orientation', o, flush=True)
json.dump(res, open(OUT + '.json', 'w'), indent=1, default=float)
