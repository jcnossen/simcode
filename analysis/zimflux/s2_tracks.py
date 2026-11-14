"""S2 input: isolated emitter track samples from the classical astigmatic localizations of the jan13 movie, in the format
of simcode.insitu.extract_tracks (rois [B,T,R,R], mask, frames, corner (x, y), xy_init (x, y in the ROI), photons_smlm).
- Link: localizations of consecutive cycles within LINK px (one emitter ON); tracks of >= MINC cycles.
- Isolation: no other localization within ISO px of the track centre in the track's cycles +- 1.
- Samples: up to T = 12 frames (4 cycles) from the first cycle of the track, R = 16 px ROI at the track centre.
- Pixel values: the per-pixel camera calibration (s1 camera.npz) is applied and re-expressed as ADU of a uniform camera
  (gain G0 = 0.338 e-/ADU, offset 99.3 ADU) because fit_insitu2 / psf_heldout take one scalar gain and offset.
usage: python s2_tracks.py [locs subfolder=f225/] -> out/zimflux/s2/tracks_jan13.npz"""
import sys, os, numpy as np, tifffile
from scipy.spatial import cKDTree
sub = sys.argv[1] if len(sys.argv) > 1 else 'f225/'
W = '/home/jc/simcode2/out/zimflux/s1/'; OUT = '/home/jc/simcode2/out/zimflux/s2/'; os.makedirs(OUT, exist_ok=True)
LINK, MINC, ISO, T, R, G0, OFF0 = 0.75, 2, 10.0, 12, 16, 0.338, 99.3
A = np.load(W + sub + 'astig.npz'); cam = np.load(W + 'camera.npz')
th = A['th']; ok = (np.abs(th[:, :2]) < 6).all(1) & (A['dev'] < 3) & (th[:, 2] > -0.075) & (th[:, 2] < 0.445)
x = A['corner'][:, 1] + 7.5 + th[:, 0]; y = A['corner'][:, 0] + 7.5 + th[:, 1]; cy = A['cycle']
allxy = np.c_[x, y]; allc = cy                       # all detections (for isolation)
gi = np.nonzero(ok)[0]; gi = gi[np.argsort(cy[gi], kind='stable')]
nc = int(cy.max()) + 1
st = np.searchsorted(cy[gi], np.arange(nc + 2))
order_all = np.argsort(allc, kind='stable'); sta = np.searchsorted(allc[order_all], np.arange(nc + 2))
trees = {}
def tree_all(c):
    if c not in trees:
        ii = order_all[sta[c]:sta[c + 1]]; trees[c] = (cKDTree(allxy[ii]), ii) if len(ii) else None
    return trees[c]
nxt = -np.ones(len(gi), int)
for c in range(nc - 1):
    a0, a1, b1 = st[c], st[c + 1], st[c + 2]
    if a1 == a0 or b1 == a1: continue
    d, j = cKDTree(allxy[gi[a1:b1]]).query(allxy[gi[a0:a1]], distance_upper_bound=LINK)
    k = np.isfinite(d); nxt[np.arange(a0, a1)[k]] = a1 + j[k]
hasp = np.zeros(len(gi), bool); hasp[nxt[nxt >= 0]] = True
tracks = []
for s in np.nonzero(~hasp)[0]:
    t = [s]
    while nxt[t[-1]] >= 0: t.append(nxt[t[-1]])
    if len(t) >= MINC: tracks.append(np.array(t))
keep = []
for t in tracks:
    c0 = allxy[gi[t]].mean(0); mine = set(gi[t].tolist()); good = True
    for c in range(max(cy[gi[t[0]]] - 1, 0), min(cy[gi[t[-1]]] + 1, nc - 1) + 1):
        tr = tree_all(c)
        if tr is None: continue
        if any(tr[1][k] not in mine for k in tr[0].query_ball_point(c0, ISO)): good = False; break
    if good: keep.append(t)
print(len(tracks), 'tracks >=', MINC, 'cycles;', len(keep), 'isolated', flush=True)
H = Wd = 256
samples = []
for t in keep:
    c0 = allxy[gi[t]].mean(0); cx0, cy0 = int(round(c0[0] - 7.5)), int(round(c0[1] - 7.5))
    if cx0 < 0 or cy0 < 0 or cx0 + R > Wd or cy0 + R > H: continue
    c_first, c_last = cy[gi[t[0]]], cy[gi[t[-1]]]
    for s0 in range(c_first, c_last + 1, T // 3):
        n = min(T // 3, c_last + 1 - s0)
        if n < MINC: continue
        samples.append((3 * s0, 3 * n, cx0, cy0, c0[0] - cx0, c0[1] - cy0, float(A['th'][gi[t], 3].mean())))
B = len(samples); print(B, 'samples', flush=True)
D = '/mnt/data/dmdsf/simcode2_out/zimflux/surfdrive/Zimflux/3Dnanorulers/'
mm = [tifffile.memmap(D + f) for f in ('nanorulers_simflux_10ms_jan13_MMStack_Default.ome.tif',
                                       'nanorulers_simflux_10ms_jan13_MMStack_Default_1.ome.tif')]
def frame(fr): return mm[fr // 20000][fr % 20000]
rois = np.zeros((B, T, R, R), np.float32); mask = np.zeros((B, T), bool); frames = -np.ones((B, T), np.int64)
off, gain = cam['offset'], cam['gain']
for i in np.argsort([s[0] for s in samples]):
    s0, n, cx0, cy0 = samples[i][:4]
    for k in range(n):
        a = frame(s0 + k)[cy0:cy0 + R, cx0:cx0 + R].astype(np.float32)
        ph = (a - off[cy0:cy0 + R, cx0:cx0 + R]) * gain[cy0:cy0 + R, cx0:cx0 + R]
        rois[i, k] = ph / G0 + OFF0; mask[i, k] = True; frames[i, k] = s0 + k
smp = np.array([s[2:] for s in samples], np.float64)
np.savez(OUT + 'tracks_jan13.npz', rois=rois, mask=mask, frames=frames, corner=smp[:, :2].astype(int), xy_init=smp[:, 2:4],
         photons_smlm=smp[:, 4], gain_equiv=G0, offset_equiv=OFF0)
print('saved', B)
