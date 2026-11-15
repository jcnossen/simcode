"""Extract isolated emitter tracks (raw ADU ROIs over consecutive frames) from a movie + SMLM localizations."""
import argparse
import numpy as np, h5py, tifffile
from scipy.spatial import cKDTree
from simcode.calibration import MOVIE_05, SMLM_05, data, out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--movie', default=MOVIE_05)
    ap.add_argument('--locs', default=SMLM_05)
    ap.add_argument('--maxframe', type=int, default=30000)
    ap.add_argument('--roi', type=int, default=11)
    ap.add_argument('--T', type=int, default=12)            # frames per sample (tracks are cut/padded to this)
    ap.add_argument('--minlen', type=int, default=6)
    ap.add_argument('--isolation', type=float, default=7.0)  # px: no other loc within this radius in the track frames +-1
    ap.add_argument('--link', type=float, default=0.8)
    ap.add_argument('--out', default=out('tracks_0.5nM.npz'))
    a = ap.parse_args()
    a.movie, a.locs = data(a.movie), data(a.locs)

    L = h5py.File(a.locs)['locs'][:]
    L = L[L['frame'] < a.maxframe]
    f = L['frame'].astype(np.int64); xy = np.stack([L['x'], L['y']], 1).astype(np.float64)
    o = np.argsort(f, kind='stable'); f, xy, L = f[o], xy[o], L[o]
    nf = f.max() + 1
    starts = np.searchsorted(f, np.arange(nf + 2))
    trees = {}
    def tree(fr):
        if fr not in trees:
            trees[fr] = cKDTree(xy[starts[fr]:starts[fr + 1]]) if starts[fr + 1] > starts[fr] else None
        return trees[fr]

    # link consecutive frames (allow 1-frame gap: modulation can push an emitter below detection)
    nxt = -np.ones(len(f), int)
    for fr in range(nf - 1):
        a_ = slice(starts[fr], starts[fr + 1])
        if a_.stop == a_.start: continue
        for gap in (1, 2):
            t = tree(fr + gap) if fr + gap < nf else None
            if t is None: continue
            need = np.arange(a_.start, a_.stop)[nxt[a_] < 0]
            if len(need) == 0: break
            d, j = t.query(xy[need], distance_upper_bound=a.link)
            ok = np.isfinite(d)
            nxt[need[ok]] = j[ok] + starts[fr + gap]
    has_prev = np.zeros(len(f), bool); has_prev[nxt[nxt >= 0]] = True

    tracks = []
    for s in np.nonzero(~has_prev)[0]:
        idx = [s]
        while nxt[idx[-1]] >= 0: idx.append(nxt[idx[-1]])
        if f[idx[-1]] - f[idx[0]] + 1 >= a.minlen:
            tracks.append(np.array(idx))
    print(f'{len(tracks)} tracks spanning >= {a.minlen} frames')

    # isolation check
    keep = []
    for t in tracks:
        c = xy[t].mean(0); f0, f1 = f[t[0]] - 1, f[t[-1]] + 1
        ok = True
        for fr in range(max(f0, 0), min(f1, nf - 1) + 1):
            tr = tree(fr)
            if tr is None: continue
            n = tr.query_ball_point(c, a.isolation)
            others = [k + starts[fr] for k in n if (k + starts[fr]) not in set(t)]
            if others: ok = False; break
        if ok: keep.append(t)
    print(f'{len(keep)} isolated tracks')

    # cut into samples of T frames (frames contiguous from first loc, includes frames where emitter is dim/missed)
    tif = tifffile.TiffFile(a.movie)
    H, W = tif.pages[0].shape
    R = a.roi
    samples = []
    for t in keep:
        c = xy[t].mean(0)
        cx, cy = int(round(c[0])) - R // 2, int(round(c[1])) - R // 2
        if cx < 0 or cy < 0 or cx + R > W or cy + R > H: continue
        f0, f1 = f[t[0]], f[t[-1]]
        for s0 in range(f0, f1 + 1, a.T):
            n = min(a.T, f1 + 1 - s0)
            if n < a.minlen: continue
            samples.append((s0, n, cx, cy, c[0] - cx, c[1] - cy, L['photons'][t].mean()))
    print(f'{len(samples)} samples of up to {a.T} frames')

    frames_needed = sorted({s0 + k for s0, n, *_ in samples for k in range(n)})
    B = len(samples)
    rois = np.zeros((B, a.T, R, R), np.uint16); mask = np.zeros((B, a.T), bool); frames = -np.ones((B, a.T), np.int64)
    fmap = {}
    for fr in frames_needed:
        fmap[fr] = tif.pages[fr].asarray()
    for i, (s0, n, cx, cy, *_ ) in enumerate(samples):
        for k in range(n):
            rois[i, k] = fmap[s0 + k][cy:cy + R, cx:cx + R]
            mask[i, k] = True; frames[i, k] = s0 + k
    smp = np.array([s[2:] for s in samples], np.float64)
    np.savez_compressed(a.out, rois=rois, mask=mask, frames=frames, corner=smp[:, :2].astype(int),
                        xy_init=smp[:, 2:4], photons_smlm=smp[:, 4])
    print('saved', a.out, rois.shape)
