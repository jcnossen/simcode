"""SMLM pass 1 on one movie file: registered 4-channel sum of each 6-frame pattern cycle (frames 6w .. 6w+5 of the
file), detection, astigmatic Gaussian MLE (x, y, N, bg, sx, sy) with CRLBs, plus the raw material for every later step:
  roi_sum [n,15,15] float32   the registered 4-channel 6-frame sum around the detection (e), centre pixel (cy, cx)
  roi_ch  [n,6,4,11,11] f16   per-frame, per-channel UNWARPED images (e) around the mapped position, corners (ix, iy)[n,4]
  fmean   [F,4]               per-frame mean signal of each channel over the illuminated area (e/px)
Read noise variance in the 6-frame 4-channel sum: 24 x rn^2 (rn 1.14 e); the bilinear warp of channels 1-3 lowers it
by a position-dependent factor (ignored: v = 24 x 1.3 e^2 is used as the noise floor).
usage: python process.py <out_dir> <nworkers> CELL:fi[:nframes] ...  (one sequential reader process + workers)
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
import os, sys, time
import numpy as np
import torch
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc import Dcimg, movie_files, crops, file_index
from register import PolyMap
from quickloc import Warp
from mle import GaussAstig, lm_fit

from simcode.setups.fourpi import acquisition; GAIN = acquisition('MT')['camera']['gain_e_per_adu']   # 4Pi camera gain (the same for MT and NPC)
RN2 = 1.3
PS, PC = 15, 11
from simcode.setups.fourpi import acquisition; XLIM = tuple(acquisition('MT')['fourpi']['xlim'])   # illuminated columns of a 4Pi channel crop
CHUNK = 600


def detect_chunk(E, a0, warp, pm):
    """E [CHUNK,4,H,W] e. Returns dict of per-detection arrays for this chunk + fmean [CHUNK,4]"""
    hs, hc = PS // 2, PC // 2
    acc = {k: [] for k in ('w', 'cx', 'cy', 'roi_sum', 'roi_ch', 'corner', 'dog', 'nn')}
    nT = E.shape[0]
    fmean = E[:, :, :, XLIM[0]:XLIM[1]].mean((2, 3))
    S6 = E.reshape(nT // 6, 6, 4, 200, 360).sum(1)
    for wl in range(nT // 6):
        w = a0 // 6 + wl
        ssum = S6[wl, 0] + sum(warp(S6[wl, c], c) for c in (1, 2, 3))
        fl = ndi.gaussian_filter(ssum, 1.5) - ndi.gaussian_filter(ssum, 5)
        sd = 1.4826 * np.median(np.abs(fl[:, XLIM[0]:XLIM[1]] - np.median(fl[:, XLIM[0]:XLIM[1]])))
        pk = (fl == ndi.maximum_filter(fl, 7)) & (fl > max(8 * sd, 12.0))
        pk[:, :XLIM[0]] = False; pk[:, XLIM[1]:] = False
        pk[:hs + 1] = False; pk[-hs - 1:] = False
        ys, xs = np.nonzero(pk)
        if len(ys) == 0: continue
        dd = np.hypot(ys[:, None] - ys[None], xs[:, None] - xs[None]); np.fill_diagonal(dd, 99)
        for k, (y, x) in enumerate(zip(ys, xs)):
            corners = []
            for c in range(4):
                xc, yc = pm.inv(c, float(x), float(y))
                ix, iy = int(round(float(xc))) - hc, int(round(float(yc))) - hc
                if not (0 <= iy and iy + PC <= 200 and 0 <= ix and ix + PC <= 360): break
                corners.append((ix, iy))
            if len(corners) < 4: continue
            acc['w'].append(w); acc['cx'].append(x); acc['cy'].append(y)
            acc['roi_sum'].append(ssum[y - hs:y + hs + 1, x - hs:x + hs + 1].copy())
            acc['roi_ch'].append(np.stack([E[6 * wl:6 * wl + 6, c, iy:iy + PC, ix:ix + PC] for c, (ix, iy) in enumerate(corners)], 1).astype(np.float16))
            acc['corner'].append(corners); acc['dog'].append(fl[y, x]); acc['nn'].append(dd[k].min())
    out = {k: (np.asarray(v) if len(v) else None) for k, v in acc.items()}
    out['fmean'] = fmean
    return out


def _reader(jobs, q, nworkers):
    cams = {}
    for cell, fi, n in jobs:
        if cell not in cams:
            cams[cell] = np.load(f'{P4}/smlm/camera/camera_{cell}.npz')['off']
        fn = [f for f in movie_files(cell) if file_index(f) == fi][0]
        d = Dcimg(fn); n = n or d.nframes
        for a0 in range(0, n, CHUNK):
            E = ((crops(d.raw(a0, CHUNK)) - cams[cell]) * GAIN).astype(np.float32)
            q.put((cell, fi, a0, n, E))
    for _ in range(nworkers): q.put(None)


def _worker(q, r):
    torch.set_num_threads(1)
    warps = {}
    while True:
        it = q.get()
        if it is None: break
        cell, fi, a0, n, E = it
        if cell not in warps:
            pm = PolyMap.load(f'{P4}/smlm/reg/reg_insitu_{cell}.json')
            warps[cell] = (Warp(lambda c, xx, yy, pm=pm: pm.inv(c, xx, yy)), pm)
        r.put((cell, fi, a0, n, detect_chunk(E, a0, *warps[cell])))


def finish(cell, fi, n, parts, out_dir):
    t0 = time.time()
    parts = [parts[a] for a in sorted(parts)]
    fmean = np.concatenate([p['fmean'] for p in parts])
    R = {k: np.concatenate([p[k] for p in parts if p[k] is not None]) for k in ('w', 'cx', 'cy', 'roi_sum', 'roi_ch', 'corner', 'dog', 'nn')}
    print(f'{cell} {fi}: {len(R["w"])} detections, fitting', flush=True)
    m = GaussAstig(PS)
    th_all, cr_all, ll_all, chi_all, dev_all = [], [], [], [], []
    for s in range(0, len(R['w']), 20000):
        dsum = torch.as_tensor(R['roi_sum'][s:s + 20000], dtype=torch.float64)
        th, ll, cr, chi2, dev, mu = lm_fit(m, dsum, m.init(dsum), 24 * RN2, iters=60)
        th_all.append(th.numpy()); cr_all.append(cr.numpy()); ll_all.append(ll.numpy()); chi_all.append(chi2.numpy()); dev_all.append(dev.numpy())
    th = np.concatenate(th_all); cr = np.concatenate(cr_all)
    np.savez(f'{out_dir}/p1_{cell}_{fi:03d}.npz', **R, gfit=th.astype(np.float32), gcrlb=cr.astype(np.float32),
             gll=np.concatenate(ll_all), gchi2=np.concatenate(chi_all), gdev=np.concatenate(dev_all), fmean=fmean,
             x=(R['cx'] + th[:, 0]).astype(np.float32), y=(R['cy'] + th[:, 1]).astype(np.float32), nframes=n, file_index=fi, cell=cell)
    print(f'{cell} {fi}: saved {len(th)} detections + Gaussian fits ({time.time() - t0:.0f}s fit)', flush=True)


if __name__ == '__main__':
    import multiprocessing as mp
    # usage: python process.py <out_dir> <nworkers> CELL:fi[:nframes] ...
    out_dir, nw = sys.argv[1], int(sys.argv[2]); os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for s in sys.argv[3:]:
        p = s.split(':'); jobs.append((p[0], int(p[1]), int(p[2]) if len(p) > 2 else None))
    torch.set_num_threads(4)
    q, r = mp.Queue(maxsize=3), mp.Queue()
    rd = mp.Process(target=_reader, args=(jobs, q, nw)); rd.start()
    ws = [mp.Process(target=_worker, args=(q, r)) for _ in range(nw)]
    for w in ws: w.start()
    got = {}; t = time.time(); done = set()
    while len(done) < len(jobs):
        cell, fi, a0, n, part = r.get()
        got.setdefault((cell, fi), {})[a0] = part
        if len(got[(cell, fi)]) % 5 == 0:
            print(f'{cell} {fi}: {len(got[(cell, fi)])}/{n // CHUNK} chunks, {time.time() - t:.0f}s', flush=True)
        if len(got[(cell, fi)]) == n // CHUNK:
            finish(cell, fi, n, got.pop((cell, fi)), out_dir); done.add((cell, fi))
    for w in ws: w.join()
    rd.join()
