"""Drift correction (DME v2 via analysis/baselines/dme_drift.py, run as a subprocess; DME imported read-only) and
renders of the localization subsets; NPC ring analysis. work = the 4Pi work folder (inputs <work>/smlm/p1, outputs
<work>/smlm/drift).
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
import os, sys, glob, subprocess, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIRST = {'MT': 5, 'NPC': 10}   # subset start file: global frame = (file - first) * 18000 + 6 w + 3


def load_locs(cell, work, kind='p2', files=None):
    P1 = f'{work}/smlm/p1'
    fs = sorted(glob.glob(f'{P1}/{kind}_{cell}_*.npz'))
    if files: fs = [f for f in fs if int(f[-7:-4]) in files]
    rows = []
    for k, f in enumerate(fs):
        D = np.load(f, allow_pickle=True)
        fi = int(f[-7:-4])
        r = dict(fi=np.full(len(D['w']), fi), k=np.full(len(D['w']), int(fi) - FIRST[cell]), **{n: D[n] for n in D.files if D[n].ndim >= 1 and len(D[n]) == len(D['w'])})
        if 'aidx' in D.files:     # amplitude fits exist for a subset only: expand to full length (NaN elsewhere)
            for n_ in ('A', 'sA', 'B', 'achi2'):
                full = np.full((len(D['w']),) + D[n_].shape[1:], np.nan, np.float32); full[D['aidx']] = D[n_]; r[n_] = full
        rows.append(r)
    out = {n: np.concatenate([r[n] for r in rows]) for n in rows[0]}
    # global frame index: consecutive files are consecutive 18000-frame blocks (the 0.7-17 s gaps between files are ignored)
    out['frame'] = out['k'] * 18000 + out['w'] * 6 + 3
    return out


def run_dme(x, y, frame, sx, sy, F, tag, work, gpu='0'):
    OUT = f'{work}/smlm'
    fn = f'{OUT}/drift/{tag}_locs.npz'; os.makedirs(f'{OUT}/drift', exist_ok=True)
    np.savez(fn, frame=frame - 3, x=x, y=y, sx=sx, sy=sy)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    r = subprocess.run([sys.executable, f'{REPO}/analysis/baselines/dme_drift.py', fn, f'{OUT}/drift/{tag}_drift.npy', str(F)],
                       env=env, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    return np.load(f'{OUT}/drift/{tag}_drift.npy')


def render(x, y, px_nm, sr_nm=5.0, sigma_nm=None, s=None, bounds=None):
    """histogram render (sr_nm per output pixel) with optional Gaussian blur of sigma_nm"""
    from scipy.ndimage import gaussian_filter
    if bounds is None: bounds = (x.min(), x.max(), y.min(), y.max())
    f = px_nm / sr_nm
    W = int((bounds[1] - bounds[0]) * f) + 1; H = int((bounds[3] - bounds[2]) * f) + 1
    img, _, _ = np.histogram2d((y - bounds[2]) * f, (x - bounds[0]) * f, bins=(H, W), range=((0, H), (0, W)))
    if sigma_nm: img = gaussian_filter(img, sigma_nm / sr_nm)
    return img
