"""Shared loaders for the critic-34 real-data checks (development critic34/common.py; CPU only, no torch): the full-movie
localization sets (ours: <out>/full_*_sc5nb_bright_real; SIMCODE (a)/(b): <out>/drift*/simcode_real/<name>.npz; SIMCODE published:
the sfhd-ndi.hdf5 next to the movie under the data root) and each set's own DME v2 drift (<out>/drift*/dmev2/<name>.npz).
Mirrors the development dme_v2_full05/25.load_set + own-DME-v2 correction."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_metrics as SM
from simcode2 import config
PX = 108.33
_O = config.root('out')
CONC = {
    '0.5': dict(F=90000, D=os.path.join(_O, 'drift') + '/',
                RES=config.resolve('dnapaint/0.5nM/cover1/1/results/simflux_5.5ms_500mW', 'data') + '/',
                OURS=os.path.join(_O, 'full_0.5nM_sc5nb_bright_real')),
    '2.5': dict(F=180000, D=os.path.join(_O, 'drift25') + '/',
                RES=config.resolve('dnapaint/2.5nM/2/results/cover1_600mW_5.5ms_cf660', 'data') + '/',
                OURS=os.path.join(_O, 'full_2.5nM_m2_sc5nb_bright_real')),
}
_OURS = {}


def ours_all(c):
    if c not in _OURS: _OURS[c] = SM.load_ours(CONC[c]['OURS'], 'est')
    return _OURS[c]


def load_set(c, name):
    """returns pos [N,2] px, time = window start + 3, crlb [N,2], extra dict"""
    C = CONC[c]
    if name.startswith('ours'):
        A = ours_all(c)
        m = (A['wstart'] % (6 if name == 'ours_disj' else 3) == 0) & (A['p'] >= 0.63)
        return np.c_[A['x_hy0'][m], A['y_hy0'][m]], (A['wstart'][m] + 3).astype(np.int64), np.c_[A['sx_cal'][m], A['sy_cal'][m]], dict(phot=A['phot'][m])
    if name == 'sc_paper_h5':
        import h5py
        L = h5py.File(C['RES'] + 'sfhd-ndi.hdf5', 'r')['locs'][:]
        return np.c_[L['x'], L['y']].astype(np.float64), L['frame'].astype(np.int64) + 3, np.c_[L['lpx'], L['lpy']], dict(phot=L['photons'])
    S = np.load(C['D'] + f'simcode_real/{name}.npz')
    return np.c_[S['x'], S['y']].astype(np.float64), S['frame'].astype(np.int64) + 3, np.c_[S['sx'], S['sy']], dict(phot=S['photons'])


def drift(c, name):
    return np.load(CONC[c]['D'] + f'dmev2/{name}.npz', allow_pickle=True)['R'].item()['drift_full']


def corrected(c, name, crop=True, drift_of=None):
    """own-DME-v2 corrected x, y (px), time, plus raw pos and index into load_set rows"""
    pos, fr, crlb, ex = load_set(c, name)
    F = CONC[c]['F']; d = drift(c, drift_of or name); f = np.clip(fr, 0, F - 1)
    x, y = pos[:, 0] - d[f, 0], pos[:, 1] - d[f, 1]
    idx = np.arange(len(x))
    if crop:
        m = (x >= 5) & (x < 115) & (y >= 5) & (y < 115); x, y, f, idx = x[m], y[m], f[m], idx[m]
    return x, y, f, idx


def nnprox(xq, yq, fq, xr, yr, fr, n=150000, seed=0, exclude_self=False):
    """consecutive-window NN precision proxy: for each query, nearest reference loc in frames f+6..f+18 (later windows of the
    same binding event), xy < 50 nm. Returns median NN (nm), fraction with a neighbour, Rayleigh+linear-bg ML sigma (nm/axis), w"""
    from scipy.spatial import cKDTree
    r = 50 / PX; s = r / 6.0
    i = np.random.default_rng(seed).choice(len(xq), min(n, len(xq)), replace=False)
    tr = cKDTree(np.c_[xr, yr, fr * s])
    L = tr.query_ball_point(np.c_[xq[i], yq[i], (fq[i] + 12) * s], r, p=np.inf)
    d = np.full(len(i), np.inf)
    for k, js in enumerate(L):
        if js:
            js = np.asarray(js); d[k] = np.hypot(xr[js] - xq[i[k]], yr[js] - yq[i[k]]).min()
    d = d[np.isfinite(d)] * PX; d = d[d < 50]
    best = None
    for sig in np.arange(3, 25, 0.25):
        ss = np.sqrt(2) * sig
        ray = d / ss ** 2 * np.exp(-d ** 2 / (2 * ss ** 2)) / (1 - np.exp(-50 ** 2 / (2 * ss ** 2)))
        lin = 2 * d / 50 ** 2
        for w in np.arange(0.3, 1.0, 0.02):
            ll = np.log(w * ray + (1 - w) * lin + 1e-300).sum()
            if best is None or ll > best[0]: best = (ll, sig, w)
    return np.median(d), len(d) / len(i), best[1], best[2]
