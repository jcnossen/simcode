"""Real 20 nM movie 2 (120000 frames; development real20/lib20.py): shared loaders for the fast-pipeline run vs the baselines (CPU; DME v2 needs GPU).
Coordinates in sensor px. 'f' = window START (6-frame windows) for every window method; decode (per frame) f = frame.
DME / FRC time 't' = window start + 3 (window methods), frame (decode per frame).
Sets
  ours  <pass><sel>   pass p1 = pass-1 one-pass final (full_20nM_m2_fast pass2_est, calib bright_real),
                      ac = pass 1 re-solved with the automatic (s, c, c_a) (full_20nM_m2_fast_autoc pass2_est),
                      p2 = pass 2 p2L (full_20nM_m2_fast pass2_est2r = est2 with the real sigma rescale of p2real);
                      sel G = global p >= 0.63, P = phase-calibrated threshold (p2prior/lib.phase_thr, equal N per photon band,
                      top p per 6x6 sub-pixel cell). Rows = all stride-3 window starts; G6 subset = wstart % 6 == 0.
  sc_b                SIMCODE L64_5 / checkpoint_3, paper camera, sign-fixed NDI (<out>/real20/simcode_real/sc_b.npz)
  run dirs: <out>/full_20nM_m2_fast(_autoc); own DME v2 drifts: <out>/real20/dmev2/<set>.npz
  smlm, sf, decode, decode_summed   published hdf5 (results/cover1_500mW_5.5ms_cf660), read-only
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_metrics as SM
from simcode import config
H = config.root('out') + '/'
D20 = H + 'real20/'
PX = 108.33
F = 120000
RES = config.resolve('dnapaint/20nM/2/results/cover1_500mW_5.5ms_cf660', 'data') + '/'   # published hdf5 (data root, read-only)
RUN = H + 'full_20nM_m2_fast'
RUN_AC = H + 'full_20nM_m2_fast_autoc'
PASS = {'p1': (RUN, 'est'), 'ac': (RUN_AC, 'est'), 'p2': (RUN, 'est2r')}
LBL = {'smlm': 'SMLM', 'sf': 'SF', 'decode': 'DECODE', 'decode_summed': 'DECODE (summed)', 'sc_b': 'SIMCODE',
       'p1': 'ours pass 1', 'ac': 'ours pass 1 autoc', 'p2': 'ours pass 2'}
PMIN = 0.63
_c = {}


def phase_thr(p, x, y, ph, thr=PMIN):
    from simcode.strict import phase_thr as _pt      # = the development p2prior/lib.phase_thr
    return _pt(p, x, y, ph, thr)


def pixu(x, y, nb=20):
    f = np.concatenate([np.mod(x, 1), np.mod(y, 1)]); f = f[np.isfinite(f)]
    if len(f) < 200: return np.nan
    h, _ = np.histogram(f, nb, (0, 1)); return float(h.min() / h.mean())


def ours_raw(ps):
    if ps not in _c:
        d, tag = PASS[ps]; _c[ps] = SM.load_ours(d, tag)
    return _c[ps]


def ours_mask(ps, sel):
    key = ('m', ps, sel)
    if key not in _c:
        A = ours_raw(ps); ok = np.isfinite(A['x_hy0']) & np.isfinite(A['y_hy0'])
        if sel == 'G': m = A['p'] >= PMIN
        else: m = phase_thr(A['p'], A['x_hy0'], A['y_hy0'], A['phot'])
        _c[key] = m & ok
    return _c[key]


def load(name):
    """-> dict f (window start / frame), t (DME time), xr, yr (raw px), sx, sy (sigma px), ph, plus extra columns"""
    if name in _c: return _c[name]
    if name[:2] in PASS and name[2] in 'GP':
        A = ours_raw(name[:2]); m = ours_mask(name[:2], name[2])
        D = dict(f=A['wstart'][m].astype(np.int64), xr=A['x_hy0'][m].astype(np.float64), yr=A['y_hy0'][m].astype(np.float64),
                 sx=A['sx_cal'][m].astype(np.float64), sy=A['sy_cal'][m].astype(np.float64), ph=A['phot'][m].astype(np.float64),
                 p=A['p'][m], xpr=A['x_prior'][m].astype(np.float64), ypr=A['y_prior'][m].astype(np.float64))
        D['t'] = D['f'] + 3
    elif name == 'sc_b':
        S = np.load(D20 + 'simcode_real/sc_b.npz')
        D = dict(f=S['frame'].astype(np.int64), xr=S['x'].astype(np.float64), yr=S['y'].astype(np.float64), sx=S['sx'].astype(np.float64),
                 sy=S['sy'].astype(np.float64), ph=S['photons'].astype(np.float64), xnr=S['net_x'].astype(np.float64), ynr=S['net_y'].astype(np.float64))
        D['t'] = D['f'] + 3
    else:
        import h5py
        L = h5py.File(RES + {'decode_summed': 'decode-summed'}.get(name, name) + '.hdf5', 'r')['locs'][:]
        D = dict(f=L['frame'].astype(np.int64), xr=L['x'].astype(np.float64), yr=L['y'].astype(np.float64), sx=L['lpx'].astype(np.float64),
                 sy=L['lpy'].astype(np.float64), ph=L['photons'].astype(np.float64))
        if name.startswith('decode'):
            # DECODE hdf5 coordinates are +1 px vs smlm / sf / ours (image cross-correlation peak (dy, dx) = (-1.0, -0.9) px,
            # pair medians -0.99 / -0.92..-0.97 px: coord_check*.txt) -> exactly -1 px (index convention; keeps frac(x))
            D['xr'] -= 1.0; D['yr'] -= 1.0
        if name == 'decode_summed':
            # decode-summed 'frame' (all % 6 == 0): exact-start pair counts vs smlm peak at smlm start = frame + 6 (coord_check3.txt)
            D['f'] = D['f'] + 6
        D['t'] = D['f'] + (0 if name == 'decode' else 3)
    _c[name] = D
    return D


def dme_name(name):
    """DME v2 is run once per method on its G6 (disjoint) rows for ours (as for the 0.5 / 2.5 nM runs); all its selections use it"""
    if name[:2] in PASS: return name[:3] + '_disj'
    return name


def drift(name):
    return np.load(D20 + f'dmev2/{dme_name(name)}.npz', allow_pickle=True)['R'].item()['drift_full']


def corrected(name, crop=True):
    D = dict(load(name)); d = drift(name); ti = np.clip(D['t'], 0, F - 1)
    D['x'] = D['xr'] - d[ti, 0]; D['y'] = D['yr'] - d[ti, 1]; D['sig'] = 0.5 * (D['sx'] + D['sy'])
    if crop:
        k = np.isfinite(D['x']) & (D['x'] >= 5) & (D['x'] < 115) & (D['y'] >= 5) & (D['y'] < 115)
        n = len(k); D = {kk: (v[k] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v) for kk, v in D.items()}
    D['n'] = len(D['x'])
    return D


def sub(D, m):
    n = D['n']; o = {k: (v[m] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v) for k, v in D.items()}
    o['n'] = int(m.sum()) if m.dtype == bool else len(m); return o


def frc(x, y, t):
    return SM.frc_split(x, y, t)
