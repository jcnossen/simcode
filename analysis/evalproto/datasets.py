"""Dataset loader of the evaluation-protocol scripts (covpool.py, nnlag.py, blockphase.py). A dataset is a json spec:

  {"name": "real 0.5 nM", "mode": "real", "grid": 3, "n_frames": 90000, "reference": "SIMCODE (b)",
   "sets": {"ours p1":     {"kind": "ours", "dir": "<simcode output dir>", "tag": "est",  "drift": "<drift>"},
            "ours p2L":    {"kind": "ours", "dir": "<simcode output dir>", "tag": "est2", "drift": "<drift>"},
            "SIMCODE (b)": {"kind": "simcode", "npz": "<sc_b.npz>", "drift": "<drift>"}}}
  sim mode: "mode": "sim", "gt": "<sim_gt.npz>" (simreal/sim_movie_r.py output), sets as above (kind "simcode" reads the NDI
  rows ndi_x / ndi_y / ndi_sx / ndi_sy of a SIMCODE run npz), no drift.
  kind "hdf5" (realproto.py baselines): {"kind": "hdf5", "file": "<smlm.hdf5 | sf.hdf5 | decode-summed.hdf5>", "drift": ...,
  "dxy": -1 (DECODE coordinates), "df": 6 (decode-summed frame -> window start)}; "baseline": true marks a set as a baseline.
Rows (as the development loaders): ours = detector p >= 0.63, f = window start, x = final - drift(f + 3), sig = mean of the
calibrated axes (nm); SIMCODE = its window rows (f = window start); 5-115 px crop after drift correction; grid G3 (window
starts % 3 == 0) or G6. Drift: DME v2 npz (R.item()['drift_full'], [F, 2] px) or a plain .npy [F, 2] px.
Sim mode adds the GT match (lib.match_fast on the prior / detection position, 1 px), per-method median offset removed (GT
400-2000 ph), e [n, 2] nm, ev (GT event), site, P (GT window photons)."""
import glob, json, os
import numpy as np
from simcode.evaluation import metrics as E

PMIN = 0.63


def _drift(fn):
    if not fn: return None
    d = np.load(fn, allow_pickle=True)
    if isinstance(d, np.lib.npyio.NpzFile):
        if 'R' in d.files: return d['R'].item()['drift_full']
        return d['drift']
    return d


def _ours_rows(d, tag):
    Q = {}
    for f in sorted(glob.glob(f'{d}/pass2_{tag}_*.npz')):
        z = np.load(f)
        for k in z.files:
            if z[k].ndim: Q.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in Q.items()}


def load_set(S, F=None, grid=3, crop=True):
    """one set spec -> dict f, x, y (drift-corrected px), xr, yr (raw), xd, yd (prior / net position, raw), sig (nm), sx, sy (px),
    phot"""
    if S['kind'] == 'ours':
        A = _ours_rows(S['dir'], S.get('tag', 'est'))
        m = (A['p'] >= S.get('pmin', PMIN)) & np.isfinite(A['x_hy0'])
        O = dict(f=A['wstart'][m].astype(np.int64), xr=A['x_hy0'][m].astype(np.float64), yr=A['y_hy0'][m].astype(np.float64),
                 xd=A['x_prior'][m].astype(np.float64), yd=A['y_prior'][m].astype(np.float64), sx=A['sx_cal'][m].astype(np.float64),
                 sy=A['sy_cal'][m].astype(np.float64), phot=A['phot'][m].astype(np.float64), p=A['p'][m])
    elif S['kind'] == 'simcode':
        R = np.load(S['npz'])
        fr = R['frame'].astype(np.int64)
        if 'ndi_x' in R.files and S.get('rows', 'ndi') == 'ndi':      # SIMCODE run npz with NDI rows (sims)
            rows = R['ndi_idx']
            O = dict(f=fr[rows], xr=R['ndi_x'].astype(np.float64), yr=R['ndi_y'].astype(np.float64), xd=R['x'][rows].astype(np.float64),
                     yd=R['y'][rows].astype(np.float64), sx=R['ndi_sx'].astype(np.float64), sy=R['ndi_sy'].astype(np.float64),
                     phot=R['ndi_photons'].astype(np.float64))
        else:                                                         # real-data sc_b.npz (final rows)
            O = dict(f=fr, xr=R['x'].astype(np.float64), yr=R['y'].astype(np.float64),
                     xd=(R['net_x'] if 'net_x' in R.files else R['x']).astype(np.float64),
                     yd=(R['net_y'] if 'net_y' in R.files else R['y']).astype(np.float64),
                     sx=R['sx'].astype(np.float64), sy=R['sy'].astype(np.float64), phot=R['photons'].astype(np.float64))
    elif S['kind'] == 'hdf5':                                         # published SMLM / SF / DECODE result hdf5 (picasso locs)
        import h5py
        L = h5py.File(S['file'], 'r')['locs'][:]
        dxy = float(S.get('dxy', 0.0))                                # DECODE hdf5: -1 px (index convention; real 20 nM coord check)
        O = dict(f=L['frame'].astype(np.int64) + int(S.get('df', 0)),  # decode-summed: window start = frame + 6
                 xr=L['x'].astype(np.float64) + dxy, yr=L['y'].astype(np.float64) + dxy, sx=L['lpx'].astype(np.float64),
                 sy=L['lpy'].astype(np.float64), phot=L['photons'].astype(np.float64))
        O['xd'], O['yd'] = O['xr'], O['yr']
    else:
        raise ValueError(S['kind'])
    d = _drift(S.get('drift'))
    if d is not None:
        fi = np.clip(O['f'] + 3, 0, len(d) - 1)
        O['x'] = O['xr'] - d[fi, 0]; O['y'] = O['yr'] - d[fi, 1]
    else:
        O['x'], O['y'] = O['xr'].copy(), O['yr'].copy()
    O['sig'] = 0.5 * (O['sx'] + O['sy']) * E.PX
    k = O['f'] % grid == 0
    if F is not None: k &= O['f'] <= F - 6
    if crop: k &= (O['x'] >= 5) & (O['x'] < 115) & (O['y'] >= 5) & (O['y'] < 115)
    return {a: b[k] for a, b in O.items()}


def load(spec, baselines=False):
    """-> (reference set, {name: set}, meta). Sets marked "baseline": true are skipped unless baselines=True."""
    D = json.load(open(spec)) if isinstance(spec, str) else spec
    grid = int(D.get('grid', 3))
    ref = D.get('reference', 'SIMCODE (b)')
    meta = dict(name=D.get('name', ''), mode=D.get('mode', 'real'), grid=grid, F=D.get('n_frames'))
    if meta['mode'] == 'sim':
        G = dict(np.load(D['gt'])); NF = int(G['nf']); meta['F'] = NF
    sets = {k: load_set(v, meta['F'] if meta['mode'] == 'sim' else None, grid) for k, v in D['sets'].items()
            if baselines or not v.get('baseline')}
    if meta['mode'] == 'sim':
        GW = E.gt_windows_fast(G, grid=3)
        site_of = G['site'] if 'site' in G else -np.ones(len(G['x']), np.int64)
        for nm, S_ in sets.items():
            gi = E.match_fast(GW, S_['f'], S_['xd'], S_['yd']); g = gi >= 0
            e = np.full((len(gi), 2), np.nan); e[g] = np.c_[S_['x'][g] - GW[gi[g], 2], S_['y'][g] - GW[gi[g], 3]]
            P = np.where(g, GW[np.maximum(gi, 0), 4], np.nan); sel = g & (P >= 400) & (P < 2000)
            off = np.nanmedian(e[sel], 0)
            S_['e'] = (e - off) * E.PX; S_['x'] = S_['x'] - off[0]; S_['y'] = S_['y'] - off[1]
            S_['ev'] = np.where(g, GW[np.maximum(gi, 0), 1], -1).astype(np.int64)
            S_['site'] = np.where(g, site_of[np.maximum(S_['ev'], 0)], -1)
            S_['P'] = P; S_['tgt'] = np.where(g, GW[np.maximum(gi, 0), 5], 0) > 0
        meta.update(G=G, GW=GW)
    S = sets.pop(ref)
    return S, sets, meta


def out_path(spec, prefix, out_dir=None):
    name = os.path.splitext(os.path.basename(spec))[0] if isinstance(spec, str) else 'dataset'
    from simcode import config
    d = out_dir or config.out_dir('evalproto')
    return os.path.join(d, f'{prefix}_{name}')
