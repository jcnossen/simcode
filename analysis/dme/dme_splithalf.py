"""DME v2 drift per localization set + the realcheck2 split-half held-out-reference error (the 'DME v2 split-half' numbers).  (0.5 nM), drift25/dme_v2_full25.py (2.5 nM),
p2real/analyze_p2.py stage dme (pass 2, 0.5 / 2.5 nM) and real20/dme20.py (20 nM): same row selections, same DME calls, same
checkerboard halves and bp / xref_err (verbatim ~/dme/gpu/loop/realcheck2.py, px -> nm with 108.33 nm), same npz layout
(R: drift_full [F, 2] px, chosen, cv, t_full, N, D {method: (half A, half B) nm}). DME v2 = v3e = learned_smoother3e of the DME
project, imported READ ONLY from $DME_PATH (default ~/dme; not part of simcode2), as analysis/baselines/dme_drift.py.

DME stage (GPU):  CUDA_VISIBLE_DEVICES=1 python analysis/dme/dme_splithalf.py run <spec.json> <set> [<set> ...]
  spec: {"n_frames": 90000, "out": "<dir>",                       -> <out>/<set>.npz (re-uses / completes an existing npz)
         "methods": ["v3e", "grid_fpb20", "grid_fpb10"],          halves to compute (dev 0.5 / 2.5 nM pass 1 + SIMCODE: all 8 of
                                                                  METHODS below; pass 2 and 20 nM: these three)
         "crlb_min_pos": false,                                   true = also drop rows with crlb <= 0 (dev 20 nM only)
         "sets": {"<set>": {"kind": "ours", "dir": "<simcode2 run dir>", "tag": "est", "grid": 6, "pmin": 0.63}, ...
                  "<set>": {"kind": "simcode", "npz": "<sc_b.npz>"},  (run_simcode_real.py ndi output, all rows)
                  "<set>": {"kind": "hdf5", "file": "<smlm.hdf5 | sfhd-ndi.hdf5 | ...>", "df": 0, "dxy": 0, "dt": 3}}}
  rows: ours = p >= pmin, finite positions, wstart % grid == 0 (6 = disjoint '_disj' sets, 3 = stride-3 '_ovl'), time = wstart
  + 3, crlb = (sx_cal, sy_cal) px; simcode / hdf5: all rows, time = frame + df + dt (dt 3 for window methods; DECODE per-frame
  dt 0), crlb = (sx, sy) / (lpx, lpy); every set: finite, crlb < 1 px, 0 <= time < F.
Report (CPU):     python analysis/dme/dme_splithalf.py report <set.npz> [...]
  per npz: DME v2 choice and the v3e split-half error per band (2-100 / 100-500 / 500-5000 / > 5000 frames) and rss < 5000
  frames for the references grid_fpb10 / grid_fpb20 (RESULTS quotes 'ref fpb10, rss < 5000')."""
import os, sys, json, time, contextlib, io
import numpy as np

PX = 108.33
BANDS = [(2, 100), (100, 500), (500, 5000), (5000, 10 ** 9)]
F = None


def checkerboard(pos, ntiles=8):          # = ~/dme/gpu/loop/realcheck.py checkerboard (diagnostic halves of realcheck2)
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    ts = (hi - lo).max() / ntiles
    t = np.floor((pos[:, :2] - lo) / ts).astype(np.int64)
    return ((t[:, 0] + t[:, 1]) % 2) == 0


def bp(x):                                # verbatim realcheck2.bp
    Fn = len(x)
    X = np.fft.rfft(x - x.mean(0), axis=0)
    per = Fn / np.maximum(np.arange(X.shape[0]), 1e-9)
    o = []
    for lo, hi in BANDS:
        m = (per >= lo) & (per < hi); m[0] = False
        o.append(np.sum(np.abs(X[m]) ** 2) * 2 / Fn ** 2 / x.shape[1])
    return np.array(o)


def xref_err(D, R):                       # verbatim realcheck2.xref_err
    NR2 = bp(D[R][0] - D[R][1]) / 2
    out = {}
    for m, (a, b) in D.items():
        if m == R:
            continue
        e2 = (bp(a - D[R][1]) + bp(b - D[R][0])) / 2 - NR2
        out[m] = np.sqrt(np.maximum(e2, 0))
    return out


def _dme():
    D = os.path.expanduser(os.environ.get('DME_PATH', '~/dme'))
    sys.path[:0] = [f'{D}/gpu', f'{D}/gpu/loop', f'{D}/gpu/loop/methods']
    from dme_grid import dme_grid_estimate
    from dme_pipeline import dme_estimate_gpu
    import learned_smoother3e as V3E
    rule = lambda N: int(np.clip(round(20000 * F / N), 10, 2000))
    grid = lambda fpb: (lambda p, f, c: dme_grid_estimate(p, f, c.mean(0), fpb(len(p)), estimatePrecision=False, numframes=F, verbose=False))
    M = {'v3e': lambda p, f, c: V3E.estimate(p, f, c, F),
         'grid_rule': grid(rule),
         'grid_fpb200': grid(lambda N: 200), 'grid_fpb50': grid(lambda N: 50),
         'grid_fpb20': grid(lambda N: 20), 'grid_fpb10': grid(lambda N: 10),
         'exact_fpb200': lambda p, f, c: dme_estimate_gpu(p, f, c, framesperbin=200, numframes=F, verbose=False, estimatePrecision=False),
         'exact_fpb50': lambda p, f, c: dme_estimate_gpu(p, f, c, framesperbin=50, numframes=F, verbose=False, estimatePrecision=False)}
    return V3E, M


def load_set(S):
    """-> pos [n, 2] px (raw), time [n], crlb [n, 2] px"""
    if S['kind'] == 'ours':
        import glob
        Q = {}
        for fn in sorted(glob.glob(f'{S["dir"]}/pass2_{S.get("tag", "est")}_*.npz')):
            d = np.load(fn)
            for k in d.files:
                if d[k].ndim: Q.setdefault(k, []).append(d[k])
        A = {k: np.concatenate(v) for k, v in Q.items()}
        m = (A['wstart'] % int(S.get('grid', 6)) == 0) & (A['p'] >= float(S.get('pmin', 0.63)))
        return np.c_[A['x_hy0'][m], A['y_hy0'][m]], (A['wstart'][m] + 3).astype(np.int64), np.c_[A['sx_cal'][m], A['sy_cal'][m]]
    if S['kind'] == 'simcode':
        R = np.load(S['npz'])
        return np.c_[R['x'], R['y']].astype(np.float64), (R['frame'].astype(np.int64) + 3), np.c_[R['sx'], R['sy']]
    if S['kind'] == 'hdf5':
        import h5py
        L = h5py.File(S['file'], 'r')['locs'][:]; dxy = float(S.get('dxy', 0.0))
        return (np.c_[L['x'] + dxy, L['y'] + dxy].astype(np.float64), L['frame'].astype(np.int64) + int(S.get('df', 0)) + int(S.get('dt', 3)),
                np.c_[L['lpx'], L['lpy']])
    raise ValueError(S['kind'])


def run(spec, name, dry=False):
    global F
    import torch
    F = int(spec['n_frames'])
    pos, fr, crlb = load_set(spec['sets'][name])
    pos = pos.astype(np.float32); crlb = crlb.astype(np.float32)
    ok = np.isfinite(pos).all(1) & np.isfinite(crlb).all(1) & (crlb.max(1) < 1.0) & (fr >= 0) & (fr < F)
    if spec.get('crlb_min_pos'): ok &= crlb.min(1) > 0
    pos, fr, crlb = pos[ok], fr[ok], crlb[ok]
    print(f'== {name}: N {len(pos)} (dropped {int((~ok).sum())}), median crlb {np.median(crlb, 0) * PX} nm', flush=True)
    if dry: return len(pos)
    V3E, METHODS = _dme()
    os.makedirs(spec['out'], exist_ok=True); fn = os.path.join(spec['out'], f'{name}.npz')
    R = dict(np.load(fn, allow_pickle=True)['R'].item()) if os.path.exists(fn) else {}
    if 'drift_full' not in R:
        torch.cuda.empty_cache(); t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            d, info = V3E.estimate(pos, fr, crlb, F, return_info=True)
        R['drift_full'] = np.asarray(d)[:F, :2]; R['chosen'] = info['chosen']
        R['cv'] = {str(k): float(v) for k, v in info['cv'].items()}; R['t_full'] = time.time() - t; R['N'] = len(pos)
        np.savez(fn, R=np.array(R, dtype=object))
    print(f'  DME v2 chosen {R["chosen"]} ({R["t_full"]:.0f} s); drift ptp {np.ptp(R["drift_full"], 0) * PX} nm', flush=True)
    A = checkerboard(pos); Dh = R.setdefault('D', {})
    for m in spec.get('methods', ['v3e', 'grid_fpb20', 'grid_fpb10']):
        if m in Dh: continue
        torch.cuda.empty_cache(); t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            Dh[m] = tuple(np.asarray(METHODS[m](pos[s], fr[s], crlb[s]))[:F, :2] * PX for s in (A, ~A))
        print(f'  halves {m:12s} {time.time() - t:5.0f} s', flush=True)
        np.savez(fn, R=np.array(R, dtype=object))
    report(fn)


def report(fn):
    R = np.load(fn, allow_pickle=True)['R'].item(); Dh = R.get('D', {})
    lab = os.path.relpath(os.path.abspath(fn)[:-4], os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(fn)))))
    print(f'{lab:30s} N {R.get("N", "?")} chosen {R.get("chosen")}')
    for ref in ('grid_fpb10', 'grid_fpb20'):
        if ref not in Dh or 'v3e' not in Dh: continue
        e = xref_err(Dh, ref)['v3e']
        print(f'  split-half v3e ref {ref}: bands {np.round(e, 2)} rss<5000 {np.sqrt((e[:3] ** 2).sum()):.2f} nm', flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'report':
        for f in sys.argv[2:]: report(f)
    elif mode in ('run', 'dry'):
        spec = json.load(open(sys.argv[2]))
        for n in sys.argv[3:]: run(spec, n, dry=(mode == 'dry'))
    else:
        raise SystemExit(__doc__)
