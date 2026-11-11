"""Fair FRC for the paper's method comparison (SMLM, SIMFLUX, DECODE, SIMCODE v2 on the same movie).

Protocol (every method identically; README.md in this directory):
  1. Rows: each method's full output table (SMLM: chi^2-filtered smlm.npy; SIMFLUX: sf.hdf5; DECODE: all rows of the decode
     locs npz; SIMCODE v2: pass-2 `sel`, the recommended full output).
  2. Disjoint windows only: all four methods localize 6-frame windows (SMLM / SIMFLUX: a window at every frame; DECODE and
     SIMCODE v2: every 3 frames), so overlapping windows re-use the same photons. Only window starts f % 6 == 0 are kept (the G6
     grid of simcode2.evalproto.lib), so no photon is counted twice and every method samples the same windows.
  3. Uncertainty filter, the same rule for all: max(sigma_x, sigma_y) < 0.15 px (16.2 nm), the paper's crlb_filter(0.15), with
     each method's own per-localization sigma (SMLM / SIMFLUX: CRLB; DECODE: its predicted sigma; SIMCODE v2: calibrated sigma).
  4. Common region: the evalproto crop (x, y in [5, 115) px) for every method (excludes the sensor edge).
  5. Drift: each method's OWN DME v2 drift (v3e, analysis/baselines/dme_drift.py), estimated on that method's rows from 2-4
     (time = window start + 3, crlb = its sigma), subtracted at window start + 3.
  6. FRC: alternating 300-frame halves (half = window start // 300 % 2), 5-nm histogram renders, 5-bin smoothed curve, 1/7
     threshold = simcode2.evalproto.lib.frc_std (= frc_split, blk 300); frc_curve() below is the same computation that also
     returns the curve and is asserted to give lib.frc's value.
  7. Counts next to the FRC: n_frc (rows in the FRC = steps 1-4), plus n_all (step 1) and n_disjoint (steps 1-2, 4).
  Secondary numbers (not plotted): FRC without the sigma filter (steps 1, 2, 4-6) and with the paper's 30-frame halves.

API: T = load(kind, path); R = evaluate(T, nframes, drift_cache=<npy>)  ->  dict(frc_nm, freq, curve, n_*, drift, mask, x_dc, y_dc)
CLI: CUDA_VISIBLE_DEVICES=1 python analysis/paper/frc_fair.py <out.json> <nframes> name=kind:path [name=kind:path ...]
     kinds: smlm (smlm.npy), sf (sf.hdf5 / any smlmtorch hdf5), decode (decode_paper.py npz), v2 (simcode2 locs_p*.npz; rows `sel`)"""
import os, sys, json, time, contextlib, io
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.evalproto import lib as EP

PX = EP.PX
SIGMA_MAX = 0.15          # px, the paper's crlb_filter(0.15)
GRID = 6                  # disjoint 6-frame windows
BLK = 300                 # FRC halves: alternating 300-frame blocks
CROP = EP.CROP            # (x0, y0, size) px


def load(kind, path):
    """-> dict x, y (px, sensor, pixel centre = integer, x = column), f (window start), sx, sy (px), photons"""
    if kind in ('smlm', 'sf'):
        from smlmtorch import Dataset
        from smlmtorch.simflux.dataset import SFDataset
        ds = SFDataset.load(path) if path.endswith('.npy') else Dataset.load(path)
        return dict(x=ds.pos[:, 0].astype(np.float64), y=ds.pos[:, 1].astype(np.float64), f=ds.frame.astype(np.int64),
                    sx=ds.crlb.pos[:, 0].astype(np.float64), sy=ds.crlb.pos[:, 1].astype(np.float64), photons=ds.photons.astype(np.float64))
    z = np.load(path)
    if kind == 'decode':
        return dict(x=z['x'].astype(np.float64), y=z['y'].astype(np.float64), f=z['frame'].astype(np.int64),
                    sx=z['sx'].astype(np.float64), sy=z['sy'].astype(np.float64), photons=z['phot'].astype(np.float64))
    if kind == 'v2':
        m = z['sel'].astype(bool)
        return dict(x=z['x'][m].astype(np.float64), y=z['y'][m].astype(np.float64), f=z['frame'][m].astype(np.int64),
                    sx=z['sx'][m].astype(np.float64), sy=z['sy'][m].astype(np.float64), photons=z['photons'][m].astype(np.float64))
    raise ValueError(kind)


def in_crop(x, y, crop=CROP):
    x0, y0, s = crop
    return (x >= x0) & (x < x0 + s) & (y >= y0) & (y < y0 + s)


def masks(T, sigma_max=SIGMA_MAX, grid=GRID, crop=CROP):
    ok = np.isfinite(T['x']) & np.isfinite(T['y']) & np.isfinite(T['sx']) & np.isfinite(T['sy'])
    disj = ok & (T['f'] % grid == 0) & in_crop(T['x'], T['y'], crop)
    filt = disj & (np.maximum(T['sx'], T['sy']) < sigma_max)
    return disj, filt


def dme_v2(x, y, t, sx, sy, F):
    """DME v2 (v3e, ~/dme read-only, as analysis/baselines/dme_drift.py) -> drift [F, 2] px, info"""
    D = os.path.expanduser(os.environ.get('DME_PATH', '~/dme'))
    p = [f'{D}/gpu', f'{D}/gpu/loop', f'{D}/gpu/loop/methods']
    for q in p:
        if q not in sys.path: sys.path.insert(0, q)
    import learned_smoother3e as V3E
    pos = np.c_[x, y].astype(np.float32); crlb = np.c_[sx, sy].astype(np.float32)
    ok = (t >= 0) & (t < F) & (crlb.max(1) < 1.0) & (crlb.min(1) > 0)
    with contextlib.redirect_stdout(io.StringIO()):
        d, info = V3E.estimate(pos[ok], t[ok].astype(np.int64), crlb[ok], F, return_info=True)
    return np.asarray(d)[:F, :2], dict(chosen=str(info.get('chosen')), n=int(ok.sum()))


def frc_curve(xa, ya, xb, yb, px=5.0):
    """simcode2.evalproto.lib.frc (same bounds, render, smoothing, 1/7 crossing) returning (resolution nm, freq 1/nm, curve)"""
    s = PX / px
    lo = np.floor(min(xa.min(), ya.min(), xb.min(), yb.min())); hi = np.ceil(max(xa.max(), ya.max(), xb.max(), yb.max()))
    n = int((hi - lo) * s); n += n % 2
    h = lambda x, y: np.histogram2d(y, x, bins=n, range=[[lo, lo + n / s], [lo, lo + n / s]])[0]
    A, B = np.fft.fftshift(np.fft.fft2(h(xa, ya))), np.fft.fftshift(np.fft.fft2(h(xb, yb)))
    yy, xx = np.indices(A.shape) - n // 2; rr = np.hypot(xx, yy).astype(int)
    num = np.bincount(rr.ravel(), (A * B.conj()).real.ravel()); da = np.bincount(rr.ravel(), (np.abs(A) ** 2).ravel())
    db = np.bincount(rr.ravel(), (np.abs(B) ** 2).ravel())
    c = np.convolve(num / np.sqrt(da * db + 1e-12), np.ones(5) / 5, 'same')
    below = np.nonzero((c[3:n // 2] < 1 / 7))[0]
    res = float('nan') if len(below) == 0 else float(n * px / (below[0] + 3))
    freq = np.arange(n // 2) / (n * px)
    return res, freq, c[:n // 2]


def frc_halves(x, y, f, blk=BLK):
    h = (f // blk) % 2 == 0
    return frc_curve(x[h], y[h], x[~h], y[~h])


def evaluate(T, nframes, drift_cache=None, sigma_max=SIGMA_MAX, secondary=True):
    disj, filt = masks(T, sigma_max)
    t = T['f'] + 3
    if drift_cache and os.path.exists(drift_cache):
        drift = np.load(drift_cache); dinfo = dict(cached=drift_cache)
    else:
        t0 = time.time()
        drift, dinfo = dme_v2(T['x'][filt], T['y'][filt], t[filt], T['sx'][filt], T['sy'][filt], nframes)
        dinfo['s'] = round(time.time() - t0, 1)
        if drift_cache:
            os.makedirs(os.path.dirname(os.path.abspath(drift_cache)), exist_ok=True); np.save(drift_cache, drift)
    ti = np.clip(t, 0, len(drift) - 1)
    xd, yd = T['x'] - drift[ti, 0], T['y'] - drift[ti, 1]
    res, freq, curve = frc_halves(xd[filt], yd[filt], T['f'][filt])
    ref = EP.frc_std(xd[filt], yd[filt], T['f'][filt])
    assert (np.isnan(res) and np.isnan(ref)) or abs(res - ref) < 1e-9, (res, ref)
    R = dict(frc_nm=res, freq=freq, curve=curve, n_frc=int(filt.sum()), n_disjoint=int(disj.sum()), n_all=int(len(T['x'])),
             drift=drift, drift_info=dinfo, drift_ptp_nm=(np.ptp(drift, 0) * PX).round(1).tolist(), mask=filt, x_dc=xd, y_dc=yd)
    if secondary:
        R['frc_nm_nofilter'] = frc_halves(xd[disj], yd[disj], T['f'][disj])[0]
        R['frc_nm_30frame_halves'] = frc_halves(xd[filt], yd[filt], T['f'][filt], blk=30)[0]
    return R


def summary(R):
    return {k: v for k, v in R.items() if k not in ('freq', 'curve', 'drift', 'mask', 'x_dc', 'y_dc')}


if __name__ == '__main__':
    out, F = sys.argv[1], int(sys.argv[2])
    res = {}
    for spec in sys.argv[3:]:
        name, kp = spec.split('=', 1); kind, path = kp.split(':', 1)
        T = load(kind, path)
        keep = T['f'] + 6 <= F                       # windows inside the first F frames (test runs on a frame range)
        T = {k: v[keep] for k, v in T.items()}
        R = evaluate(T, F, drift_cache=os.path.splitext(out)[0] + f'_{name}_drift.npy')
        res[name] = dict(kind=kind, path=path, **summary(R))
        print(name, json.dumps(res[name]), flush=True)
    json.dump(dict(protocol=dict(sigma_max_px=SIGMA_MAX, grid=GRID, halves_frames=BLK, crop=CROP, drift='own DME v2 (v3e)'),
                   nframes=F, methods=res), open(out, 'w'), indent=1)
