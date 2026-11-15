"""
Drift estimation with DME v2: access to the `dme` package (github.com/jcnossen/drift-estimation, version >= 2.0, MIT), which is
not part of this repository. It is looked up, in order, in
  1. $DME_PATH or the config key "dme" (simcode.json): a checkout of that repository (the directory that holds dme/),
  2. the installed package (pip install git+https://github.com/jcnossen/drift-estimation).

    from simcode.util.drift import dme_v2, drift_of_table
    drift = dme_v2(pos, frame, crlb, numframes)          # [F, D], units of pos; corrected = pos - drift[frame]
    drift = drift_of_table(T, nframes)                   # [F, 2] px of a localization table (subtract at frame + 3)

    zdrift = zdrift_struct(x, y, z, w)                  # sample axial drift from the structure itself (4Pi tables; per block of windows)

drift_of_table is the lateral drift estimate used everywhere (pipeline tables, baselines, 4Pi tables): rows = good and
disjoint windows when the table has those columns (else all rows), time = window start + 3, crlb = the table's sigma (px),
rows with a sigma >= 1 px, a non-positive sigma or a non-finite value dropped.
"""
import importlib, os, sys
import numpy as np
from .. import config

URL = 'github.com/jcnossen/drift-estimation'
_pkg = None


def dme_package():
    """the imported dme package (version >= 2)"""
    global _pkg
    if _pkg is not None:
        return _pkg
    p = config.root('dme')
    if p:
        p = os.path.abspath(os.path.expanduser(p))
        if not os.path.isfile(os.path.join(p, 'dme', '__init__.py')):
            raise RuntimeError(f'DME_PATH / config "dme" = {p} is not a checkout of {URL} (no dme/__init__.py)')
        if p not in sys.path: sys.path.insert(0, p)
    m = sys.modules.get('dme') or importlib.import_module('dme')
    v = str(getattr(m, '__version__', '1'))
    if int(v.split('.')[0]) < 2:
        raise RuntimeError(f'the dme package at {getattr(m, "__file__", "?")} is version {v}; DME v2 is needed. Set DME_PATH (or '
                           f'"dme" in simcode.json) to a checkout of {URL}, or pip install git+https://{URL}')
    _pkg = m
    return m


def version():
    """'<version> (<path>, git <commit>)' of the dme package in use, for logs"""
    m = dme_package(); d = os.path.dirname(os.path.dirname(os.path.abspath(m.__file__))); c = ''
    try:
        import subprocess
        c = ', git ' + subprocess.run(['git', '-C', d, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return f'{m.__version__} ({d}{c})'


def dme_v2(pos, frame, crlb, numframes=None, return_info=False, device=None, verbose=False):
    """DME v2 (temporal resolution chosen from the data, learned temporal prior) -> drift [F, D] float32 in the units of pos.
    pos [N, D], frame [N] int, crlb [N, D] or [D]. return_info: also dict(chosen=(kind, framesperbin, segment frames), ...)"""
    dme_package()
    from dme.dme_prior import dme_prior_estimate
    from dme.dme_gpu import default_device
    d, info = dme_prior_estimate(np.asarray(pos, np.float32), np.asarray(frame, np.int32), np.asarray(crlb, np.float32),
                                 numframes=numframes, device=default_device(device), verbose=verbose, return_info=True)
    if return_info:
        L = info.get('segment_frames')
        info = dict(info, chosen=('smooth' if L else 'dme', info.get('start_framesperbin'), L or 0), version=version())
        return d, info
    return d


def dme_grid(pos, frame, crlb, framesperbin, numframes=None, **kw):
    """DME v2 with a fixed spline knot spacing (dme.dme_grid.dme_grid_estimate; crlb = one [D] vector)"""
    dme_package()
    from dme.dme_grid import dme_grid_estimate
    return dme_grid_estimate(pos, frame, crlb, framesperbin, numframes=numframes, **{'verbose': False, **kw})


def dme_exact(pos, frame, crlb, framesperbin, numframes=None, **kw):
    """the 2021 algorithm (exact pairwise objective) on the GPU (dme.dme_pipeline.dme_estimate_gpu)"""
    dme_package()
    from dme.dme_pipeline import dme_estimate_gpu
    return dme_estimate_gpu(pos, frame, crlb, framesperbin, numframes=numframes, **{'verbose': False, **kw})


def drift_rows(x, y, frame, sx, sy, nframes, t_offset=3):
    """the rows that enter the drift estimate -> (pos [n,2] float32, time [n] int64 = frame + t_offset, crlb [n,2] float32):
    finite position and sigma, 0 < sigma < 1 px on both axes, time inside [0, nframes)"""
    pos = np.c_[x, y].astype(np.float32); fr = (np.asarray(frame) + t_offset).astype(np.int64)
    crlb = np.c_[sx, sy].astype(np.float32)
    ok = np.isfinite(pos).all(1) & np.isfinite(crlb).all(1) & (crlb.max(1) < 1.0) & (crlb.min(1) > 0) & (fr >= 0) & (fr < nframes)
    return pos[ok], fr[ok], crlb[ok]


def drift_of_table(T, nframes=None, return_info=False, t_offset=3, device=None):
    """lateral drift [F, 2] (px; subtract drift[frame + 3]) of a localization table T (dict or npz with frame, x, y, sx, sy
    in px; if it has the columns good and disjoint only those rows are used) with DME v2. nframes default: last frame + 6.
    return_info: also dict(n rows, chosen, version)."""
    keys = T.files if hasattr(T, 'files') else T
    F = int(nframes) if nframes else int(np.max(T['frame'])) + 6
    m = (np.asarray(T['good']) & np.asarray(T['disjoint'])) if ('good' in keys and 'disjoint' in keys) else np.ones(len(T['frame']), bool)
    pos, fr, crlb = drift_rows(T['x'][m], T['y'][m], T['frame'][m], T['sx'][m], T['sy'][m], F, t_offset)
    d, info = dme_v2(pos, fr, crlb, F, return_info=True, device=device)
    d = np.asarray(d)[:F, :2]
    return (d, dict(n=int(len(pos)), chosen=info['chosen'], version=info['version'])) if return_info else d


def zdrift_struct(x, y, z, w, bin_nm=60.0, blk_w=500, n_it=8, clip=150.0):
    """Sample axial drift from the localizations (generalises the NPC per-pore step to any sample): per block of blk_w
    windows, the median of z - (median z of the same bin_nm lateral bin over the whole movie), smoothed over blocks
    (sd 1 block), iterated n_it times; zero mean. x, y, z in nm, w = window index per localization. Returns the drift (nm)
    per block; subtract drift[w // blk_w]. Blocks with fewer than 200 rows within clip nm get no update."""
    import numpy as np, pandas as pd
    from scipy.ndimage import gaussian_filter1d
    ix = np.floor((x - x.min()) / bin_nm).astype(np.int64); iy = np.floor((y - y.min()) / bin_nm).astype(np.int64)
    pix = np.unique(ix * (iy.max() + 1) + iy, return_inverse=True)[1]
    blk = (w // blk_w).astype(np.int64); nb = int(blk.max()) + 1; drift = np.zeros(nb)
    for _ in range(n_it):
        zc = z - drift[blk]
        ref = pd.Series(zc).groupby(pix).median().to_numpy()
        r = zc - ref[pix]; m = np.abs(r) < clip
        g = pd.Series(r[m]).groupby(blk[m]); md = np.array(g.median().reindex(range(nb)).fillna(0).to_numpy(), dtype=float)
        md[np.array(g.size().reindex(range(nb)).fillna(0).to_numpy()) < 200] = 0
        drift = drift + gaussian_filter1d(md, 1.0); drift -= drift.mean()
    return drift
