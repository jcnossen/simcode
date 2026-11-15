"""Spot ROIs of the Cell11 movie for the in-situ 4Pi PSF fit (from the crop cache, analysis/fourpi_plain/p4cache.py), at
the current SIMCODE localizations (pipeline/Cell11/process/locs_<file>.npz).

Per localization with N >= nmin and nearest other candidate of its frame >= nn_min px: the P x P patch of every channel
centred on round(the channel's mapped position) (the file's registration map, as ChannelFront.patches) and the sub-pixel
rest of that mapped position (pos [n, 4, 2] px, emitter position relative to the patch centre pixel).

  python analysis/fourpi_plain/p4spots.py [--P 17] [--nmin 800] [--nn 7]
      -> <out>/plain4pi/p4psf/spots_P<P>.npz: adu [n, 4, P, P] uint16 (camera ADU; e = (adu - offset) gain), pos [n, 4, 2],
         corner [n, 4, 2], and per spot the table columns (fidx = file, frame, x, y, z, N, bg [4], Phi0, chi2, dof, zprior, sx, sy,
         sz, nn, zref, wrap, good, Tsum), offset, gain, readnoise
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from analysis.fourpi_plain.p4cache import acq, cache_path, CachedReader, CACHE

LOCS = os.path.join(os.path.dirname(CACHE), '..', 'pipeline', 'Cell11', 'process')
COLS = ('frame', 'x', 'y', 'z', 'N', 'bg', 'Phi0', 'chi2', 'dof', 'zprior', 'sx', 'sy', 'sz', 'nn', 'zref', 'wrap', 'good', 'Tsum')


def spots_file(fi, x, y, frame, pm, rd, P):
    """patches [n, 4, P, P] ADU, pos [n, 4, 2], corner [n, 4, 2], ok [n] of the localizations (x, y) in channel-0 px"""
    C, H, W = rd.shape
    h = P // 2
    mx = np.stack([x] + [pm.inv(c, x, y)[0] for c in range(1, C)], 1)
    my = np.stack([y] + [pm.inv(c, x, y)[1] for c in range(1, C)], 1)
    cx, cy = np.round(mx).astype(int), np.round(my).astype(int)
    ok = ((cx - h >= 0) & (cx + h < W) & (cy - h >= 0) & (cy + h < H)).all(1)
    out = np.zeros((len(x), C, P, P), np.uint16)
    order = np.argsort(frame)
    for f0 in range(0, rd.n, 500):
        sel = order[(frame[order] >= f0) & (frame[order] < f0 + 500)]
        sel = sel[ok[sel]]
        if len(sel) == 0: continue
        A = rd.a[f0:f0 + 500]
        for c in range(C):
            yy = cy[sel, c, None] - h + np.arange(P)[None]
            xx = cx[sel, c, None] - h + np.arange(P)[None]
            out[sel, c] = A[frame[sel, None, None] - f0, c, yy[:, :, None], xx[:, None, :]]
    pos = np.stack([mx - cx, my - cy], -1)
    corner = np.stack([cx - h, cy - h], -1)
    return out, pos, corner, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--P', type=int, default=17)
    p.add_argument('--nmin', type=float, default=800.0)
    p.add_argument('--nn', type=float, default=7.0)
    p.add_argument('--files', type=int, nargs='+', default=[0, 1, 2, 3])
    a = p.parse_args()
    c, S = acq()
    cam = S.camera(c)
    acc = {k: [] for k in ('adu', 'pos', 'corner', 'fidx') + COLS}
    for fi in a.files:
        d = np.load(os.path.join(LOCS, f'locs_{fi:03d}.npz'))
        sel = np.nonzero((d['N'] >= a.nmin) & (d['nn'] >= a.nn) & np.isfinite(d['z']))[0]
        rd = CachedReader(cache_path(fi))
        pm = S.regmap(c, fi)
        adu, pos, corner, ok = spots_file(fi, d['x'][sel], d['y'][sel], d['frame'][sel].astype(int), pm, rd, a.P)
        sel = sel[ok]
        acc['adu'].append(adu[ok]); acc['pos'].append(pos[ok]); acc['corner'].append(corner[ok])
        acc['fidx'].append(np.full(len(sel), fi, np.int16))
        for k in COLS: acc[k].append(d[k][sel])
        print(f'file {fi}: {len(sel)} spots', flush=True)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    fo = os.path.join(os.path.dirname(CACHE), f'spots_P{a.P}.npz')
    np.savez(fo, **out, offset=cam['offset'], gain=cam['gain'], readnoise=cam['readnoise_e'])
    print(f'{len(out["x"])} spots -> {fo}')


if __name__ == '__main__':
    main()
