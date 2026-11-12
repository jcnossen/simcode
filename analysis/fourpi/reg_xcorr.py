"""In-situ registration refinement by tile cross-correlation of time-averaged channel images.
Mean image of K frames per channel (e), warped into ch0 coordinates with the current map; for each tile (40 x 40 px,
step 20, illuminated area) the sub-pixel shift between the warped channel c and ch0 (phase correlation, upsampling 100,
after a 1.5 px DoG band-pass that removes the smooth background). The per-emitter 4Pi shape differences average out
over many emitters / phases. Correction: inv_new(p) = inv_old(p + d(p)), d fitted by a quadratic polynomial over
tiles; fwd refitted from inv on a grid. usage: python reg_xcorr.py <cell> <ref_fi> [check_fi ...]
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
import os, sys, json
import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc import Dcimg, crops, movie_files, file_index
from register import PolyMap, feats, fit_poly
from quickloc import Warp

OUT = f'{P4}/smlm/reg'
GAIN = 0.23


def mean_img(cell, fi, off, f0=0, n=3600):
    d = Dcimg([f for f in movie_files(cell) if file_index(f) == fi][0])
    acc = 0
    for a0 in range(f0, f0 + n, 600):
        acc = acc + ((crops(d.raw(a0, 600)) - off) * GAIN).sum(0)
    return acc / n


def tile_shifts(M, pm, xr=(70, 290), yr=(10, 190), T=40, step=20):
    warp = Warp(lambda c, xx, yy: pm.inv(c, xx, yy))
    Wm = [warp(M[c], c) for c in range(4)]
    bp = [ndi.gaussian_filter(w, 1.0) - ndi.gaussian_filter(w, 4) for w in Wm]
    rows = []
    for y0 in range(yr[0], yr[1] - T + 1, step):
        for x0 in range(xr[0], xr[1] - T + 1, step):
            ref = bp[0][y0:y0 + T, x0:x0 + T]
            win = np.outer(np.hanning(T), np.hanning(T))
            for c in (1, 2, 3):
                sh, err, _ = phase_cross_correlation(ref * win, bp[c][y0:y0 + T, x0:x0 + T] * win, upsample_factor=100, normalization=None)
                # sh = shift to apply to channel c to match ref: content of c sits at p - sh -> d = -sh (x, y)
                rows.append((c, x0 + T / 2, y0 + T / 2, -sh[1], -sh[0], err))
    return np.array(rows)


def refine(pm, S, order=2):
    cf, ci = {}, {}
    yy, xx = np.mgrid[0:200:5, 0:360:5].astype(float)
    xx, yy = xx.ravel(), yy.ravel()
    rep = {}
    for c in (1, 2, 3):
        s = S[S[:, 0] == c]
        cx = fit_poly(s[:, 1:3], s[:, 3:5], order)                            # d(p) in ch0 px
        r = s[:, 3:5] - feats(s[:, 1], s[:, 2], order) @ cx
        d = feats(xx, yy, order) @ cx
        xi, yi = pm.inv(c, xx + d[:, 0], yy + d[:, 1])
        ci[c] = fit_poly(np.c_[xx, yy], np.c_[xi, yi], 3)
        cf[c] = fit_poly(np.c_[xi, yi], np.c_[xx, yy], 3)
        rep[c] = dict(tile_shift_median=np.median(s[:, 3:5], 0).round(4).tolist(), tile_shift_sd=s[:, 3:5].std(0).round(4).tolist(),
                      tile_shift_maxabs=np.abs(s[:, 3:5]).max(0).round(4).tolist(), poly_resid_sd=r.std(0).round(4).tolist())
    return PolyMap(cf, ci, 3), rep


if __name__ == '__main__':
    cell, ref = sys.argv[1], int(sys.argv[2]); checks = [int(v) for v in sys.argv[3:]]
    off = np.load(f'{P4}/smlm/camera/camera_{cell}.npz')['off']
    pm = PolyMap.load(f'{OUT}/reg_beads_{cell}.json')
    M = mean_img(cell, ref, off)
    report = {}
    for it in range(3):
        S = tile_shifts(M, pm)
        pm_new, rep = refine(pm, S)
        report[f'iter{it}_residual_of_input_map'] = rep
        print(cell, ref, 'iter', it, 'residual tile shifts of the input map:', json.dumps(rep), flush=True)
        pm = pm_new
    S = tile_shifts(M, pm)
    report['final_tiles'] = {c: dict(median=np.median(S[S[:, 0] == c, 3:5], 0).round(4).tolist(), sd=S[S[:, 0] == c, 3:5].std(0).round(4).tolist()) for c in (1, 2, 3)}
    print(cell, ref, 'final map residual', json.dumps(report['final_tiles']), flush=True)
    np.save(f'{OUT}/tiles_{cell}_{ref}.npy', S)
    for fi in checks:
        S = tile_shifts(mean_img(cell, fi, off), pm)
        report[f'file{fi}'] = {c: dict(median=np.median(S[S[:, 0] == c, 3:5], 0).round(4).tolist(), sd=S[S[:, 0] == c, 3:5].std(0).round(4).tolist()) for c in (1, 2, 3)}
        print(cell, fi, 'residual of the file-%d map' % ref, json.dumps(report[f'file{fi}']), flush=True)
        np.save(f'{OUT}/tiles_{cell}_{fi}.npy', S)
    pm.save(f'{OUT}/reg_insitu_{cell}.json', source=f'bead map refined by tile cross-correlation of mean images, {cell} file {ref} frames 0-3600', report=report)
