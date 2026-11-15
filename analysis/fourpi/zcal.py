"""Bead z-stacks (61 slices, 20 nm stage steps, 50 ms; one bead per stack):
  1. registered 4-channel sum per slice (bilinear warp of ch1-3 with the in-situ map) -> astigmatic Gaussian MLE per
     slice -> sigma_x, sigma_y vs stage z (the astigmatism calibration, fitted with the usual
     s(z) = s0 sqrt(1 + ((z - c) / d)^2 + A ((z - c) / d)^3 + B ((z - c) / d)^4) curves);
  2. per-channel photometry (r = 3 px aperture and r = 1 px centre) vs z -> channel fractions, interference contrast;
  3. saves the per-channel ROIs (registered, 21 x 21) for the PSF steps.
usage: python zcal.py  (both stacks)
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
import os, sys, json
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc import zstack
from register import PolyMap
from quickloc import Warp
from mle import GaussAstig, lm_fit

OUT = f'{P4}/smlm/zcal'
from simcode.setups.fourpi import acquisition; GAIN = acquisition('MT')['camera']['gain_e_per_adu']   # 4Pi camera gain (the same for MT and NPC)
def astig_curve(p, z):
    s0, c, d, A, B = p
    u = (z - c) / d
    return s0 * np.sqrt(np.clip(1 + u ** 2 + A * u ** 3 + B * u ** 4, 1e-6, None))


def fit_curve(z, s, c0):
    best = None
    for c in (c0 - 0.2, c0, c0 + 0.2):
        r = least_squares(lambda p: astig_curve(p, z) - s, [s.min(), c, 0.3, 0, 0],
                          bounds=([0.5, -1, 0.05, -5, -5], [3, 1, 2, 5, 5]))
        if best is None or r.cost < best.cost: best = r
    return best.x


def run(cell, R=21):
    os.makedirs(OUT, exist_ok=True)
    cam = np.load(f'{P4}/smlm/camera/camera_{cell}.npz')
    pm = PolyMap.load(f'{P4}/smlm/reg/reg_insitu_{cell}.json')
    warp = Warp(lambda c, xx, yy: pm.inv(c, xx, yy))
    Z = zstack(cell)                                            # [61,4,H,W] ADU
    sat = int((Z >= 65535).sum())
    E = (Z - cam['off'][None]) * GAIN
    W = np.stack([[warp(E[k, c], c) for c in range(4)] for k in range(61)])     # registered [61,4,H,W]
    S = W.sum(1)
    mx = S.max(0); f = ndi.gaussian_filter(mx, 2); y0, x0 = np.unravel_index(np.argmax(f[15:185, 15:345]), (170, 330)); y0 += 15; x0 += 15
    h = R // 2
    rois = W[:, :, y0 - h:y0 + h + 1, x0 - h:x0 + h + 1]           # [61,4,R,R]
    zst = (np.arange(61) - 30) * 0.02
    # Gaussian MLE per slice on the 4-channel sum
    m = GaussAstig(R, smax=8)
    d = torch.as_tensor(rois.sum(1), dtype=torch.float64)
    th, ll, cr, chi2, dev, mu = lm_fit(m, d, m.init(d), 4 * 1.3, iters=100)
    th = th.numpy()
    # photometry per channel
    yv, xv = np.mgrid[-h:h + 1, -h:h + 1]
    res = {}
    for nm, rad in (('r3', 3.0), ('r1', 1.0), ('r6', 6.0)):
        ap = (np.hypot(yv[None] - th[:, 1, None, None], xv[None] - th[:, 0, None, None]) <= rad)
        ring = (np.hypot(yv, xv) >= 8)
        bg = np.median(rois[..., ring], -1)
        I = ((rois - bg[..., None, None]) * ap[:, None]).sum((-1, -2))
        res[nm] = I
    sx, sy = th[:, 4], th[:, 5]
    good = (th[:, 2] > 0.05 * th[:, 2].max()) & (sx < 6) & (sy < 6)
    px = fit_curve(zst[good], sx[good], zst[np.argmin(sx)]); py = fit_curve(zst[good], sy[good], zst[np.argmin(sy)])
    out = dict(cell=cell, bead_xy=[int(x0), int(y0)], saturated_px=sat, zstage=zst.tolist(), sx=sx.tolist(), sy=sy.tolist(),
               N=th[:, 2].tolist(), bg=th[:, 3].tolist(), x=th[:, 0].tolist(), y=th[:, 1].tolist(), good=good.tolist(),
               curve_x=px.tolist(), curve_y=py.tolist(),
               **{f'I_{k}': v.tolist() for k, v in res.items()})
    json.dump(out, open(f'{OUT}/zcal_{cell}.json', 'w'))
    np.save(f'{OUT}/zstack_rois_{cell}.npy', rois.astype(np.float32))
    print(f'{cell}: bead at ({x0},{y0}), saturated px {sat}; sx min {sx[good].min():.2f} at z {zst[good][np.argmin(sx[good])]:+.2f}, '
          f'sy min {sy[good].min():.2f} at z {zst[good][np.argmin(sy[good])]:+.2f}; curve x {np.round(px,3)}, y {np.round(py,3)}')
    return out


if __name__ == '__main__':
    for cell in sys.argv[1:] or ('NPC', 'MT'):
        run(cell)
