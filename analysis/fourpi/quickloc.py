"""Quick per-window emitter photometry used by the frame-order check and the pattern checks (no fitting):
registered 4-channel sum of 6-frame windows (stored affine or a polynomial map), peak detection, intensity-weighted
centroid, per-frame per-channel aperture photometry (r = 2.5 px, local ring background).
"""
import os, sys
import numpy as np
import yaml
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc import crops, bead_align_file

from simcode.setups.fourpi import acquisition; GAIN = acquisition('MT')['camera']['gain_e_per_adu']   # 4Pi camera gain (the same for MT and NPC)


def affine_maps(cell):
    y = os.path.join(os.path.dirname(bead_align_file(cell)), 'mappings_astig_fit.yaml')
    mp = yaml.safe_load(open(y))
    return [np.array(mp[f'image_{i}']['T_inv']) for i in range(1, 4)], [np.array(mp[f'image_{i}']['T_fwd']) for i in range(1, 4)]


class Warp:
    """ch c -> ch0 resampling. coords(c) gives, for every ch0 pixel (row y, col x), the (row, col) in channel c."""
    def __init__(self, inv_fun, H=200, W=360):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        self.coords = [None] + [np.stack(inv_fun(c, xx, yy)[::-1]).astype(np.float32) for c in (1, 2, 3)]

    def __call__(self, img, c):
        if c == 0: return img
        return ndi.map_coordinates(img, self.coords[c], order=1, mode='nearest')


def affine_warp(cell):
    Tinv, _ = affine_maps(cell)
    def inv(c, xx, yy):
        T = Tinv[c - 1]
        return T[0, 0] * xx + T[0, 1] * yy + T[0, 2], T[1, 0] * xx + T[1, 1] * yy + T[1, 2]
    return Warp(inv)


def window_photometry(E, warp, xlim=(60, 300), thr=25.0, iso=8, r_ap=2.5):
    """E [6k, 4, H, W] photo-electrons (offset removed). Returns rows: (window, x, y, I[6,4] aperture e, bg[6,4] e/px)"""
    nW = E.shape[0] // 6
    yv, xv = np.mgrid[-6:7, -6:7]; rr = np.hypot(yv, xv)
    ap = rr <= r_ap; ring = (rr > 4.5) & (rr <= 6)
    Wf = np.stack([np.stack([warp(E[t, c], c) for c in range(4)]) for t in range(E.shape[0])])   # [T,4,H,W]
    out = []
    H, W = E.shape[-2:]
    for w in range(nW):
        P = Wf[6 * w:6 * w + 6]
        s6 = P.sum((0, 1))
        fl = ndi.gaussian_filter(s6, 1.5) - ndi.gaussian_filter(s6, 5)
        pk = (fl == ndi.maximum_filter(fl, 7)) & (fl > thr)
        ys, xs = np.nonzero(pk)
        keep = [(y, x) for y, x in zip(ys, xs) if 8 <= y < H - 8 and xlim[0] <= x < xlim[1]
                and not (((np.hypot(ys - y, xs - x) > 0) & (np.hypot(ys - y, xs - x) < iso)).any())]
        for y, x in keep:
            roi = P[:, :, y - 6:y + 7, x - 6:x + 7]
            bg = np.median(roi[..., ring], -1)                                 # [6,4]
            I = ((roi - bg[..., None, None]) * ap).sum((-1, -2))
            s = (roi.sum((0, 1)) - bg.sum() ) * ap
            sw = s.clip(0).sum()
            cx = x + (s.clip(0) * xv).sum() / sw; cy = y + (s.clip(0) * yv).sum() / sw
            out.append((w, cx, cy, I, bg))
    return out


def rows_to_arrays(rows):
    w = np.array([r[0] for r in rows]); x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    I = np.stack([r[3] for r in rows]); bg = np.stack([r[4] for r in rows])
    return w, x, y, I, bg


def slot_coeffs(x, y, I, k0, k1, min_mean=150.0):
    """Z[j, a] = 2 <(I_j / mean I - 1) exp(-i k_a . r)> over emitters 'on' in all 6 frames (channel-summed)"""
    It = I.sum(-1) if I.ndim == 3 else I
    m = It.mean(1)
    ok = (m > min_mean) & (It.min(1) > 0.25 * m)
    r = It[ok] / m[ok, None] - 1
    X = np.c_[x[ok], y[ok]]
    Z = np.stack([2 * (r * np.exp(-1j * (X @ k))[:, None]).mean(0) for k in (k0, k1)], 1)
    return Z, int(ok.sum())


def refine_k(x, y, I, k_init, slots, span=0.06, n=41, min_mean=150.0):
    It = I.sum(-1) if I.ndim == 3 else I
    m = It.mean(1); ok = (m > min_mean) & (It.min(1) > 0.25 * m)
    r = It[ok][:, slots] / m[ok, None] - 1; X = np.c_[x[ok], y[ok]]
    best = (-1, None)
    for dx in np.linspace(-span, span, n):
        for dy in np.linspace(-span, span, n):
            k = np.array([k_init[0] + dx, k_init[1] + dy])
            s = (np.abs((r * np.exp(-1j * (X @ k))[:, None]).mean(0)) ** 2).sum()
            if s > best[0]: best = (s, k)
    return best[1]
