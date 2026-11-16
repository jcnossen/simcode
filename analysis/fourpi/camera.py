"""Per-pixel camera offset / read-noise maps for the 4 channel crops (no dark frames exist in this data set).

Model per raw pixel p: ADU = off_p + g * (photo-electrons) + read noise (sd rn_p), g = 4.35 ADU/e (0.23 e/ADU).
Per pixel, over K movie frames: L_p = sigma-clipped temporal mean (clip at median + 3 sd: removes emitter-on frames),
V_p = clipped temporal variance.
  * read noise: dark strips (raw columns 20-200, left of channel 0) give rn_p^2 = V_p directly; elsewhere rn^2 is taken
    as the dark-strip mean (the sCMOS per-pixel rn spread cannot be measured under light without dark frames).
  Findings (2026-11-11): the mean-variance offset level is NOT usable here (excess temporal variance in ch2/ch3 gives
  offsets of 97-98 ADU, below the 100.5 ADU dark level); per-pixel FPN is stable within a day (r = 0.97, MT files 5 vs 60)
  but not between the MT and NPC days (r = 0.15); the column FPN is stable across days (r = 0.94). See build().
usage: python camera.py [out_dir] --work <4Pi work folder> --data <raw 4Pi movie folder>  (default out_dir <work>/smlm/camera)
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
import sys, os, argparse
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc import Dcimg, movie_files, X0, CW, MIRROR

G_ADU_PER_E = 4.35


def clipped_stats(fn, f0s, n):
    d = Dcimg(fn)
    a = np.concatenate([d.raw(f, n) for f in f0s]).astype(np.float32)      # [K, 200, 2304]
    a[:, 0, :] = np.nan                                                     # row 0 carries embedded header pixels
    m = np.nanmedian(a, 0)
    s = 1.4826 * np.nanmedian(np.abs(a - m), 0)
    ok = (a < m + 3 * s) & (a > m - 5 * s)
    cnt = ok.sum(0)
    L = np.where(ok, a, 0).sum(0) / cnt
    V = np.where(ok, (a - L) ** 2, 0).sum(0) / (cnt - 1) / 0.9733           # +3 sd one-sided clip of a Gaussian
    return L, V, cnt / len(a)


def to_crops(img):
    out = []
    for c in range(4):
        x = img[..., X0[c]:X0[c] + CW]
        out.append(x[..., ::-1] if MIRROR[c] else x)
    return np.stack(out, -3)


def build(out_dir, work, data, K=1200):
    """offset map = global level OFF0 (left dark strip, raw columns 20-200, both movies 100.49-100.51 ADU) + the movie's
    own column FPN (column means over rows 1-199 of L - median_filter(L, 1 x 15)); the per-pixel residual FPN (0.9 ADU sd,
    stable within a day, not across days) is NOT corrected in the lit area, where sample structure leaks into any
    per-pixel estimate without dark frames. Read noise: dark-strip same-step-difference variance (24.6-24.9 ADU^2).
    work: the 4Pi work folder, data: the folder of the raw 4Pi movies (the movie file lists)."""
    os.makedirs(out_dir, exist_ok=True)
    for cell, fi in (('MT', 5), ('NPC', 10)):
        fn = [f for f in movie_files(cell, work, data) if f.endswith(f'_{fi:03d}.dcimg')][0]
        L, V, frac = clipped_stats(fn, [600, 9000, 16000], K // 3)
        L[0] = L[1]
        OFF0 = float(np.mean(L[1:, 20:200]))
        hfx = L - median_filter(L, size=(1, 15), mode='nearest')
        col = hfx[1:].mean(0)
        off_raw = OFF0 + np.broadcast_to(col[None], L.shape)
        pix = (L - median_filter(L, size=(9, 9), mode='nearest')) - col[None]
        rn2 = float(np.mean(V[1:, 20:200]))
        off = to_crops(off_raw).astype(np.float32)
        bg_e = (gaussian_filter(to_crops(L), (0, 8, 8)) - off) * (1 / G_ADU_PER_E)
        np.savez_compressed(f'{out_dir}/camera_{cell}.npz', off=off, off0=OFF0, colfpn=to_crops(np.broadcast_to(col[None], L.shape)).astype(np.float32),
                            pixfpn=to_crops(pix).astype(np.float32), L=to_crops(L).astype(np.float32), bg_e=bg_e.astype(np.float32),
                            rn2_adu=rn2, gain_e_per_adu=1 / G_ADU_PER_E)
        print(f'{cell}: OFF0 {OFF0:.2f} ADU, column FPN sd {col.std():.3f} ADU, per-pixel residual sd (dark strip) '
              f'{pix[1:,20:200].std():.3f} ADU, rn^2 {rn2:.2f} ADU^2 (rn {np.sqrt(rn2)/G_ADU_PER_E:.2f} e)')
        for ch in range(4):
            sl = (slice(10, 190), slice(100, 240))
            print(f'   ch{ch}: bg (L - offset) {bg_e[ch][sl].mean():.2f} e/px/frame [p5 {np.percentile(bg_e[ch][sl],5):.2f}, p95 {np.percentile(bg_e[ch][sl],95):.2f}]')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir', nargs='?')
    ap.add_argument('--work', required=True, help='4Pi work folder')
    ap.add_argument('--data', required=True, help='folder of the raw 4Pi movies')
    A = ap.parse_args()
    build(A.out_dir or f'{A.work}/smlm/camera', A.work, A.data)
