"""SI figure si_review-single-frame (reviewer figure): one raw camera frame of the low-density movie (1809/2, the Fig 3 data set)
in photo-electrons, with an inset of one diffraction-limited spot and a 2 um scale bar; caption material with the PSF width.

The preprint's version (a 0.5 nM frame, a PNG without a stored script) is regenerated here on the paper's Fig 3 movie with the
v2 camera calibration (gain, per-pixel offset map of models/insitu/calib_1809_2_psf_v2.json) and the in-situ PSF v2:
- frame: the frame in [--frame, --frame + --search) (default 30001, 60) with the best isolated inset spot (below); the image is (ADU - offset map) x gain, clipped at the 0.5 / 99.8 percentiles,
  grey scale (as the preprint), the run's crop (x, y in [5, 115) px) only;
- inset spot: the SIMCODE v2 localization with the largest 3 x 3 px signal in this frame (paper selection: p >= 0.63, max(sigma) <= 12 nm, in the crop) of a window that
  contains the frame, with no other pass-2 row (any p) of these windows at 1.5-8 px and an isolated spot in the frame image and >= 8 px from the crop edge
  (or --spot x,y in px); 9 x 9 px inset;
- PSF width: FWHM of the PSF v2 model (angle-averaged in-situ fit, pixel-integrated, i.e. as seen in the image) at the median
  emitter height of the v2 selection, along x and y, from a 0.02 px scan of the centre pixel; also the FWHM of the spot itself
  (1D Gaussian fits of the background-subtracted inset rows / columns through the spot, informative only);
- --mark 1: a variant with the selected v2 localizations of the windows containing the frame as circles (<out>_locs.*).

usage: python analysis/paper/si_single_frame.py [--frame F] [--run <v2 run dir>] [--out <prefix>] [--mark 1]
  default out: out/paper_figs_v2/si_review-single-frame  (.pdf / .svg / .png + _caption.txt)"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'analysis', 'baselines')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch, Circle

MOVIE = '/mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/simflux_5.5ms_500mW.tif'
RUN = '/mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/runs/simcode_v2_psfv2_20261115'
CALIB = 'insitu/calib_1809_2_psf_v2.json'
CROP = (5, 5, 110)


def read_frame(path, i):
    """frame i (>= 1) of an uncompressed TIFF with a constant page stride after page 0 (page 0 carries the longer first IFD):
    data offset of page 1 + (i - 1) x the stride (checked on pages 1-4 and against tifffile on page 4), read by memmap (walking
    30 000 IFDs is slow on the data disk)"""
    import tifffile
    with tifffile.TiffFile(path) as t:
        p = [t.pages[k] for k in range(5)]
        off = [q.dataoffsets[0] for q in p]; shape, dtype = p[0].shape, p[0].dtype
        st = off[2] - off[1]
        assert all(q.compression == 1 for q in p) and off[3] - off[2] == st and off[4] - off[3] == st and i >= 1
        assert np.array_equal(np.memmap(path, dtype, 'r', off[1] + 3 * st, shape), p[4].asarray())
    return np.array(np.memmap(path, dtype, 'r', off[1] + (i - 1) * st, shape), np.float64)


def psf_fwhm(calib_path, z_um, device='cpu'):
    """pixel-integrated FWHM (px) of the PSF model along x and y at height z_um: value of the centre pixel vs sub-pixel offset"""
    import torch, zpsf
    psf = zpsf.load(calib_path, device=device)
    d = np.arange(-3.0, 3.0001, 0.02)
    out = []
    for ax in (0, 1):
        t = torch.as_tensor(d, dtype=torch.float32, device=device); z0 = torch.zeros_like(t)
        xc, yc = (t, z0) if ax == 0 else (z0, t)
        img = psf.eval(xc, yc, 15, torch.full_like(t, z_um))[0].cpu().numpy()
        v = img[:, 7, 7]; v = v / v.max(); above = d[v >= 0.5]
        out.append(float(above.max() - above.min()))
    return out


def gauss_fwhm(profile):
    """FWHM (px) of a 1D Gaussian + constant fitted to a profile"""
    from scipy.optimize import curve_fit
    x = np.arange(len(profile)) - len(profile) // 2
    f = lambda x, a, x0, s, b: a * np.exp(-0.5 * (x - x0) ** 2 / s ** 2) + b
    p, _ = curve_fit(f, x, profile, p0=[profile.max() - profile.min(), 0.0, 1.2, 0.0],
                     bounds=([0, -2, 0.3, -np.inf], [np.inf, 2, 5, np.inf]))
    return float(2.3548 * abs(p[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', type=int, default=30001)
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--calib', default=CALIB)
    ap.add_argument('--spot', default=None, help='x,y (px, full-frame coordinates) of the inset spot')
    ap.add_argument('--mark', type=int, default=1)
    ap.add_argument('--search', type=int, default=60, help='frames searched from --frame for the best isolated spot')
    ap.add_argument('--out', default=os.path.join(REPO, 'out', 'paper_figs_v2', 'si_review-single-frame'))
    a = ap.parse_args()
    from simcode.config import resolve
    from simcode.dataset import Dataset
    from simcode.psf import read_calib, camera_maps
    cp = resolve(a.calib); c = read_calib(cp)
    maps = camera_maps(cp, c)
    gain = float(c['camera']['gain_e_per_adu']); px_nm = c['pixelsize_um'] * 1e3
    x0, y0, s = CROP
    A = Dataset.load(a.run); V = A.select(12.0).numpy(); A = A.numpy()
    zsel = V['pos'][:, 2] if V['pos'].shape[1] > 2 else None
    z_med_um = float(c['focus_um'] + np.nanmedian(zsel) * c['pixelsize_um']) if zsel is not None else float(c['emitter_z_median_um'])

    def frame_rows(fr):
        wa = (A['frame'] <= fr) & (fr < A['frame'] + 6)          # all pass-2 rows (any p) of the windows containing the frame
        w = (V['frame'] <= fr) & (fr < V['frame'] + 6)           # the paper's selection
        return A['pos'][wa, 0], A['pos'][wa, 1], V['pos'][w, 0], V['pos'][w, 1], V['photons'][w]

    def pick(img, xa, ya, xs, ys):
        """(signal, index) of the selected localization with the largest isolated in-frame signal"""
        def frame_signal(xi, yi):        # 3 x 3 px sum around the localization minus the 9 x 9 border median; -inf if not isolated
            jx, jy = int(round(xi)), int(round(yi)); b = img[jy - 4:jy + 5, jx - 4:jx + 5]
            bb = b - np.median(np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]]))
            c3 = bb[3:6, 3:6].sum()
            outer = [bb[i:i + 3, j:j + 3].sum() for i in range(7) for j in range(7) if abs(i - 3) >= 3 or abs(j - 3) >= 3]
            if max(outer) >= 0.3 * c3: return -np.inf             # another 3 x 3 box of the inset above 30 % of it
            dd = np.hypot(xa - xi, ya - yi)
            return -np.inf if ((dd > 1.5) & (dd < 8)).any() else c3   # no other pass-2 row at 1.5-8 px
        sig = np.array([frame_signal(xi, yi) if (x0 + 8 <= xi < x0 + s - 8 and y0 + 8 <= yi < y0 + s - 8) else -np.inf
                        for xi, yi in zip(xs, ys)])
        i = int(np.argmax(sig)) if len(sig) else -1
        return (sig[i], i) if len(sig) else (-np.inf, -1)

    frames = [a.frame] if (a.spot or a.search <= 1) else list(range(a.frame, a.frame + a.search))
    best = None
    for fr in frames:
        im_ = (read_frame(MOVIE, fr) - maps['offset_adu']) * gain       # photo-electrons
        xa, ya, xs_, ys_, ph_ = frame_rows(fr)
        sg, i = (0.0, -1) if a.spot else pick(im_, xa, ya, xs_, ys_)
        if best is None or sg > best[0]: best = (sg, i, fr, im_, xs_, ys_, ph_)
    _, best_i, frame, img, xs, ys, phs = best
    a.frame = frame
    sub = img[y0:y0 + s, x0:x0 + s]
    if a.spot:
        sx, sy = (float(v) for v in a.spot.split(','))
    else:
        best = best_i
        sx, sy = float(xs[best]), float(ys[best])
    # inset box, 9 x 9 px around the spot's pixel
    R = 4; ix, iy = int(round(sx)), int(round(sy))
    ins = img[iy - R:iy + R + 1, ix - R:ix + R + 1]
    bgv = np.median(np.concatenate([ins[0], ins[-1], ins[:, 0], ins[:, -1]]))
    fw_spot = [gauss_fwhm(ins[R] - bgv), gauss_fwhm(ins[:, R] - bgv)]
    fw_psf = psf_fwhm(cp, z_med_um)

    lo, hi = np.percentile(sub, [0.5, 99.8])
    for mark in ([0, 1] if a.mark else [0]):
        fig = plt.figure(figsize=(6.0, 6.0))
        ax = fig.add_axes([0, 0, 1, 1])
        ext = [x0 - 0.5, x0 + s - 0.5, y0 + s - 0.5, y0 - 0.5]
        ax.imshow(sub, cmap='gray', vmin=lo, vmax=hi, extent=ext, interpolation='nearest')
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.axis('off')
        if mark:
            for xi, yi in zip(xs, ys):
                if x0 <= xi < x0 + s and y0 <= yi < y0 + s:
                    ax.add_patch(Circle((xi, yi), 2.0, fill=False, edgecolor='#00d0ff', lw=0.8))
        # box around the spot in the frame
        bx = (ix - R - 0.5, iy - R - 0.5)
        ax.add_patch(Rectangle(bx, 2 * R + 1, 2 * R + 1, fill=False, edgecolor='red', lw=2, ls=(0, (1, 1))))
        # inset at the lower left (or lower right if the spot is there)
        left = sx > x0 + s / 2
        ia = fig.add_axes([0.06 if left else 0.64, 0.08, 0.3, 0.3])
        ia.imshow(ins, cmap='gray', interpolation='nearest', extent=[-R - 0.5, R + 0.5, R + 0.5, -R - 0.5])
        ia.set_xticks([]); ia.set_yticks([])
        for sp in ia.spines.values(): sp.set_edgecolor('red'); sp.set_linewidth(2.5)
        bl, br, bt, bb_ = bx[0], bx[0] + 2 * R + 1, bx[1], bx[1] + 2 * R + 1        # box left / right / top / bottom (data)
        pairs = (((1, 1), (bl, bt)), ((1, 0), (bl, bb_))) if left else (((0, 1), (br, bt)), ((0, 0), (br, bb_)))
        for (xa, ya), (xb, yb) in pairs:
            fig.add_artist(ConnectionPatch((xa, ya), (xb, yb), coordsA=ia.transAxes, coordsB=ax.transData, color='red', lw=1.5,
                                           ls=(0, (1, 1))))
        # 2 um scale bar, lower right (or lower left)
        L = 2000 / px_nm; xb0 = (x0 + s - 6 - L) if left else (x0 + 6)
        ax.add_patch(Rectangle((xb0, y0 + s - 8), L, 2.0, color='#d4c000'))
        out = a.out + ('_locs' if mark else '')
        for ext_ in ('pdf', 'svg', 'png'):
            fig.savefig(f'{out}.{ext_}', dpi=200 if ext_ == 'png' else 300)
        plt.close(fig)

    nloc = int(((xs >= x0) & (xs < x0 + s) & (ys >= y0) & (ys < y0 + s)).sum())
    cap = f"""Caption material: si_review-single-frame (script analysis/paper/si_single_frame.py)

Draft caption:
A single raw frame of the low-density DNA-PAINT data set (main text Fig. 3; frame {a.frame} of 120 000, {s} x {s} px region
analysed in the paper), in photo-electrons (camera offset map subtracted, gain {gain:.4f} e-/ADU). Inset: one
diffraction-limited spot ({2 * R + 1} x {2 * R + 1} px). With 60x magnification and 6.5 um camera pixels the pixel size is
{px_nm:.2f} nm. The PSF (in-situ vectorial PSF fit at the median emitter height, {z_med_um * 1e3:.0f} nm) has a full width at half
maximum of {np.mean(fw_psf):.2f} px ({np.mean(fw_psf) * px_nm:.0f} nm) as sampled by the camera pixels. Scale bar: 2 um.

Facts:
- movie {MOVIE}, frame index {a.frame} (0-based); frame index mod 6 = {a.frame % 6} (cycle of 6 frames = 2 pattern orientations x 3 phase steps).
- display: grey scale, linear, clipped at the 0.5 / 99.8 percentiles of the region ({lo:.1f} / {hi:.1f} e-).
- inset spot at x, y = {sx:.2f}, {sy:.2f} px (full-frame px), the isolated SIMCODE localization (of a window containing this frame) with the
  largest signal in this frame (3 x 3 px sum minus background) and no other
  3 x 3 px box of the 9 x 9 px inset above 30 % of it (paper selection: p >= 0.63, calibrated max(sigma) <= 12 nm), no other pass-2 row
  of these windows (any p) at 1.5-8 px; its photons {float(phs[best]) if not a.spot else float('nan'):.0f}
  (network photons per 6-frame window, PSF-model intensity units).
- PSF FWHM (in-situ PSF model as loaded by analysis/baselines/zpsf.py, pixel-integrated, at z = {z_med_um * 1e3:.0f} nm): x {fw_psf[0]:.2f} px, y {fw_psf[1]:.2f} px
  ({fw_psf[0] * px_nm:.0f} / {fw_psf[1] * px_nm:.0f} nm).
- spot FWHM (1D Gaussian fit to the inset's centre row / column, background = border median): x {fw_spot[0]:.2f} px,
  y {fw_spot[1]:.2f} px (single noisy spot; informative only).
- variant si_review-single-frame_locs.*: cyan circles = the {nloc} selected SIMCODE localizations of the 6 windows that contain
  this frame (window starts every 3 frames, so most emitters appear in two windows).
- calibration {cp}; run {a.run}.
"""
    open(a.out + '_caption.txt', 'w').write(cap)
    # working notes (not caption text): corrections relative to earlier caption versions
    open(a.out + '_notes_internal.txt', 'w').write(
        'Working notes behind si_review-single-frame (internal: corrections relative to earlier versions; not caption text).\n\n'
        '- the earlier caption\'s "approximately 1.3 pixels full-width at half-maximum ... 141 nm" was wrong: 1.3 px is the Gaussian\n'
        '  sigma of the earlier Gaussian simulation PSF (FWHM 2.355 x 1.3 = 3.06 px = 332 nm); replaced by the PSF-model value\n'
        f'  ({np.mean(fw_psf):.2f} px = {np.mean(fw_psf) * px_nm:.0f} nm, in-situ PSF at z = {z_med_um * 1e3:.0f} nm).\n')
    print(cap)


if __name__ == '__main__':
    main()
