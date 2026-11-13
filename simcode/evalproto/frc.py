"""
Fourier ring correlation of two localization sets: the one implementation used by the evaluation, the paper figures and
the 4Pi scripts.

  frc(xa, ya, xb, yb, px=5.0, units='px', px_nm=108.33, return_curve=False)
      the reference estimator: both sets are rendered as histograms on a square grid of px-nm pixels that covers both sets,
      FRC per integer ring of the 2D FFT, 5-ring moving average, resolution = n px / (first ring >= 3 below 1/7).
      units 'px': coordinates in camera pixels of px_nm nm, grid from floor(min) to ceil(max) (camera px);
      units 'nm': coordinates in nm, grid from the smallest coordinate with one spare pixel (the 4Pi figures).
      Returns the resolution in nm (nan if the curve never drops below 1/7); with return_curve also the ring frequencies
      (1/nm) and the smoothed curve up to Nyquist.
  frc_halves(x, y, f, blk=300, ...)   the two sets = alternating blocks of blk frames of one localization set
  frc_hann(xa, ya, xb, yb, ext_a, ext_b, pix=4.0)
      variant for a rectangular field (nm): square-root Hann window, rings of width 1 / (largest side), half-ring frequency
      centres, crossing only counted above 1/300 nm. Gives different numbers than frc(); it is the estimator of the 4Pi
      showcase evaluation and is kept so that those numbers do not change.
  frc_fov(x, y, half, x0, x1, px_nm=108.33, pix_nm=5.0)
      variant on a fixed square field [x0, x1) camera px with 300 radial bins up to Nyquist, crossing counted above
      1/500 nm (analysis/hd_compare/score.py); kept for the same reason.
"""
import numpy as np

PX_NM = 108.33
THRESHOLD = 1 / 7


def _ring_curve(ha, hb):
    """histograms [n, n] -> 5-ring smoothed FRC per integer ring"""
    n = ha.shape[0]
    A, B = np.fft.fftshift(np.fft.fft2(ha)), np.fft.fftshift(np.fft.fft2(hb))
    yy, xx = np.indices(A.shape) - n // 2; rr = np.hypot(xx, yy).astype(int)
    num = np.bincount(rr.ravel(), (A * B.conj()).real.ravel()); da = np.bincount(rr.ravel(), (np.abs(A) ** 2).ravel())
    db = np.bincount(rr.ravel(), (np.abs(B) ** 2).ravel())
    return np.convolve(num / np.sqrt(da * db + 1e-12), np.ones(5) / 5, 'same')


def frc(xa, ya, xb, yb, px=5.0, units='px', px_nm=PX_NM, return_curve=False):
    if units == 'px':
        s = px_nm / px
        lo = np.floor(min(xa.min(), ya.min(), xb.min(), yb.min())); hi = np.ceil(max(xa.max(), ya.max(), xb.max(), yb.max()))
        n = int((hi - lo) * s); n += n % 2
        rng = [[lo, lo + n / s], [lo, lo + n / s]]
    elif units == 'nm':
        lo = min(xa.min(), xb.min(), ya.min(), yb.min()); hi = max(xa.max(), xb.max(), ya.max(), yb.max())
        n = int((hi - lo) / px) + 2; n += n % 2
        rng = [[lo, lo + n * px], [lo, lo + n * px]]
    else:
        raise ValueError(units)
    h = lambda x, y: np.histogram2d(y, x, bins=n, range=rng)[0]
    c = _ring_curve(h(xa, ya), h(xb, yb))
    below = np.nonzero((c[3:n // 2] < THRESHOLD))[0]
    res = float('nan') if len(below) == 0 else float(n * px / (below[0] + 3))
    if return_curve:
        return res, np.arange(n // 2) / (n * px), c[:n // 2]
    return res


def frc_halves(x, y, f, blk=300, **kw):
    """FRC between the localizations of the even and the odd blocks of blk frames (f = frame of each localization)"""
    h = (f // blk) % 2 == 0
    return frc(x[h], y[h], x[~h], y[~h], **kw)


def frc_hann(xa, ya, xb, yb, ext_a, ext_b, pix=4.0):
    """-> (freq 1/nm, unsmoothed frc, resolution nm); coordinates in nm inside [0, ext_a) x [0, ext_b)"""
    na, nb = int(ext_a / pix), int(ext_b / pix)
    def img(x, y):
        ok = (x >= 0) & (x < ext_a) & (y >= 0) & (y < ext_b)
        I = np.zeros((nb, na)); np.add.at(I, ((y[ok] / pix).astype(int), (x[ok] / pix).astype(int)), 1); return I
    win = np.outer(np.hanning(nb), np.hanning(na)) ** 0.5
    A = np.fft.fft2(img(xa, ya) * win); B = np.fft.fft2(img(xb, yb) * win)
    fy = np.fft.fftfreq(nb, pix); fx = np.fft.fftfreq(na, pix)
    fr = np.hypot(fy[:, None], fx[None])
    df = 1.0 / (max(na, nb) * pix)
    k = (fr / df).astype(int).ravel(); nmax = int(0.5 / pix / df)
    num = np.bincount(k, (A * B.conj()).real.ravel(), nmax + 1)[:nmax]
    da = np.bincount(k, (np.abs(A) ** 2).ravel(), nmax + 1)[:nmax]; db = np.bincount(k, (np.abs(B) ** 2).ravel(), nmax + 1)[:nmax]
    c = num / np.sqrt(np.maximum(da * db, 1e-30))
    f = (np.arange(nmax) + 0.5) * df
    cs = np.convolve(c, np.ones(5) / 5, 'same')
    lo = np.nonzero((cs < THRESHOLD) & (f > 1 / 300))[0]
    res = float(1 / f[lo[0]]) if len(lo) else float('nan')
    return f, c, res


def frc_fov(x, y, half, x0, x1, px_nm=PX_NM, pix_nm=5.0):
    """FRC between the rows with half True and False on the fixed field [x0, x1) camera px -> resolution nm"""
    zoom = px_nm / pix_nm; W = int((x1 - x0) * zoom)
    yy, xx = np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(W), indexing='ij')
    nbins = 300; rbin = np.minimum((np.sqrt(xx ** 2 + yy ** 2) / 0.5 * nbins).astype(int), nbins)
    ims = []
    for m in (half, ~half):
        h, _, _ = np.histogram2d((y[m] - x0) * zoom, (x[m] - x0) * zoom, bins=(W, W), range=[[0, W], [0, W]])
        ims.append(np.fft.fft2(h))
    num = np.bincount(rbin.ravel(), np.real(ims[0] * np.conj(ims[1])).ravel(), nbins + 1)
    den = np.sqrt(np.bincount(rbin.ravel(), (np.abs(ims[0]) ** 2).ravel(), nbins + 1) * np.bincount(rbin.ravel(), (np.abs(ims[1]) ** 2).ravel(), nbins + 1))
    c = np.convolve((num / np.maximum(den, 1e-12))[:nbins], np.ones(5) / 5, mode='same')
    fq = (np.arange(nbins) + 0.5) / nbins * 0.5 / pix_nm
    below = np.nonzero((c < THRESHOLD) & (fq > 1 / 500))[0]
    return float(1 / fq[below[0]]) if len(below) else np.nan
