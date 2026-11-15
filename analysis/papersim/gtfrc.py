"""Ground-truth FRC and the fidelity diagnostics of a localization set against a known structure (simulations).

Dimension-agnostic: positions [n, D], sigma [n, D] (per-axis reported precision, same unit as the positions), structure
points [m, D]; D = 2 (papersim) or 3 (e.g. a 3D density sweep). All lengths are in one coordinate unit (camera px in
papersim); `unit_nm` converts the reported resolutions to nm.

  gt_frc(pos, gt, lo, hi, pixel, unit_nm)        FRC between the rendered reconstruction and the rendered ground truth (all
        labelling sites), both as histograms on the same fixed field [lo, hi) with the same pixel; FRC per shell of width
        1 / (largest field side), 5-shell moving average, resolution = 1 / (first shell, from shell 3 on, below the
        threshold, default 1/7) -- the estimator of simcode.evaluation.frc.frc on a fixed field (a common Gaussian blur of
        both renders would cancel per shell, so none is applied). Missing coverage (correlated power lost at all
        frequencies), false positives / ghosts (uncorrelated power) and imprecision (high-frequency fall-off) all lower it;
        no matching radius. -> dict(res_nm, freq (1/nm), curve)
  time_to_fidelity(pos, frame, gt, ..., grid, targets)   gt_frc on the rows with frame < N for every N of the grid ->
        resolution vs frames, and the frames needed to reach each target resolution (log-interpolated between grid points;
        None = not reached within the grid)
  gate_pairs(pos, sigma, ref, k)                 all (row, ref point) pairs with Mahalanobis distance <= k under the row's
        own diagonal sigma (the sigma gate used below)
  sigma_jaccard(pos, sigma, gt, k=3)             Jaccard with a sigma gate instead of a fixed radius: a localization is
        correct if a structure point lies within k of its own sigma; a structure point is recovered if it lies within the
        k-sigma ellipse of some localization; J = (correct + recovered) / (n + m) (the weighting of the preprint's
        compute_jaccard_points)
  ghost_fraction(pos, sigma, frame, emitters, k=3)   share of localizations with no emitter that is ON in their own window
        (emitters: dict window start -> [m, D] positions) within k of their own sigma
  segments(gt, length, break_dist)               structure points (ordered along each filament) -> segment id of each point
        (filaments split at jumps > break_dist, then into pieces of `length` along the arc) and the segment arc lengths
  coverage_cv(pos, sigma, gt, seg, seg_len, weight, length, k=3)   coverage uniformity: each localization that passes the
        sigma gate is assigned to its nearest gated structure point and so to a segment; recovered / expected per segment,
        expected = the segment's share of `weight` (e.g. the ground-truth emitter-windows of each point) times all assigned
        localizations; segments shorter than length / 2 dropped. -> CV of the ratio, its Poisson floor sqrt(mean 1 /
        expected) and the excess sqrt(CV^2 - floor^2) (biased rejection shows up as excess)."""
import numpy as np
from scipy.spatial import cKDTree

THRESHOLD = 1 / 7


# --------------------------------------------------------------------------------------------------- rendering / FRC
def render(pos, lo, hi, pixel):
    """histogram of the rows of pos [n, D] on the field [lo, hi) (per-axis sequences) with cubic pixels of size pixel"""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    nb = np.maximum(np.round((hi - lo) / pixel).astype(int), 1)
    return np.histogramdd(np.asarray(pos, float), bins=nb, range=list(zip(lo, lo + nb * pixel)))[0]


def frc_images(a, b, pixel, smooth=5):
    """FRC of two images of the same shape: (shell frequencies (1 / unit), smoothed curve) up to Nyquist"""
    A, B = np.fft.fftn(a), np.fft.fftn(b)
    df = 1.0 / (max(a.shape) * pixel)
    fr = np.sqrt(sum(g ** 2 for g in np.meshgrid(*[np.fft.fftfreq(n, pixel) for n in a.shape], indexing='ij')))
    k = (fr / df).astype(int).ravel(); nmax = int(0.5 / pixel / df)
    num = np.bincount(k, (A * B.conj()).real.ravel(), nmax + 1)[:nmax]
    da = np.bincount(k, (np.abs(A) ** 2).ravel(), nmax + 1)[:nmax]; db = np.bincount(k, (np.abs(B) ** 2).ravel(), nmax + 1)[:nmax]
    c = num / np.sqrt(np.maximum(da * db, 1e-30))
    if smooth > 1: c = np.convolve(c, np.ones(smooth) / smooth, 'same')
    return np.arange(nmax) * df, c


def crossing(freq, curve, threshold=THRESHOLD, start=3):
    """1 / (first frequency from shell `start` on with the curve below threshold); nan if never"""
    below = np.nonzero(curve[start:] < threshold)[0]
    return float('nan') if len(below) == 0 else float(1.0 / freq[below[0] + start])


def gt_frc(pos, gt, lo, hi, pixel, unit_nm=1.0, threshold=THRESHOLD, gt_img=None):
    """FRC(rendered reconstruction, rendered ground truth) -> dict(res_nm, freq (1/nm), curve, n)"""
    g = render(gt, lo, hi, pixel) if gt_img is None else gt_img
    if len(pos) == 0: return dict(res_nm=float('nan'), freq=[], curve=[], n=0)
    freq, c = frc_images(render(pos, lo, hi, pixel), g, pixel)
    freq = freq / unit_nm
    return dict(res_nm=crossing(freq, c, threshold), freq=freq.tolist(), curve=c.tolist(), n=int(len(pos)))


def time_to_fidelity(pos, frame, gt, lo, hi, pixel, unit_nm=1.0, grid=(1000, 2000, 4000, 8000, 15000, 30000, 60000),
                     targets=(40, 30), threshold=THRESHOLD):
    """GT-FRC resolution (nm) on the rows with frame < N for N in grid; frames to reach each target (nm), None = not reached"""
    g = render(gt, lo, hi, pixel)
    res = [gt_frc(pos[frame < N], None, lo, hi, pixel, unit_nm, threshold, gt_img=g)['res_nm'] for N in grid]
    reach = {}
    for t in targets:
        r = np.array(res); ok = np.isfinite(r) & (r <= t)
        if not ok.any(): reach[str(t)] = None; continue
        i = int(np.argmax(ok))
        if i == 0 or not np.isfinite(r[i - 1]): reach[str(t)] = float(grid[i]); continue      # already at the first grid point
        # log-frames interpolation between the bracketing grid points (res decreasing from r[i-1] > t to r[i] <= t)
        u = (r[i - 1] - t) / (r[i - 1] - r[i])
        reach[str(t)] = float(np.exp(np.log(grid[i - 1]) + u * (np.log(grid[i]) - np.log(grid[i - 1]))))
    return dict(frames=list(map(int, grid)), res_nm=res, frames_to=reach)


# --------------------------------------------------------------------------------------------------- sigma gate
def gate_pairs(pos, sigma, ref, k=3.0, tree=None, batch=200000):
    """(row index, ref index, squared Mahalanobis distance) of all pairs with distance <= k, diagonal sigma of the row"""
    pos = np.asarray(pos, float); sigma = np.asarray(sigma, float)
    tree = tree or cKDTree(np.asarray(ref, float)); ref = tree.data
    I, J, D = [], [], []
    for s in range(0, len(pos), batch):
        p = pos[s:s + batch]; sg = sigma[s:s + batch]
        lst = tree.query_ball_point(p, k * sg.max(1), return_sorted=False)
        ln = np.fromiter((len(v) for v in lst), int, len(lst))
        if ln.sum() == 0: continue
        i = np.repeat(np.arange(len(p)), ln); j = np.fromiter((x for v in lst for x in v), int, ln.sum())
        d2 = (((p[i] - ref[j]) / sg[i]) ** 2).sum(1); ok = d2 <= k * k
        I.append(i[ok] + s); J.append(j[ok]); D.append(d2[ok])
    if not I: return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def _valid(pos, sigma):
    return np.isfinite(pos).all(1) & np.isfinite(sigma).all(1) & (sigma > 0).all(1)


def sigma_jaccard(pos, sigma, gt, k=3.0):
    v = _valid(pos, sigma); pos, sigma = pos[v], sigma[v]
    i, j, _ = gate_pairs(pos, sigma, gt, k)
    correct = np.zeros(len(pos), bool); correct[i] = True
    found = np.zeros(len(gt), bool); found[j] = True
    n, m = len(pos), len(gt)
    return dict(jaccard=float((correct.sum() + found.sum()) / max(n + m, 1)), frac_correct=float(correct.mean()) if n else float('nan'),
                frac_gt_recovered=float(found.mean()), k=k, n=int(n))


def ghost_fraction(pos, sigma, frame, emitters, k=3.0):
    """emitters: dict window start -> [m, D] ON-emitter positions of that window"""
    v = _valid(pos, sigma); pos, sigma, frame = pos[v], sigma[v], np.asarray(frame)[v]
    ghost = np.ones(len(pos), bool)
    o = np.argsort(frame, kind='stable'); fs = frame[o]
    cut = np.r_[0, np.nonzero(np.diff(fs))[0] + 1, len(fs)]
    for a, b in zip(cut[:-1], cut[1:]):
        E = emitters.get(int(fs[a]))
        if E is None or len(E) == 0: continue
        ii = o[a:b]
        d2 = (((pos[ii, None, :] - np.asarray(E)[None]) / sigma[ii, None, :]) ** 2).sum(-1)
        ghost[ii] = d2.min(1) > k * k
    return dict(ghost_fraction=float(ghost.mean()) if len(pos) else float('nan'), n=int(len(pos)), k=k)


# --------------------------------------------------------------------------------------------------- coverage
def segments(gt, length, break_dist):
    """gt [m, D] ordered along each filament -> (segment id per point, arc length of each segment)"""
    gt = np.asarray(gt, float)
    step = np.r_[0.0, np.linalg.norm(np.diff(gt, axis=0), axis=1)]
    fil = np.cumsum(step > break_dist); step[step > break_dist] = 0.0
    seg = np.zeros(len(gt), int); seg_len = []; nxt = 0
    for f in np.unique(fil):
        ii = np.nonzero(fil == f)[0]; s = np.cumsum(step[ii])
        q = np.floor(s / length).astype(int)
        seg[ii] = q + nxt
        for u in range(q.max() + 1):
            su = s[q == u]; seg_len.append(float(su.max() - su.min()) if len(su) else 0.0)
        nxt += q.max() + 1
    return seg, np.array(seg_len)


def coverage_cv(pos, sigma, gt, seg, seg_len, weight, length, k=3.0):
    v = _valid(pos, sigma); pos, sigma = pos[v], sigma[v]
    i, j, d2 = gate_pairs(pos, sigma, gt, k)
    o = np.lexsort((d2, i)); i, j = i[o], j[o]
    first = np.r_[True, i[1:] != i[:-1]]; j = j[first]                       # nearest gated structure point per row
    rec = np.bincount(seg[j], minlength=len(seg_len)).astype(float)
    W = np.bincount(seg, np.asarray(weight, float), minlength=len(seg_len))
    keep = (seg_len >= 0.5 * length) & (W > 0)
    if rec[keep].sum() == 0: return dict(cv=float('nan'), poisson_cv=float('nan'), excess_cv=float('nan'), n_segments=int(keep.sum()), n_assigned=0)
    exp = W[keep] * rec[keep].sum() / W[keep].sum(); r = rec[keep] / exp
    cv = float(r.std() / r.mean()); floor = float(np.sqrt(np.mean(1.0 / exp)))
    return dict(cv=cv, poisson_cv=floor, excess_cv=float(np.sqrt(max(cv ** 2 - floor ** 2, 0.0))), n_segments=int(keep.sum()),
                n_assigned=int(len(j)), mean_expected=float(exp.mean()), frac_empty_segments=float(np.mean(rec[keep] == 0)))
