"""Shared pieces of the pixel-bias (pixel-locking) study: sub-pixel coordinate, uniformity statistics, loaders, plotting.
Convention: pixel centres at integers; sub-pixel coordinate f = x - round(x) in [-0.5, 0.5)."""
import os, sys, glob
import numpy as np
from scipy import stats as SS

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, 'analysis', 'paper'), os.path.join(REPO, 'analysis', 'papersim')):
    if p not in sys.path: sys.path.insert(0, p)
OUT = os.path.join(REPO, 'out', 'pixbias')
PX = 108.33
PH_BINS = (0, 500, 1000, 2000, 4000, np.inf)     # network window photons (label units)


def sub(x):
    x = np.asarray(x, np.float64)
    f = x - np.round(x)
    f[f >= 0.5] -= 1.0
    return f


def ustats(f, nb=20, s=None):
    """uniformity statistics of a sub-pixel sample f in [-0.5, 0.5): n, chi2/dof + p (nb bins), KS D + p, first two Fourier
    coefficients c_k = E exp(2 pi i k f) (|c_k|, Re c_k: > 0 = density peaks at the pixel centre), noise floor 1 / sqrt(n),
    and the implied bias amplitude A (nm) of a bias b(f) = -A sin(2 pi f) (A > 0 = pull towards the pixel centre):
    |c_1| = pi A (small A); with s (per-row sigma, px) also corrected for the noise attenuation exp(-2 pi^2 s^2)."""
    f = np.asarray(f, np.float64); f = f[np.isfinite(f)]; n = len(f)
    if n < 50: return dict(n=n)
    h, _ = np.histogram(f, nb, (-0.5, 0.5)); e = n / nb
    chi2 = float(((h - e) ** 2 / e).sum())
    ks = SS.kstest(f + 0.5, 'uniform')
    c1 = np.mean(np.exp(2j * np.pi * f)); c2 = np.mean(np.exp(4j * np.pi * f))
    r = dict(n=n, chi2_dof=chi2 / (nb - 1), chi2_p=float(SS.chi2.sf(chi2, nb - 1)), ks_D=float(ks.statistic), ks_p=float(ks.pvalue),
             c1_abs=float(abs(c1)), c1_re=float(c1.real), c1_im=float(c1.imag), c2_abs=float(abs(c2)), c2_re=float(c2.real),
             noise=float(1 / np.sqrt(n)), A_nm=float(np.sign(c1.real) * abs(c1) / np.pi * PX))
    if s is not None:
        s = np.asarray(s, np.float64); s = s[np.isfinite(s)]
        att = float(np.mean(np.exp(-2 * np.pi ** 2 * np.clip(s, 0, 0.5) ** 2))) if len(s) else 1.0
        r['att'] = att; r['A_nm_att'] = r['A_nm'] / max(att, 1e-3)
    return r


def ph_bin_masks(ph, bins=PH_BINS):
    return [(f'{bins[i]:g}-{bins[i + 1]:g}', (ph >= bins[i]) & (ph < bins[i + 1])) for i in range(len(bins) - 1)]


def cdf_equalise(x, groups=None, nq=2001):
    """post-hoc correction (a): per group (e.g. photon bin) map the sub-pixel coordinate through its own empirical CDF,
    x' = round(x) + F(f) - 0.5 (fit on the data itself, no ground truth). Returns x' (float64)."""
    x = np.asarray(x, np.float64); f = sub(x); xc = x.copy()
    gs = [np.ones(len(x), bool)] if groups is None else groups
    for g in gs:
        if g.sum() < 100: continue
        q = np.quantile(f[g], np.linspace(0, 1, nq))
        F = np.interp(f[g], q, np.linspace(0, 1, nq))
        xc[g] = (x[g] - f[g]) + F - 0.5
    return xc


def hist2d(ax, fx, fy, nb=25, lim=0.25, title=''):
    h, _, _ = np.histogram2d(fx, fy, nb, [[-0.5, 0.5], [-0.5, 0.5]])
    h = h / max(h.mean(), 1e-12)
    im = ax.imshow(h.T, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5), cmap='RdBu_r', vmin=1 - lim, vmax=1 + lim)
    ax.set_title(title, fontsize=8); ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([-0.5, 0, 0.5])
    return im


def hist1d(ax, f, nb=40, label='', color=None):
    h, e = np.histogram(f, nb, (-0.5, 0.5)); h = h / max(h.mean(), 1e-12)
    ax.step(0.5 * (e[1:] + e[:-1]), h, where='mid', label=label, color=color, lw=1)


def load_rows(d, tag, keys):
    fs = sorted(glob.glob(f'{d}/pass2_{tag}_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
    Q = {k: [] for k in keys}
    for f in fs:
        z = np.load(f)
        for k in keys:
            if k in z.files: Q[k].append(z[k])
    return {k: np.concatenate(v) for k, v in Q.items() if v}


def load_pass1(d, keys=('wstart', 'x', 'y', 'x_det', 'y_det', 'p', 'phot', 'sx', 'sy')):
    fs = sorted(glob.glob(f'{d}/pass1_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
    Q = {k: [] for k in keys}
    for f in fs:
        z = np.load(f)
        for k in keys: Q[k].append(z[k])
    return {k: np.concatenate(v) for k, v in Q.items()}


def join_p1_det(P1, E):
    """attach the pass-1 detector position to the pass-1 solve rows (pass2_est = pass-1 rows, x_prior = pass1 x exactly)"""
    import pandas as pd
    a = pd.DataFrame(dict(w=P1['wstart'], x=P1['x'].astype(np.float32), y=P1['y'].astype(np.float32), xd=P1['x_det'], yd=P1['y_det']))
    a = a.drop_duplicates(['w', 'x', 'y'])
    b = pd.DataFrame(dict(w=E['wstart'], x=E['x_prior'].astype(np.float32), y=E['y_prior'].astype(np.float32), i=np.arange(len(E['wstart']))))
    m = b.merge(a, on=['w', 'x', 'y'], how='left')
    m = m.sort_values('i')
    return m['xd'].to_numpy(np.float64), m['yd'].to_numpy(np.float64)
