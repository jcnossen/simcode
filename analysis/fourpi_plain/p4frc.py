"""Half-split FRC of the Cell11 localizations (SIMCODE plain 4Pi run(s) and the published table of Li & Ries) on the NPC
field: two halves of disjoint frames in alternating blocks of B frames, 2D histogram renders (px nm) of the common
lateral field, FRC over rings, resolution at the 1/7 threshold (first crossing after the curve has been above it,
smoothed over 5 rings). A random per-localization split (not fair: the same on-event lands in both halves) is reported
for reference.

  python analysis/fourpi_plain/p4frc.py --runs name:<run dir> [...] --out <dir> --data <extracted Zenodo folder> [--blocks 1 10 100 500] [--px 5]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from compare_published import load_ours, load_theirs, sel, match, NF


def frc_curve(xa, ya, xb, yb, ext, px):
    x0, x1, y0, y1 = ext
    n = int(min(x1 - x0, y1 - y0) / px) // 2 * 2
    rng = ((y0, y0 + n * px), (x0, x0 + n * px))
    A, _, _ = np.histogram2d(ya, xa, bins=(n, n), range=rng)
    B, _, _ = np.histogram2d(yb, xb, bins=(n, n), range=rng)
    w = np.outer(np.hanning(n), np.hanning(n)) ** 0.5            # mild apodization (edge)
    FA, FB = np.fft.fftshift(np.fft.fft2(A * w)), np.fft.fftshift(np.fft.fft2(B * w))
    k = np.arange(n) - n // 2
    R = np.hypot(k[:, None], k[None, :]).astype(int)
    nr = n // 2
    num = np.bincount(R.ravel(), (FA * np.conj(FB)).real.ravel(), minlength=nr)[:nr]
    da = np.bincount(R.ravel(), (np.abs(FA) ** 2).ravel(), minlength=nr)[:nr]
    db = np.bincount(R.ravel(), (np.abs(FB) ** 2).ravel(), minlength=nr)[:nr]
    frc = num / np.sqrt(da * db + 1e-30)
    q = np.arange(nr) / (n * px)                                  # cycles / nm
    cnt = np.bincount(R.ravel(), minlength=nr)[:nr]
    return q, frc, cnt


def resolution(q, frc, thr=1 / 7, smooth=5):
    f = np.convolve(frc, np.ones(smooth) / smooth, mode='same')
    above = np.nonzero(f > thr)[0]
    above = above[above > 2]
    if len(above) == 0: return np.nan
    i = above[0]
    while i + 1 < len(f) and f[i + 1] > thr: i += 1
    if i + 1 >= len(f): return np.nan
    t = (f[i] - thr) / (f[i] - f[i + 1])
    qq = q[i] + t * (q[i + 1] - q[i])
    return 1.0 / qq if qq > 0 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--blocks', type=int, nargs='+', default=[1, 10, 100, 500])
    p.add_argument('--px', type=float, default=5.0)
    p.add_argument('--nmin', type=float, default=500.0)
    p.add_argument('--sig_max', type=float, default=25.0)
    p.add_argument('--zref_max', type=float, default=120.0)
    p.add_argument('--chi2_max', type=float, default=1e9, help='ours: chi2/dof cut (the final comparison: none)')
    p.add_argument('--matched', type=int, default=1, help='also the matched subsets (same emitters and frames in both tables)')
    p.add_argument('--files', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--data', required=True, help='folder of the extracted Zenodo data (their published tables)')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw, pub = load_theirs(a.data)
    tmax = NF * len(a.files)
    pub = sel(pub, pub['t'] < tmax)
    sets = {'published': pub}
    for spec in a.runs:
        nm, run = spec.split(':', 1)
        O = load_ours(run, a.files)
        keep = (O['chi2'] < a.chi2_max) & (np.maximum(O['sx'], O['sy']) < a.sig_max) & (O['N'] >= a.nmin) & ~(np.abs(O['dzref']) > a.zref_max)
        Os = sel(O, keep)
        ia, ib = match(Os, pub, 300.0)
        dxy = (float(np.median(pub['x'][ib] - Os['x'][ia])), float(np.median(pub['y'][ib] - Os['y'][ia])))
        Os['x'] = Os['x'] + dxy[0]; Os['y'] = Os['y'] + dxy[1]
        sets[nm] = Os
        if a.matched:   # the same emitters / frames in both tables: localizations matched one-to-one within 100 nm
            ia, ib = match(Os, pub, 100.0)
            sets[f'{nm} matched'] = sel(Os, ia)
            sets[f'published matched ({nm})'] = sel(pub, ib)
    # common field: the central part covered by every set (2-98 % quantiles of the published table, intersected)
    lo = np.max([[np.percentile(d['x'], 2), np.percentile(d['y'], 2)] for d in sets.values()], 0)
    hi = np.min([[np.percentile(d['x'], 98), np.percentile(d['y'], 98)] for d in sets.values()], 0)
    ext = (lo[0], hi[0], lo[1], hi[1])
    R = dict(field_nm=[float(v) for v in ext], px_nm=a.px, sets={})
    curves = {}
    for nm, d in sets.items():
        m = (d['x'] >= ext[0]) & (d['x'] < ext[1]) & (d['y'] >= ext[2]) & (d['y'] < ext[3])
        x, y, t = d['x'][m], d['y'][m], d['t'][m]
        area = (ext[1] - ext[0]) * (ext[3] - ext[2]) / 1e6
        r = dict(n=int(m.sum()), density_per_um2=float(m.sum() / area), splits={})
        for B in a.blocks + ['random']:
            if B == 'random':
                h = np.random.default_rng(0).random(len(x)) < 0.5
            else:
                h = (t // B) % 2 == 0
            q, f, cnt = frc_curve(x[h], y[h], x[~h], y[~h], ext, a.px)
            res = resolution(q, f)
            r['splits'][str(B)] = dict(res_nm=float(res), n_half=[int(h.sum()), int((~h).sum())])
            curves[(nm, str(B))] = (q, f)
        R['sets'][nm] = r
        print(nm, json.dumps(r), flush=True)
    json.dump(R, open(os.path.join(a.out, 'frc.json'), 'w'), indent=1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cols = ['#2a78d6', '#eb6834', '#1a9e77', '#7a5195']
    splits = [str(b) for b in a.blocks] + ['random']
    fig, ax = plt.subplots(1, len(splits), figsize=(3.2 * len(splits), 3), sharey=True)
    for k, s in enumerate(splits):
        for i, nm in enumerate(sets):
            q, f = curves[(nm, s)]
            fs = np.convolve(f, np.ones(5) / 5, mode='same')
            ax[k].plot(q * 1000, fs, color=cols[i % 4], lw=1.5, label=f'{nm} ({R["sets"][nm]["splits"][s]["res_nm"]:.0f} nm)')
        ax[k].axhline(1 / 7, color='0.5', lw=1, ls='--')
        ax[k].set_xlim(0, 1000 / (2 * a.px)); ax[k].set_ylim(-0.1, 1.0)
        ax[k].set_title(f'split: {"random locs (not fair)" if s == "random" else f"alternating {s}-frame blocks"}', fontsize=8)
        ax[k].set_xlabel('spatial frequency (1/um)', fontsize=8); ax[k].legend(fontsize=6.5, frameon=False)
    ax[0].set_ylabel('FRC (5-ring smoothed)', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'frc.png'), dpi=130)


if __name__ == '__main__':
    main()
