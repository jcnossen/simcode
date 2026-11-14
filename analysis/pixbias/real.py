"""Pixel locking on the real paper movies: sub-pixel histograms + uniformity statistics of every SIMCODE v2 position estimate
(pass-1 / pass-2 detector, head prior, explicit solve) by selection and photon bin, and of the classical baselines.
usage: python analysis/pixbias/real.py 1809_2 [2.5nM_2 ...]   -> out/pixbias/real_<ds>.json, fig_real_<ds>.png"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pblib as B
from simcode import strict as SP
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

RUN = {'1809_2': 'v2_matched', '2.5nM_2': 'v2_matched', '20nM_2': 'v2_m25'}


def v2_sets(ds):
    d = os.path.join(B.REPO, 'out', 'paper2', ds, RUN[ds])
    S = {}
    for ps, tag, tab in (('P1', 'est', 'locs_p1.npz'), ('P2', 'est2', 'locs_p2.npz')):
        keys = ['wstart', 'x_hy0', 'y_hy0', 'x_prior', 'y_prior', 'phot', 'p', 'sx_cal', 'sy_cal'] + (['x_det', 'y_det'] if ps == 'P2' else [])
        E = B.load_rows(d, tag, keys)
        T = np.load(os.path.join(d, tab))
        assert len(T['x']) == len(E['x_hy0']) and np.allclose(T['x'][:1000], E['x_hy0'][:1000]), (ps, len(T['x']), len(E['x_hy0']))
        if ps == 'P1':
            P1 = B.load_pass1(d)
            E['x_det'], E['y_det'] = B.join_p1_det(P1, E)
            print(ds, 'P1 det join: matched', np.isfinite(E['x_det']).mean(), flush=True)
            del P1
        sel = dict(p30=np.ones(len(T['x']), bool), p63=T['good'].astype(bool), sig12=SP.select(T, 'main'))
        sx = np.asarray(T['sx'], np.float64)
        for est, (kx, ky) in dict(det=('x_det', 'y_det'), prior=('x_prior', 'y_prior'), solve=('x_hy0', 'y_hy0')).items():
            S[f'{ps}_{est}'] = dict(x=E[kx].astype(np.float64), y=E[ky].astype(np.float64), ph=E['phot'].astype(np.float64), sel=sel, s=sx)
    return S


def base_sets(ds):
    import frc_fair
    d = os.path.join(B.REPO, 'out', 'paper2', ds)
    S = {}
    for name, kind, path in (('smlm', 'smlm', f'{d}/smlm/smlm.npy'), ('simflux', 'sf', f'{d}/sf/sf.hdf5'), ('decode', 'decode', f'{d}/decode/locs.npz')):
        if not os.path.exists(path): continue
        T = frc_fair.load(kind, path)
        ins = (T['x'] >= 5) & (T['x'] < 115) & (T['y'] >= 5) & (T['y'] < 115)
        filt = ins & (np.maximum(T['sx'], T['sy']) < 0.15)
        S[name] = dict(x=T['x'], y=T['y'], ph=T['photons'], sel=dict(all=ins, s015=filt), s=T['sx'])
    return S


def analyse(S):
    R = {}
    for name, D in S.items():
        R[name] = {}
        for sn, m in D['sel'].items():
            m = m & np.isfinite(D['x']) & np.isfinite(D['y'])
            fx, fy = B.sub(D['x'][m]), B.sub(D['y'][m])
            r = dict(x=B.ustats(fx, s=D['s'][m]), y=B.ustats(fy, s=D['s'][m]), ph={})
            if name.startswith('P'):
                bins = B.PH_BINS
            else:
                q = np.quantile(D['ph'][m], [0, .25, .5, .75, 1]); q[-1] = np.inf; bins = tuple(q)
            for lab, b in B.ph_bin_masks(D['ph'][m], bins):
                r['ph'][lab] = dict(x=B.ustats(fx[b]), y=B.ustats(fy[b]))
            R[name][sn] = r
    return R


def figure(ds, S, R, fn):
    names = list(S)
    fig, axs = plt.subplots(len(names), 4, figsize=(15, 2.6 * len(names)))
    for i, n in enumerate(names):
        D = S[n]; sk = 'sig12' if 'sig12' in D['sel'] else 's015'; sa = 'p63' if 'p63' in D['sel'] else 'all'
        m = D['sel'][sk] & np.isfinite(D['x'])
        fx, fy = B.sub(D['x'][m]), B.sub(D['y'][m])
        im = B.hist2d(axs[i, 0], fx, fy, title=f'{n} [{sk}] n={m.sum()}')
        if i == 0: fig.colorbar(im, ax=axs[i, 0], fraction=0.046)
        for j, (f_, lab) in enumerate(((fx, 'fx'), (fy, 'fy'))):
            ax = axs[i, 1 + j]
            ma = D['sel'][sa] & np.isfinite(D['x'])
            B.hist1d(ax, B.sub((D['x'] if lab == 'fx' else D['y'])[ma]), label=sa, color='C0')
            B.hist1d(ax, f_, label=sk, color='C3')
            ax.axhline(1, color='k', lw=0.5); ax.set_ylim(0.6, 1.4); ax.set_title(f'{n} {lab}: |c1| {sa} {R[n][sa][lab[1]]["c1_abs"]:.4f}, {sk} {R[n][sk][lab[1]]["c1_abs"]:.4f}', fontsize=8)
            if i == 0: ax.legend(fontsize=7)
        ax = axs[i, 3]
        for sn, c in ((sa, 'C0'), (sk, 'C3')):
            P = R[n][sn]['ph']; labs = list(P)
            for ax_, ls in (('x', '-'), ('y', '--')):
                v = [P[l][ax_].get('A_nm', np.nan) for l in labs]; e = [P[l][ax_].get('noise', np.nan) / np.pi * B.PX for l in labs]
                ax.errorbar(np.arange(len(labs)), v, yerr=e, color=c, ls=ls, marker='o', ms=3, label=f'{sn} {ax_}')
        ax.axhline(0, color='k', lw=0.5); ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels(labs, fontsize=6, rotation=20)
        ax.set_ylabel('implied bias amp. A (nm)\n(sign(Re c1)|c1|/pi px)', fontsize=7)
        if i == 0: ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f'{ds}: sub-pixel distributions (pixel centre = 0); 2D: density / mean, colour 0.75-1.25', fontsize=10)
    fig.tight_layout(); fig.savefig(fn, dpi=90); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(B.OUT, exist_ok=True)
    for ds in sys.argv[1:]:
        S = v2_sets(ds); S.update(base_sets(ds))
        R = analyse(S)
        json.dump(R, open(os.path.join(B.OUT, f'real_{ds}.json'), 'w'), indent=1)
        figure(ds, S, R, os.path.join(B.OUT, f'fig_real_{ds}.png'))
        for n in R:
            for sn in R[n]:
                r = R[n][sn]
                print(f'{ds} {n:12s} {sn:6s} n={r["x"]["n"]:9d}  x: |c1| {r["x"]["c1_abs"]:.4f} A {r["x"]["A_nm"]:+.2f} nm chi2/dof {r["x"]["chi2_dof"]:9.1f} '
                      f'| y: |c1| {r["y"]["c1_abs"]:.4f} A {r["y"]["A_nm"]:+.2f} nm chi2/dof {r["y"]["chi2_dof"]:9.1f}  (floor {r["x"]["noise"]:.4f})', flush=True)
