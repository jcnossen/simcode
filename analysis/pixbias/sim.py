"""Pixel bias with ground truth: paper simulation movies (out/papersim/<D>/full: SIMCODE v2 pass 1 / pass 2 chunk files,
classical baselines) and the gate scenes (out/pixbias/gate, pass-1 sc front-end with the m1809 networks).
Per estimate: bias curve <est - GT> vs the GT sub-pixel position (per axis, 10 bins, SE bands), its first-harmonic amplitude
A (nm; e = -A sin(2 pi f_gt) + ..., A > 0 = pull to the pixel centre), by photon bin; sub-pixel histograms of estimate and GT;
leakage of the prior bias into the solve (A_solve / A_prior and the regression slope of e_solve on e_prior); and the
post-hoc CDF equalisation (fit on the estimates, no GT) checked against the GT bias curve.
usage: python analysis/pixbias/sim.py D1 Dexp1 gate   -> out/pixbias/sim_<name>.json, fig_sim_<name>_*.png"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pblib as B
from simcode import strict as SP
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

PS = os.path.join(B.REPO, 'out', 'papersim')
RMAX = 1.0      # px match radius
ISO = 2.0       # px: no other ON GT emitter within this distance of the matched one
NB = 10


def gt_rows(gt, nframes, w, xr, yr, ph_min=100.0):
    """per row: GT position of the nearest ON emitter of its window (>= ph_min window photons, within RMAX, isolated)"""
    import metrics as M
    us = np.unique(w); GW = M.gt_windows(gt, nframes, us)
    tx = np.full(len(w), np.nan); ty = np.full(len(w), np.nan); tph = np.full(len(w), np.nan)
    o = np.argsort(w, kind='stable'); ws = w[o]; st = np.r_[0, np.nonzero(np.diff(ws))[0] + 1, len(ws)]
    for a, b in zip(st[:-1], st[1:]):
        f = ws[a]
        if f not in GW: continue
        gx, gy, gph = GW[f]
        if not len(gx): continue
        ii = o[a:b]
        d = np.hypot(xr[ii, None] - gx[None], yr[ii, None] - gy[None]); j = d.argmin(1); dm = d[np.arange(len(ii)), j]
        dd = np.hypot(gx[:, None] - gx[None], gy[:, None] - gy[None]); np.fill_diagonal(dd, np.inf)
        iso = (dd.min(1) > ISO) & (gph >= ph_min)
        ok = (dm < RMAX) & iso[j]
        tx[ii[ok]] = gx[j[ok]]; ty[ii[ok]] = gy[j[ok]]; tph[ii[ok]] = gph[j[ok]]
    return tx, ty, tph


def bias_curve(e, fg, nb=NB):
    """binned mean error (trim |e| < 0.5 px) vs GT sub-pixel position; harmonic fit e = a sin + b cos + c (unbinned)"""
    k = np.isfinite(e) & (np.abs(e) < 0.5); e = e[k]; fg = fg[k]
    if len(e) < 100: return None
    ib = np.clip(((fg + 0.5) * nb).astype(int), 0, nb - 1)
    mu = np.array([e[ib == i].mean() if (ib == i).sum() > 5 else np.nan for i in range(nb)])
    se = np.array([e[ib == i].std() / np.sqrt(max((ib == i).sum(), 1)) for i in range(nb)])
    X = np.c_[np.sin(2 * np.pi * fg), np.cos(2 * np.pi * fg), np.ones(len(fg))]
    c, *_ = np.linalg.lstsq(X, e, rcond=None)
    r = e - X @ c; cov = np.linalg.inv(X.T @ X) * r.var()
    return dict(n=int(len(e)), mu_nm=(mu * B.PX).tolist(), se_nm=(se * B.PX).tolist(), A_nm=float(-c[0] * B.PX),
                A_se_nm=float(np.sqrt(cov[0, 0]) * B.PX), cos_nm=float(c[1] * B.PX), mean_nm=float(c[2] * B.PX),
                rms_nm=float(np.sqrt(np.mean(e ** 2)) * B.PX), rms_resid_nm=float(np.sqrt(np.mean(r ** 2)) * B.PX))


# ------------------------------------------------------------------ data sets: dict pass -> dict(est -> (x, y)), gt, ph, masks
def papersim(D):
    run = os.path.join(PS, D, 'full'); d = os.path.join(run, 'v2_m1809')
    gt = dict(np.load(os.path.join(PS, D, 'gt.npz'))); nframes = int(gt['nframes'])
    out = {}
    for ps, tag, tab in (('P1', 'est', 'locs_p1_sim.npz'), ('P2', 'est2', 'locs_p2_sim.npz')):
        keys = ['wstart', 'x_hy0', 'y_hy0', 'x_prior', 'y_prior', 'phot', 'p', 'sx_cal', 'sy_cal'] + (['x_det', 'y_det'] if ps == 'P2' else [])
        E = B.load_rows(d, tag, keys); T = np.load(os.path.join(d, tab))
        assert len(T['x']) == len(E['x_hy0']), (len(T['x']), len(E['x_hy0']))
        if ps == 'P1':
            E['x_det'], E['y_det'] = B.join_p1_det(B.load_pass1(d), E)
        w = E['wstart'].astype(np.int64)
        xr = np.where(np.isfinite(E['x_det']), E['x_det'], E['x_prior']); yr = np.where(np.isfinite(E['y_det']), E['y_det'], E['y_prior'])
        tx, ty, tph = gt_rows(gt, nframes, w, xr, yr)
        est = dict(det=(E['x_det'], E['y_det']), prior=(E['x_prior'], E['y_prior']), solve=(E['x_hy0'], E['y_hy0']))
        sel = dict(p63=T['good'].astype(bool), sig12=SP.select(T, 'main'))
        out[ps] = dict(est={k: (np.asarray(a, np.float64), np.asarray(b, np.float64)) for k, (a, b) in est.items()},
                       tx=tx, ty=ty, ph=E['phot'].astype(np.float64), sel=sel)
    # baselines (rows inside the FOV with sigma < 0.15 px), 6-frame windows at their own starts
    import frc_fair
    for name, kind, path in (('smlm', 'smlm', f'{run}/base/smlm/smlm.npy'), ('simflux', 'sf', f'{run}/base/sf/sf.hdf5'),
                             ('decode', 'decode', f'{run}/decode/locs.npz')):
        if not os.path.exists(path): continue
        T = frc_fair.load(kind, path)
        k = np.isfinite(T['x']) & (np.maximum(T['sx'], T['sy']) < 0.15) & (T['f'] + 6 <= nframes)
        if k.sum() > 400000:
            k[np.nonzero(k)[0][np.random.default_rng(0).permutation(k.sum())[400000:]]] = False
        x, y, f = T['x'][k], T['y'][k], T['f'][k]
        tx, ty, tph = gt_rows(gt, nframes, f, x, y)
        out[name] = dict(est=dict(loc=(x, y)), tx=tx, ty=ty, ph=T['photons'][k], sel=dict(s015=np.ones(len(x), bool)))
    return out


def gate(scen):
    out = {}
    X = {k: [] for k in ('det', 'prior', 'solve', 'tx', 'ty', 'ph', 'p', 's')}
    for seed in (301, 302):
        fn = os.path.join(B.OUT, 'gate', f'{scen}__s{seed}_m1809.npz')
        if not os.path.exists(fn): continue
        z = np.load(fn); off = seed * 100000
        gf = z['gt_fov'].astype(np.int64); gx, gy = z['gt_x'].astype(np.float64), z['gt_y'].astype(np.float64)
        b = z['sc_b'].astype(np.int64)
        xd, yd = z['sc_x_det'].astype(np.float64), z['sc_y_det'].astype(np.float64)
        tx = np.full(len(b), np.nan); ty = np.full(len(b), np.nan)
        go = np.argsort(gf, kind='stable'); gs = gf[go]
        for fv in np.unique(b):
            ii = np.nonzero(b == fv)[0]; a0, a1 = np.searchsorted(gs, [fv, fv + 1]); jj = go[a0:a1]
            if not len(jj): continue
            vis = (z['gt_target'][jj] > 0)
            d = np.hypot(xd[ii, None] - gx[jj][None], yd[ii, None] - gy[jj][None]); j = d.argmin(1); dm = d[np.arange(len(ii)), j]
            dd = np.hypot(gx[jj][:, None] - gx[jj][None], gy[jj][:, None] - gy[jj][None]); np.fill_diagonal(dd, np.inf)
            iso = (dd.min(1) > ISO) & vis
            ok = (dm < RMAX) & iso[j]
            tx[ii[ok]] = gx[jj][j[ok]]; ty[ii[ok]] = gy[jj][j[ok]]
        X['det'].append((xd, yd)); X['prior'].append((z['sc_x_prior'], z['sc_y_prior'])); X['solve'].append((z['sc_x_sc'], z['sc_y_sc']))
        X['tx'].append(tx); X['ty'].append(ty); X['ph'].append(z['sc_phot']); X['p'].append(z['sc_score']); X['s'].append(z['sc_scal_sc'].max(1))
    est = {k: (np.concatenate([a[0] for a in X[k]]).astype(np.float64), np.concatenate([a[1] for a in X[k]]).astype(np.float64)) for k in ('det', 'prior', 'solve')}
    p = np.concatenate(X['p']); s = np.concatenate(X['s'])
    out['P1'] = dict(est=est, tx=np.concatenate(X['tx']), ty=np.concatenate(X['ty']), ph=np.concatenate(X['ph']).astype(np.float64),
                     sel=dict(p63=p >= 0.63, sig12=(p >= 0.63) & (s * B.PX <= 12)))
    return out


# ------------------------------------------------------------------ analysis
def analyse(DS):
    R = {}
    for ps, G in DS.items():
        R[ps] = {}
        for sn, m0 in G['sel'].items():
            m = m0 & np.isfinite(G['tx'])
            r = {}
            bins = B.PH_BINS if ps.startswith('P') else tuple(np.r_[np.quantile(G['ph'][m], [0, .25, .5, .75]), np.inf])
            for en, (x, y) in G['est'].items():
                mm = m & np.isfinite(x)
                rr = dict(n=int(mm.sum()))
                for ax, v, t in (('x', x, G['tx']), ('y', y, G['ty'])):
                    e = v[mm] - t[mm]; fg = B.sub(t[mm])
                    rr[ax] = dict(all=bias_curve(e, fg), ph={lab: bias_curve(e[b], fg[b]) for lab, b in B.ph_bin_masks(G['ph'][mm], bins)},
                                  hist_est=B.ustats(B.sub(v[mm])), hist_gt=B.ustats(fg), hist_est_all=B.ustats(B.sub(v[m0 & np.isfinite(v)])))
                    # post-hoc CDF equalisation, fit on ALL rows of this selection (no GT), per photon bin
                    va = v[m0]; groups = [b for _, b in B.ph_bin_masks(G['ph'][m0], bins)]
                    vc = np.full(len(v), np.nan); vc[m0] = B.cdf_equalise(va, groups)
                    ec = vc[mm] - t[mm]
                    rr[ax]['eq'] = dict(all=bias_curve(ec, fg), ph={lab: bias_curve(ec[b], fg[b]) for lab, b in B.ph_bin_masks(G['ph'][mm], bins)},
                                        hist_est=B.ustats(B.sub(vc[mm])))
                r[en] = rr
            # leakage prior -> solve
            if 'prior' in G['est'] and 'solve' in G['est']:
                lk = {}
                for ax, i, t in (('x', 0, G['tx']), ('y', 1, G['ty'])):
                    ep = G['est']['prior'][i] - t; es = G['est']['solve'][i] - t
                    k = m & (np.abs(ep) < 0.5) & (np.abs(es) < 0.5)
                    lk[ax] = {}
                    for lab, b in B.ph_bin_masks(G['ph'][k], bins):
                        if b.sum() < 200: continue
                        a_, c_ = np.polyfit(ep[k][b], es[k][b], 1)
                        lk[ax][lab] = dict(slope=float(a_), corr=float(np.corrcoef(ep[k][b], es[k][b])[0, 1]), n=int(b.sum()))
                r['leak'] = lk
            R[ps][sn] = r
    return R


def fig_bias(name, R, fn):
    rows = [(ps, sn) for ps in R for sn in R[ps]]
    fig, axs = plt.subplots(len(rows), 4, figsize=(16, 3.0 * len(rows)), squeeze=False)
    fc = (np.arange(NB) + 0.5) / NB - 0.5
    for i, (ps, sn) in enumerate(rows):
        r = R[ps][sn]; ests = [e for e in r if e != 'leak']
        for j, ax_ in enumerate(('x', 'y')):
            ax = axs[i, j]
            for c, en in enumerate(ests):
                bc = r[en][ax_]['all']
                if bc is None: continue
                mu, se = np.array(bc['mu_nm']), np.array(bc['se_nm'])
                ax.plot(fc, mu - bc['mean_nm'], color=f'C{c}', label=f'{en} A={bc["A_nm"]:+.2f}±{bc["A_se_nm"]:.2f}')
                ax.fill_between(fc, mu - bc['mean_nm'] - 1.96 * se, mu - bc['mean_nm'] + 1.96 * se, color=f'C{c}', alpha=0.2)
                bq = r[en][ax_]['eq']['all']
                if bq is not None: ax.plot(fc, np.array(bq['mu_nm']) - bq['mean_nm'], color=f'C{c}', ls=':', lw=1)
            ax.axhline(0, color='k', lw=0.5); ax.set_xlabel(f'GT sub-pixel {ax_} (px)'); ax.set_ylabel('<est - GT> - mean (nm)')
            ax.set_title(f'{name} {ps} [{sn}] {ax_}  (dotted: after CDF equalisation)', fontsize=8); ax.legend(fontsize=6)
        ax = axs[i, 2]
        for c, en in enumerate(ests):
            for ax_, ls in (('x', '-'), ('y', '--')):
                P = r[en][ax_]['ph']; labs = [l for l in P if P[l] is not None]
                if not labs: continue
                ax.errorbar(np.arange(len(labs)), [P[l]['A_nm'] for l in labs], yerr=[1.96 * P[l]['A_se_nm'] for l in labs], color=f'C{c}', ls=ls, marker='o', ms=3,
                            label=f'{en} {ax_}')
            ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels(labs, fontsize=6, rotation=20)
        ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('A (nm), >0 = pull to pixel centre'); ax.set_title('first-harmonic bias by photon bin', fontsize=8); ax.legend(fontsize=6, ncol=2)
        ax = axs[i, 3]
        for c, en in enumerate(ests):
            # histogram of the estimate's sub-pixel coordinate (matched rows) vs GT
            pass
        ax.set_axis_off()
        txt = [f'{en}: hist |c1| x {r[en]["x"]["hist_est"]["c1_abs"]:.4f} (GT {r[en]["x"]["hist_gt"]["c1_abs"]:.4f}) y {r[en]["y"]["hist_est"]["c1_abs"]:.4f}; '
               f'implied A {r[en]["x"]["hist_est"]["A_nm"]:+.2f}/{r[en]["y"]["hist_est"]["A_nm"]:+.2f} nm; after eq. A {r[en]["x"]["eq"]["all"]["A_nm"]:+.2f}/{r[en]["y"]["eq"]["all"]["A_nm"]:+.2f}; '
               f'rms {r[en]["x"]["all"]["rms_nm"]:.1f}->{r[en]["x"]["eq"]["all"]["rms_nm"]:.1f} nm'
               for en in ests if r[en]['x']['all'] is not None]
        if 'leak' in r:
            txt.append('leak slope e_solve~e_prior x: ' + ', '.join(f'{l}:{v["slope"]:.2f}' for l, v in r['leak']['x'].items()))
        ax.text(0, 1, '\n'.join(txt), fontsize=6, va='top', family='monospace', wrap=True)
    fig.tight_layout(); fig.savefig(fn, dpi=90); plt.close(fig)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        if name == 'gate':
            for sc in ('iso', 'rs05qm'):
                DS = gate(sc); R = analyse(DS); nm = f'gate_{sc}'
                json.dump(R, open(os.path.join(B.OUT, f'sim_{nm}.json'), 'w'), indent=1)
                fig_bias(nm, R, os.path.join(B.OUT, f'fig_sim_{nm}.png'))
                print(nm, 'done', flush=True)
        else:
            DS = papersim(name); R = analyse(DS)
            json.dump(R, open(os.path.join(B.OUT, f'sim_{name}.json'), 'w'), indent=1)
            fig_bias(name, R, os.path.join(B.OUT, f'fig_sim_{name}.png'))
            print(name, 'done', flush=True)
