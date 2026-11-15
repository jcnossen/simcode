"""Excess intensity variance (the c terms), 2026-11-12. The automatic estimate theta = (s, c, c_a) of simcode/solve/autoc.py (unchanged code) on
  real   the saved pass-1 outputs of the production runs: 1809_2 (m1809 set), 2.5nM_2 (m25 set), 20nM_2 (psf3dmix set, the existing run):
         3 repeats on disjoint chunk subsets (5 chunks each, <= 120000 rows, spread over the movie) -> mean +- sd
  sim    (a) papersim D1 through the whole pipeline (constant emitter brightness, matched PSF, pattern estimated from the data): control
         (b) gate scene rs05qm dumps (gate_dump.py, pass-1 head of the m1809 set, seed 301): exact model without any emitter fluctuation (ctl),
             with a known per-frame fluctuation (f10 / f20: sd 0.10 / 0.20), a known per-angle factor (a10: sd 0.10), both (f10a10), the method's
             own model error only (mis: PSF + halo jitter off... PSF / camera / pattern-estimate error on, emitter fluctuation off), and the full
             mismatch scene (full: frame 0.10, step 0.07, angle 0.14 + PSF / camera / pattern errors); all rows; ctl / f10 / mis also on 3 FOV groups -> sd
  figure the direct picture: variance of the per-frame residual normalised by the network-only prediction, r0 = e_t / sqrt(V0_tt) (V0 = residual
         covariance for theta = (1, 0, 0)), against the model intensity mu_t; lines: 1 (network covariance as is), s0 (best single scale with
         c = c_a = 0), and the fitted theta's prediction median(V_theta,tt / V0_tt). Second row: the same for the angle contrast (sum angle 1 -
         sum angle 2) against the window model photons (this is what separates c_a from c).
usage: CUDA_VISIBLE_DEVICES=2 python analysis/ablation_cov/autoc_est.py [real,sim,fig] -> out/ablation_cov/autoc_est.json, fig_excess_variance.png"""
import glob, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis', 'calib'))
from simcode.solve import autoc as AC
from simcode import config
from simcode.util import proctitle
from scipy.optimize import minimize_scalar
proctitle.set_title('covabl')
OUT = os.path.join(REPO, 'out', 'ablation_cov'); dev = 'cuda'
REAL = {'1809_2': ('out/paper2/1809_2/v2_matched', 'calib/calib_sc_hd_m1809.json'), '2.5nM_2': ('out/paper2/2.5nM_2/v2_matched', 'calib/calib_sc_hd_m25.json'),
        '20nM_2': ('out/paper2/20nM_2/v2', 'calib/calib_sc_hd_psf3dmix.json')}
SIMP = {'D1 (pipeline, constant brightness)': ('out/papersim/D1/full/v2_m1809', 'calib/calib_sc_hd_m1809.json')}
GATE = [('ctl', 'exact model, no fluctuation', (0, 0)), ('f10', 'per-frame sd 0.10', (0.10, 0)), ('f20', 'per-frame sd 0.20', (0.20, 0)), ('a10', 'per-angle sd 0.10', (0, 0.10)),
        ('f10a10', 'per-frame 0.10 + per-angle 0.10', (0.10, 0.10)), ('mis', 'PSF / camera / pattern error only, no fluctuation', (0, 0)),
        ('full', 'full mismatch (frame 0.10, step 0.07, angle 0.14)', (float(np.hypot(0.10, 0.07)), 0.14))]
mute = lambda *a, **k: None
NREP = 3          # repeats on disjoint chunk subsets (one estimate takes ~5 GPU-minutes)


def gate_rows(tag, fovsel=None):
    D = np.load(f'{OUT}/fe_ctl/rs05qm__s301_s1_{tag}.npz')
    k = D['sc_score'] >= 0.5
    iso = AC.iso_flag(D['sc_b'], D['sc_x_det'], D['sc_y_det'])       # among all stored candidates (p >= 0.2)
    if fovsel is not None: k &= fovsel(D['sc_b'].astype(int))
    k = np.nonzero(k)[0]; tt = AC.tt
    o = dict(I=tt(D['sc_I'][k]), L=tt(D['sc_L'][k]), x=tt(D['sc_x'][k]), y=tt(D['sc_y'][k]), lsxy=tt(D['sc_lsxy'][k]), x_det=tt(D['sc_x_det'][k]), y_det=tt(D['sc_y_det'][k]))
    return [(o, tt(D['patt_est']), torch.tensor(D['sc_b'][k].astype(np.int64), device=dev), D['sc_I'][k].sum(1), iso[k])]


def est(rows, cal0):
    th, hist, S = AC.estimate(rows, cal0, n_iter=3, log=mute)
    return th, hist[-1]['n_rows'], S


def s_only(S):
    B = AC.snr_bins(S)
    def f(ls):
        rF, rC = AC.bin_medians(S, (float(np.exp(ls)), 0.0, 0.0), B)
        return sum((m - 0.5) ** 2 for _, _, m in rF if np.isfinite(m)) + sum((m - 0.5) ** 2 for _, _, m in rC if np.isfinite(m))
    r = minimize_scalar(f, bounds=(np.log(0.3), np.log(10.0)), method='bounded'); return float(np.exp(r.x)), float(r.fun)


def picture(S, th):
    """binned normalised residual variance (robust) and the predictions"""
    rv = lambda x: float((1.4826 * x.abs().median()) ** 2)
    V0 = AC.resid_cov(S, (1.0, 0.0, 0.0)); Vt = AC.resid_cov(S, th)
    v0 = V0.diagonal(dim1=1, dim2=2).clamp(min=1e-6); vt = Vt.diagonal(dim1=1, dim2=2).clamp(min=1e-6)
    r0 = S['e'] / v0.sqrt(); mu = S['mu']
    ed = np.geomspace(8, 1500, 15); F = []
    for lo, hi in zip(ed[:-1], ed[1:]):
        m = (mu >= lo) & (mu < hi)
        if int(m.sum()) < 300: continue
        F.append(dict(mu=float(mu[m].median()), n=int(m.sum()), emp=rv(r0[m]), pred=float((vt[m] / v0[m]).median()), rel_sd=float(1.4826 * (S['e'][m] / mu[m]).abs().median()),
                      mean_rel=float((S['e'][m] / mu[m]).median())))
    w = AC.wvec(S); q = lambda V: (w[:, None, :] @ V @ w[:, :, None])[:, 0, 0].clamp(min=1e-6)
    h0 = (w * S['e']).sum(1) / q(V0).sqrt(); N = mu.sum(1); rat = q(Vt) / q(V0)
    ed = np.geomspace(150, 6000, 12); Cc = []
    for lo, hi in zip(ed[:-1], ed[1:]):
        m = (N >= lo) & (N < hi)
        if int(m.sum()) < 300: continue
        Cc.append(dict(N=float(N[m].median()), n=int(m.sum()), emp=rv(h0[m]), pred=float(rat[m].median()), rel_sd=float(1.4826 * ((w * S['e']).sum(1)[m] / N[m]).abs().median()),
                       mean_rel=float(((w * S['e']).sum(1)[m] / N[m]).median())))
    s0, J0 = s_only(S); Jt = None
    return dict(frame=F, contrast=Cc, s_only=s0, J_s_only=J0)


def main():
    what = (sys.argv[1] if len(sys.argv) > 1 else 'real,sim,fig').split(',')
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None        # movie names / gate tags of this process (parallel streams)
    if 'fig' in what or 'merge' in what:                                # merge the per-stream files
        R = {}
        for f in sorted(glob.glob(f'{OUT}/autoc_est_*.json')): R.update(json.load(open(f)))
        order = list(REAL) + list(SIMP) + ['gate:' + g[0] for g in GATE]
        R = {k: R[k] for k in order if k in R}
        json.dump(R, open(f'{OUT}/autoc_est.json', 'w'), indent=1, default=float)
        if 'fig' in what: fig(R)
        return
    fn = f'{OUT}/autoc_est_{(only or ["all"])[0]}.json'; R = json.load(open(fn)) if os.path.exists(fn) else {}
    save = lambda: json.dump(R, open(fn, 'w'), indent=1, default=float)
    if 'real' in what or 'simpipe' in what:
        for name, (d, c0) in list(REAL.items()) * ('real' in what) + list(SIMP.items()) * ('sim' in what or 'simpipe' in what):
            if only and name.split(' ')[0] not in only: continue
            d = os.path.join(REPO, d); cal0 = json.load(open(config.resolve(c0)))
            nch = len(glob.glob(d + '/pass1_*.npz')); reps = []
            for r in range(NREP):
                ch = sorted(set((np.arange(5) * (nch // 5) + r * max(nch // 20, 1)).clip(0, nch - 1).tolist()))
                rows = AC.pipeline_rows(d, chunks=ch, nmax=120000, seed=r)
                th, n, S = est(rows, cal0); reps.append(dict(chunks=ch, theta=th, n_rows=n))
                print(f'{name} rep {r} chunks {ch}: s {th[0]:.3f} c {th[1]:.3f} c_a {th[2]:.3f} (isolated non-kinetic rows {n})', flush=True)
                if r == 0: pic = picture(S, th); pic['theta'] = th
                del rows, S; torch.cuda.empty_cache()
            T = np.array([q['theta'] for q in reps])
            R[name] = dict(dir=d, cal0=c0, reps=reps, mean=T.mean(0).tolist(), sd=T.std(0, ddof=1).tolist(), picture=pic)
            print(f'== {name}: s {T[:, 0].mean():.3f}+-{T[:, 0].std(ddof=1):.3f}  c {T[:, 1].mean():.3f}+-{T[:, 1].std(ddof=1):.3f}  c_a {T[:, 2].mean():.3f}+-{T[:, 2].std(ddof=1):.3f} | s-only fit s0 {pic["s_only"]:.2f}', flush=True)
            save()
    if 'sim' in what:
        cal0 = json.load(open(config.resolve('calib/calib_sc_hd_m1809.json')))
        for tag, desc, truth in GATE:
            if (only and tag not in only) or 'gate:' + tag in R: continue
            th, n, S = est(gate_rows(tag), cal0); pic = picture(S, th); pic['theta'] = th
            reps = [est(gate_rows(tag, lambda b, g=g: b % 3 == g), cal0)[0] for g in range(3)] if tag in ('ctl', 'f10', 'mis') else [th, th, th]
            T = np.array(reps)
            R['gate:' + tag] = dict(desc=desc, truth=truth, theta=th, n_rows=n, reps=reps, sd=T.std(0, ddof=1).tolist(), picture=pic)
            print(f'gate {tag:7s} ({desc}): truth c {truth[0]:.3f} c_a {truth[1]:.3f} | all rows: s {th[0]:.3f} c {th[1]:.3f} c_a {th[2]:.3f} (n {n}) | 3 FOV groups: c {T[:, 1].mean():.3f}+-{T[:, 1].std(ddof=1):.3f} '
                  f'c_a {T[:, 2].mean():.3f}+-{T[:, 2].std(ddof=1):.3f} | s-only s0 {pic["s_only"]:.2f}', flush=True)
            save()


def fig(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.titlesize': 9})
    cols = [(k, f'Real {k}') for k in REAL if k in R] + [(k, t) for k, t in (('gate:ctl', 'Sim control: no fluctuation'), ('gate:f10', 'Sim: per-frame sd 0.10 put in')) if k in R]
    fig_, ax = plt.subplots(2, len(cols), figsize=(3.5 * len(cols), 6.6), gridspec_kw=dict(wspace=0.3, hspace=0.42), squeeze=False)
    for j, (k, title) in enumerate(cols):
        P = R[k]['picture']; th = P['theta']
        for i, (key, xk, xl) in enumerate((('frame', 'mu', 'model intensity of the frame (photons)'), ('contrast', 'N', 'model window photons'))):
            a = ax[i, j]; Q = P[key]; x = [q[xk] for q in Q]
            a.axhline(1.0, color=MUTED, lw=1.5, ls=':', label='c = 0: network covariance as is')
            a.axhline(P['s_only'], color=MUTED, lw=1.5, ls='--', label=f'c = 0, best scale s = {P["s_only"]:.2f}')
            a.plot(x, [q['pred'] for q in Q], color=ORANGE, lw=2, label=f'fitted: s {th[0]:.2f}, c {th[1]:.3f}, c_a {th[2]:.3f}')
            a.plot(x, [q['emp'] for q in Q], color=BLUE, lw=0, marker='o', ms=6.5, markeredgecolor=SURF, markeredgewidth=1.2, label='measured')
            a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(xl); a.grid(color=GRID, lw=0.6, which='major')
            a.set_ylim(0.4, max(12, 1.5 * max(q['emp'] for q in Q)))
            if j == 0: a.set_ylabel('variance of the per-frame residual /\nnetwork-only prediction' if i == 0 else 'variance of the angle contrast /\nnetwork-only prediction')
            a.set_title(title + (': per frame' if i == 0 else ': angle 1 - angle 2'), loc='left'); a.legend(frameon=False, fontsize=7, loc='upper left')
    fig_.savefig(f'{OUT}/fig_excess_variance.png', dpi=140, bbox_inches='tight'); print('saved fig')


if __name__ == '__main__':
    main()
