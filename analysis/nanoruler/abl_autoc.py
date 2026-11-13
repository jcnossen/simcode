"""Nanoruler ablation, part C: the automatic estimate theta = (s, c, c_a) of simcode2/autoc.py (unchanged estimator) on a pipeline run of
this setup, and the residual-variance-versus-intensity picture (= analysis/ablation_cov/autoc_est.py for this setup).

Differences to autoc.pipeline_rows / autoc_est.py: ONLY cycle-aligned windows (wstart % 6 == 0) are used (the cache of the real movie is
stored reordered within each cycle, other windows are not six consecutive acquisition frames); repeats on NREP disjoint chunk subsets
(chunks r, r + NREP, ...; <= 120000 rows each) -> mean, sd over repeats, se = sd / sqrt(NREP); an optional pattern file replaces the
run's pattern_est.pt (part D). Start calibration cal0 = the sim-sigma calibration of the nr80 head (its kap_I2 bins are part of Sigma_net).

usage: CUDA_VISIBLE_DEVICES=g python analysis/nanoruler/abl_autoc.py est <set> [--pattern file.pt --name tag] [--nrep 4]
             -> out/nanoruler/ablation/autoc_<set>[_<tag>].json
       CUDA_VISIBLE_DEVICES=g python analysis/nanoruler/abl_autoc.py branch <set> [--pattern file.pt --name tag]   (bright-branch fit, see branch())
       python analysis/nanoruler/abl_autoc.py fig <name1,name2,..> [out.png]     (names = json suffixes) -> fig_excess_variance.png, tab_autoc.md"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A


def rows(d, chunks, nmax=120000, seed=0, pattern=None):
    """autoc.pipeline_rows restricted to cycle-aligned windows"""
    import torch
    from simcode2 import autoc as AC
    from simcode2.pipeline.common import pattern_frames
    fs = sorted(glob.glob(f'{d}/pass1_*.npz'), key=lambda f: int(f.split('_')[-1][:-4])); fs = [fs[i] for i in chunks if i < len(fs)]
    Pt = torch.load(pattern or f'{d}/pattern_est.pt', weights_only=False)
    if 'pipeline' in Pt: Pt = Pt['pipeline']
    rng = np.random.default_rng(seed); per = nmax // len(fs); out = []; tt = AC.tt; T = AC.T
    for f in fs:
        R = np.load(f); X0, Y0 = int(R['X0']), int(R['Y0'])
        iso_all = AC.iso_flag(R['wstart'], R['x_det'], R['y_det'])
        keep = np.nonzero(R['out'].astype(bool) & (R['wstart'] % 6 == 0))[0]
        if len(keep) > per: keep = np.sort(rng.choice(keep, per, replace=False))
        ws = R['wstart'][keep]; uw, inv = np.unique(ws, return_inverse=True)
        PW = pattern_frames(Pt, (uw[:, None] + np.arange(T)[None]).reshape(-1), X0, Y0, AC.dev).reshape(len(uw), T, 5)
        o = dict(I=tt(R['I'][keep]), L=tt(R['L'][keep]), x=tt(R['x'][keep] - X0), y=tt(R['y'][keep] - Y0),
                 lsxy=torch.log(tt(np.c_[R['sx'][keep], R['sy'][keep]])), x_det=tt(R['x_det'][keep] - X0), y_det=tt(R['y_det'][keep] - Y0))
        out.append((o, PW, torch.tensor(inv, device=AC.dev), R['I'][keep].sum(1), iso_all[keep]))
    return out


def rows_p2(head, d, chunks, nmax=120000, seed=0, pattern=None):
    """the same for the pass-2 head outputs captured by abl_capture.py (isolation among the stored pass-2 candidates)"""
    import torch
    from simcode2 import autoc as AC
    from simcode2.pipeline.common import pattern_frames
    fs = sorted(glob.glob(f'{head}/head_*.npz'), key=lambda f: int(f.split('_')[-1][:-4])); fs = [fs[i] for i in chunks if i < len(fs)]
    Pt = torch.load(pattern or f'{d}/pattern_est.pt', weights_only=False)
    if 'pipeline' in Pt: Pt = Pt['pipeline']
    rng = np.random.default_rng(seed); per = nmax // len(fs); out = []; tt = AC.tt; T = AC.T
    for f in fs:
        H = np.load(f); X0, Y0 = int(H['X0']), int(H['Y0']); ws = H['wstart'].astype(np.int64)
        iso_all = AC.iso_flag(ws, H['o_x_det'], H['o_y_det'])
        keep = np.nonzero(ws % 6 == 0)[0]
        if len(keep) > per: keep = np.sort(rng.choice(keep, per, replace=False))
        uw, inv = np.unique(ws[keep], return_inverse=True)
        PW = pattern_frames(Pt, (uw[:, None] + np.arange(T)[None]).reshape(-1), X0, Y0, AC.dev).reshape(len(uw), T, 5)
        o = {k: tt(H['o_' + k][keep]) for k in ('I', 'L', 'x', 'y', 'lsxy', 'x_det', 'y_det')}
        out.append((o, PW, torch.tensor(inv, device=AC.dev), H['o_I'][keep].sum(1), iso_all[keep]))
    return out


def get_rows(S, chunks, seed=0, pattern=None):
    return rows_p2(S['head'], S['run'], chunks, seed=seed, pattern=pattern) if S['tag'] == 'est2' else rows(S['run'], chunks, seed=seed, pattern=pattern)


def estimate(setn, pattern=None, name='', nrep=4):
    import torch
    import autoc_est as AE
    A.title()
    S = A.SETS[setn]; d = S['run']; cal0 = json.load(open(S['prod']))       # start calibration = the sim-sigma calibration of this pass
    nch = len(glob.glob(d + '/pass1_*.npz')); reps = []; pic = None
    for r in range(nrep):
        ch = list(range(r, nch, nrep))
        rw = get_rows(S, ch, seed=r, pattern=pattern)
        th, n, Sres = AE.est(rw, cal0); reps.append(dict(chunks=ch, theta=th, n_rows=n))
        print(f'{setn} {name} rep {r} chunks {ch}: s {th[0]:.3f} c {th[1]:.3f} c_a {th[2]:.3f} (isolated non-kinetic rows {n})', flush=True)
        if r == 0: pic = AE.picture(Sres, th); pic['theta'] = th
        del rw, Sres; torch.cuda.empty_cache()
    T = np.array([q['theta'] for q in reps]); sd = T.std(0, ddof=1) if nrep > 1 else np.zeros(3)
    R = dict(set=setn, dir=d, pattern=pattern or f'{d}/pattern_est.pt', cal0=S['prod'], reps=reps, mean=T.mean(0).tolist(), sd=sd.tolist(),
             se=(sd / np.sqrt(nrep)).tolist(), picture=pic, note='cycle-aligned windows only; simcode2.autoc.estimate, 3 fixed-point iterations')
    fn = f'{A.OUT}/autoc_{setn}{"_" + name if name else ""}.json'; A.jdump(R, fn)
    print(f'== {setn} {name}: s {T[:, 0].mean():.3f} +- {sd[0]:.3f}  c {T[:, 1].mean():.3f} +- {sd[1]:.3f}  c_a {T[:, 2].mean():.3f} +- {sd[2]:.3f} (sd over {nrep} '
          f'disjoint chunk subsets) | best single scale with c = 0: {pic["s_only"]:.2f} -> {fn}', flush=True)
    return R


def branch(setn, pattern=None, name='', mu_lo=40.0):
    """The bright branch of the residual-variance picture, without the frames near the fringe minima: per bin of the model intensity mu
    the robust variance of e / sqrt(V0) (V0 = residual covariance for theta = (1, 0, 0)) and the median of mu^2 / V0; then a straight-line
    fit over the bins with mu >= mu_lo:  emp = s_b + c_eff^2 * median(mu^2 / V0)  (c_eff = the per-frame relative excess, c and c_a
    together), bootstrap over the rows (200 resamples). Also the median relative residual e / mu per bin (a bias of the pattern model).
    One solve at the fitted theta of autoc_<set>[_<name>].json, chunks of repeat 0."""
    import torch
    from simcode2 import autoc as AC
    A.title()
    assert not (pattern and not name), 'give --name with --pattern (the result is stored in autoc_<set>_<name>.json)'
    S = A.SETS[setn]; fn = f'{A.OUT}/autoc_{setn}{"_" + name if name else ""}.json'; R = json.load(open(fn)); th = R['reps'][0]['theta']
    cal0 = json.load(open(S['prod']))
    rw = get_rows(S, R['reps'][0]['chunks'], seed=0, pattern=pattern)
    Sr = AC.solve_rows(rw, AC.with_theta(cal0, th))
    V0 = AC.resid_cov(Sr, (1.0, 0.0, 0.0)); v0 = V0.diagonal(dim1=1, dim2=2).clamp(min=1e-6)
    r0 = (Sr['e'] / v0.sqrt()).flatten().cpu().numpy(); mu = Sr['mu'].flatten().cpu().numpy(); g = (Sr['mu'] ** 2 / v0).flatten().cpu().numpy()
    rel = (Sr['e'] / Sr['mu'].clamp(min=1.0)).flatten().cpu().numpy()
    ed = np.geomspace(8, 1500, 15); rv = lambda x: float((1.4826 * np.median(np.abs(x))) ** 2)
    def bins(idx):
        out = []
        for lo, hi in zip(ed[:-1], ed[1:]):
            m = idx[(mu[idx] >= lo) & (mu[idx] < hi)]
            if len(m) < 300: continue
            out.append((float(np.median(mu[m])), rv(r0[m]), float(np.median(g[m])), float(np.median(rel[m])), len(m)))
        return np.array(out)
    def fit(B):
        k = B[:, 0] >= mu_lo
        if k.sum() < 3: return np.nan, np.nan
        a = np.polyfit(B[k, 2], B[k, 1], 1); return float(a[1]), float(np.sign(a[0]) * np.sqrt(abs(a[0])))
    n = len(mu); B = bins(np.arange(n)); s_b, c_eff = fit(B); rng = np.random.default_rng(0); nrow = n // 6
    bs = []
    for _ in range(200):
        ii = rng.integers(0, nrow, nrow); bs.append(fit(bins((ii[:, None] * 6 + np.arange(6)[None]).ravel())))
    bs = np.array(bs)
    R['branch'] = dict(mu_lo=mu_lo, bins=[dict(mu=b[0], emp=b[1], mu2_over_v0=b[2], median_rel_resid=b[3], n=int(b[4])) for b in B], s_b=s_b, c_eff=c_eff,
                       s_b_sd=float(np.nanstd(bs[:, 0])), c_eff_sd=float(np.nanstd(bs[:, 1])), note='emp = s_b + c_eff^2 mu^2 / V0 over bins with mu >= mu_lo; sd = row bootstrap')
    A.jdump(R, fn)
    print(f'{setn} {name}: bright branch (mu >= {mu_lo:.0f}): s_b {s_b:.2f} +- {R["branch"]["s_b_sd"]:.2f}, c_eff {c_eff:.3f} +- {R["branch"]["c_eff_sd"]:.3f}')
    print('   mu       ' + ' '.join(f'{b[0]:6.0f}' for b in B)); print('   emp      ' + ' '.join(f'{b[1]:6.2f}' for b in B))
    print('   mu2/V0   ' + ' '.join(f'{b[2]:6.1f}' for b in B)); print('   med e/mu ' + ' '.join(f'{b[3]:+6.3f}' for b in B))


TITLES = {'p1': 'Real movie, pass 1', 'p2': 'Real movie, pass 2', 'sima_p1': 'Sim nr_a (fluctuation put in)', 'simc0_p1': 'Sim nr_c0 (no fluctuation)', 'sima_p2': 'Sim nr_a, pass 2', 'simc0_p2': 'Sim nr_c0, pass 2',
          'p1_ps': 'Real: per-step pattern', 'p1_psd': 'Real: per-step phases', 'p1_shd': 'Real: in-situ shared fit', 'p1_k60': 'Real: 60-frame drift blocks', 'p1_k30': 'Real: 30-frame drift blocks'}


def fig(names, out=None, titles=None, table=True):
    titles = titles or TITLES
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker
    BLUE, ORANGE, INK, MUTED, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.titlesize': 9, 'text.color': INK, 'axes.labelcolor': INK})
    Rs = [(n, json.load(open(f'{A.OUT}/autoc_{n}.json'))) for n in names if os.path.exists(f'{A.OUT}/autoc_{n}.json')]
    nrow = 3 if any('branch' in R for _, R in Rs) else 2
    fig_, ax = plt.subplots(nrow, len(Rs), figsize=(3.7 * len(Rs), 3.3 * nrow), gridspec_kw=dict(wspace=0.3, hspace=0.42), squeeze=False)
    for j, (n, R) in enumerate(Rs):
        P = R['picture']; th = P['theta']; title = (titles or {}).get(n, n)
        for i, (key, xk, xl) in enumerate((('frame', 'mu', 'model intensity of the frame (photons)'), ('contrast', 'N', 'model window photons'))):
            a = ax[i, j]; Q = P[key]; x = [q[xk] for q in Q]
            a.axhline(1.0, color=MUTED, lw=1.5, ls=':', label='c = 0: network covariance as is')
            a.axhline(P['s_only'], color=MUTED, lw=1.5, ls='--', label=f'c = 0, best scale s = {P["s_only"]:.2f}')
            a.plot(x, [q['pred'] for q in Q], color=ORANGE, lw=2, label=f'fitted: s {th[0]:.2f}, c {th[1]:.3f}, c_a {th[2]:.3f}')
            a.plot(x, [q['emp'] for q in Q], color=BLUE, lw=0, marker='o', ms=6.5, markeredgecolor=SURF, markeredgewidth=1.2, label='measured')
            a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(xl); a.grid(color=GRID, lw=0.6, which='major'); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            a.set_ylim(0.4, max(12, 1.5 * max(q['emp'] for q in Q)))
            if j == 0: a.set_ylabel('variance of the per-frame residual /\nnetwork-only prediction' if i == 0 else 'variance of the orientation contrast /\nnetwork-only prediction')
            a.set_title(title + ': per frame' if i == 0 else 'orientation 1 - orientation 2', loc='left'); a.legend(frameon=False, fontsize=7, loc='upper left')
        if nrow == 3:       # third row: bias of the model intensity (median relative residual), and the bright-branch fit
            a = ax[2, j]; a.axhline(0.0, color=MUTED, lw=1.5, ls=':')
            if 'branch' in R:
                Bq = R['branch']; a.plot([q['mu'] for q in Bq['bins']], [q['median_rel_resid'] for q in Bq['bins']], color=BLUE, lw=2, marker='o', ms=6.5, markeredgecolor=SURF, markeredgewidth=1.2)
                a.text(0.97, 0.93, f'frames with mu >= {Bq["mu_lo"]:.0f}:\nrelative excess {Bq["c_eff"]:.3f} +- {Bq["c_eff_sd"]:.3f}\nscale {Bq["s_b"]:.2f} +- {Bq["s_b_sd"]:.2f}', transform=a.transAxes, ha='right', va='top', fontsize=7.5, color=INK)
            a.set_xscale('log'); a.set_xlabel('model intensity of the frame (photons)'); a.grid(color=GRID, lw=0.6, which='major'); a.set_title('bias of the model intensity', loc='left')
            if j == 0: a.set_ylabel('median of (network intensity - model) / model')
    out = out or f'{A.OUT}/fig_excess_variance.png'; fig_.savefig(out, dpi=140, bbox_inches='tight'); print('saved', out)
    if not table: return
    L = ['**Automatic estimate theta = (s, c, c_a), simcode2/autoc.py, cycle-aligned windows.** mean +- sd over disjoint chunk subsets (se = sd / sqrt(repeats)).', '',
         '| data | pattern | repeats | s | c | c_a | isolated non-kinetic rows per repeat | best single scale with c = 0 | bright branch (mu >= 40): scale | relative excess c_eff |', '|---|---|---|---|---|---|---|---|---|---|']
    for n, R in Rs:
        m, s = R['mean'], R['sd']
        L.append(f'| {(titles or {}).get(n, n)} | {os.path.relpath(R["pattern"], A.REPO)} | {len(R["reps"])} | {m[0]:.2f} +- {s[0]:.2f} | {m[1]:.3f} +- {s[1]:.3f} | {m[2]:.3f} +- {s[2]:.3f} | '
                 f'{int(np.mean([q["n_rows"] for q in R["reps"]]))} | {R["picture"]["s_only"]:.2f} | '
                 + (f'{R["branch"]["s_b"]:.2f} +- {R["branch"]["s_b_sd"]:.2f} | {R["branch"]["c_eff"]:.3f} +- {R["branch"]["c_eff_sd"]:.3f} |' if 'branch' in R else '- | - |'))
    open(os.path.splitext(out)[0].replace('fig_excess_variance', 'tab_autoc') + '.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'branch':
        g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
        branch(a[1], g('--pattern'), g('--name', '')); sys.exit()
    if a[0] == 'est':
        g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
        estimate(a[1], g('--pattern'), g('--name', ''), int(g('--nrep', 4)))
    else:
        fig(a[1].split(','), a[2] if len(a) > 2 else None, table='--notable' not in a)
