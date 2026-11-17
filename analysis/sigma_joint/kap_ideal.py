"""sigma study (2026-11-17): what would photon-count-dependent factors look like if the double counting of the prior and the
intensities (they come from one network with correlated errors) were handled by factors in an independent solve, as kap_p /
kap_I2 / sig_ph did until 2026-11-17? Per emitter and axis, the 2-estimate picture of the joint-covariance networks:
  sigma_p   the prior sd (network)
  sigma_q   the sd of the position implied by the intensities alone: Fisher of the intensity block at the solved theta (x, y, B;
            non-kinetic rows), with the marginal intensity covariance C C^T + L L^T + the nuisance c term of the solve
  rho       the correlation of the two position errors implied by the joint covariance (rho_a = (G C)_aa / sigma_q,a, G the
            intensity-only estimator's gain)
  r = sigma_p / sigma_q
BLUE weight on the prior w = (1 - rho r) / (1 + r^2 - 2 rho r); the prior-variance factor that makes the independent solve give that
weight alpha = (1 - rho / r) / (1 - rho r) (undefined / negative when rho >= r: the optimal weight on the intensities is <= 0);
true BLUE variance sigma_p^2 (1 - rho^2) / (1 + r^2 - 2 rho r) vs the variance an independent solve with prior variance alpha sigma_p^2
claims, alpha sigma_p^2 / (1 + alpha r^2) -> sigma factor sf = sqrt(true / claimed) (sf0: the same vs the plain independent claim,
alpha = 1). This is the DOUBLE-COUNTING part only (1D per axis, linearized); the brightness-variation / crowding gap of the solve
(movie_diag.py) is separate.
  collect: python kap_ideal.py collect <model> <out npz> <sim> [...]   (pass 2: the joint run's front re-run on the movie with
           return_head; pass 1: the stored pass-1 rows) -> per row: network outputs, the per-row pattern, photons, truth
  replot:  python kap_ideal.py replot <model> <report json> <out png>
  report:  python kap_ideal.py report <model> <npz> <out txt> <out png>   distributions per photon bin, R^2 of photons (+ r, bg,
           crowding), the photon-only alpha(N) / sf(N) applied in the solve vs the per-emitter joint solve (z, precision), the
           production factors until 2026-11-17 for comparison"""
import json, os, subprocess, sys
import numpy as np, torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalsim import gt_windows, load_gt, gt_of, SL, SJ

PX = 108.33
PH = [0, 300, 500, 800, 1200, 2000, 3000, 5000, 1e9]
dev = 'cuda'


def movie_of(sim, model):
    C = f'{SL}/calibs'
    cal = dict(m1809=f'{C}/calib_1809_2_psf_v2_simcam.json', m25=f'{C}/calib_2p5nM_m2_psf_v2_simcam.json', g13=f'{C}/calib_gauss1.3_simcam.json')[model]
    if sim == 'ah1': return os.path.expanduser('~/simcode2/out/psf_v2/eval/sims/m05r3v2/sim_movie.tif'), cal
    if sim == 'bh1': return os.path.expanduser('~/simcode2/out/psf_v2/eval/sims/m25rc3v2/sim_movie.tif'), cal
    p = os.path.expanduser(f'~/simcode2/out/papersim/{sim}')
    if os.path.isdir(p): return f'{p}/movie.tif', f'{p}/calib_sim.json'
    return f'{SL}/sims/{sim}/sim_movie.tif', cal


def match_truth(R, X0, Y0, gt, crop):
    """GT position (region coords) and crowding (another true emitter ON in the window within 3 px) of the rows -> gx, gy (NaN
    unmatched), crowded"""
    from simcode.evaluation import metrics as E
    x, y, t0, t1, nf = load_gt(gt)
    GW = gt_windows(x, y, t0, t1, nf)
    gi = E.match_fast(GW, R['wstart'].astype(np.int64), R['x_hy0'].astype(np.float64), R['y_hy0'].astype(np.float64), r=1.0)
    m = gi >= 0
    gx = np.full(len(gi), np.nan); gy = np.full(len(gi), np.nan); crowd = np.zeros(len(gi), bool)
    gx[m] = GW[gi[m], 2] - X0; gy[m] = GW[gi[m], 3] - Y0
    T_ = cKDTree(np.c_[GW[:, 2], GW[:, 3], GW[:, 0] * 1e4])
    d, _ = T_.query(np.c_[GW[gi[m], 2], GW[gi[m], 3], GW[gi[m], 0] * 1e4], k=2, distance_upper_bound=3.0)
    crowd[m] = np.isfinite(d[:, 1])
    return gx, gy, crowd


@torch.no_grad()
def collect(model, outp, sims):
    from simcode import config, table as SP
    from simcode.io.movie import open_movie
    from simcode.processing.common import region, load_calib, pattern_frames
    from simcode.processing.inference_fast import Pass2FrontFast
    M = config.model_set(f'{SJ}/models/{model}_joint')
    c1 = json.load(open(config.resolve(M['cal_sim']))); c2 = json.load(open(config.resolve(M['cal_sim_pass2'])))
    p2 = None
    acc = {1: {}, 2: {}}
    add = lambda ps, k, v: acc[ps].setdefault(k, []).append(np.asarray(v))
    for sim in sims:
        run = f'{SJ}/runs/{model}_joint/{sim}'
        mov, calp = movie_of(sim, model)
        movie = open_movie(mov); cal = load_calib(calp)
        if p2 is None or p2._calib != calp:
            p2 = Pass2FrontFast(calp, emitter=M['emitter_pass2'], detector=M['detector_pass2'], compile='off'); p2._calib = calp
        crop = tuple(int(v) for v in SP.run_crop(run)); X0, Y0, S = region(movie, crop, 5)
        Pt = torch.load(f'{run}/pattern_est.pt', weights_only=False)
        f0s = sorted(int(f.rsplit('_', 1)[1][:-4]) for f in os.listdir(run) if f.startswith('pass1_') and f.endswith('.npz'))
        for f0 in f0s:
            P1 = dict(np.load(f'{run}/pass1_{f0}.npz'))
            R, H = p2.run_chunk(movie, cal, f0, min(f0 + 6000, movie.n), Pt, c2, P1, c1, crop=crop, margin=5, return_head=True)
            o = H['o']; PW = H['PWall'][H['bw']]
            gx, gy, crowd = match_truth(R, X0, Y0, gt_of(sim), crop)
            for k in ('I', 'L', 'C', 'x', 'y', 'lsxy', 'bg'): add(2, k, o[k].float().cpu().numpy())
            add(2, 'PW', PW.cpu().numpy()); add(2, 'p', R['p']); add(2, 'xs', R['x_hy0'] - X0); add(2, 'ys', R['y_hy0'] - Y0)
            add(2, 'sx', R['sx_cal']); add(2, 'sy', R['sy_cal']); add(2, 'gx', gx); add(2, 'gy', gy); add(2, 'crowd', crowd)
            add(2, 'inc', SP.in_crop(R['x_hy0'], R['y_hy0'], crop)); add(2, 'sim', np.full(len(gx), sims.index(sim)))
            # pass 1: the stored rows (out) and their pattern solve (pass2_est)
            Q = np.load(f'{run}/pass2_est_{f0}.npz'); k = P1['out'].astype(bool)
            fr = (Q['wstart'][:, None] + np.arange(6)[None]).reshape(-1)
            PW1 = pattern_frames(Pt, fr, X0, Y0).reshape(-1, 6, 5).cpu().numpy()
            R1 = dict(wstart=Q['wstart'], x_hy0=Q['x_hy0'], y_hy0=Q['y_hy0'])
            g1x, g1y, cr1 = match_truth(R1, X0, Y0, gt_of(sim), crop)
            for kk in ('I', 'L', 'C', 'bg'): add(1, kk, P1[kk][k])
            add(1, 'x', P1['x'][k] - X0); add(1, 'y', P1['y'][k] - Y0); add(1, 'lsxy', np.log(np.c_[P1['sx'], P1['sy']][k]))
            add(1, 'PW', PW1); add(1, 'p', Q['p']); add(1, 'xs', Q['x_hy0'] - X0); add(1, 'ys', Q['y_hy0'] - Y0)
            add(1, 'sx', Q['sx_cal']); add(1, 'sy', Q['sy_cal']); add(1, 'gx', g1x); add(1, 'gy', g1y); add(1, 'crowd', cr1)
            add(1, 'inc', SP.in_crop(Q['x_hy0'], Q['y_hy0'], crop)); add(1, 'sim', np.full(len(g1x), sims.index(sim)))
            print(model, sim, f0, 'pass-2 rows', len(gx), 'pass-1 rows', len(g1x), flush=True)
            del R, H, o
            torch.cuda.empty_cache()
    out = {}
    for ps in (1, 2):
        for k, v in acc[ps].items(): out[f'p{ps}_{k}'] = np.concatenate(v)
    np.savez(outp, sims=np.array(sims), **out)


@torch.no_grad()
def two_estimates(D, cal, chunk=32768):
    """per row and axis: sigma_p, sigma_q, rho at the MAP of the joint solve (non-kinetic rows; kinetic -> NaN); + the solve"""
    from simcode.solve import solve_fast as MF
    from simcode.solve import pattern as Cm
    tt = lambda a: torch.as_tensor(np.asarray(a), device=dev, dtype=torch.float32)
    n = len(D['I']); res = {k: [] for k in ('sp', 'sq', 'rho', 'kin', 'r', 's')}
    cmult, cang = float(cal['cmult']), float(cal['cmult_ang'])
    for c0 in range(0, n, chunk):
        sl = slice(c0, c0 + chunk)
        o = {k: tt(D[k][sl]) for k in ('I', 'L', 'C', 'x', 'y', 'lsxy', 'bg')}
        PW = tt(D['PW'][sl]); b = torch.arange(len(o['I']), device=dev)
        so = MF.solve_emitters_fast(o, PW, b, cal, compile='off')
        xy = so['r_map']; mu = so['mu']
        il, dx, dy = Cm.illum_at(PW, xy[:, 0], xy[:, 1])
        B = mu.sum(1) / il.sum(1).clamp(min=1e-6)
        J = torch.stack([B[:, None] * dx, B[:, None] * dy, il], -1)                       # [n,T,3] d mu / d (x, y, B)
        L, Cj = o['L'], o['C']
        S = L @ L.transpose(1, 2) + Cj @ Cj.transpose(1, 2) + 1e-3 * torch.eye(6, device=dev)
        mup = mu.clamp(min=0)
        S = S + torch.diag_embed((cmult * mup) ** 2)
        if cang > 0:
            oh = torch.nn.functional.one_hot(Cm.angle_k(PW)[1], 2).float()
            S = S + cang ** 2 * (mup[:, :, None] * mup[:, None, :]) * (oh @ oh.transpose(1, 2))
        SiJ = torch.linalg.solve(S, J)
        F = J.transpose(1, 2) @ SiJ
        cov = torch.linalg.inv(F + 1e-9 * torch.eye(3, device=dev))
        G = cov @ SiJ.transpose(1, 2)                                                       # [n,3,T]
        sq = cov.diagonal(dim1=1, dim2=2)[:, :2].clamp(min=0).sqrt()
        GC = (G[:, :2] @ Cj).diagonal(dim1=1, dim2=2)                                       # [n,2]
        rho = (GC / sq.clamp(min=1e-9)).clamp(-0.999, 0.999)
        res['sp'].append(o['lsxy'].exp()); res['sq'].append(sq); res['rho'].append(rho); res['kin'].append(so['kin'].float())
        res['r'].append(so['r']); res['s'].append(so['s_cal'])
    return {k: torch.cat(v).cpu().numpy() for k, v in res.items()}


def ideal(sp, sq, rho):
    r = sp / sq
    den = 1 + r ** 2 - 2 * rho * r
    alpha = (1 - rho / r) / (1 - rho * r)
    var_true = sp ** 2 * (1 - rho ** 2) / den
    claim = alpha * sp ** 2 / (1 + alpha * r ** 2)
    sf = np.sqrt(var_true / np.where(claim > 0, claim, np.nan))
    sf0 = np.sqrt(var_true * (1 + r ** 2) / sp ** 2)
    return r, alpha, sf, sf0, rho >= r


def r2_groups(y, g):
    ok = np.isfinite(y); y, g = y[ok], g[ok]
    m = np.bincount(g, weights=y, minlength=g.max() + 1) / np.maximum(np.bincount(g, minlength=g.max() + 1), 1)
    return float(1 - np.sum((y - m[g]) ** 2) / np.sum((y - y.mean()) ** 2))


def r2_ols(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(1); y, X = y[ok], X[ok]
    X = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2))


def old_factors(model, ps):
    br = 'paper' if model in ('m1809', 'm25') else 'master'
    g = lambda p: json.loads(subprocess.check_output(['git', '-C', os.path.expanduser('~/simcode2'), 'show', f'{br}:models/{model}/{p}']))
    M = g('model.json')[model]
    return g(M['cal_sim' if ps == 1 else 'cal_sim_pass2'])


def binval(tab, key, ph):
    return np.interp(np.log(np.maximum(ph, 1)), np.log(np.array(tab['ph'])), np.array(tab[key]))


def report(model, npzp, outt, outpng):
    from simcode import config
    Z = np.load(npzp)
    M = config.model_set(f'{SJ}/models/{model}_joint')
    cals = {1: json.load(open(config.resolve(M['cal_sim']))), 2: json.load(open(config.resolve(M['cal_sim_pass2'])))}
    L = []; pr = lambda s: (print(s, flush=True), L.append(s))
    pr(f'# {model}: held-out sim movies {list(Z["sims"])}; joint-covariance networks; rows p >= 0.63 inside the crop')
    pr('# double counting only (2-estimate picture, 1D per axis); the brightness / crowding gap of the solve is separate (movie_diag)')
    plotd = {}; res_all = {}
    for ps in (2, 1):
        D = {k[3:]: Z[k] for k in Z.files if k.startswith(f'p{ps}_')}
        keep = (D['p'] >= 0.63) & D['inc']
        D = {k: v[keep] for k, v in D.items()}
        T = two_estimates(D, cals[ps])
        ph = D['I'].sum(1); nk = T['kin'] < 0.5
        sp, sq, rho = T['sp'], T['sq'], T['rho']
        r, alpha, sf, sf0, neg = ideal(sp, sq, rho)
        phb = np.repeat(ph[:, None], 2, 1)
        bg = np.repeat(np.log(np.maximum(D['bg'].mean(1), 0.01))[:, None], 2, 1); crowd = np.repeat(D['crowd'][:, None], 2, 1).astype(float)
        nk2 = np.repeat(nk[:, None], 2, 1)
        pr(f'\n## pass {ps}: {len(ph)} rows ({100 * nk.mean():.0f} % non-kinetic, used for the 2-estimate quantities), both axes pooled')
        pr('   bin           n   r p10/50/90          rho p10/50/90        alpha p10/50/90 (rho<r)  sf p10/50/90 (alpha applied)  sf0 p50  frac rho>=r')
        old = old_factors(model, ps)
        rows = []
        for lo, hi in zip(PH[:-1], PH[1:]):
            m = nk2 & (phb >= lo) & (phb < hi)
            if m.sum() < 200: continue
            v = m & ~neg
            q = lambda a, mm: np.nanpercentile(a[mm], [10, 50, 90])
            rr, ro, al, s_ = q(r, m), q(rho, m), q(alpha, v), q(sf, v)
            pc = float(np.median(phb[m]))
            ao = binval(old['kap_bins'], 'kap_p', pc) ** 2 / binval(old['kap_bins'], 'kap_I2', pc)
            so = binval(old['sig_ph'], 'f', pc)
            rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), ph_med=pc, r=rr.tolist(), rho=ro.tolist(), alpha=al.tolist(), sf=s_.tolist(),
                             sf0=float(np.nanmedian(sf0[m])), frac_neg=float(neg[m].mean()), alpha_old=float(ao), sig_ph_old=float(so),
                             kap_p_old=float(binval(old['kap_bins'], 'kap_p', pc)), kap_I2_old=float(binval(old['kap_bins'], 'kap_I2', pc))))
            pr(f'   {lo:>5g}-{hi:<6g}{m.sum():8d}  {rr[0]:5.2f}/{rr[1]:5.2f}/{rr[2]:5.2f}   {ro[0]:5.2f}/{ro[1]:5.2f}/{ro[2]:5.2f}   '
               f'{al[0]:5.2f}/{al[1]:5.2f}/{al[2]:5.2f}           {s_[0]:5.2f}/{s_[1]:5.2f}/{s_[2]:5.2f}           {np.nanmedian(sf0[m]):5.2f}   {neg[m].mean():.3f}')
        pr('   production until 2026-11-17 (old networks; at the bin median photons): alpha_old = kap_p^2 / kap_I2 (the prior / intensity variance\n'
           '   weighting of the independent solve), sig_ph (the per-photon-bin sigma factor; sig_a max(q/dof, 1)^sig_g came on top):')
        pr('   ' + '  '.join(f"{b['lo']:g}-{b['hi']:g}: alpha_old {b['alpha_old']:.2f} (kap_p {b['kap_p_old']:.2f}, kap_I2 {b['kap_I2_old']:.2f}) sig_ph {b['sig_ph_old']:.2f}"
                             for b in rows))
        # R^2 of log alpha / log sf
        v = nk2 & ~neg & np.isfinite(alpha) & (alpha > 0)
        la, ls = np.log(np.where(v, alpha, np.nan)), np.log(np.where(v, sf, np.nan))
        lp = np.log(np.maximum(phb, 1)); lr = np.log(r)
        edges = np.quantile(phb[v], np.linspace(0, 1, 21))
        g20 = np.clip(np.searchsorted(edges, phb, side='right') - 1, 0, 19)
        for nm, y in (('log alpha', la), ('log sf', ls)):
            y_ = np.where(v, y, np.nan).ravel()
            r2p = r2_groups(y_[v.ravel()], g20.ravel()[v.ravel()])
            r2pr = r2_ols(y_, np.c_[lp.ravel(), lp.ravel() ** 2, lr.ravel(), lr.ravel() ** 2, lp.ravel() * lr.ravel()])
            r2all = r2_ols(y_, np.c_[lp.ravel(), lp.ravel() ** 2, lr.ravel(), lr.ravel() ** 2, lp.ravel() * lr.ravel(), bg.ravel(), crowd.ravel()])
            pr(f'   R^2 of {nm} (per emitter and axis, rho < r): photons alone (20 photon quantile bins) {r2p:.3f}; photons + r {r2pr:.3f}; '
               f'+ bg + crowding {r2all:.3f}')
        res = dict(rows=rows)
        # photon-only factors applied in an independent solve vs the per-emitter joint solve
        if ps == 2:
            from simcode.solve import solve_fast as MF
            pcs = np.log([b['ph_med'] for b in rows]); amed = np.array([b['alpha'][1] for b in rows]); smed = np.array([b['sf'][1] for b in rows])
            aN = np.exp(np.interp(np.log(np.maximum(ph, 1)), pcs, np.log(amed))); sN = np.exp(np.interp(np.log(np.maximum(ph, 1)), pcs, np.log(smed)))
            tt = lambda a: torch.as_tensor(np.asarray(a), device=dev, dtype=torch.float32)
            rr_, ss_ = [], []
            for c0 in range(0, len(ph), 32768):
                sl = slice(c0, c0 + 32768)
                o = {k: tt(D[k][sl]) for k in ('I', 'L', 'x', 'y', 'bg')}
                o['C'] = torch.zeros(len(o['I']), 6, 2, device=dev)
                o['lsxy'] = tt(D['lsxy'][sl]) + 0.5 * tt(np.log(aN[sl]))[:, None]
                so = MF.solve_emitters_fast(o, tt(D['PW'][sl]), torch.arange(len(o['I']), device=dev), cals[2], compile='off')
                rr_.append(so['r'].cpu().numpy()); ss_.append(so['s_cal'].cpu().numpy() * sN[sl, None])
            rN, sNs = np.concatenate(rr_), np.concatenate(ss_)
            g = np.c_[D['gx'], D['gy']]; mt = np.isfinite(g).all(1)
            def zp(rpos, s):
                e = rpos[mt] - g[mt]; e -= np.median(e, 0); z = e / s[mt]
                out = []
                for lo, hi in zip(PH[:-1], PH[1:]):
                    b = (ph[mt] >= lo) & (ph[mt] < hi)
                    if b.sum() < 200: continue
                    zz = z[b].ravel(); c = zz[np.abs(zz) < 4]
                    out.append((lo, float(np.sqrt(np.mean(c ** 2)) / 0.99893), float(1.4826 * np.median(np.abs(e[b].ravel()))) * PX,
                                float(100 * np.mean(np.hypot(*e[b].T) * PX >= 50))))
                return out
            J_ = zp(T['r'], T['s']); N_ = zp(rN, sNs)
            pr('   photon-only alpha(N) / sf(N) (the bin medians, log-photon interpolation) in an independent solve (C = 0) vs the per-emitter joint solve,')
            pr('   matched rows: z = rms(error / sigma) |z| < 4 | robust error sd nm | gross > 50 nm %')
            for a, b in zip(J_, N_):
                pr(f'   {a[0]:>5g}: joint z {a[1]:.2f} sd {a[2]:6.2f} gross {a[3]:5.2f} | photon-only factors z {b[1]:.2f} sd {b[2]:6.2f} ({100 * (b[2] / a[2] - 1):+.1f} %) gross {b[3]:5.2f}')
            res['solve_cmp'] = dict(joint=J_, photon_only=N_)
            plotd = dict(rows=rows, J=J_, N=N_)
        res_all[f'pass{ps}'] = res
    open(outt, 'w').write('\n'.join(L) + '\n')
    # plot: pass-2 alpha and sf distributions vs photons, with the production factors
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rows = plotd['rows']; x = [b['ph_med'] for b in rows]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, key, lab, oldk in ((ax[0], 'alpha', 'ideal prior-variance factor alpha', 'alpha_old'), (ax[1], 'sf', 'ideal sigma factor (alpha applied)', 'sig_ph_old')):
        a.fill_between(x, [b[key][0] for b in rows], [b[key][2] for b in rows], color='#2a78d6', alpha=0.2, lw=0, label='p10-p90 per emitter')
        a.plot(x, [b[key][1] for b in rows], marker='o', color='#2a78d6', lw=2, label='median (photon-only factor)')
        a.plot(x, [b[oldk] for b in rows], marker='s', color='#eb6834', lw=2, ls='--', label='production until now (old networks)')
        a.set_title(lab, fontsize=9)
    ax[2].plot(x, [b['frac_neg'] for b in rows], marker='o', color='#6250d6', lw=2)
    ax[2].set_title('fraction with rho >= r (alpha undefined)', fontsize=9)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel('network window photons'); a.grid(color='#e4e3df', lw=0.6)
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_xticks([300, 1000, 3000]); a.set_xticklabels(['300', '1000', '3000'])
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc='lower center', ncol=3)
    fig.suptitle(f'{model} pass 2: photon-count dependence of the double-counting factors (held-out sims)', fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(outpng, dpi=150)
    json.dump(res_all, open(outt.replace('.txt', '.json'), 'w'), indent=1, default=float)


def replot(model, js, outpng):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rows = json.load(open(js))['pass2']['rows']; x = [b['ph_med'] for b in rows]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, key, lab, oldk in ((ax[0], 'alpha', 'ideal prior-variance factor alpha', 'alpha_old'), (ax[1], 'sf', 'ideal sigma factor (alpha applied)', 'sig_ph_old')):
        a.fill_between(x, [b[key][0] for b in rows], [b[key][2] for b in rows], color='#2a78d6', alpha=0.2, lw=0, label='p10-p90 per emitter')
        a.plot(x, [b[key][1] for b in rows], marker='o', color='#2a78d6', lw=2, label='median (photon-only factor)')
        a.plot(x, [b[oldk] for b in rows], marker='s', color='#eb6834', lw=2, ls='--', label='production until now (old networks): kap_p^2 / kap_I2, sig_ph')
        a.set_title(lab, fontsize=9)
    ax[2].plot(x, [b['frac_neg'] for b in rows], marker='o', color='#6250d6', lw=2)
    ax[2].set_title('fraction with rho >= r (alpha undefined)', fontsize=9)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel('network window photons'); a.grid(color='#e4e3df', lw=0.6)
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_xticks([300, 1000, 3000]); a.set_xticklabels(['300', '1000', '3000'])
        for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc='lower center', ncol=3)
    fig.suptitle(f'{model} pass 2: photon-count dependence of the double-counting factors (held-out sims)', fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(outpng, dpi=150)


if __name__ == '__main__':
    if sys.argv[1] == 'collect': collect(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif sys.argv[1] == 'replot': replot(sys.argv[2], sys.argv[3], sys.argv[4])
    else: report(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
