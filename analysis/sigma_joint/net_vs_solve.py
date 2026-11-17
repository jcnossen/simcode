"""sigma study (2026-11-17): how close do the emitter networks and the pattern solve get to the Cramer-Rao bounds, and how much does
the solve add to the network? Held-out simulated movies regenerated with their per-frame truth (sim_truth.py; out/sigma_joint/
truth/<sim>), the joint-covariance model (out/sigma_joint/models/<model>_joint) and its pipeline runs (out/sigma_joint/runs).
Per GT window matched in both passes (p >= 0.63, inside the crop, 1 px, the window's true event):
  net_p1   the pass-1 emitter network's position prior (prior_off of the pass-1 calibration removed), predicted sigma = its sd
  net_p2   the pass-2 emitter network's position prior (re-run of the pass-2 front, return_head)
  solve_p1 / solve_p2   the pattern solve of pass 1 / pass 2 (the run's x, y; sigma = the posterior sd)
  CRLB_SMLM   single emitter, the 6-frame sum, theta (x, y, I, bg), the true per-frame photons summed
  CRLB_SF     the 'SIMCODE bound' of si_crlb_precision: SIMFLUX with one bg per frame at the TRUE pattern (incl. the pattern-phase /
              depth field), theta (x, y, B, bg_1..6), mu_t = I_t il_t(x) / il_t(x0) g(x) + bg_t with the true per-frame photons I_t
  both: Poisson, 16 x 16 px ROI, the model's PSF (analysis/baselines/zpsf.py) at the emitter's true height, bg_t = the pass-2
  network's per-frame background of the row; isolated bounds (no neighbours)
Photons = detected photons of the window (sum of the true per-frame photons x the PSF's collected fraction at the emitter's
height). Errors minus the per-movie median per estimate. Per photon bin, all rows and isolated + on-through-window rows: robust sd
(1.4826 MAD, x and y pooled), z = rms(error / predicted sigma) (|z| < 4), median CRLBs, robust sd / median CRLB.
Kinetic model (pass 2): fraction of rows with the kinetic model selected per photon bin; on always-on GT emitters (ON through the
window) the precision and z of the selected model vs forcing the non-kinetic model vs forcing the kinetic model.
usage: python net_vs_solve.py <model> <out json> <out png> <sim> [...]   |   python net_vs_solve.py replot <json> <png>"""
import json, os, sys
import numpy as np, torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, 'analysis', 'baselines')); sys.path.insert(0, os.path.join(REPO, 'analysis', 'paper'))
from evalsim import gt_windows, load_gt, SJ
from kap_ideal import movie_of

PX = 108.33
PH = [0, 300, 500, 800, 1200, 2000, 3000, 5000, 1e9]
dev = 'cuda'
PSFSPEC = dict(m1809=os.path.join(REPO, 'models/m1809/calib_1809_2_psf_v2.json'), m25=os.path.join(REPO, 'models/insitu/calib_2p5nM_m2_psf_v2.json'),
               g13='gauss:1.3')
METHODS = ('net_p1', 'net_p2', 'solve_p1', 'solve_p2')


def matched(wstart, x, y, GW):
    from simcode.evaluation import metrics as E
    return E.match_fast(GW, wstart.astype(np.int64), x.astype(np.float64), y.astype(np.float64), r=1.0)


@torch.no_grad()
def crlbs(psf, x, y, z, Ig, bg, il, dilx, dily, P=16):
    """per row: CRLB_SMLM, CRLB_SF [n,2] (px). x, y, z [n]; Ig, bg, il, dilx, dily [n,6] (true per-frame photons, bg, true
    illumination and its x / y derivative at the true position)"""
    tt = lambda a: torch.as_tensor(np.asarray(a), device=dev, dtype=torch.float32)
    out_sm, out_sf, coll = [], [], []
    for c0 in range(0, len(x), 8192):
        s = slice(c0, c0 + 8192)
        xc = tt(x[s] - np.floor(x[s]) - 0.5); yc = tt(y[s] - np.floor(y[s]) - 0.5)
        g, gx, gy = psf.eval(xc, yc, P, tt(z[s]))
        G, GX, GY = (v.flatten(1) for v in (g, gx, gy))                           # [m,P*P]
        I = tt(Ig[s]); B = tt(bg[s]); L = tt(il[s]).clamp(min=1e-3); DX = tt(dilx[s]); DY = tt(dily[s])
        # SMLM on the sum
        It = I.sum(1, keepdim=True); bt = B.sum(1, keepdim=True)
        mu = It * G + bt
        J = torch.stack([It * GX, It * GY, G, torch.ones_like(G)], 1)
        F = torch.einsum('nkp,nlp->nkl', J / mu[:, None], J)
        out_sm.append(torch.linalg.inv(F + 1e-9 * torch.eye(4, device=dev)).diagonal(dim1=1, dim2=2)[:, :2].clamp(min=0).sqrt())
        # SIMFLUX, bg per frame, true pattern
        mu = I[..., None] * G[:, None] + B[..., None]                             # [m,6,P*P]
        Jx = I[..., None] * ((DX / L)[..., None] * G[:, None] + GX[:, None])
        Jy = I[..., None] * ((DY / L)[..., None] * G[:, None] + GY[:, None])
        Js = I[..., None] * G[:, None]
        m = len(I); Jb = torch.zeros(m, 6, 6, G.shape[1], device=dev); Jb[:, torch.arange(6), torch.arange(6)] = 1.0
        J = torch.cat([torch.stack([Jx, Jy, Js], 1), Jb], 1).flatten(2)          # [m,9,6*P*P]
        F = torch.einsum('nkp,nlp->nkl', J / mu.flatten(1)[:, None], J)
        out_sf.append(torch.linalg.inv(F + 1e-9 * torch.eye(9, device=dev)).diagonal(dim1=1, dim2=2)[:, :2].clamp(min=0).sqrt())
        g31, _, _ = psf.eval(torch.zeros_like(xc), torch.zeros_like(yc), 31, tt(z[s]))
        coll.append(g31.flatten(1).sum(1))
    cat = lambda v: torch.cat(v).cpu().numpy()
    return cat(out_sm), cat(out_sf), cat(coll)


@torch.no_grad()
def collect_sim(model, sim, psf):
    from simcode import config, table as SP
    from simcode.io.movie import open_movie
    from simcode.processing.common import region, load_calib
    from simcode.processing.inference_fast import Pass2FrontFast
    from simcode.solve import solve_fast as MF
    M = config.model_set(f'{SJ}/models/{model}_joint')
    c1 = json.load(open(config.resolve(M['cal_sim']))); c2 = json.load(open(config.resolve(M['cal_sim_pass2'])))
    run = f'{SJ}/runs/{model}_joint/{sim}'; tdir = f'{SJ}/truth/{sim}'
    mov, calp = movie_of(sim, model)
    movie = open_movie(mov); cal = load_calib(calp)
    crop = tuple(int(v) for v in SP.run_crop(run)); X0, Y0, S = region(movie, crop, 5)
    p2 = Pass2FrontFast(calp, emitter=M['emitter_pass2'], detector=M['detector_pass2'], compile='off')
    Pt = torch.load(f'{run}/pattern_est.pt', weights_only=False)
    P1 = dict(np.load(f'{run}/pass1_0.npz')); Q = dict(np.load(f'{run}/pass2_est_0.npz'))
    R, H = p2.run_chunk(movie, cal, 0, movie.n, Pt, c2, P1, c1, crop=crop, margin=5, return_head=True)
    o, PWall, bw = H['o'], H['PWall'], H['bw']
    # pass-2 solve: selected kinetic model, forced non-kinetic, forced kinetic
    V = {}
    for nm, thr in (('sel', 9.21), ('nokin', 1e9), ('kin', -1e9)):
        rr, ss, kk = [], [], []
        for c0 in range(0, len(o['I']), 16384):
            sl = slice(c0, c0 + 16384)
            so = MF.solve_emitters_fast({k: v[sl] for k, v in o.items()}, PWall, bw[sl], c2, thr=thr, compile='off')
            rr.append(so['r']); ss.append(so['s_cal']); kk.append(so['kin'].float())
        V[nm] = dict(r=torch.cat(rr).cpu().numpy() + [X0, Y0], s=torch.cat(ss).cpu().numpy(), kin=torch.cat(kk).cpu().numpy())
    print(sim, 'pass-2 rows', len(R['x_hy0']), 'max |selected - run| px', float(np.abs(V['sel']['r'][:, 0] - R['x_hy0']).max()), flush=True)
    G = np.load(f'{tdir}/sim_gt.npz'); TF = np.load(f'{tdir}/truth_frames.npz')
    gxa, gya, t0, t1, nf = load_gt(f'{tdir}/sim_gt.npz')
    GW = gt_windows(gxa, gya, t0, t1, nf)
    b2 = (R['p'] >= 0.63) & SP.in_crop(R['x_hy0'], R['y_hy0'], crop)
    k1 = P1['out'].astype(bool)
    b1 = (Q['p'] >= 0.63) & SP.in_crop(Q['x_hy0'], Q['y_hy0'], crop)
    g2 = np.full(len(b2), -1); g2[b2] = matched(R['wstart'][b2], R['x_hy0'][b2], R['y_hy0'][b2], GW)
    g1 = np.full(len(b1), -1); g1[b1] = matched(Q['wstart'][b1], Q['x_hy0'][b1], Q['y_hy0'][b1], GW)
    com, i2, i1 = np.intersect1d(g2[g2 >= 0], g1[g1 >= 0], return_indices=True)
    r2 = np.nonzero(g2 >= 0)[0][i2]; r1 = np.nonzero(g1 >= 0)[0][i1]
    w = GW[com, 0].astype(np.int64); ev = GW[com, 1].astype(np.int64); tx, ty = GW[com, 2], GW[com, 3]
    tz = G['z'][ev]
    key = TF['ev'] * nf + TF['frame']; oo = np.argsort(key); key, phv = key[oo], TF['ph'][oo]
    q = ev[:, None] * nf + w[:, None] + np.arange(6)[None]
    j = np.clip(np.searchsorted(key, q), 0, len(key) - 1)
    Ig = np.where(key[j] == q, phv[j], 0.0)
    fr = w[:, None] + np.arange(6)[None]
    pt = G['patt'][fr].astype(np.float64).copy()
    if 'dph' in TF.files:
        ang = (fr % 6) // 3
        pt[..., 2] = pt[..., 2] + TF['dph'][ev[:, None], ang]; pt[..., 3] = np.minimum(pt[..., 3] * TF['dmf'][ev[:, None], ang], 1.0)
    arg = pt[..., 0] * tx[:, None] + pt[..., 1] * ty[:, None] - pt[..., 2]
    il = pt[..., 4] * (1 + pt[..., 3] * np.cos(arg)); dd = -pt[..., 4] * pt[..., 3] * np.sin(arg)
    bg = o['bg'][torch.as_tensor(r2, device=dev)].float().cpu().numpy().clip(min=0.05)
    csm, csf, coll = crlbs(psf, tx, ty, tz, Ig, bg, il, dd * pt[..., 0], dd * pt[..., 1])
    T_ = cKDTree(np.c_[GW[:, 2], GW[:, 3], GW[:, 0] * 1e4])
    dnn, _ = T_.query(np.c_[tx, ty, w * 1e4], k=2, distance_upper_bound=3.0)
    onthru = (t0[ev] <= w) & (t1[ev] >= w + 6)
    po1, po2 = np.array(c1['prior_off']), np.array(c2['prior_off'])
    sel = torch.as_tensor(r2, device=dev)
    e = dict(net_p1=np.c_[P1['x'][k1][r1] - po1[0], P1['y'][k1][r1] - po1[1]],
             net_p2=np.c_[o['x'][sel].cpu().numpy() + X0 - po2[0], o['y'][sel].cpu().numpy() + Y0 - po2[1]],
             solve_p1=np.c_[Q['x_hy0'][r1], Q['y_hy0'][r1]], solve_p2=np.c_[R['x_hy0'][r2], R['y_hy0'][r2]],
             p2_sel=V['sel']['r'][r2], p2_nokin=V['nokin']['r'][r2], p2_kin=V['kin']['r'][r2])
    s = dict(net_p1=np.c_[P1['sx'][k1][r1], P1['sy'][k1][r1]], net_p2=o['lsxy'][sel].exp().cpu().numpy(),
             solve_p1=np.c_[Q['sx_cal'][r1], Q['sy_cal'][r1]], solve_p2=np.c_[R['sx_cal'][r2], R['sy_cal'][r2]],
             p2_sel=V['sel']['s'][r2], p2_nokin=V['nokin']['s'][r2], p2_kin=V['kin']['s'][r2])
    out = dict(ph=Ig.sum(1) * coll, iso=~np.isfinite(dnn[:, 1]), onthru=onthru, crlb_smlm=csm, crlb_sf=csf, kin=V['sel']['kin'][r2])
    for k in e:
        err = e[k] - np.c_[tx, ty]; err -= np.median(err, 0)
        out['e_' + k] = err; out['s_' + k] = s[k]
    return out


def rob(v): return float(1.4826 * np.median(np.abs(v))) if len(v) else float('nan')


def zcore(z):
    v = z.ravel(); c = v[np.abs(v) < 4]
    return float(np.sqrt(np.mean(c ** 2)) / 0.99893) if len(c) > 20 else float('nan')


def plot(res, outpng):
    import matplotlib; matplotlib.use('Agg')
    import figstyle as FS
    FS.apply()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    model = res['model']
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    cols = dict(net_p1='#eb6834', net_p2='#e87ba4', solve_p1='#1baf7a', solve_p2='#2a78d6')
    labs = dict(net_p1='network, pass 1 (prior)', net_p2='network, pass 2 (prior)', solve_p1='solve, pass 1', solve_p2='solve, pass 2')
    for a, sn, title in ((ax[0], 'all', 'all matched rows'), (ax[1], 'iso_onthru', 'isolated, on through the window')):
        rows = res['subsets'][sn]; x = [r['ph_med'] for r in rows]
        a.loglog(x, [r['crlb_smlm_nm'] for r in rows], '--', color='0.45', lw=1.2, label='CRLB SMLM')
        a.loglog(x, [r['crlb_sf_nm'] for r in rows], '-.', color='0.2', lw=1.2, label='CRLB SIMFLUX (true pattern, bg per frame)')
        for k in METHODS:
            a.loglog(x, [r[k]['sd_nm'] for r in rows], marker='o', ms=3.5, lw=1.5, color=cols[k], label=labs[k])
        a.set_title(title); a.set_xlabel('detected photons per 6-frame window')
        a.set_xticks([300, 1000, 3000]); a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); a.xaxis.set_minor_formatter(NullFormatter())
        a.set_yticks([2, 5, 10, 20]); a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); a.yaxis.set_minor_formatter(NullFormatter())
    ax[0].set_ylabel('localization precision (robust sd, nm)')
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc='lower center', ncol=3, fontsize=7)
    fig.suptitle(f'{model}: networks vs solve vs CRLB (held-out simulated movies)', fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(outpng, dpi=200)


def main():
    model, outj, outpng, sims = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    import zpsf
    psf = zpsf.load(PSFSPEC[model], zmax=0.4)
    parts = [collect_sim(model, s, psf) for s in sims]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    res = dict(model=model, sims=sims, n=int(len(D['ph'])), subsets={}, kinetic={})
    L = []; pr = lambda t: (print(t, flush=True), L.append(t))
    pr(f'# {model}: held-out sims {sims}; GT windows matched in both passes: {len(D["ph"])}; photons = detected photons of the window')
    for sn, m0 in (('all', np.ones(len(D['ph']), bool)), ('iso_onthru', D['iso'] & D['onthru'])):
        rows = []
        pr(f'\n## {sn} ({100 * m0.mean():.0f} % of the rows): robust sd nm (z = rms(err / predicted sigma)) | CRLB nm | sd / CRLB')
        pr('   photons       n    net_p1          net_p2          solve_p1        solve_p2      | CRLB SMLM  SF   | net_p1/SMLM net_p2/SF solve_p1/SF solve_p2/SF')
        for lo, hi in zip(PH[:-1], PH[1:]):
            m = m0 & (D['ph'] >= lo) & (D['ph'] < hi)
            if m.sum() < 100: continue
            r = dict(lo=lo, hi=hi, n=int(m.sum()), ph_med=float(np.median(D['ph'][m])),
                     crlb_smlm_nm=float(np.median(D['crlb_smlm'][m])) * PX, crlb_sf_nm=float(np.median(D['crlb_sf'][m])) * PX)
            for k in METHODS:
                e = D['e_' + k][m]; r[k] = dict(sd_nm=rob(e.ravel()) * PX, rms98_nm=float(np.sqrt(np.mean(np.sort(e.ravel() ** 2)[:int(0.98 * e.size)]))) * PX,
                                                z=zcore(e / D['s_' + k][m]), sigma_med_nm=float(np.median(D['s_' + k][m])) * PX)
            rows.append(r)
            pr(f"   {lo:>5g}-{hi:<6g}{r['n']:7d}  " + '  '.join(f"{r[k]['sd_nm']:6.2f} ({r[k]['z']:.2f})" for k in METHODS) +
               f" | {r['crlb_smlm_nm']:6.2f} {r['crlb_sf_nm']:6.2f} | {r['net_p1']['sd_nm'] / r['crlb_smlm_nm']:.2f}        "
               f"{r['net_p2']['sd_nm'] / r['crlb_sf_nm']:.2f}      {r['solve_p1']['sd_nm'] / r['crlb_sf_nm']:.2f}        {r['solve_p2']['sd_nm'] / r['crlb_sf_nm']:.2f}")
        res['subsets'][sn] = rows
    # kinetic model
    pr('\n## kinetic model (pass 2): fraction selected (all rows) | always-on GT emitters (ON through the window): robust sd nm (z) for the')
    pr('   selected model / forced non-kinetic / forced kinetic')
    krows = []
    for lo, hi in zip(PH[:-1], PH[1:]):
        m = (D['ph'] >= lo) & (D['ph'] < hi)
        if m.sum() < 100: continue
        mo = m & D['onthru']
        r = dict(lo=lo, hi=hi, n=int(m.sum()), frac_kin=float(D['kin'][m].mean()), frac_kin_onthru=float(D['kin'][mo].mean()),
                 frac_kin_blink=float(D['kin'][m & ~D['onthru']].mean()) if (m & ~D['onthru']).any() else float('nan'))
        for k in ('p2_sel', 'p2_nokin', 'p2_kin'):
            r[k] = dict(sd_nm=rob(D['e_' + k][mo].ravel()) * PX, z=zcore(D['e_' + k][mo] / D['s_' + k][mo]),
                        gross50_pct=float(100 * np.mean(np.hypot(*D['e_' + k][mo].T) * PX >= 50)))
            mb = m & ~D['onthru']
            r[k + '_blink'] = dict(sd_nm=rob(D['e_' + k][mb].ravel()) * PX, z=zcore(D['e_' + k][mb] / D['s_' + k][mb])) if mb.sum() > 50 else None
        krows.append(r)
        pr(f"   {lo:>5g}-{hi:<6g} kin selected {100 * r['frac_kin']:5.1f} % (on-through {100 * r['frac_kin_onthru']:5.1f} %, blinking "
           f"{100 * r['frac_kin_blink']:5.1f} %) | on-through: sel {r['p2_sel']['sd_nm']:6.2f} ({r['p2_sel']['z']:.2f}) nokin "
           f"{r['p2_nokin']['sd_nm']:6.2f} ({r['p2_nokin']['z']:.2f}) kin {r['p2_kin']['sd_nm']:6.2f} ({r['p2_kin']['z']:.2f})"
           + (f" | blinking: sel {r['p2_sel_blink']['sd_nm']:6.2f} ({r['p2_sel_blink']['z']:.2f}) nokin {r['p2_nokin_blink']['sd_nm']:6.2f} "
              f"({r['p2_nokin_blink']['z']:.2f}) kin {r['p2_kin_blink']['sd_nm']:6.2f} ({r['p2_kin_blink']['z']:.2f})" if r['p2_sel_blink'] else ''))
    res['kinetic'] = krows
    json.dump(res, open(outj, 'w'), indent=1)
    open(outj.replace('.json', '.txt'), 'w').write('\n'.join(L) + '\n')
    plot(res, outpng)
    print('wrote', outpng, outj)


if __name__ == '__main__':
    if sys.argv[1] == 'replot': plot(json.load(open(sys.argv[2])), sys.argv[3])
    else: main()
