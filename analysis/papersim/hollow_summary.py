"""Summary of the hollow-microtubule sweep (gen_hollow.py + run_all.py + hollow_analysis.py): dip ratio, fitted wall separation,
localization precision and counts vs photons per method and radius; the analytic expectation; zoom renders of the same regions
at every photon level.

Analytic expectation: the projected wall profile is the arcsine density on [-R, R] (u = R cos theta); blurred by a Gaussian
localization error sigma its dip ratio (hollow_analysis.dip_ratio on the continuous profile) rises with sigma / R and reaches 1
(no dip) at sigma / R = S1 = 0.29; a clear dip (ratio 0.9; sigma -> 0 gives 0.77) needs sigma / R <= S08 = 0.23. The SIMFLUX CRLB (and the SMLM CRLB) of the
averaged 1809/2 PSF on a 16 x 16 ROI at the condition's mean background then gives the photon count at which the bound
reaches those sigmas (sigma = mean of the x and y CRLB over random sub-pixel positions and pattern phases).
usage: python analysis/papersim/hollow_summary.py [--netset default] [--cond 1809_2]
outputs: out/papersim/hollow/summary_<netset>.json / .png, renders_<netset>_R<r>.png"""
import os, sys, json, glob, argparse, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import hollow_analysis as H
import metrics as M
from scipy.special import ndtr

OUTD = os.path.join(C.OUT, 'hollow')
LAB = dict(smlm='SMLM', sf='SIMFLUX (est. pattern)', sf_gtp='SIMFLUX (GT pattern)', decode='DECODE', v2='SIMCODE v2 strict', v2q45='SIMCODE v2, u <= 0.45')
COL = dict(smlm='tab:blue', sf='tab:green', sf_gtp='olive', decode='tab:orange', v2='tab:red', v2q45='darkred')


def dip_curve(ratios=np.linspace(0.05, 1.2, 116)):
    R = 1.0; u = np.linspace(-3, 3, 6001); th = (np.arange(2048) + 0.5) / 2048 * np.pi
    out = []
    for r in ratios:
        f = np.exp(-0.5 * ((u[:, None] - R * np.cos(th)[None]) / r) ** 2).mean(1)
        c = np.abs(u) < 0.3 * R; w = (np.abs(u) > 0.7 * R) & (np.abs(u) < 1.3 * R)
        out.append(f[c].mean() / f[w].mean())
    out = np.array(out)
    s1 = float(np.interp(1.0, out, ratios)); s08 = float(np.interp(0.9, out, ratios))   # 'clear dip' = ratio 0.9 (the noise-free limit of this ratio is 0.77)
    return ratios, out, s1, s08


def crlb_vs_photons(bg_frame, photons, n=2000, seed=0):
    import torch
    _s = importlib.util.spec_from_file_location('paper_pipeline', os.path.join(C.REPO, 'analysis', 'baselines', 'paper_pipeline.py'))
    PP = importlib.util.module_from_spec(_s); _s.loader.exec_module(PP)
    import s10
    psf = PP.zpsf.load(C.PSF_CALIB); P = 16; rng = np.random.default_rng(seed)
    res = {}
    for N in photons:
        x = 7 + rng.random(n); y = 7 + rng.random(n)
        Pg = C.gt_pattern([0, 90]); off = rng.random((n, 2)) * 2 * np.pi
        s = np.arange(6); a_ = s // 3
        ph = Pg['phi'][None] + np.repeat(off, 3, 1) - Pg['k'][a_, 0][None] * 0 - np.pi / 2
        mod = np.stack([np.broadcast_to(Pg['k'][a_, 0], (n, 6)), np.broadcast_to(Pg['k'][a_, 1], (n, 6)), ph,
                        np.full((n, 6), C.DEPTH), np.full((n, 6), 1 / 6)], -1)
        t = lambda v: torch.as_tensor(np.asarray(v, np.float32), device='cuda')
        with torch.no_grad():
            mu, J = PP.smlm_model(psf, P)(torch.stack([t(x), t(y), t(np.full(n, N)), t(np.full(n, 6 * bg_frame))], 1))
            sm = s10.crlb(mu, J)[:, :2].mean().item()
            mu, J = PP.sf_model(psf, P, t(mod))(torch.stack([t(x), t(y), t(np.full(n, N)), t(np.full(n, bg_frame))], 1))
            sf = s10.crlb(mu, J)[:, :2].mean().item()
        res[float(N)] = dict(smlm_nm=sm * C.PX_NM, sf_nm=sf * C.PX_NM)
    return res


def runs(netset, cond):
    out = {}
    for d in sorted(glob.glob(os.path.join(OUTD, f'{cond}_R*_x*'))):
        f = os.path.join(d, 'full', f'hollow_{netset}.json')
        if os.path.exists(f):
            j = json.load(open(f)); out[(j['radius_nm'], j['phot_scale'])] = j
    return out


def render_zooms(netset, cond, R, scales, methods=('smlm', 'sf_gtp', 'decode', 'v2q45'), box_nm=600, pix_nm=2.0):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    d0 = os.path.join(OUTD, f'{cond}_R{R:g}_x{scales[0]:g}')
    gt = dict(np.load(os.path.join(d0, 'gt.npz'))); CL = H.centreline(gt)
    # two zoom centres: clean centreline points with the most clean points within 300 nm, >= 1.5 um apart
    cand = np.nonzero(CL['clean'])[0]
    score = np.array([len(CL['tree'].query_ball_point(CL['c'][i], 300 / C.PX_NM)) * CL['clean'][CL['tree'].query_ball_point(CL['c'][i], 300 / C.PX_NM)].mean() for i in cand[::20]])
    order = cand[::20][np.argsort(-score)]
    centres = [order[0]]
    for i in order:
        if all(np.hypot(*(CL['c'][i] - CL['c'][j])) * C.PX_NM > 1500 for j in centres): centres.append(i)
        if len(centres) == 2: break
    hw = box_nm / 2 / C.PX_NM
    rows = [('GT', None)] + [(f'x{s:g}', s) for s in scales]
    fig, axs = plt.subplots(len(rows), 2 * len(methods), figsize=(2.0 * 2 * len(methods), 2.0 * len(rows)), squeeze=False)
    nb = int(box_nm / pix_nm)
    for ri, (lab, s) in enumerate(rows):
        for zi, ci in enumerate(centres):
            cx, cy = CL['c'][ci]
            for mi, m in enumerate(methods):
                ax = axs[ri, zi * len(methods) + mi]; ax.set_xticks([]); ax.set_yticks([])
                if s is None:
                    x, y = gt['pts'][:, 0], gt['pts'][:, 1]
                else:
                    p = M.method_paths(os.path.join(OUTD, f'{cond}_R{R:g}_x{s:g}', 'full'), netset).get(m)
                    if p is None:
                        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
                        if zi == 0 and mi == 0: ax.set_ylabel(lab, fontsize=8)
                        continue
                    T = M.load_method(m, p); g = json.load(open(os.path.join(OUTD, f'{cond}_R{R:g}_x{s:g}', 'gen.json')))
                    _, _, f = M.frc_set(T, 40, g['nframes']); x, y = T['x'][f], T['y'][f]
                img = np.histogram2d(y, x, bins=nb, range=[[cy - hw, cy + hw], [cx - hw, cx + hw]])[0]
                img = gaussian_filter(img, 1.0)
                ax.imshow(img, cmap='inferno', vmax=np.percentile(img, 99.5) + 1e-9)
                if ri == 0: ax.set_title(('GT ' if s is None else '') + LAB.get(m, m) + f' (zoom {zi + 1})', fontsize=6)
                if zi == 0: ax.set_ylabel(lab, fontsize=8)
    fig.suptitle(f'{cond}, R {R:g} nm, {box_nm} nm boxes, FRC-set rows (v2: u <= 0.45), 2 nm pixels, 2 nm blur; rows = photon level', fontsize=8)
    plt.tight_layout(); fn = os.path.join(OUTD, f'renders_{netset}_{cond}_R{R:g}.png'); plt.savefig(fn, dpi=110); plt.close(fig)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--netset', default='default'); ap.add_argument('--cond', default='1809_2')
    ap.add_argument('--root', default='hollow', help="out/papersim/<root>: 'hollow' (30 000 frames) or 'hollow120k' (final, 120 000 frames)")
    a = ap.parse_args()
    global OUTD; OUTD = os.path.join(C.OUT, a.root)
    S = json.load(open(H_STATS := os.path.join(C.REPO, 'out', 'paper2', 'decode_real_stats.json')))[a.cond]
    bg = S['bg_sum6_per_px']['0.5'] / 6; N1 = S['photons_sum6']['0.5']
    ratios, dips, s1, s08 = dip_curve()
    ph = N1 * np.array([0.5, 1, 2, 4, 8, 16, 32, 64])
    CR = crlb_vs_photons(bg, ph)
    AN = dict(sigma_over_R_no_dip=s1, sigma_over_R_dip09=s08, bg_per_frame=bg, median_photons_x1=N1, crlb=CR)
    for Rn in (12.5, 25.0):
        for key, so in (('no_dip', s1), ('dip09', s08)):
            sig = so * Rn
            for kind in ('sf_nm', 'smlm_nm'):
                v = np.array([CR[float(p)][kind] for p in ph])
                AN[f'photons_{kind[:-3]}_crlb_R{Rn:g}_{key}'] = float(np.exp(np.interp(-np.log(sig), -np.log(v), np.log(ph)))) if v.min() < sig else float('inf')
    RUN = runs(a.netset, a.cond)
    tab = []
    for (Rn, s), j in sorted(RUN.items()):
        for m, r in j['methods'].items():
            if 'dip' not in r: continue
            tab.append(dict(R=Rn, scale=s, photons_median=j['window_photons_ge300']['p50'], method=m, n=r['n'], dip=r['dip'], dip_ci95=r['dip_ci95'],
                            R_fit=r['fit_two']['R'], R_ci95=r['R_ci95'], sigma_fit=r['fit_two']['sigma'], p_lrt=r['p_lrt'], dAIC=r['dAIC'],
                            resolved=r['resolved'], rms_gt_nm=r.get('rms_gt_nm'), frc_nm=r.get('frc_nm')))
    json.dump(dict(netset=a.netset, cond=a.cond, analytic=AN, table=tab), open(os.path.join(OUTD, f'summary_{a.netset}_{a.cond}.json'), 'w'), indent=1)
    print(json.dumps(AN, indent=1))
    for t in tab:
        print(f"R {t['R']:5.1f} x{t['scale']:<4g} ph {t['photons_median']:7.0f} {t['method']:7s} n {t['n']:6d} dip {t['dip']:.2f} [{t['dip_ci95'][0]:.2f},{t['dip_ci95'][1]:.2f}] "
              f"R' {t['R_fit']:5.1f} [{t['R_ci95'][0]:.1f},{t['R_ci95'][1]:.1f}] sig {t['sigma_fit']:5.1f} p {t['p_lrt']:.1e} rmsGT {t['rms_gt_nm']} {'RESOLVED' if t['resolved'] else ''}")
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for i, Rn in enumerate((12.5, 25.0)):
        for m in LAB:
            T = sorted([t for t in tab if t['R'] == Rn and t['method'] == m], key=lambda t: t['scale'])
            if not T: continue
            p = [t['photons_median'] for t in T]
            axs[i, 0].errorbar(p, [t['dip'] for t in T], yerr=np.abs(np.array([t['dip_ci95'] for t in T]).T - [t['dip'] for t in T]), color=COL[m], marker='o', label=LAB[m], capsize=2)
            axs[i, 1].plot(p, [t['R_fit'] for t in T], 'o-', color=COL[m], label=LAB[m])
            axs[i, 2].plot(p, [np.mean(t['rms_gt_nm']) if t['rms_gt_nm'] else np.nan for t in T], 'o-', color=COL[m], label=LAB[m])
        axs[i, 0].axhline(1, color='k', lw=0.8); axs[i, 0].set_ylabel(f'R {Rn:g} nm: dip ratio (95 % CI)')
        axs[i, 1].axhline(Rn, color='k', lw=0.8); axs[i, 1].set_ylabel("fitted wall radius R' (nm)")
        for so, ls in ((s1, '--'), (s08, ':')):
            axs[i, 2].axhline(so * Rn, color='k', ls=ls, lw=0.8, label=f'sigma for dip {"1.0" if ls == "--" else "0.9"}')
        axs[i, 2].plot(ph, [CR[float(q)]['sf_nm'] for q in ph], 'k-', lw=0.8, label='CRLB SIMFLUX'); axs[i, 2].plot(ph, [CR[float(q)]['smlm_nm'] for q in ph], color='0.5', lw=0.8, label='CRLB SMLM')
        axs[i, 2].set_ylabel('RMS error vs GT (nm)'); axs[i, 2].set_yscale('log')
        for ax in axs[i]: ax.set_xscale('log'); ax.grid(True, which='both', alpha=0.3); ax.set_xlabel('median window photons')
        axs[i, 0].legend(fontsize=6); axs[i, 2].legend(fontsize=6)
    fig.suptitle(f'hollow microtubules, {a.cond} background ({bg:.2f} ph/px/frame), net set {a.netset}', fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUTD, f'summary_{a.netset}_{a.cond}.png'), dpi=90); plt.close(fig)
    for Rn in (12.5, 25.0):
        sc = sorted(s for (r_, s) in RUN if r_ == Rn)
        if sc: print('renders', render_zooms(a.netset, a.cond, Rn, sc))


if __name__ == '__main__':
    main()
