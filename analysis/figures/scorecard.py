"""Sim-realism scorecard (2026-11-10, "sim realism + p2L prior weight"): the SAME statistics on real and sim pass-1 outputs.
Pass 1 = det_n1 + sc5nb head (pass1_<f0>.npz) + its pattern solve (pass2_est_<f0>.npz) with the cmult-0.10 calibration
(real: calib_sc_hd_sc5nbb15_pm_bright_real.json; sim: ..._bright.json, identical except the sigma factor, so positions / q_dof /
dnll are comparable; the b22 sim runs used in realgap/simreal.py have cmult 0.22, which flattens q_dof).
Statistics (all CPU):
  phot      network window photons (label units) quantiles + log histogram, solve rows p >= 0.63 in the 5-115 crop
  dnll      frac dnll < 2 and median dnll per ours-photon bin
  q         median q_dof per ours-photon bin
  det       detector level (pass1 rows p >= 0.5, all windows of a subsample): detections / window, frac nn < 1.5 px / < 2.5 px
  bg        head bg median (pass1 bg, mean over frames)
  prior     robust sd (1.4826 MAD, nm) of final - prior per photon bin; pixel-phase uniformity (min/mean of 20-bin frac-x / -y
            histograms, mean of x and y) of final and prior
  raw       raw-frame stats (n20/density_bg.raw_stats definition): mean, p10, median-image high-pass sd, excess temporal
            variance EV (mean / median / p90 / share of total EV in the top 10 % pixels) -> spatial EV map saved
usage: python scorecard.py name=<run_dir>:<f0>:<movie|maps.npz>:<raw_f0>:<raw_n>[:offmap:rnmap] ...  (json merged into
       <out>/simreal2/scorecard.json by name); python scorecard.py --plot a,b,c,... out.png; --print [a,b,...]
       (run dirs / movies: absolute, or relative to the current directory; camera maps default models/insitu/camera_maps_0.5nM.npz)
plots/realism_scorecard.png = two --plot panels stacked (the development composite; same panels, header layout re-created):
  python scorecard.py --stack 'real05,m05qh,m05r@0.5 nM: real (black) vs m05qh (old) vs m05r (realistic)' \\
      'real25,m25qh,m25r@2.5 nM: real (black) vs m25qh (old) vs m25r (realistic)' <out>/plots/realism_scorecard.png"""
import json, os, sys
import numpy as np

from simcode2 import config
HERE = os.path.join(config.root('out'), 'simreal2')         # scorecard.json (development: next to the script)
SP = os.path.join(config.root('out'), 'scratch')
CAM05 = config.model('insitu/camera_maps_0.5nM.npz')
PX = 108.33
G = 0.56
PB = ((300, 700), (700, 1200), (1200, 2000), (2000, 4000))
mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) > 20 else np.nan


def nn_same_window(w, x, y):
    o = np.argsort(w, kind='stable'); w, x, y = w[o], x[o], y[o]
    nn = np.full(len(w), 99.0)
    cut = np.flatnonzero(np.diff(w)) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(w)]):
        if b - a < 2: continue
        X, Y = x[a:b], y[a:b]
        D = np.hypot(X[:, None] - X[None], Y[:, None] - Y[None]); np.fill_diagonal(D, 1e9)
        nn[a:b] = D.min(1)
    r = np.empty_like(nn); r[o] = nn
    return r


def uniformity(v):
    h, _ = np.histogram(np.mod(v, 1.0), 20, (0, 1))
    return float(h.min() / h.mean()) if h.sum() > 200 else np.nan


def raw_stats(movie, f0, n, offmap, rnmap):
    import tifffile
    from scipy import ndimage as ndi
    t = tifffile.TiffFile(movie)
    adu = np.stack([t.pages[i].asarray() for i in range(f0, f0 + n)]).astype(np.float32)
    off = np.load(offmap)['offset_adu'].astype(np.float32); rn2 = np.load(rnmap)['readnoise_e'].astype(np.float32) ** 2
    e = (adu - off) * G
    mean = e.mean(0); med = np.median(e, 0); p10 = np.percentile(e, 10, 0)
    B = 600; ev = []
    for b0 in range(0, n - B + 1, B):
        for s in range(6):
            fr = e[b0 + ((s - (f0 + b0)) % 6)::6][:B // 6]
            ev.append(fr.var(0, ddof=1) - fr.mean(0) - rn2)
    ev = np.mean(ev, 0)
    return dict(mean=mean, med=med, p10=p10, ev=ev)


def raw_summary(M):
    from scipy import ndimage as ndi
    c = slice(5, 115)
    mean, med, p10, ev = M['mean'], M['med'], M['p10'], M['ev']
    E = ev[c, c].ravel(); Es = np.sort(E)[::-1]
    return dict(mean=float(mean[c, c].mean()), p10=float(p10[c, c].mean()), above_p10=float(mean[c, c].mean() - p10[c, c].mean()),
                med_hp_sd=float((med - ndi.median_filter(med, 9))[c, c].std()), med_lf_sd=float(ndi.gaussian_filter(med, 6)[c, c].std()),
                ev_mean=float(E.mean()), ev_median=float(np.median(E)), ev_p90=float(np.percentile(E, 90)),
                ev_top10_share=float(Es[:len(Es) // 10].sum() / max(Es.sum(), 1e-9)),
                ev_over_light=float(E.mean() / max(mean[c, c].mean() - p10[c, c].mean(), 1e-3)))


def card(run, f0, raw_src, raw_f0, raw_n, offmap=None, rnmap=None, name='x'):
    P1 = np.load(f'{run}/pass1_{f0}.npz'); S = np.load(f'{run}/pass2_est_{f0}.npz')
    o = {}
    # ---- solve rows
    m = S['p'] >= 0.63
    x, y = S['x_hy0'], S['y_hy0']
    m &= (x >= 5) & (x < 115) & (y >= 5) & (y < 115) & np.isfinite(x)
    ph, dn, q = S['phot'][m], S['dnll'][m], S['q_dof'][m]
    o['n_rows'] = int(m.sum()); o['n_windows'] = int(len(np.unique(S['wstart'])))
    o['rows_per_window'] = o['n_rows'] / o['n_windows']
    o['phot_q'] = [float(v) for v in np.percentile(ph, [10, 25, 50, 75, 90])]
    he = np.geomspace(100, 20000, 41); h, _ = np.histogram(ph, he); o['phot_hist'] = (h / h.sum()).tolist(); o['phot_edges'] = he.tolist()
    dx = (S['x_hy0'][m] - S['x_prior'][m]) * PX; dy = (S['y_hy0'][m] - S['y_prior'][m]) * PX
    sxp = None
    o['bins'] = {}
    for lo, hi in PB:
        b = (ph >= lo) & (ph < hi)
        if b.sum() < 50: continue
        o['bins'][f'{lo}-{hi}'] = dict(n=int(b.sum()), dnll_lt2=float(np.mean(dn[b] < 2)), dnll_lt3=float(np.mean(dn[b] < 3)),
                                       dnll_med=float(np.median(dn[b])), q_med=float(np.median(q[b])),
                                       d_prior_final_sd=float(0.5 * (mad(dx[b]) + mad(dy[b]))),
                                       sx_cal_med=float(np.median(np.hypot(S['sx_cal'][m][b], S['sy_cal'][m][b]) / np.sqrt(2) * PX)),
                                       unif_final=0.5 * (uniformity(S['x_hy0'][m][b]) + uniformity(S['y_hy0'][m][b])),
                                       unif_prior=0.5 * (uniformity(S['x_prior'][m][b]) + uniformity(S['y_prior'][m][b])))
    o['unif_final_all'] = 0.5 * (uniformity(S['x_hy0'][m]) + uniformity(S['y_hy0'][m]))
    o['unif_prior_all'] = 0.5 * (uniformity(S['x_prior'][m]) + uniformity(S['y_prior'][m]))
    # ---- detector level
    w = P1['wstart']; uw = np.unique(w)
    sel_w = uw[np.linspace(0, len(uw) - 1, min(400, len(uw))).astype(int)]
    k = np.isin(w, sel_w) & (P1['p'] >= 0.5)
    nn = nn_same_window(w[k], P1['x'][k], P1['y'][k])
    X, Y = P1['x'][k], P1['y'][k]; ins = (X >= 5) & (X < 115) & (Y >= 5) & (Y < 115)
    o['det_per_window'] = float(ins.sum() / len(sel_w))
    o['nn_lt15'] = float((nn[ins] < 1.5).mean()); o['nn_lt25'] = float((nn[ins] < 2.5).mean())
    o['det_phot_q'] = [float(v) for v in np.percentile(P1['phot'][k][ins], [10, 50, 90])]
    o['head_bg_med'] = float(np.median(P1['bg'][k][ins].mean(1)))
    o['head_sxy_med_nm'] = float(np.median(np.hypot(P1['sx'][k][ins], P1['sy'][k][ins]) / np.sqrt(2) * PX))
    pp = P1['p'][k][ins]; o['p_hist'] = [float(np.mean((pp >= a_) & (pp < b_))) for a_, b_ in ((0.5, 0.63), (0.63, 0.8), (0.8, 0.95), (0.95, 1.01))]
    php = P1['phot'][k][ins]; sxh = np.hypot(P1['sx'][k][ins], P1['sy'][k][ins]) / np.sqrt(2) * PX
    o['head_sx_bins'] = {f'{lo}-{hi}': float(np.median(sxh[(php >= lo) & (php < hi) & (pp >= 0.63)])) for lo, hi in PB if np.sum((php >= lo) & (php < hi) & (pp >= 0.63)) > 50}
    # ---- raw frames
    if raw_src.endswith('.npz'):
        M = dict(np.load(raw_src))
    else:
        M = raw_stats(raw_src, raw_f0, raw_n, offmap, rnmap)
        np.savez_compressed(f'{SP}/simreal2/rawmaps_{name}.npz', **M)
    o['raw'] = raw_summary(M); o['raw_maps'] = raw_src if raw_src.endswith('.npz') else f'{SP}/simreal2/rawmaps_{name}.npz'
    o['src'] = dict(run=run, f0=f0, raw=raw_src, raw_f0=raw_f0, raw_n=raw_n)
    return o


def fmt(nm, r):
    s = [f'{nm}: rows/win {r["rows_per_window"]:.1f}  det/win {r["det_per_window"]:.0f}  nn<1.5 {r["nn_lt15"]:.3f}  nn<2.5 {r["nn_lt25"]:.3f}  '
         f'phot q10/50/90 {r["phot_q"][0]:.0f}/{r["phot_q"][2]:.0f}/{r["phot_q"][4]:.0f}  head bg {r["head_bg_med"]:.2f}  '
         f'unif final/prior {r["unif_final_all"]:.2f}/{r["unif_prior_all"]:.2f}',
         f'    p hist [.5,.63,.8,.95,1] {np.round(r.get("p_hist", []), 3).tolist()}  head sx (p>=.63) per bin ' + ' '.join(f'{b_}:{v_:.1f}' for b_, v_ in r.get('head_sx_bins', {}).items())]
    R = r['raw']
    s.append(f'    raw: mean {R["mean"]:.2f} p10 {R["p10"]:.2f} above_p10 {R["above_p10"]:.2f} hp_sd {R["med_hp_sd"]:.2f} lf_sd {R["med_lf_sd"]:.2f} | '
             f'EV mean {R["ev_mean"]:.2f} med {R["ev_median"]:.2f} p90 {R["ev_p90"]:.2f} top10 {R["ev_top10_share"]:.2f} EV/light {R["ev_over_light"]:.2f}')
    for b, v in r['bins'].items():
        s.append(f'    {b:>10}: n {v["n"]:6d} dnll<2 {v["dnll_lt2"]:.3f} med dnll {v["dnll_med"]:5.1f} q {v["q_med"]:.2f} '
                 f'sd(final-prior) {v["d_prior_final_sd"]:.2f} nm sx_cal {v["sx_cal_med"]:.1f} unif f/p {v["unif_final"]:.2f}/{v["unif_prior"]:.2f}')
    return '\n'.join(s)


def plot(names, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.load(open(f'{HERE}/scorecard.json'))
    fig, ax = plt.subplots(2, 4, figsize=(19, 8.5)); ax = ax.ravel()
    sty = {}
    cols = {'real': ['#000000', '#555555'], 'sim': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']}
    ci = {'real': 0, 'sim': 0}
    for nm in names:
        kind = 'real' if nm.startswith('real') else 'sim'
        c = cols[kind][ci[kind] % len(cols[kind])]; ci[kind] += 1
        sty[nm] = dict(color=c, ls='-' if kind == 'real' else '--', lw=2.2 if kind == 'real' else 1.5, marker='o', ms=4)
    for nm in names:
        r = R[nm]; st = sty[nm]
        e = np.array(r['phot_edges']); cen = np.sqrt(e[1:] * e[:-1])
        ax[0].plot(cen, r['phot_hist'], label=nm, **{**st, 'marker': None})
        bc = [np.sqrt(float(b.split('-')[0]) * float(b.split('-')[1])) for b in r['bins']]
        ax[1].plot(bc, [v['dnll_lt2'] for v in r['bins'].values()], **st)
        ax[2].plot(bc, [v['q_med'] for v in r['bins'].values()], **st)
        ax[3].plot(bc, [v['d_prior_final_sd'] for v in r['bins'].values()], **st)
        ax[7].plot(bc, [v['unif_final'] for v in r['bins'].values()], **st)
        ax[7].plot(bc, [v['unif_prior'] for v in r['bins'].values()], **{**st, 'marker': 'x', 'alpha': 0.6})
    xs = np.arange(len(names))
    ax[4].bar(xs - 0.2, [R[n]['nn_lt15'] for n in names], 0.4, color=[sty[n]['color'] for n in names], label='nn < 1.5 px')
    ax[4].bar(xs + 0.2, [R[n]['det_per_window'] / 1000 for n in names], 0.4, color=[sty[n]['color'] for n in names], alpha=0.4, hatch='//', label='det / window / 1000')
    ax[4].set_xticks(xs, names, rotation=40, ha='right', fontsize=8); ax[4].set_title('nn<1.5 px frac (solid); det/win/1000 (hatch)')
    ax[5].bar(xs - 0.2, [R[n]['raw']['ev_mean'] for n in names], 0.4, color=[sty[n]['color'] for n in names])
    ax[5].bar(xs + 0.2, [R[n]['raw']['ev_median'] for n in names], 0.4, color=[sty[n]['color'] for n in names], alpha=0.4, hatch='//')
    ax[5].set_xticks(xs, names, rotation=40, ha='right', fontsize=8); ax[5].set_title('raw EV mean (solid), median (hatch)')
    ax5b = ax[5].twinx(); ax5b.plot(xs, [R[n]['raw']['ev_top10_share'] for n in names], 'k^', label='top-10 % pixel share'); ax5b.set_ylim(0, 1); ax5b.set_ylabel('EV share of top 10 % pixels (^)')
    ax[6].bar(xs - 0.2, [R[n]['head_bg_med'] for n in names], 0.4, color=[sty[n]['color'] for n in names])
    ax[6].bar(xs + 0.2, [R[n]['raw']['above_p10'] for n in names], 0.4, color=[sty[n]['color'] for n in names], alpha=0.4, hatch='//')
    ax[6].set_xticks(xs, names, rotation=40, ha='right', fontsize=8); ax[6].set_title('head bg (solid); raw mean-p10 (hatch)')
    for a, t in zip(ax[:4], ('network window photons (p >= 0.63 rows)', 'frac dnll < 2 (fringe-ambiguous) vs photons',
                              'median q_dof vs photons', 'robust sd(final - prior) nm vs photons')):
        a.set_xscale('log'); a.set_title(t, fontsize=10); a.set_xlabel('ours network photons (label units)'); a.grid(alpha=0.3)
    ax[1].set_yscale('log')
    ax[7].set_xscale('log'); ax[7].set_title('pixel uniformity: final (o), prior (x)'); ax[7].grid(alpha=0.3); ax[7].set_xlabel('ours network photons')
    ax[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=75)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit()
    if sys.argv[1] == '--plot':
        plot(sys.argv[2].split(','), sys.argv[3]); sys.exit()
    if sys.argv[1] == '--stack':          # --stack 'a,b,c@title' 'd,e,f@title' out.png: --plot panels stacked, each block = 25 px title line + panel + 5 px pad
        import tempfile, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.image as mi
        ims, ttl = [], []
        for i, g in enumerate(sys.argv[2:-1]):
            nm, _, tt = g.partition('@')
            f = os.path.join(tempfile.mkdtemp(), f'p{i}.png'); plot(nm.split(','), f); ims.append(mi.imread(f)); ttl.append(tt)
        HD, PD = 25, 5; W = max(im.shape[1] for im in ims); Ht = sum(im.shape[0] + HD + PD for im in ims)
        fig = plt.figure(figsize=(W / 100, Ht / 100), dpi=100); y = Ht
        for im, tt in zip(ims, ttl):
            fig.text(6 / W, (y - HD / 2) / Ht, tt, fontsize=7, va='center', ha='left')
            y -= HD + im.shape[0]; fig.figimage(im, xo=0, yo=y); y -= PD
        fig.savefig(sys.argv[-1], dpi=100); sys.exit()
    os.makedirs(f'{SP}/simreal2', exist_ok=True)
    if sys.argv[1] == '--print':
        R = json.load(open(f'{HERE}/scorecard.json'))
        for nm in (sys.argv[2].split(',') if len(sys.argv) > 2 else R): print(fmt(nm, R[nm]))
        sys.exit()
    R = json.load(open(f'{HERE}/scorecard.json')) if os.path.exists(f'{HERE}/scorecard.json') else {}
    for a in sys.argv[1:]:
        nm, spec = a.split('=', 1); p = spec.split(':')
        run, f0, raw, rf0, rn = p[0], int(p[1]), p[2], int(p[3]), int(p[4])
        offm = p[5] if len(p) > 5 else CAM05; rnm = p[6] if len(p) > 6 else CAM05
        r = card(run, f0, raw, rf0, rn, offm, rnm, name=nm)
        R = json.load(open(f'{HERE}/scorecard.json')) if os.path.exists(f'{HERE}/scorecard.json') else {}
        R[nm] = r; json.dump(R, open(f'{HERE}/scorecard.json', 'w'), indent=1)
        print(fmt(nm, r), flush=True)
