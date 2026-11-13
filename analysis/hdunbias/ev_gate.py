"""Gate A/B of the pass-1 head arms (out/hdunbias/gate/<scen>__s<seed>_<tag>.npz, analysis/hdunbias/gate.sh; detector
unchanged -> same detections and GT for every tag, paired comparison). Standard table from analysis/detband/ab.py (recall,
FP, bias A of det / prior / solve overall and < 500 ph, RMS / CRLB at 500-5000 ph); plus per photon bin (TPs, score >= THR,
matched on x_det within 1 px; bias on isolated targets nn > 2 px, else all): prior bias A, prior RMS / SMLM CRLB (gt_sm_m),
solve RMS / SIMFLUX CRLB (gt_sf_m), each as sqrt(mean(e^2) / mean(crlb^2)) on |e| < 0.5 px, and prior / solve gross
fraction (|e| > 0.5 px).
usage: python analysis/hdunbias/ev_gate.py base ctl J P JP -> out/hdunbias/ev_gate.json / .txt, fig_gate_bias.png"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'analysis', 'detband')); sys.path.insert(0, os.path.join(REPO, 'analysis', 'pixbias'))
import ab
import pblib as B
from sim import bias_curve
from simcode.hd.gate_report import load, match
OUT = os.path.join(REPO, 'out', 'hdunbias'); ab.D = os.path.join(OUT, 'gate'); ab.OUT = OUT
PHB = ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 1e9))


def extra(tag, sc):
    R, nf = load(ab.D, sc, '_' + tag)
    tgt = R['gt_target'] > 0
    idx = np.nonzero(R['sc_score'] >= ab.THR)[0]
    og = match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], R['sc_x_det'][idx], R['sc_y_det'][idx], 1.0)
    tp = (og >= 0) & tgt[np.maximum(og, 0)]
    oi = idx[tp]; ti = og[tp]
    tx, ty = R['gt_x'][ti].astype(np.float64), R['gt_y'][ti].astype(np.float64)
    ph = R['gt_phot'][ti]; nn = R['gt_nn'][ti]; iso = np.isfinite(nn) & (nn > 2.0)
    if iso.sum() < 500: iso = np.ones(len(ti), bool)
    E = dict(prior=(R['sc_x_prior'][oi], R['sc_y_prior'][oi]), solve=(R['sc_x_sc'][oi], R['sc_y_sc'][oi]), det=(R['sc_x_det'][oi], R['sc_y_det'][oi]))
    C = dict(prior=R['gt_sm_m'][ti], solve=R['gt_sf_m'][ti], det=R['gt_sm_m'][ti])
    out = {}
    for lo, hi in PHB:
        b = (ph >= lo) & (ph < hi)
        if b.sum() < 100: continue
        r = dict(n=int(b.sum()))
        for en, (x, y) in E.items():
            e = np.stack([x - tx, y - ty], 1).astype(np.float64); c = C[en]
            ok = b[:, None] & np.isfinite(c) & (np.abs(e) < 0.5)
            r[f'{en}_rms_crlb'] = float(np.sqrt(np.mean(e[ok] ** 2) / np.mean(c[ok] ** 2)))
            r[f'{en}_rms_nm'] = float(np.sqrt(np.mean(e[ok] ** 2)) * B.PX)
            r[f'{en}_gross'] = float(np.mean((np.abs(e[b]) > 0.5).any(1)))
            bb = b & iso
            A = []
            for ax in (0, 1):
                bc = bias_curve(e[bb, ax], B.sub((tx, ty)[ax][bb]))
                A.append(bc['A_nm'] if bc else np.nan)
            r[f'{en}_A'] = float(np.mean(A))
        out[f'{lo}-{hi:.0f}'] = r
    # bias curve of the prior (pooled axes, iso), overall and < 500 ph
    cur = {}
    for nm, k in (('all', iso), ('lt500', iso & (ph < 500))):
        mu = []
        for ax, t in ((0, tx), (1, ty)):
            e = (E['prior'][ax] - t).astype(np.float64)[k]; bc = bias_curve(e, B.sub(t[k]))
            mu.append(np.array(bc['mu_nm']) if bc else np.full(10, np.nan))
        cur[nm] = (0.5 * (mu[0] + mu[1])).tolist()
    return dict(phbins=out, curve=cur)


if __name__ == '__main__':
    tags = sys.argv[1:]
    RES = {t: {} for t in tags}
    for t in tags:
        for sc in ab.SCEN:
            try:
                RES[t][sc] = ab.one(t, sc); RES[t][sc].update(extra(t, sc))
            except (ValueError, IndexError, KeyError, FileNotFoundError) as ex:
                print('skip', t, sc, repr(ex)[:120])
    json.dump(RES, open(f'{OUT}/ev_gate.json', 'w'), indent=1)
    txt = ab.table(RES, tags)
    L = [txt, '# per photon bin: A_prior (nm) | prior RMS / SMLM CRLB | solve RMS / SF CRLB | prior gross % | solve gross %']
    for sc in ab.SCEN:
        L.append(f'## {sc}')
        bins = [f'{lo}-{hi:.0f}' for lo, hi in PHB]
        L.append('tag    ' + ' '.join(f'{b:>32s}' for b in bins))
        for t in tags:
            r = RES[t].get(sc)
            if r is None: continue
            cells = []
            for bn in bins:
                q = r['phbins'].get(bn)
                cells.append(f'{q["prior_A"]:+5.1f} {q["prior_rms_crlb"]:5.2f} {q["solve_rms_crlb"]:5.2f} {100*q["prior_gross"]:5.2f} {100*q["solve_gross"]:5.2f}' if q else ' ' * 32)
            L.append(f'{t:6s} ' + ' '.join(f'{c:>32s}' for c in cells))
        L.append('')
    txt = '\n'.join(L); open(f'{OUT}/ev_gate.txt', 'w').write(txt); print(txt)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, len(ab.SCEN), figsize=(4 * len(ab.SCEN), 7), squeeze=False)
    fc = (np.arange(10) + 0.5) / 10 - 0.5
    for j, sc in enumerate(ab.SCEN):
        for i, t in enumerate(tags):
            r = RES[t].get(sc)
            if r is None: continue
            for k, nm in enumerate(('all', 'lt500')):
                mu = np.array(r['curve'][nm]); axs[k, j].plot(fc, mu - np.nanmean(mu), color=f'C{i}', marker='o', ms=3, label=t)
        for k, nm in enumerate(('all photons', '< 500 ph')):
            axs[k, j].set_title(f'{sc}: prior <est - GT> - mean (nm), {nm}', fontsize=8); axs[k, j].axhline(0, color='k', lw=0.5)
            axs[k, j].set_xlabel('GT sub-pixel f (px)')
    axs[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{OUT}/fig_gate_bias.png', dpi=90); plt.close(fig)
