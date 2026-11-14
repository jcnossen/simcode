"""Pass-1 detector --band A/B and pass-1 head prior-range A/B on the gate scenes (out/detband/gate/<scen>__s<seed>_<tag>.npz,
written by analysis/detband/gate_ab.sh; same GT for every tag, so the comparison is paired).
Per tag and scene (detections with score >= THR, one-to-one matched on the detector position within 1 px):
  recall of the visible targets, overall / < 500 ph, and vs the GT distance to the nearest pixel edge (per axis, both axes
  pooled): edge ratio = recall(|f| > 0.45) / recall(|f| < 0.25) (1 = no edge dip);
  FP (unmatched to ANY active GT within 1 px, interior) per 1000 targets, and dup = unmatched detections within 1 px of a
  GT that is matched to another detection (double detection of one emitter);
  candidates whose pixel is not the nearest pixel (|x_det - ix| > 0.5, the head's 'off' input out of its training range);
  first-harmonic bias A (nm, sim.bias_curve; isolated targets nn > 2 px) of detector / prior / solve, and the RMS of the
  prior and the solve; pass-1 solve and prior RMS / CRLB_SF_multi (gate_report_xdet.prec_stats; 500-5000 ph, CRLB < 25 nm).
usage: python analysis/detband/ab.py d0 b10 ...  -> out/detband/ab_gate.json, ab_gate.txt, fig_ab_gate.png"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pixbias'))
import pblib as B
from sim import bias_curve
from simcode.hd.gate_report import load, match
from simcode.hd.gate_report_xdet import prec_stats

D = os.path.join(B.REPO, 'out', 'detband', 'gate')
OUT = os.path.join(B.REPO, 'out', 'detband')
SCEN = ('iso', 'rs05qm', 'pair1', 'pair2')
THR = float(os.environ.get('AB_THR', '0.63'))
S = 64
EB = np.linspace(0, 0.5, 11)      # |f| bins


def one(tag, sc):
    R, nf = load(D, sc, '_' + tag)
    tgt = R['gt_target'] > 0; NT = int(tgt.sum())
    idx = np.nonzero(R['sc_score'] >= THR)[0]
    ob, ox, oy = R['sc_b'][idx], R['sc_x_det'][idx], R['sc_y_det'][idx]
    og = match(R['gt_fov'], R['gt_x'], R['gt_y'], ob, ox, oy, 1.0)
    inter = (ox >= 9) & (ox <= S - 10) & (oy >= 9) & (oy <= S - 10)
    tp = (og >= 0) & tgt[np.maximum(og, 0)]
    hit = np.zeros(len(tgt), bool); hit[og[tp]] = True
    matched_any = np.zeros(len(tgt), bool); matched_any[og[og >= 0]] = True
    # FP / duplicates among the unmatched interior detections
    un = (og < 0) & inter
    dup = np.zeros(len(idx), bool)
    if un.any():
        ui = np.nonzero(un)[0]
        for f in np.unique(ob[ui]):
            a = ui[ob[ui] == f]; g = np.nonzero(R['gt_fov'] == f)[0]
            if not len(g): continue
            d = np.hypot(ox[a, None] - R['gt_x'][g][None], oy[a, None] - R['gt_y'][g][None])
            near = (d < 1.0) & matched_any[g][None]
            dup[a] = near.any(1)
    r = dict(n_targets=NT, recall=float(hit[tgt].mean()), fp_per_1k=1000 * float((un & ~dup).sum()) / NT, dup_per_1k=1000 * float(dup.sum()) / NT)
    ph = R['gt_phot']
    r['recall_lt500'] = float(hit[tgt & (ph < 500)].mean()) if (tgt & (ph < 500)).sum() > 50 else None
    r['recall_ge500'] = float(hit[tgt & (ph >= 500)].mean())
    # recall vs GT |sub-pixel| (pooled over the axes)
    fx, fy = np.abs(B.sub(R['gt_x'])), np.abs(B.sub(R['gt_y']))
    rec = []
    for lo, hi in zip(EB[:-1], EB[1:]):
        h, n = 0, 0
        for f in (fx, fy):
            m = tgt & (f >= lo) & (f < hi); h += hit[m].sum(); n += m.sum()
        rec.append(h / max(n, 1))
    r['recall_vs_absf'] = [float(v) for v in rec]
    def rr(lo, hi, extra=None):
        h, n = 0, 0
        for f in (fx, fy):
            m = tgt & (f >= lo) & (f < hi) & (extra if extra is not None else True); h += hit[m].sum(); n += m.sum()
        return h / max(n, 1)
    r['edge_ratio'] = float(rr(0.45, 0.5) / max(rr(0, 0.25), 1e-9))
    r['edge_ratio_lt500'] = float(rr(0.45, 0.5, ph < 500) / max(rr(0, 0.25, ph < 500), 1e-9)) if (tgt & (ph < 500)).sum() > 50 else None
    # candidates off their nearest pixel (head 'off' input |.| > 0.5)
    if 'sc_ix' in R:
        offx = R['sc_x_det'][idx] - R['sc_ix'][idx]; offy = R['sc_y_det'][idx] - R['sc_iy'][idx]
        r['frac_off_gt05'] = float(np.mean((np.abs(offx) > 0.5) | (np.abs(offy) > 0.5)))
        r['frac_off_gt06'] = float(np.mean((np.abs(offx) > 0.6) | (np.abs(offy) > 0.6)))
    # positions of the true positives
    oi = idx[tp]; ti = og[tp]
    tx, ty = R['gt_x'][ti].astype(np.float64), R['gt_y'][ti].astype(np.float64)
    nn = R['gt_nn'][ti]; iso = np.isfinite(nn) & (nn > 2.0)
    if iso.sum() < 500: iso = np.ones(len(ti), bool)     # pair scenes: all matched targets (bias then includes the neighbour pull)
    E = dict(det=(R['sc_x_det'][oi], R['sc_y_det'][oi]), prior=(R['sc_x_prior'][oi], R['sc_y_prior'][oi]), solve=(R['sc_x_sc'][oi], R['sc_y_sc'][oi]))
    BI = {}
    for en, (x, y) in E.items():
        q = {}
        for ax, v, t in (('x', x, tx), ('y', y, ty)):
            e = (v.astype(np.float64) - t)[iso]; fg = B.sub(t[iso])
            bc = bias_curve(e, fg)
            bl = bias_curve(e[ph[ti][iso] < 500], fg[ph[ti][iso] < 500]) if (ph[ti][iso] < 500).sum() > 200 else None
            q[ax] = dict(A=bc['A_nm'], A_se=bc['A_se_nm'], rms=bc['rms_nm'], mu=bc['mu_nm'],
                         A_lt500=bl['A_nm'] if bl else None)
        BI[en] = dict(A=0.5 * (q['x']['A'] + q['y']['A']), A_se=0.5 * np.hypot(q['x']['A_se'], q['y']['A_se']),
                      A_lt500=0.5 * (q['x']['A_lt500'] + q['y']['A_lt500']) if q['x']['A_lt500'] is not None else None,
                      rms=float(np.sqrt(0.5 * (q['x']['rms'] ** 2 + q['y']['rms'] ** 2))), mu=q['x']['mu'], mu_y=q['y']['mu'])
    r['bias'] = BI
    # RMS / CRLB (gate_report_xdet convention)
    c = R['gt_sf_m'][ti]; phm = ph[ti]
    tr = np.stack([tx, ty], 1)
    es = np.stack([R['sc_x_sc'][oi], R['sc_y_sc'][oi]], 1) - tr
    ep = np.stack([R['sc_x_prior'][oi], R['sc_y_prior'][oi]], 1) - tr
    res = (phm >= 500) & (phm < 5000) & np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * B.PX < 25) & np.isfinite(es).all(1)
    if res.sum() > 20:
        q = prec_stats(es[res], c[res], R['sc_scal_sc'][oi][res], ep[res]); qp = prec_stats(ep[res], c[res])
        r['solve_rms_crlb'] = q['rms']; r['solve_trim_crlb'] = q['trim']; r['solve_zmed'] = q['zmed']; r['gross_c5'] = q['gross_c5']
        r['prior_rms_crlb'] = qp['rms']; r['prior_trim_crlb'] = qp['trim']; r['n_prec'] = int(res.sum())
    # sig12 selection (main) of the pass-1 solve: count of TPs and RMS (nm)
    s12 = R['sc_scal_sc'][oi].max(1) * B.PX <= 12
    r['tp_sig12_per_target'] = float(s12.sum() / NT)
    r['solve_rms_nm_sig12'] = float(np.sqrt(np.mean(np.clip(es[s12], -0.5, 0.5) ** 2)) * B.PX)
    return r


def table(RES, tags):
    L = []
    cols = [('recall', 'rec', '{:.4f}'), ('recall_lt500', 'rec<500', '{:.4f}'), ('edge_ratio', 'edge', '{:.3f}'),
            ('edge_ratio_lt500', 'edge<500', '{:.3f}'), ('fp_per_1k', 'FP/1k', '{:.1f}'), ('dup_per_1k', 'dup/1k', '{:.1f}'),
            ('frac_off_gt05', 'off>.5', '{:.4f}')]
    for sc in SCEN:
        L.append(f'## {sc} (THR {THR})')
        L.append('tag    ' + ' '.join(f'{h:>8s}' for _, h, _ in cols) + '   A_det  A_pri  A_sol (nm; <500ph)    rmsP  rmsS(nm)  P/CRLB S/CRLB S_trim  gross  tp_s12  rmsS12')
        for t in tags:
            r = RES[t].get(sc)
            if r is None: continue
            s = ' '.join(f'{(fmt.format(r[k]) if r.get(k) is not None else "-"):>8s}' for k, _, fmt in cols)
            bi = r['bias']
            a = ' '.join(f'{bi[e]["A"]:+5.2f}' for e in ('det', 'prior', 'solve'))
            al = ' '.join(f'{bi[e]["A_lt500"]:+5.1f}' if bi[e]['A_lt500'] is not None else '   - ' for e in ('det', 'prior', 'solve'))
            p = f'{r.get("prior_rms_crlb", np.nan):6.3f} {r.get("solve_rms_crlb", np.nan):6.3f} {r.get("solve_trim_crlb", np.nan):6.3f} {r.get("gross_c5", np.nan):5.2f}'
            L.append(f'{t:6s} {s}  {a} ({al})  {bi["prior"]["rms"]:5.1f} {bi["solve"]["rms"]:5.1f}     {p}  {r["tp_sig12_per_target"]:.4f}  {r["solve_rms_nm_sig12"]:.2f}')
        L.append('')
    return '\n'.join(L)


def fig(RES, tags, fn):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, len(SCEN), figsize=(4 * len(SCEN), 7), squeeze=False)
    ec = 0.5 * (EB[1:] + EB[:-1]); fc = (np.arange(10) + 0.5) / 10 - 0.5
    for j, sc in enumerate(SCEN):
        for i, t in enumerate(tags):
            r = RES[t].get(sc)
            if r is None: continue
            axs[0, j].plot(ec, r['recall_vs_absf'], marker='o', ms=3, color=f'C{i}', label=t)
            for k, (en, ls) in enumerate((('det', '-'), ('prior', '--'), ('solve', ':'))):
                mu = 0.5 * (np.array(r['bias'][en]['mu']) + np.array(r['bias'][en]['mu_y']))
                axs[1, j].plot(fc, mu - mu.mean(), color=f'C{i}', ls=ls, label=f'{t} {en}' if j == 0 else None)
        axs[0, j].set_title(f'{sc}: recall vs GT |sub-pixel| (THR {THR})', fontsize=8); axs[0, j].set_xlabel('|f| (px; 0.5 = pixel edge)')
        axs[1, j].set_title(f'{sc}: <est - GT> - mean (nm), iso targets', fontsize=8); axs[1, j].set_xlabel('GT sub-pixel f (px)')
        axs[1, j].axhline(0, color='k', lw=0.5)
    axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(fn, dpi=90); plt.close(fig)


if __name__ == '__main__':
    tags = sys.argv[1:]
    RES = {t: {} for t in tags}
    for t in tags:
        for sc in SCEN:
            try:
                RES[t][sc] = one(t, sc)
            except (ValueError, IndexError, KeyError) as ex:
                print('skip', t, sc, repr(ex)[:120])
    nm = os.environ.get('AB_NAME', 'gate')
    json.dump(RES, open(f'{OUT}/ab_{nm}.json', 'w'), indent=1)
    txt = table(RES, tags); open(f'{OUT}/ab_{nm}.txt', 'w').write(txt); print(txt)
    fig(RES, tags, f'{OUT}/fig_ab_{nm}.png')
