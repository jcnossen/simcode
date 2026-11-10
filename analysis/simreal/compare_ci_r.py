"""[autoc/compare_ci.py + one guard (skips empty
bootstrap draws in small bins; identical output otherwise); only the import of compare.py (analysis/baselines/compare.py) changed]
simcode_base2 headline with bootstrap CIs (critic 33 #4): compare.py's protocol (W3 windows, 1 px detection matching, global
offset removed, CRLB_ref of the gate scene), plus
  - per label-photon bin: our matched n, SIMCODE's matched n (both vs GT), recall
  - PAIRED (primary): the same GT targets matched by both; RMS/CRLB_ref for SIMCODE and ours and the difference, 95 % CI by
    an event-block bootstrap (resampling GT emitter events = the independent units of a single-FOV movie; all windows of an
    event move together; nboot 400)
  - matched recall: ours top-n (n = SIMCODE's matched count in the bin, re-drawn per bootstrap) by sigma_cal sqrt(max(q, .3))
  - zcal / gross (ours, with CIs)
  - matched-FP CURVE: ours recall (targets >= 200 label ph) vs FP fraction over detection-p thresholds; SIMCODE-net curve over
    its photon threshold (it has no detection score in the npz); SIMCODE filtered outputs as points.
usage: python analysis/simreal/compare_ci_r.py <sim_dir> <gate_scene> <out_prefix> ours=<run_dir> [ours_gtP=<run_dir>] SPEC=kind:path ..."""
import sys
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
import compare as C       # analysis/baselines/compare.py

PX = 108.33
HB = [(200, 700), (700, 2000), (2000, 4000), (4000, 1e9), (200, 2000)]
FB = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9)]
NB = 400


def bname(lo, hi):
    return f'{lo:.0f}-{hi:.0f}' if hi < 1e8 else f'>{lo:.0f}'


def wrms(R, ii, w):
    """RMS / C over rows ii with weights w (per row)"""
    if len(ii) == 0 or w.sum() <= 0: return np.nan
    e, c = R['e'][ii], R['C'][ii]
    return float(np.sqrt((w[:, None] * e ** 2).sum() / w.sum() / 2) / np.sqrt((w * c ** 2).sum() / w.sum()))


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 10 else (np.nan, np.nan)


def main():
    sd, scene, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    specs = [a.split('=', 1) for a in sys.argv[4:]]
    G = dict(np.load(f'{sd}/sim_gt.npz')); NF = int(G['nf'])
    GW = C.gt_windows(G, np.arange(0, NF - 5, 3))
    crf, _ = C.crlb_ref(scene)
    ev = GW[:, 1].astype(int); n_ev = ev.max() + 1
    rng = np.random.default_rng(0)
    W = [np.bincount(rng.integers(0, n_ev, n_ev), minlength=n_ev).astype(float) for _ in range(NB)]
    M, R, D = {}, {}, {}
    for name, ks in specs:
        if '=' not in name and ':' not in ks:
            kind, path = ('ours_est' if name == 'ours' else 'ours_ref'), ks
        else:
            kind, path = ks.split(':', 1)
        Mm = C.load_method(kind, path, NF)
        r = C.evaluate(G, Mm, GW, crf, 2.0, kind.startswith('sc'))
        R[name] = r['_rows']; D[name] = r['det']; M[name] = (kind, path)
    ours = 'ours'; oref = 'ours_gtP' if 'ours_gtP' in R else 'ours'
    L = [f'# {sd} | CRLB_ref gate {scene} | event-block bootstrap ({NB}) 95 % CIs | label-photon bins']
    # ---- n per bin and recall
    L.append('\n== matched n per bin (vs GT) and recall: ' + ' | '.join(R))
    t = GW[:, 5] > 0; P_all = GW[:, 4]
    for lo, hi in FB + [(200, 2000)]:
        nt = int((t & (P_all >= lo) & (P_all < hi)).sum())
        cells = [f'{name} n {int(((R[name]["P"] >= lo) & (R[name]["P"] < hi)).sum()):6d} rec {((R[name]["P"] >= lo) & (R[name]["P"] < hi)).sum() / max(nt, 1):.3f}'
                 for name in R]
        L.append(f'  {bname(lo, hi):>10s} targets {nt:6d} | ' + ' | '.join(cells))
    # ---- ours zcal / gross per bin with CI
    L.append('\n== ours: zcal (1.4826 med |e / s_cal|) / gross |e| > 4 s (%) per bin, 95 % CI')
    for name in [n for n in (ours, oref) if n in R]:
        A = R[name]
        for lo, hi in FB:
            m = np.nonzero((A['P'] >= lo) & (A['P'] < hi))[0]
            if len(m) < 20: continue
            z = np.abs(A['e'][m] / A['s'][m]); g = np.hypot(*(A['e'][m] / A['s'][m]).T) > 4
            evr = ev[A['g'][m]]
            zb = []; gb = []
            for w in W[:200]:
                ww = w[evr]; k = ww > 0
                if not k.any(): continue
                zz = np.repeat(z[k], ww[k].astype(int), 0); zb.append(1.4826 * np.median(zz)); gb.append(100 * np.average(g[k], weights=ww[k]))
            c1, c2 = ci(zb), ci(gb)
            L.append(f'  {name:9s} {bname(lo, hi):>10s} n {len(m):6d} zcal {1.4826 * np.median(z):.2f} [{c1[0]:.2f}, {c1[1]:.2f}] gross {100 * g.mean():5.2f} % [{c2[0]:.2f}, {c2[1]:.2f}]')
    # ---- paired + matched recall per SIMCODE spec
    H = {}
    for name, (kind, path) in M.items():
        if not kind.startswith('sc') or kind == 'sc_net' or 'ndi' in kind: continue
        o0 = oref if '_gt' in path else ours
        A, B = R[o0], R[name]
        com, ia, ib = np.intersect1d(A['g'], B['g'], return_indices=True)
        L.append(f'\n== {name} vs {o0}: PAIRED (primary; same targets) | matched recall (n = SIMCODE matched; ours top-n by sigma_cal sqrt(q))')
        L.append(f'  {"bin":>10s} {"n":>6s} {"SIMCODE":>18s} {"ours":>18s} {"ours - SC":>22s} || {"n":>6s} {"SIMCODE":>18s} {"ours":>18s} {"ours - SC":>22s}')
        H[name] = {}
        for lo, hi in HB:
            m = (A['P'][ia] >= lo) & (A['P'][ia] < hi)
            pa, pb = ia[m], ib[m]
            evp = ev[A['g'][pa]]
            sc0, ou0 = wrms(B, pb, np.ones(len(pb))), wrms(A, pa, np.ones(len(pa)))
            bs, bo, bd = [], [], []
            for w in W:
                ww = w[evp]; a_, b_ = wrms(B, pb, ww), wrms(A, pa, ww); bs.append(a_); bo.append(b_); bd.append(b_ - a_)
            # matched recall
            ms = np.nonzero((B['P'] >= lo) & (B['P'] < hi))[0]; mo = np.nonzero((A['P'] >= lo) & (A['P'] < hi))[0]
            n = len(ms); mr = None
            if n >= 15 and len(mo) >= n:
                order = mo[np.argsort(A['rank'][mo], kind='stable')]
                keep = order[:n]
                s1, o1 = wrms(B, ms, np.ones(n)), wrms(A, keep, np.ones(n))
                es, eo = ev[B['g'][ms]], ev[A['g'][order]]
                cs, co, cd = [], [], []
                for w in W:
                    ws_ = w[es]; nb_ = ws_.sum(); wo = w[eo]; cw = np.cumsum(wo)
                    k = np.searchsorted(cw, nb_); wk = wo.copy(); wk[k + 1:] = 0
                    if k < len(wk): wk[k] = max(nb_ - (cw[k - 1] if k > 0 else 0), 0)
                    a_, b_ = wrms(B, ms, ws_), wrms(A, order, wk); cs.append(a_); co.append(b_); cd.append(b_ - a_)
                mr = (n, s1, ci(cs), o1, ci(co), o1 - s1, ci(cd))
            H[name][bname(lo, hi)] = dict(paired=dict(n=int(m.sum()), sc=sc0, sc_ci=ci(bs), ours=ou0, ours_ci=ci(bo), d=ou0 - sc0, d_ci=ci(bd)),
                                          matched=None if mr is None else dict(n=mr[0], sc=mr[1], sc_ci=mr[2], ours=mr[3], ours_ci=mr[4], d=mr[5], d_ci=mr[6]))
            f3 = lambda v, c: f'{v:5.2f} [{c[0]:.2f},{c[1]:.2f}]'
            fd = lambda v, c: f'{v:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]'
            s = f'  {bname(lo, hi):>10s} {int(m.sum()):6d} {f3(sc0, ci(bs)):>18s} {f3(ou0, ci(bo)):>18s} {fd(ou0 - sc0, ci(bd)):>22s} || '
            s += (f'{mr[0]:6d} {f3(mr[1], mr[2]):>18s} {f3(mr[3], mr[4]):>18s} {fd(mr[5], mr[6]):>22s}' if mr else f'{n:6d} (too few)')
            L.append(s)
    # ---- matched-FP curve
    L.append('\n== matched-FP curve (targets >= 200 label ph; FP = unmatched W3 outputs)')
    for name in [n for n in (ours,) if n in R]:
        A = R[name]; o = np.argsort(-A['det_p'], kind='stable')
        ctp = np.cumsum(A['det_tp'][o]); cfp = np.cumsum(A['det_fp'][o]); nt = A['n_t200']
        fpf = cfp / np.maximum(ctp + cfp, 1)
        L.append(f'  ours (by detection p): ' + ' '.join(
            f'p>{pt:.2f}: rec {ctp[k] / nt:.3f} FPf {fpf[k]:.4f};' for pt in (0.95, 0.9, 0.8, 0.7, 0.63, 0.5)
            for k in [max(np.searchsorted(-A['det_p'][o], -pt, 'right') - 1, 0)]))
        L.append('  ours recall at FP fraction: ' + ' '.join(
            f'{f_:.3f}:{(ctp[np.nonzero(fpf <= f_)[0][-1]] / nt if (fpf <= f_).any() else np.nan):.3f}' for f_ in (0.001, 0.002, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)))
        for sn, (kind, path) in M.items():
            if kind != 'sc_net': continue
            B = R[sn]; Mm = C.load_method(kind, path, NF); w3 = (Mm['w'] % 3 == 0) & (Mm['w'] <= NF - 6)
            ph = Mm['phot'][w3]; o2 = np.argsort(-ph, kind='stable')
            c2t = np.cumsum(B['det_tp'][o2]); c2f = np.cumsum(B['det_fp'][o2]); f2 = c2f / np.maximum(c2t + c2f, 1)
            L.append(f'  {sn} (by its photons): recall at FP fraction: ' + ' '.join(
                f'{f_:.3f}:{(c2t[np.nonzero(f2 <= f_)[0][-1]] / nt if (f2 <= f_).any() else np.nan):.3f}' for f_ in (0.001, 0.002, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03))
                + f' | all: recall {D[sn]["recall"]:.3f} FPf {D[sn]["FP_frac"]:.4f}')
        for sn, (kind, path) in M.items():
            if kind not in ('sc_mrg', 'sc_ndi'): continue
            d = D[sn]; ok = np.nonzero(cfp <= d['FP'])[0]
            L.append(f'  {sn}: recall {d["recall"]:.3f} FP {d["FP"]} | ours at FP <= {d["FP"]}: ' + (f'{ctp[ok[-1]] / nt:.3f}' if len(ok) else 'n/a'))
    import json
    json.dump(H, open(outp + '.json', 'w'), indent=1, default=float)
    open(outp + '.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
