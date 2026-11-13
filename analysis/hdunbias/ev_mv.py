"""Movie check of the pass-1 head arms (out/hdunbias/movie_<D>_<tag>, analysis/hdunbias/movie.py; detector unchanged).
(1) per estimate (P1: det / prior / solve; P2: det / prior / solve) and selection (p63, sig12): GT bias A (nm; isolated
    matched rows, analysis/pixbias/sim.py gt_rows + bias_curve), overall and per photon bin, and RMS (nm, |e| < 0.5 px)
    per photon bin;
(2) recall / FP / final RMS of the P1 and P2 tables (analysis/detband/mv.py evaluate).
usage: python analysis/hdunbias/ev_mv.py Dexp1 base ctl J P JP -> out/hdunbias/ev_mv_<D>.json / .txt"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
for p in ('pixbias', 'papersim', 'detband'): sys.path.insert(0, os.path.join(REPO, 'analysis', p))
import pblib as B
from sim import bias_curve, gt_rows
import mv as MV
import metrics as M
from simcode import strict as SP
OUT = os.path.join(REPO, 'out', 'hdunbias')


def rows(d, gt, nframes):
    out = {}
    for ps, tag, tab in (('P1', 'est', 'locs_p1_sim.npz'), ('P2', 'est2', 'locs_p2_sim.npz')):
        keys = ['wstart', 'x_hy0', 'y_hy0', 'x_prior', 'y_prior', 'phot'] + (['x_det', 'y_det'] if ps == 'P2' else [])
        E = B.load_rows(d, tag, keys); T = dict(np.load(os.path.join(d, tab)))
        assert len(T['x']) == len(E['x_hy0'])
        if ps == 'P1': E['x_det'], E['y_det'] = B.join_p1_det(B.load_pass1(d), E)
        w = E['wstart'].astype(np.int64)
        xr = np.where(np.isfinite(E['x_det']), E['x_det'], E['x_prior']); yr = np.where(np.isfinite(E['y_det']), E['y_det'], E['y_prior'])
        tx, ty, tph = gt_rows(gt, nframes, w, xr, yr)
        est = dict(det=(E['x_det'], E['y_det']), prior=(E['x_prior'], E['y_prior']), solve=(E['x_hy0'], E['y_hy0']))
        out[ps] = dict(est=est, tx=tx, ty=ty, ph=E['phot'].astype(np.float64), sel=dict(p63=T['good'].astype(bool), sig12=SP.select(T, 'main')), T=T)
    return out


def stats(G):
    R = {}
    for sn, m0 in G['sel'].items():
        m = m0 & np.isfinite(G['tx'])
        r = {}
        for en, (x, y) in G['est'].items():
            x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
            mm = m & np.isfinite(x)
            q = dict(n=int(mm.sum()))
            for lab, b in [('all', np.ones(len(x), bool))] + B.ph_bin_masks(G['ph']):
                k = mm & b
                if k.sum() < 200: continue
                A, rms = [], []
                for v, t in ((x, G['tx']), (y, G['ty'])):
                    bc = bias_curve(v[k] - t[k], B.sub(t[k])); A.append(bc['A_nm']); rms.append(bc['rms_nm'])
                    if lab == 'all': q.setdefault('curve', []).append(bc['mu_nm'])
                q[lab] = dict(n=int(k.sum()), A=float(np.mean(A)), rms=float(np.sqrt(np.mean(np.square(rms)))))
            r[en] = q
        R[sn] = r
    return R


if __name__ == '__main__':
    D = sys.argv[1]; tags = sys.argv[2:]
    md = os.path.join(REPO, 'out', 'papersim', D)
    gt = dict(np.load(os.path.join(md, 'gt.npz'))); nf = int(gt['nframes']); W = int(gt['W'])
    GW = M.gt_windows(gt, nf, np.arange(0, nf - 5, 6))
    RES = {}
    for t in tags:
        d = os.path.join(OUT, f'movie_{D}_{t}')
        X = rows(d, gt, nf)
        RES[t] = {ps: dict(bias=stats(X[ps]), tab={sn: MV.evaluate(X[ps]['T'], sel, GW, W) for sn, sel in X[ps]['sel'].items()}) for ps in X}
    json.dump(RES, open(os.path.join(OUT, f'ev_mv_{D}.json'), 'w'), indent=1)
    L = [f'# {D}: bias A (nm) / RMS (nm) of the matched isolated rows; table recall / FP / final RMS (disjoint windows, >= 150 ph)']
    bins = ['all'] + [lab for lab, _ in B.ph_bin_masks(np.zeros(1))]
    for ps in ('P1', 'P2'):
        for sn in ('p63', 'sig12'):
            L.append(f'## {ps} {sn}: A / RMS per photon bin ({", ".join(bins)}); recall, rec<500, FP/1k, tab rms_nm, tab A_nm')
            for en in ('det', 'prior', 'solve'):
                for t in tags:
                    q = RES[t][ps]['bias'][sn][en]
                    cells = ' '.join(f'{q[b]["A"]:+5.2f}/{q[b]["rms"]:5.1f}' if b in q else '     -     ' for b in bins)
                    tb = RES[t][ps]['tab'][sn]
                    tail = f'  rec {tb["recall"]:.4f} {tb["recall_lt500"]:.4f} FP {tb["fp_per_1k"]:6.1f} rms {tb["rms_nm"]:5.2f} A {tb["A_nm"]:+.2f}' if en == 'solve' else ''
                    L.append(f'{en:6s} {t:5s} {cells}{tail}')
            L.append('')
    txt = '\n'.join(L); print(txt); open(os.path.join(OUT, f'ev_mv_{D}.txt'), 'w').write(txt)
