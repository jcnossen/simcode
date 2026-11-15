"""Task 3 (quantify only, no implementation): sub-pixel (d <= 1 px) targets that stay merged after the pass-2 heads, and the
per-target resolvability bound (development p2/bound.py logic; B pattern-free, C pattern known; Neyman-Pearson
simple-vs-simple with oracle hypotheses = an UPPER bound, nearest neighbour only).
Status per target (>= 200 label ph, nearest visible GT neighbour at d = gt_nn): matched / merged / undetected, 1 px
one-to-one on (a) the detector point x_det (identical for p2s and p2L: same detector) and (b) the final solved position
(explicit solve, own calibration, p >= 0.5). bg for the bound: pair scenes 6.1 (flat 0.5 nM, as bound.py); rs scenes the
median head bg_t of the candidates in the target's FOV.
Inputs: the pass-2 gate dumps <out>/gate_dumps (analysis/calib/plib.py). Output: <out_dir>/merged_d1.txt / .json (default <out>/pass2_gate/long;
vs_simcode_p2.py reads merged_d1.json from there for its ideal_C column). Development p2/long/merged_d1.py.
usage: CUDA_VISIBLE_DEVICES=1 python analysis/gate/merged_d1.py [scenes (pair1,rs25qm,rs05qm)] [out_dir]"""
import json, os, sys
import numpy as np, torch
from scipy.stats import norm
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2eval as E        # analysis/gate/p2eval.py (cals, miss_anatomy)
P = E.P                   # analysis/calib/plib.py
from simcode import config


SCEN = (sys.argv[1] if len(sys.argv) > 1 else 'pair1,rs25qm,rs05qm').split(',')
D = sys.argv[2] if len(sys.argv) > 2 else config.out_dir('pass2_gate', 'long')
C = E.cals()
FR = dict(p1=('p1', 'p1'), p2s=('p2', 'p2s'), p2L=('p2L', 'p2L'))
dev = 'cuda'


class BD:
    """pair-bound PSF / illumination of the development p2/bound.py (only the parts used here): vectorial PSF of the
    0.5 nM in-situ calibration on a 21 x 21 ROI, read noise 1.57 e"""
    RN2 = 1.57 ** 2
    _psf = None

    @staticmethod
    @torch.no_grad()
    def Pf(xyz):
        if BD._psf is None:
            from simcode import psf as spsf
            psf = spsf.PSF.load(config.model('insitu/calib_0.5nM.json'), roisize=21, oversample=3, device=dev)   # opt-in calib PSF terms
            BD._psf, BD._focus = psf, psf.focus_um
        return torch.cat([BD._psf(p, torch.full((len(p),), BD._focus, device=dev)) for p in xyz.split(2048)])

    @staticmethod
    def illum(pt, x, y):
        return pt[..., 4] * (1 + pt[..., 3] * torch.cos(pt[..., 0] * x[:, None] + pt[..., 1] * y[:, None] - pt[..., 2]))


def anatomy(R, xo, yo, sel):
    """matched / merged / undetected on positions xo, yo of candidates sel (bool)"""
    idx = np.nonzero(sel)[0]
    og = P.match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], xo[idx], yo[idx], 1.0)
    cand_of = np.full(len(R['gt_x']), -1); cand_of[og[og >= 0]] = idx[og >= 0]
    st = (cand_of >= 0).astype(int)
    _, nbr = E.miss_anatomy(R)
    for g in np.nonzero(st == 0)[0]:
        q = nbr[g]
        if q >= 0 and cand_of[q] >= 0:
            c = cand_of[q]
            if np.hypot(xo[c] - R['gt_x'][g], yo[c] - R['gt_y'][g]) < 1.0: st[g] = 2
    return st


@torch.no_grad()
def bound(R, bgv):
    gx, gy = R['gt_xt'], R['gt_yt']
    tgt = np.nonzero((R['gt_target'] > 0) & (R['gt_nn'] < 1.5))[0]
    part = np.full(len(tgt), -1)
    for f in np.unique(R['gt_fov'][tgt]):
        gi = np.nonzero(R['gt_fov'] == f)[0]; vis = gi[R['gt_phot'][gi] >= 100]
        if len(vis) < 2: continue
        tr = cKDTree(np.c_[gx[vis], gy[vis]]); sel = np.nonzero(R['gt_fov'][tgt] == f)[0]
        d, j = tr.query(np.c_[gx[tgt[sel]], gy[tgt[sel]]], 2)
        part[sel] = np.where(vis[j[:, 0]] == tgt[sel], vis[j[:, 1]], vis[j[:, 0]])
    ok = part >= 0; tgt, part = tgt[ok], part[ok]
    T = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
    out = dict(ti=tgt, d=np.hypot(gx[tgt] - gx[part], gy[tgt] - gy[part]), ph=R['gt_phot'][tgt], ph2=R['gt_phot'][part])
    PB, PC = [], []
    for s in np.array_split(np.arange(len(tgt)), max(1, len(tgt) // 2000)):
        t_, p_ = tgt[s], part[s]
        x1, y1, z1, x2, y2, z2 = T(gx[t_]), T(gy[t_]), T(R['gt_z'][t_]), T(gx[p_]), T(gy[p_]), T(R['gt_z'][p_])
        I1, I2 = T(R['gt_I'][t_]), T(R['gt_I'][p_])
        pt = T(R['patt_true'][R['gt_fov'][t_].astype(int)])
        mx, my = torch.round((x1 + x2) / 2), torch.round((y1 + y2) / 2)
        w1, w2 = I1.sum(1), I2.sum(1); W = (w1 + w2).clamp(min=1e-6)
        xc, yc, zc = (w1 * x1 + w2 * x2) / W, (w1 * y1 + w2 * y2) / W, (w1 * z1 + w2 * z2) / W
        f = lambda x, y, z: BD.Pf(torch.stack([x - mx, y - my, z], 1))
        p1, p2, p0 = f(x1, y1, z1), f(x2, y2, z2), f(xc, yc, zc)
        s1 = I1[..., None, None] * p1[:, None] + I2[..., None, None] * p2[:, None]
        l1, l2, lc = BD.illum(pt, x1, y1), BD.illum(pt, x2, y2), BD.illum(pt, xc, yc)
        u = I1 / l1.clamp(min=1e-3) + I2 / l2.clamp(min=1e-3)
        bg = T(bgv[t_])[:, None, None, None]
        for nm, I0, L_ in (('B', I1 + I2, PB), ('C', u * lc, PC)):
            s0 = I0[..., None, None] * p0[:, None]
            dp = (((s1 - s0) ** 2) / (s0.clamp(min=0) + bg + BD.RN2)).sum((1, 2, 3)).sqrt().cpu().numpy()
            L_.append(norm.cdf(dp - norm.ppf(0.99)))
    br = np.where(out['ph'] > out['ph2'], 1.0, np.where(out['ph'] == out['ph2'], 0.5, 0.0))
    for nm, L_ in (('B', PB), ('C', PC)):
        ps = np.concatenate(L_); out[f'P_{nm}'] = ps; out[f'ideal_{nm}'] = ps + (1 - ps) * br
    return out


L = ['# d <= 1 px targets (>= 200 label ph, eval seeds 301-303): matched / merged / undetected (1 px one-to-one, p >= 0.5)',
     '# xdet = on the detector point (same for p2s / p2L), fin = on the final solved position; bound: mean P_split and ideal recall',
     '# (B pattern-free, C pattern known; NP oracle upper bound); "recoverable" = n x (ideal_C - matched_fin of p2L) targets']
REP = {}
for sc in SCEN:
    Rf = {n: P.load(sc, f) for n, (f, _) in FR.items() if os.path.exists(f'{P.FE}/{sc}__s301_{f}.npz') and FR[n][1] in C}
    LAST = 'p2L' if 'p2L' in Rf else 'p2s'
    ST = {}
    for n, R in Rf.items():
        ST[n + '|xdet'] = anatomy(R, R['sc_x_det'], R['sc_y_det'], R['sc_score'] >= 0.5)
        ia = np.nonzero(R['sc_score'] >= 0.5)[0]
        s = P.solve(R, ia, C[FR[n][1]], 'own')
        xf = np.full(len(R['sc_b']), np.nan); yf = xf.copy(); xf[ia] = s['x']; yf[ia] = s['y']
        ST[n + '|fin'] = anatomy(R, xf, yf, R['sc_score'] >= 0.5)
    R = Rf[LAST]
    if sc.startswith('pair'):
        bgv = np.full(len(R['gt_x']), 6.1)
    else:
        fb = np.full(int(R['n_fov']), np.nan)
        for f in np.unique(R['sc_b']):
            fb[int(f)] = np.median(R['sc_bg'][R['sc_b'] == f])
        bgv = np.nan_to_num(fb[R['gt_fov'].astype(int)], nan=np.nanmedian(fb))
    Bd = bound(R, bgv)
    tg = (R['gt_target'] > 0) & (R['gt_phot'] >= 200)
    L.append(f'\n=== {sc}' + ('' if sc.startswith('pair') else f' (bound bg: per-FOV median head bg, median {np.median(bgv):.1f} ph/px/frame)'))
    for lo, hi in ((0, 0.5), (0.5, 1.0), (0, 1.0), (1.0, 1.5)):
        for plo, phi in ((200, 700), (700, 2000), (2000, 1e9), (200, 1e9)):
            m = tg & (R['gt_nn'] >= lo) & (R['gt_nn'] < hi) & (R['gt_phot'] >= plo) & (R['gt_phot'] < phi)
            if m.sum() < 10: continue
            r = dict(n=int(m.sum()))
            s_ = f'  d {lo:g}-{hi:g} ph {plo}-{phi:g} n {m.sum():5d}'
            for k, st in ST.items():
                r[k] = [int((st[m] == 1).sum()), int((st[m] == 2).sum()), int((st[m] == 0).sum())]
                if k.startswith('p1') or k.startswith('p2s|xdet') or True:
                    s_ += f' | {k} {r[k][0]}/{r[k][1]}/{r[k][2]}'
            ib = np.isin(Bd['ti'], np.nonzero(m)[0])
            if ib.sum() >= 10:
                for nm in ('B', 'C'):
                    r[f'P_{nm}'] = float(Bd[f'P_{nm}'][ib].mean()); r[f'ideal_{nm}'] = float(Bd[f'ideal_{nm}'][ib].mean())
                r['nb'] = int(ib.sum())
                rec = r['n'] * r['ideal_C'] - r[LAST + '|fin'][0]
                r['recoverable_C'] = float(rec)
                s_ += (f' || bound n {ib.sum()} P_split B {r["P_B"]:.2f} C {r["P_C"]:.2f} ideal B {r["ideal_B"]:.2f} C {r["ideal_C"]:.2f}'
                       f' | ideal_C x n {r["n"] * r["ideal_C"]:.0f} vs {LAST} matched {r[LAST + "|fin"][0]} -> recoverable {rec:.0f} ({rec / r["n"]:.0%})')
            REP[f'{sc}|d{lo:g}-{hi:g}|{plo}-{phi:g}'] = r
            L.append(s_)
open(f'{D}/merged_d1.txt', 'w').write('\n'.join(L) + '\n')
json.dump(REP, open(f'{D}/merged_d1.json', 'w'), indent=1)
print('\n'.join(L))
