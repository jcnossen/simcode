"""Movie check of the detector A/B: papersim movie runs out/detband/movie_<D>_<tag> (analysis/detband/movie.py).
Disjoint windows (frame % 6 == 0); targets = GT emitters ON in the window with >= 150 window photons, 2 px inside the FOV;
rows = pass-1 (locs_p1_sim) and final pass-2 (locs_p2_sim) tables, selections p63 ('good') and sig12 (strict.select main);
one-to-one matching per window within 100 nm (gate_report.match). Reports recall, recall vs GT |sub-pixel| (edge ratio
= recall(|f| > 0.45) / recall(|f| < 0.25), pooled axes), FP (unmatched rows) per 1000 targets, matched RMS (nm), first-
harmonic bias A (nm) of the final position.
usage: python analysis/detband/mv.py Dexp1 d0 b10 ...  -> out/detband/mv_<D>.json / .txt"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'analysis', 'pixbias')); sys.path.insert(0, os.path.join(REPO, 'analysis', 'papersim'))
import pblib as B
from sim import bias_curve
import metrics as M
from simcode.hd.gate_report import match
from simcode import strict as SP

EB = np.linspace(0, 0.5, 11)


def evaluate(T, sel, GW, W):
    w = T['frame'].astype(np.int64)
    m = sel & (w % 6 == 0)
    keys = sorted(GW)
    gb, gx, gy, gph = [], [], [], []
    for f in keys:
        x, y, ph = GW[f]; gb.append(np.full(len(x), f)); gx.append(x); gy.append(y); gph.append(ph)
    gb, gx, gy, gph = (np.concatenate(v) for v in (gb, gx, gy, gph))
    act = gph >= 30
    gb, gx, gy, gph = gb[act], gx[act], gy[act], gph[act]
    tg = (gph >= 150) & (gx >= 2) & (gx < W - 2) & (gy >= 2) & (gy < W - 2)
    idx = np.nonzero(m)[0]
    og = match(gb, gx, gy, w[idx], T['x'][idx].astype(np.float64), T['y'][idx].astype(np.float64), 100 / B.PX)
    inter = (T['x'][idx] >= 2) & (T['x'][idx] < W - 2) & (T['y'][idx] >= 2) & (T['y'][idx] < W - 2)
    hit = np.zeros(len(gx), bool); hit[og[og >= 0]] = True
    NT = int(tg.sum())
    r = dict(n_targets=NT, n_rows=int(len(idx)), recall=float(hit[tg].mean()), fp_per_1k=1000 * float(((og < 0) & inter).sum()) / NT)
    fx, fy = np.abs(B.sub(gx)), np.abs(B.sub(gy))
    def rr(lo, hi, ex=True):
        h = n = 0
        for f in (fx, fy):
            k = tg & (f >= lo) & (f < hi) & ex; h += hit[k].sum(); n += k.sum()
        return h / max(n, 1)
    r['recall_vs_absf'] = [float(rr(a, b)) for a, b in zip(EB[:-1], EB[1:])]
    r['edge_ratio'] = float(rr(0.45, 0.5) / rr(0, 0.25))
    r['edge_ratio_lt500'] = float(rr(0.45, 0.5, gph < 500) / rr(0, 0.25, gph < 500))
    r['recall_lt500'] = float(hit[tg & (gph < 500)].mean())
    oi = idx[og >= 0]; ti = og[og >= 0]
    ex = T['x'][oi] - gx[ti]; ey = T['y'][oi] - gy[ti]
    r['rms_nm'] = float(np.sqrt(0.5 * np.mean(ex ** 2 + ey ** 2)) * B.PX)
    bx = bias_curve(ex.astype(np.float64), B.sub(gx[ti])); by = bias_curve(ey.astype(np.float64), B.sub(gy[ti]))
    r['A_nm'] = 0.5 * (bx['A_nm'] + by['A_nm']); r['A_se'] = 0.5 * np.hypot(bx['A_se_nm'], by['A_se_nm'])
    return r


if __name__ == '__main__':
    D = sys.argv[1]; tags = sys.argv[2:]
    md = os.path.join(REPO, 'out', 'papersim', D)
    gt = dict(np.load(os.path.join(md, 'gt.npz'))); nf = int(gt['nframes']); W = int(gt['W'])
    GW = M.gt_windows(gt, nf, np.arange(0, nf - 5, 6))
    RES = {}
    for t in tags:
        d = os.path.join(REPO, 'out', 'detband', f'movie_{D}_{t}')
        RES[t] = {}
        for ps, fn in (('P1', 'locs_p1_sim.npz'), ('P2', 'locs_p2_sim.npz')):
            T = dict(np.load(os.path.join(d, fn)))
            for sn, sel in (('p63', T['good'].astype(bool)), ('sig12', SP.select(T, 'main'))):
                RES[t][f'{ps}_{sn}'] = evaluate(T, sel, GW, W)
    json.dump(RES, open(os.path.join(REPO, 'out', 'detband', f'mv_{D}.json'), 'w'), indent=1)
    L = [f'# {D} (disjoint windows; targets >= 150 ph)', 'tag  set        n_rows  recall  rec<500  edge  edge<500  FP/1k  rms_nm  A_nm']
    for t in tags:
        for k, r in RES[t].items():
            L.append(f'{t:4s} {k:10s} {r["n_rows"]:7d}  {r["recall"]:.4f}  {r["recall_lt500"]:.4f}  {r["edge_ratio"]:.3f}  {r["edge_ratio_lt500"]:.3f}  '
                     f'{r["fp_per_1k"]:6.1f}  {r["rms_nm"]:6.2f}  {r["A_nm"]:+.2f}±{r["A_se"]:.2f}')
    txt = '\n'.join(L); print(txt); open(os.path.join(REPO, 'out', 'detband', f'mv_{D}.txt'), 'w').write(txt)
