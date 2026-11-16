"""Referee checks on the simulation figure (Fig 2, fig_sim_frcopt.py selections; webSMLM hollow-tube sets Dexp*_v2psf_wsm).

Rows: the cached base sets of frcopt.py (out/papersim/frcopt/<set>_rows.npz) cut at each method's figure threshold
(opt[fam]['gt_sust'], fam = 'sig_p' for SIMCODE, 'sig' otherwise; = fig_sim_frcopt.py), asserted against the stored n and GT-FRC.
(a) one-to-one matching per disjoint 6-frame window to the ground-truth emitter-windows (every blink event ON in the window,
    metrics.gt_windows; >= 150 window photons as a secondary reference): Hungarian assignment on the 2D distance with a gate
    (default 50 nm; 20 nm secondary). TP = matched pairs; duplicate = an unmatched localization with a GT emitter of its window
    within the gate (that emitter already taken); false positive = no GT emitter within the gate. Recall = TP / GT emitter-windows,
    precision = TP / n, emitter-window Jaccard = TP / (n + N_gt - TP). Preprint Jaccard (Eq. 9) = metrics.compute_jaccard_points
    (all labelling sites vs the localizations, 10 nm, whole movie) at the figure selection (the figure's MET value is on all rows).
(b) GT-FRC (rendered set vs rendered ground truth, gtfrc.gt_frc) read at 1/7 and at 0.5 (the threshold for a comparison with a
    noise-free reference: Rosenthal & Henderson 2003, C_ref = sqrt(2 FSC / (1 + FSC)), FSC = 1/7 -> 0.5), first and sustained
    crossing (sustained = stays below for 0.01 / nm); also with each method's threshold re-optimized for the 0.5 criterion.
(c) block bootstrap (density 1): windows (6 frames) resampled with replacement, the same draws for all methods (paired), B
    resamples; secondary: 300-frame blocks. GT-FRC 1/7 sustained (and 0.5) percentile 95 % CIs, the SIMCODE margin over the next
    best; time to fidelity: for each N of the grid, windows in [0, N) resampled -> frames to 40 / 30 / 20 nm CIs.
usage: python analysis/refcheck/sim_checks.py [--sets Dexp0.5,Dexp1,Dexp2,Dexp4] [--B 200] [--Bttf 100]
output: out/refcheck/sim_checks.json"""
import os, sys, json, argparse, time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
PS = os.path.join(REPO, 'analysis', 'papersim')
sys.path[:0] = [PS, REPO, os.path.join(REPO, 'analysis', 'paper')]
import common as C, metrics as M, gtfrc as G

FRCD = os.path.join(C.OUT, 'frcopt'); OUTD = os.path.join(REPO, 'out', 'refcheck')
METH = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2', 'SIMCODE')]
PIX = 5.0 / C.PX_NM
GRID = (250, 500, 1000, 2000, 4000, 8000, 15000, 30000, 45000, 60000)


def sustained(freq, curve, thr, width=0.01, start=3):
    f = np.asarray(freq); c = np.asarray(curve); w = max(int(width / max(f[1] - f[0], 1e-12)), 1)
    for i in range(start, len(c) - w):
        if (c[i:i + w] < thr).all(): return float(1 / f[i])
    return float('nan')


def first(freq, curve, thr, start=3):
    f = np.asarray(freq); c = np.asarray(curve); b = np.nonzero(c[start:] < thr)[0]
    return float('nan') if not len(b) else float(1 / f[b[0] + start])


def crossings(freq, curve):
    return dict(t7_first=first(freq, curve, 1 / 7), t7_sust=sustained(freq, curve, 1 / 7),
                t05_first=first(freq, curve, 0.5), t05_sust=sustained(freq, curve, 0.5))


class GTF:
    """GT-FRC with optional per-row weights (bootstrap); same render / FRC as gtfrc.gt_frc"""
    def __init__(self, W, pts):
        self.W = float(W); self.nb = int(round(W / PIX))
        self.rng = [(0.0, self.nb * PIX)] * 2
        g = np.histogramdd(pts, bins=(self.nb, self.nb), range=self.rng)[0]
        self.Gf = np.fft.fftn(g)
        fr = np.sqrt(sum(q ** 2 for q in np.meshgrid(*[np.fft.fftfreq(self.nb, PIX)] * 2, indexing='ij')))
        df = 1.0 / (self.nb * PIX); self.k = (fr / df).astype(int).ravel(); self.nmax = int(0.5 / PIX / df)
        self.dg = np.bincount(self.k, (np.abs(self.Gf) ** 2).ravel(), self.nmax + 1)[:self.nmax]
        self.freq = np.arange(self.nmax) * df / C.PX_NM

    def curve(self, x, y, w=None):
        a = np.histogramdd(np.c_[x, y], bins=(self.nb, self.nb), range=self.rng, weights=w)[0]
        A = np.fft.fftn(a)
        num = np.bincount(self.k, (A * self.Gf.conj()).real.ravel(), self.nmax + 1)[:self.nmax]
        da = np.bincount(self.k, (np.abs(A) ** 2).ravel(), self.nmax + 1)[:self.nmax]
        c = num / np.sqrt(np.maximum(da * self.dg, 1e-30))
        return self.freq, np.convolve(c, np.ones(5) / 5, 'same')


def match_windows(x, y, f, GW, gate_px, ph_min=150.0):
    """one-to-one (Hungarian) per window -> counts"""
    n = len(x); tp = 0; dup = 0; fp = 0; ngt = 0; ngt150 = 0; tp150 = 0; err = []
    o = np.argsort(f, kind='stable'); fs = f[o]; cut = np.r_[0, np.nonzero(np.diff(fs))[0] + 1, n]
    seen = set()
    for a, b in zip(cut[:-1], cut[1:]):
        fw = int(fs[a]); seen.add(fw); ii = o[a:b]
        gx, gy, gph = GW.get(fw, (np.zeros(0), np.zeros(0), np.zeros(0)))
        ngt += len(gx); ngt150 += int((gph >= ph_min).sum())
        if not len(gx): fp += len(ii); continue
        d = np.hypot(x[ii, None] - gx[None], y[ii, None] - gy[None])
        cost = np.where(d <= gate_px, d, 1e6)
        r, c = linear_sum_assignment(cost); ok = cost[r, c] <= gate_px
        r, c = r[ok], c[ok]; tp += len(r); tp150 += int((gph[c] >= ph_min).sum()); err.append(d[r, c])
        un = np.ones(len(ii), bool); un[r] = False
        near = (d <= gate_px).any(1)
        dup += int((un & near).sum()); fp += int((un & ~near).sum())
    for fw, (gx, gy, gph) in GW.items():           # windows without any localization
        if fw not in seen: ngt += len(gx); ngt150 += int((gph >= ph_min).sum())
    e = np.concatenate(err) * C.PX_NM if err else np.zeros(0)
    return dict(n=int(n), n_gt=int(ngt), n_gt_ph150=int(ngt150), tp=int(tp), tp_ph150=int(tp150), duplicates=int(dup),
                false_pos=int(fp), recall=tp / max(ngt, 1), recall_ph150=tp150 / max(ngt150, 1), precision=tp / max(n, 1),
                dup_rate=dup / max(n, 1), fp_rate=fp / max(n, 1), jaccard_ew=tp / max(n + ngt - tp, 1),
                rms_err_tp_nm=float(np.sqrt(np.mean(e ** 2) / 2)) if len(e) else float('nan'))


def frames_to(grid, res, targets=(40, 30, 20)):
    out = {}
    r = np.array(res, float)
    for t in targets:
        ok = np.isfinite(r) & (r <= t)
        if not ok.any(): out[str(t)] = None; continue
        i = int(np.argmax(ok))
        if i == 0 or not np.isfinite(r[i - 1]): out[str(t)] = float(grid[i]); continue
        u = (r[i - 1] - t) / (r[i - 1] - r[i])
        out[str(t)] = float(np.exp(np.log(grid[i - 1]) + u * (np.log(grid[i]) - np.log(grid[i - 1]))))
    return out


def load_set(name):
    R = json.load(open(os.path.join(FRCD, f'{name}.json'))); Z = np.load(os.path.join(FRCD, f'{name}_rows.npz'))
    B, SEL = {}, {}
    for k, _ in METH:
        ck = 'v2_sig_p' if k == 'v2' else k; fam = 'sig_p' if k == 'v2' else 'sig'
        B[k] = {c.split('__', 1)[1]: Z[c] for c in Z.files if c.split('__', 1)[0] == ck}
        o = R['methods'][k]['opt'][fam]['gt_sust']
        SEL[k] = dict(thr=o['thr'], n=o['n'], gt_sust=o['gt_sust'], gt_first=o['gt_first'], fam=fam, n_base=R['methods'][k]['n_base'])
    return R, B, SEL


def run_set(name, a, rng):
    md = os.path.join(C.OUT, name); g0 = json.load(open(os.path.join(md, 'gen.json')))
    gt = dict(np.load(os.path.join(md, 'gt.npz'))); W = int(g0['W']); nf = int(g0['nframes']); pts = gt['pts'][:, :2]
    R, B, SEL = load_set(name)
    starts = np.arange(0, nf - 5, 6)
    GW = M.gt_windows(gt, nf, starts)
    S = GTF(W, pts)
    out = dict(set=name, density=g0['density'], W=W, nframes=nf, n_labels=int(len(pts)), n_events=int(len(gt['ev_t0'])),
               n_gt_emitter_windows=int(sum(len(v[0]) for v in GW.values())),
               n_gt_emitter_windows_ph150=int(sum((v[2] >= 150).sum() for v in GW.values())), methods={})
    sel_m = {}
    for k, lab in METH:
        b = B[k]; s = SEL[k]; m = b['key'] <= s['thr']; sel_m[k] = m
        fq, cv = S.curve(b['x'][m], b['y'][m]); cr = crossings(fq, cv)
        assert m.sum() == s['n'] and abs(cr['t7_sust'] - s['gt_sust']) < 1e-6 and abs(cr['t7_first'] - s['gt_first']) < 1e-6, (name, k, m.sum(), s, cr)
        r = dict(label=lab, threshold_nm=s['thr'] * C.PX_NM, n=int(m.sum()), n_base=s['n_base'], gtfrc=cr)
        # sampling ceiling: the same rows moved to the true position of their nearest ON emitter (rows without one dropped)
        t = m & np.isfinite(b['ex'])
        fq0, cv0 = S.curve(b['x'][t] - b['ex'][t], b['y'][t] - b['ey'][t])
        r['gtfrc_true_positions'] = dict(n=int(t.sum()), **crossings(fq0, cv0),
                                         curve_at={f'{nm}nm': float(np.interp(1 / nm, fq0, cv0)) for nm in (100, 50, 30, 20)})
        r['gtfrc_curve_at'] = {f'{nm}nm': float(np.interp(1 / nm, fq, cv)) for nm in (100, 50, 30, 20)}
        # error transfer = measured curve / its true-position ceiling (sampling and the tube spectrum divide out)
        rat = np.asarray(cv) / np.maximum(np.asarray(cv0), 1e-3)
        r['gtfrc_ratio_to_ceiling'] = dict(t05_first=first(fq, rat, 0.5), t05_sust=sustained(fq, rat, 0.5),
                                           t7_first=first(fq, rat, 1 / 7), t7_sust=sustained(fq, rat, 1 / 7))
        for gate in (50.0, 20.0):
            r[f'match_{gate:g}nm'] = match_windows(b['x'][m], b['y'][m], b['f'][m], GW, gate / C.PX_NM)
        r['jaccard_preprint_10nm'] = float(M.compute_jaccard_points(pts, np.c_[b['x'][m], b['y'][m]], 10 / C.PX_NM))
        # threshold re-optimized for the 0.5 criterion (sustained), on the same quantile grid as frcopt
        kb = b['key']; best = None
        for q in np.round(np.arange(1.0, 0.0999, -0.025), 4):
            thr = float(np.quantile(kb, q)); mm = kb <= thr
            if mm.sum() < 200: continue
            v = crossings(*S.curve(b['x'][mm], b['y'][mm]))
            if best is None or (np.isfinite(v['t05_sust']) and v['t05_sust'] < best['t05_sust'] - 1e-9):
                best = dict(q=float(q), threshold_nm=thr * C.PX_NM, n=int(mm.sum()), **v)
        r['opt_t05_sust'] = best
        out['methods'][k] = r
        print(f"{name} {lab}: n {r['n']} GT-FRC 1/7 {cr['t7_sust']:.1f} 0.5 {cr['t05_sust']:.1f} (first {cr['t05_first']:.1f}) | "
              f"recall {r['match_50nm']['recall']:.3f} dup {r['match_50nm']['dup_rate']:.3f} fp {r['match_50nm']['fp_rate']:.3f} "
              f"J10 {r['jaccard_preprint_10nm']:.3f}", flush=True)
    if a.boot and abs(g0['density'] - 1) < 1e-9:
        out['bootstrap'] = bootstrap(B, sel_m, S, nf, a, rng)
    if a.jack and (abs(g0['density'] - 1) < 1e-9 or a.jack_all):
        out['jackknife'] = jackknife(B, sel_m, S, nf, a, rng)
    return out


def bootstrap(B, sel, S, nf, a, rng):
    """paired block bootstrap over windows (6 frames) and 300-frame blocks"""
    res = {}
    for blk in (6, 300):
        nb = int(np.ceil(nf / blk)); t0 = time.time()
        vals = {k: dict(t7_sust=[], t05_sust=[], t7_first=[]) for k, _ in METH}
        for it in range(a.B):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
            for k, _ in METH:
                b = B[k]; m = sel[k]; wr = w[b['f'][m] // blk]
                v = crossings(*S.curve(b['x'][m], b['y'][m], wr))
                for c in vals[k]: vals[k][c].append(v[c])
        R = {}
        for k, _ in METH:
            R[k] = {c: dict(median=float(np.nanmedian(v)), lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)),
                            sd=float(np.nanstd(v)), nan=int(np.isnan(v).sum())) for c, v in vals[k].items()}
        for c in ('t7_sust', 't05_sust'):
            o = [kk for kk, _ in METH if kk != 'v2']
            nb_best = np.nanmin(np.array([vals[kk][c] for kk in o]), 0); sc = np.array(vals['v2'][c])
            gain = (nb_best - sc) / nb_best
            R[f'margin_{c}'] = dict(next_best_minus_simcode_nm=dict(median=float(np.nanmedian(nb_best - sc)), lo=float(np.nanpercentile(nb_best - sc, 2.5)),
                                                                     hi=float(np.nanpercentile(nb_best - sc, 97.5))),
                                    rel_gain=dict(median=float(np.nanmedian(gain)), lo=float(np.nanpercentile(gain, 2.5)), hi=float(np.nanpercentile(gain, 97.5))),
                                    p_simcode_best=float(np.mean(sc < nb_best)))
        res[f'blk{blk}'] = R
        print(f'bootstrap blk {blk}: {time.time() - t0:.0f} s', json.dumps({k: R[k]['t7_sust'] for k, _ in METH}), flush=True)
    # time to fidelity (6-frame windows; each N resampled within [0, N))
    t0 = time.time(); grid = [n for n in GRID if n < nf] + [nf]
    TT = {k: [] for k, _ in METH}; RES = {k: [] for k, _ in METH}
    for it in range(a.Bttf):
        rr = {k: [] for k, _ in METH}
        for N in grid:
            nw = N // 6; w = np.bincount(rng.integers(0, nw, nw), minlength=nw).astype(float)
            for k, _ in METH:
                b = B[k]; m = sel[k] & (B[k]['f'] < 6 * nw); wr = w[b['f'][m] // 6]
                rr[k].append(crossings(*S.curve(b['x'][m], b['y'][m], wr))['t7_sust'] if m.sum() else np.nan)
        for k, _ in METH: TT[k].append(frames_to(grid, rr[k])); RES[k].append(rr[k])
    ttf = {}
    for k, _ in METH:
        ttf[k] = {}
        for t in ('40', '30', '20'):
            v = np.array([np.nan if d[t] is None else d[t] for d in TT[k]])
            ttf[k][t] = dict(frac_reached=float(np.isfinite(v).mean()), median=float(np.nanmedian(v)) if np.isfinite(v).any() else None,
                             lo=float(np.nanpercentile(v, 2.5)) if np.isfinite(v).sum() > 2 else None,
                             hi=float(np.nanpercentile(v, 97.5)) if np.isfinite(v).sum() > 2 else None)
        ttf[k]['res_nm_median'] = np.nanmedian(np.array(RES[k], float), 0).tolist()
    res['ttf'] = dict(grid=grid, methods=ttf, B=a.Bttf)
    print(f'bootstrap ttf: {time.time() - t0:.0f} s', json.dumps({k: ttf[k]['30'] for k, _ in METH}), flush=True)
    return res


def jk(v, n, d, point):
    """delete-d jackknife over blocks: SE = sqrt((n - d) / d * var(theta_j)); 95 % CI = point +- 1.96 SE"""
    v = np.asarray(v, float); se = float(np.sqrt((n - d) / d * np.nanvar(v)))
    return dict(point=float(point), se=se, lo=float(point - 1.96 * se), hi=float(point + 1.96 * se), mean_sub=float(np.nanmean(v)), J=int(len(v)))


def jackknife(B, sel, S, nf, a, rng, keep=0.9):
    """paired delete-d jackknife (d = 10 % of the blocks dropped without replacement, the same blocks for every method); no
    duplicated rows, so no shot-noise bias as with the with-replacement bootstrap"""
    res = {}
    pt = {k: crossings(*S.curve(B[k]['x'][sel[k]], B[k]['y'][sel[k]])) for k, _ in METH}
    for blk in (6, 300):
        nb = int(np.ceil(nf / blk)); d = nb - int(round(keep * nb)); V = {k: dict(t7_sust=[], t05_sust=[]) for k, _ in METH}
        for it in range(a.J):
            kb = np.ones(nb, bool); kb[rng.choice(nb, d, replace=False)] = False
            for k, _ in METH:
                b = B[k]; m = sel[k] & kb[b['f'] // blk]
                v = crossings(*S.curve(b['x'][m], b['y'][m]))
                for c in V[k]: V[k][c].append(v[c])
        R = {k: {c: jk(V[k][c], nb, d, pt[k][c]) for c in V[k]} for k, _ in METH}
        o = [kk for kk, _ in METH if kk != 'v2']
        for c in ('t7_sust',):
            nbst = np.nanmin(np.array([V[kk][c] for kk in o]), 0); sc = np.array(V['v2'][c])
            p0 = min(pt[kk][c] for kk in o) - pt['v2'][c]
            R[f'margin_{c}'] = dict(next_best_minus_simcode_nm=jk(nbst - sc, nb, d, p0),
                                    rel_gain=jk((nbst - sc) / nbst, nb, d, p0 / min(pt[kk][c] for kk in o)))
        res[f'blk{blk}'] = R
        print(f'jackknife blk {blk}:', json.dumps({k: (round(R[k]['t7_sust']['lo'], 2), round(R[k]['t7_sust']['hi'], 2)) for k, _ in METH}), flush=True)
    # frames to 40 / 30 / 20 nm: the time course on each jackknife subsample (windows, 6 frames)
    grid = [n for n in GRID if n < nf] + [nf]; nb = nf // 6; d = nb - int(round(keep * nb))
    pt_ttf = {}; TT = {k: [] for k, _ in METH}
    for k, _ in METH:
        b = B[k]; m = sel[k]
        pt_ttf[k] = frames_to(grid, [crossings(*S.curve(b['x'][m & (b['f'] < N)], b['y'][m & (b['f'] < N)]))['t7_sust'] for N in grid])
    for it in range(a.Jttf):
        kb = np.ones(nb + 1, bool); kb[rng.choice(nb, d, replace=False)] = False
        for k, _ in METH:
            b = B[k]; rr = []
            for N in grid:
                m = sel[k] & (b['f'] < N) & kb[b['f'] // 6]
                rr.append(crossings(*S.curve(b['x'][m], b['y'][m]))['t7_sust'])
            TT[k].append(frames_to(grid, rr))
    ttf = {}
    for k, _ in METH:
        ttf[k] = {}
        for t in ('40', '30', '20'):
            v = np.array([np.nan if q[t] is None else q[t] for q in TT[k]]); p0 = pt_ttf[k][t]
            if p0 is None or np.isnan(v).mean() > 0.5:
                ttf[k][t] = dict(point=p0, frac_reached=float(np.isfinite(v).mean())); continue
            lv = np.log(v[np.isfinite(v)]); se = float(np.sqrt((nb - d) / d * np.var(lv)))
            ttf[k][t] = dict(point=p0, lo=float(p0 * np.exp(-1.96 * se)), hi=float(p0 * np.exp(1.96 * se)), se_log=se,
                             frac_reached=float(np.isfinite(v).mean()))
    res['ttf'] = dict(grid=grid, methods=ttf, J=a.Jttf, note='CI on log(frames), delete-10 % jackknife over 6-frame windows')
    print('jackknife ttf:', json.dumps({k: ttf[k]['30'] for k, _ in METH}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='Dexp0.5,Dexp1,Dexp2,Dexp4'); ap.add_argument('--suffix', default='_v2psf_wsm')
    ap.add_argument('--B', type=int, default=200); ap.add_argument('--Bttf', type=int, default=100)
    ap.add_argument('--boot', type=int, default=1); ap.add_argument('--jack', type=int, default=0); ap.add_argument('--jack_all', type=int, default=0)
    ap.add_argument('--J', type=int, default=100); ap.add_argument('--Jttf', type=int, default=50)
    ap.add_argument('--out', default=os.path.join(OUTD, 'sim_checks.json'))
    a = ap.parse_args(); rng = np.random.default_rng(20261115)
    os.makedirs(OUTD, exist_ok=True)
    res = dict(protocol=__doc__, sets={})
    for s in a.sets.split(','):
        res['sets'][s] = run_set(s + a.suffix, a, rng)
        json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print('written', a.out)


if __name__ == '__main__':
    main()
