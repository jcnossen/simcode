"""Selection sweep per method: is there a stricter, FRC-optimal selection, and does the half-split FRC (usable on real data)
find the same operating point as the GT-FRC (simulation only)?

For every method of a papersim density run (SMLM, SIMFLUX with the estimated pattern, DECODE, SIMCODE) the rows of the
method's Fig 2 base set (disjoint 6-frame windows inside the field [0, W), max(sx, sy) < 0.15 px with the method's own sigma;
SIMCODE: the selection base below) are thresholded on the method's OWN reported uncertainty, from all rows down to 10 % kept
(quantile grid Q of the base set). At each threshold:
  gt_first / gt_sust  GT-FRC (gtfrc.gt_frc: rendered reconstruction vs rendered ground truth, 5 nm pixels, field [0, W),
                      5-shell smoothing, 1/7): first crossing from shell 3 (= the Fig 2 titles) and the sustained crossing
                      (metrics.frc_sustained: the curve stays below 1/7 for 0.01 / nm; hollow tubes have a spectral null near
                      65 nm that the first crossing can stop at)
  hs_first / hs_sust  half-split FRC without ground truth = the fair real-data protocol (frc_fair.frc_halves: alternating
                      300-frame halves of the disjoint windows, 5 nm renders, 1/7; no drift in the simulations)
  n, frac             rows kept; frac = n / n of the method's Fig 2 base set (SIMCODE: p >= 0.63, max(sx, sy) < 0.15 px)
  fp                  false-positive fraction = no emitter ON in the row's window within 3 of its own sigma (gtfrc.ghost_fraction)
  rms_tp_nm           RMS error per axis of the true positives (the non-false-positive rows) against the nearest ON emitter
Families (threshold variable):
  smlm / sf / decode  'sig': max(sx, sy) (CRLB; DECODE: its predicted sigma); decode also 'prob' (its detection probability)
  SIMCODE ('v2')      'sig_p'    max(calibrated sx, sy) on the rows with p >= 0.63 (production rule; the 12 nm point = Fig 2)
                      'sig_all'  max(calibrated sx, sy), no p condition (all stored rows, p >= 0.3)
                      'sig_pagg' max(calibrated sx, sy) on the rows with p_agg >= 0.63 (aggregated detection probability of
                                 analysis/pixel_bias_simcode/stage_solves.py: p + the p of the NMS-suppressed cells of the same
                                 emitter), when <out>/pagg/<set>.npz exists
                      'p_sig12'  score p threshold at max(sigma) <= 12 nm (production sigma)
                      'grid'     p in P_GRID x sigma quantile (production sigma)
                      Calibrated sigma: the pass-2 sx_cal, sy_cal times the model's selection-aware rescale g(u), u ranked
                      within the family's own base rows (simcode.table.build_table does this on the p >= 0.63 rows), so
                      'sig_p' reproduces the production table exactly.
Optimum per method / family: argmin of the GT-FRC (ties: more rows) and, separately, argmin of the half-split FRC.
usage: python analysis/papersim/frcopt.py --set Dexp1_v2psf_wsm [--model m1809v2]
output: out/papersim/frcopt/<set>.json (all sweep rows, curves at the current and optimal points), <set>_rows.npz (per-row
        cache of the base sets: positions, sigma, thresholds, GT error, false-positive flag) for the figure script"""
import os, sys, json, argparse, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis', 'paper'))
import common as C, metrics as M, gtfrc as G
from gtfrc_run import emitter_windows, PIXEL_NM, K

CRIT = ('gt_first', 'gt_sust', 'gt_interp', 'gt_band', 'hs_first', 'hs_sust', 'hs_interp', 'hs_band')
Q = np.round(np.arange(1.0, 0.0999, -0.025), 4)            # kept fraction of the base set
P_GRID = (0.3, 0.5, 0.63, 0.75, 0.85, 0.9, 0.95)
OUTD = os.path.join(C.OUT, 'frcopt')


# ------------------------------------------------------------------------------------------------ loading
def v2_rows(d, W, pagg_file=None):
    """pass-2 rows of a SIMCODE run (simcode.table.load_rows of est2 = the row order of locs_p2.hdf5) -> dict with x, y, f, p,
    sx0, sy0 (calibrated, before the selection-aware rescale), p_agg (nan without the stage file) and the rescale g(u) helper"""
    from simcode.table import load_rows, tile_rank, model_sig_u
    R = load_rows(d, 'est2')
    T = dict(x=R['x_hy0'].astype(np.float64), y=R['y_hy0'].astype(np.float64), f=R['wstart'].astype(np.int64),
             p=R['p'].astype(np.float64), sx0=R['sx_cal'].astype(np.float32), sy0=R['sy_cal'].astype(np.float32))
    T['p_agg'] = np.full(len(T['x']), np.nan)
    if pagg_file and os.path.exists(pagg_file):
        S = np.load(pagg_file)
        assert len(S['p2_p_agg']) == len(T['x']), (len(S['p2_p_agg']), len(T['x']))
        assert np.array_equal(S['p2_wstart'].astype(np.int64), T['f'])
        T['p_agg'] = S['p2_p_agg'].astype(np.float64)
    su = json.load(open(model_sig_u(d, 'est2')))
    crop = (0.0, 0.0, float(W))

    def sigma(base):
        """sx, sy rescaled by g(u), u = per-tile rank within base (only base rows are rescaled, as build_table)"""
        sx = T['sx0'].copy(); sy = T['sy0'].copy(); ii = np.nonzero(base)[0]
        u = tile_rank(T['x'][ii], T['y'][ii], 0.5 * (sx[ii] + sy[ii]).astype(np.float64), crop)
        g = np.interp(u, su['u_mid'], su['g']).astype(np.float32); sx[ii] *= g; sy[ii] *= g
        return sx.astype(np.float64), sy.astype(np.float64)
    T['sigma'] = sigma
    return T


def truth(x, y, f, sx, sy, EW):
    """per row: error (px) to the nearest ON emitter of its window (nan if none) and the false-positive flag (no ON emitter
    within K of the row's own sigma, = gtfrc.ghost_fraction)"""
    n = len(x); ex = np.full(n, np.nan); ey = np.full(n, np.nan); fp = np.ones(n, bool)
    o = np.argsort(f, kind='stable'); fs = f[o]; cut = np.r_[0, np.nonzero(np.diff(fs))[0] + 1, n]
    for a, b in zip(cut[:-1], cut[1:]):
        E = EW.get(int(fs[a]))
        if E is None or len(E) == 0: continue
        ii = o[a:b]
        dx = x[ii, None] - E[None, :, 0]; dy = y[ii, None] - E[None, :, 1]
        j = np.argmin(dx * dx + dy * dy, 1); r = np.arange(len(ii))
        ex[ii] = dx[r, j]; ey[ii] = dy[r, j]
        fp[ii] = ((dx / sx[ii, None]) ** 2 + (dy / sy[ii, None]) ** 2).min(1) > K * K
    return ex, ey, fp


# ------------------------------------------------------------------------------------------------ evaluation
BAND = (0.025, 0.075)     # 1/nm (40 - 13 nm): band of the band-mean FRC


def band_mean(freq, curve, band=BAND):
    """mean of the (5-shell smoothed) FRC curve over the band: a smooth fidelity score (the 1/7 crossing of a slowly decaying,
    noisy curve is ill-conditioned and snaps to the hollow tube's spectral nulls)"""
    f = np.asarray(freq); c = np.asarray(curve); k = (f >= band[0]) & (f <= band[1])
    return float(c[k].mean()) if k.any() else float('nan')


def interp_crossing(freq, curve, threshold=1 / 7, start=3):
    """first 1/7 crossing from shell `start` on, linearly interpolated between the shells around it (nm)"""
    f = np.asarray(freq); c = np.asarray(curve); b = np.nonzero(c[start:] < threshold)[0]
    if not len(b): return float('nan')
    i = b[0] + start; u = (c[i - 1] - threshold) / max(c[i - 1] - c[i], 1e-12)
    return float(1 / (f[i - 1] + u * (f[i] - f[i - 1])))

class Scorer:
    def __init__(self, W, pts):
        self.lo, self.hi = (0.0, 0.0), (float(W), float(W)); self.pix = PIXEL_NM / C.PX_NM
        self.gimg = G.render(pts, self.lo, self.hi, self.pix)

    def __call__(self, B, m, curves=False):
        import frc_fair
        n = int(m.sum()); r = dict(n=n)
        if n < 200: return r
        g = G.gt_frc(np.c_[B['x'][m], B['y'][m]], None, self.lo, self.hi, self.pix, C.PX_NM, gt_img=self.gimg)
        r['gt_first'] = g['res_nm']; r['gt_sust'] = M.frc_sustained(g['freq'], g['curve']); r['gt_interp'] = interp_crossing(g['freq'], g['curve'])
        r['gt_band'] = band_mean(g['freq'], g['curve'])
        h, hf, hc = frc_fair.frc_halves(B['x'][m], B['y'][m], B['f'][m])
        r['hs_first'] = float(h); r['hs_sust'] = M.frc_sustained(hf, hc); r['hs_interp'] = interp_crossing(hf, hc)
        r['hs_band'] = band_mean(hf, hc)
        r['fp'] = float(B['fp'][m].mean())
        t = m & ~B['fp']; e2 = (B['ex'][t] ** 2 + B['ey'][t] ** 2) * C.PX_NM ** 2
        r['rms_tp_nm'] = float(np.sqrt(e2.mean() / 2)) if t.any() else float('nan')
        r['med_sig_nm'] = float(np.median(B['key'][m]) * C.PX_NM) if 'key' in B else float('nan')
        if curves:
            r['gt_curve'] = dict(freq=g['freq'], curve=g['curve']); r['hs_curve'] = dict(freq=np.asarray(hf).tolist(), curve=np.asarray(hc).tolist())
        return r


def best(rows, key):
    """argmin of rows[key] (finite), ties -> more rows kept"""
    ok = [i for i, r in enumerate(rows) if np.isfinite(r.get(key, np.nan))]
    if not ok: return None
    sg = -1 if key.endswith('_band') else 1          # band means: higher is better
    return min(ok, key=lambda i: (round(sg * rows[i][key], 6), -rows[i]['n']))


def sweep_family(S, B, base, key, Qs=Q, lower_is_better=True, extra=None):
    """threshold `key` (B[key]) at quantiles of the base rows; lower_is_better: keep key <= thr, else key >= thr"""
    rows = []
    kb = B[key][base]
    for q in Qs:
        thr = float(np.quantile(kb, q)) if lower_is_better else float(np.quantile(kb, 1 - q))
        m = base & ((B[key] <= thr) if lower_is_better else (B[key] >= thr))
        r = dict(q=float(q), thr=thr, **(extra or {})); r.update(S(B, m)); rows.append(r)
    return rows


def run_set(name, model, out=OUTD):
    md = os.path.join(C.OUT, name); run = os.path.join(md, 'full')
    g = json.load(open(os.path.join(md, 'gen.json'))); W = int(g['W']); nf = int(g['nframes'])
    gt = dict(np.load(os.path.join(md, 'gt.npz'))); pts = gt['pts'][:, :2]
    EW = emitter_windows(gt, nf); S = Scorer(W, pts)
    P = M.method_paths(run, model); res = dict(set=name, model=model, W=W, nframes=nf, Q=Q.tolist(), methods={}); cache = {}
    t0 = time.time()
    for k in ('smlm', 'sf', 'decode'):
        T = M.load_method(k, P[k]); _, _, base = M.frc_set(T, W, nf)
        if k == 'decode':
            from simcode.dataset import Dataset, decode_table
            T['prob'] = Dataset.from_decode(decode_table(P[k]), window=6).numpy()['score'].astype(np.float64)
        i = np.nonzero(base)[0]
        B = {c: T[c][i] for c in ('x', 'y', 'f', 'sx', 'sy')}
        if 'prob' in T: B['prob'] = T['prob'][i]
        B['key'] = np.maximum(B['sx'], B['sy']); B['ex'], B['ey'], B['fp'] = truth(B['x'], B['y'], B['f'], B['sx'], B['sy'], EW)
        allm = np.ones(len(i), bool)
        R = dict(n_base=int(len(i)), current=S(B, allm, curves=True), fam={})
        R['fam']['sig'] = sweep_family(S, B, allm, 'key')
        if 'prob' in B: R['fam']['prob'] = sweep_family(S, B, allm, 'prob', lower_is_better=False)
        res['methods'][k] = R; cache[k] = B
        print(f'{name} {k}: base {len(i)}  current GT {R["current"]["gt_first"]:.1f}/{R["current"]["gt_sust"]:.1f}  ({time.time() - t0:.0f} s)', flush=True)
    # SIMCODE
    d = os.path.dirname(P['v2sig']); T = v2_rows(d, W, os.path.join(out, 'pagg', f'{name}.npz'))
    inside = (T['x'] >= 0) & (T['x'] < W) & (T['y'] >= 0) & (T['y'] < W)
    disj = inside & (T['f'] % 6 == 0) & (T['f'] + 6 <= nf)
    sig_lim = M.SIGMA_MAX; s12 = 12.0 / C.PX_NM
    variants = {'sig_p': T['p'] >= 0.63, 'sig_all': np.ones(len(T['x']), bool)}
    if np.isfinite(T['p_agg']).any(): variants['sig_pagg'] = T['p_agg'] >= 0.63
    R = dict(fam={}, variants={})
    for vn, vb in variants.items():
        sx, sy = T['sigma'](vb & inside)
        key = np.maximum(sx, sy); base = disj & vb & (key < sig_lim)
        i = np.nonzero(base)[0]
        B = dict(x=T['x'][i], y=T['y'][i], f=T['f'][i], sx=sx[i], sy=sy[i], key=key[i], p=T['p'][i], p_agg=T['p_agg'][i])
        B['ex'], B['ey'], B['fp'] = truth(B['x'], B['y'], B['f'], B['sx'], B['sy'], EW)
        cache['v2_' + vn] = B; allm = np.ones(len(i), bool)
        rows = sweep_family(S, B, allm, 'key')
        # the 12 nm point of this variant (production: = Fig 2)
        m12 = B['key'] <= s12; r12 = dict(q=float(m12.mean()), thr=s12); r12.update(S(B, m12, curves=(vn == 'sig_p')))
        R['variants'][vn] = dict(n_base=int(len(i)), at12=r12)
        R['fam'][vn] = rows
        print(f'{name} v2 {vn}: base {len(i)}  12 nm: n {r12["n"]} GT {r12.get("gt_first", np.nan):.1f}/{r12.get("gt_sust", np.nan):.1f}  ({time.time() - t0:.0f} s)', flush=True)
    R['n_base'] = R['variants']['sig_p']['n_base']; R['current'] = R['variants']['sig_p']['at12']
    # p family and the (p, sigma) grid on the production sigma (rows p < 0.63 keep their unrescaled sigma, as in the table)
    sxp, syp = T['sigma'](variants['sig_p'] & inside)
    i = np.nonzero(disj & (np.maximum(sxp, syp) < sig_lim))[0]
    Bp = dict(x=T['x'][i], y=T['y'][i], f=T['f'][i], sx=sxp[i], sy=syp[i], key=np.maximum(sxp, syp)[i], p=T['p'][i])
    Bp['ex'], Bp['ey'], Bp['fp'] = truth(Bp['x'], Bp['y'], Bp['f'], Bp['sx'], Bp['sy'], EW)
    cache['v2_prod'] = Bp
    m12 = Bp['key'] <= s12
    R['fam']['p_sig12'] = [dict(q=float(np.mean(Bp['p'][m12] >= pt)), thr=pt, **S(Bp, m12 & (Bp['p'] >= pt))) for pt in
                           (0.3, 0.4, 0.5, 0.55, 0.6, 0.63, 0.67, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97)]
    grid = []
    for pt in P_GRID:
        b = Bp['p'] >= pt
        for r in sweep_family(S, Bp, b, 'key', Qs=Q[::2]): r['p_thr'] = pt; grid.append(r)
    R['fam']['grid'] = grid
    res['methods']['v2'] = R
    print(f'{name} v2 p / grid done ({time.time() - t0:.0f} s)', flush=True)
    # optima
    for k, R in res['methods'].items():
        R['opt'] = {}
        for fam, rows in R['fam'].items():
            o = {}
            for crit in CRIT:
                j = best(rows, crit)
                if j is not None: o[crit] = dict(idx=j, **{c: rows[j].get(c) for c in ('q', 'thr', 'p_thr', 'n', 'fp', 'rms_tp_nm') + CRIT})
            R['opt'][fam] = o
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f'{name}.json'); json.dump(res, open(fn, 'w'), indent=1, default=float)
    np.savez_compressed(os.path.join(out, f'{name}_rows.npz'), **{f'{k}__{c}': v for k, B in cache.items() for c, v in B.items()})
    print('written', fn, f'{time.time() - t0:.0f} s')
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--set', required=True); ap.add_argument('--model', default='m1809v2')
    a = ap.parse_args()
    for s in a.set.split(','): run_set(s, a.model)


if __name__ == '__main__':
    main()
