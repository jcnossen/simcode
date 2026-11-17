"""One predefined selection rule for every method (Fig 2 of the *_wsm3d sets, user decision 2026-11-15): no threshold tuned on the
ground truth. Each method's own basic quality filters (those of its pipeline: SMLM chi-square, SIMFLUX modulation-error and
|SIMFLUX - SMLM| < 0.5 px, DECODE's detection threshold, SIMCODE's detection threshold p >= 0.63) plus ONE common precision cut on
the method's own reported uncertainty, max(sigma_x, sigma_y) < 0.15 px (16.25 nm): SMLM / SIMFLUX = CRLB, DECODE = its predicted
sigma, SIMCODE = its reported sigma (the result table locs_p2_net: the pass-2 emitter network's sd, frcopt.v2_rows; until
2026-11-17 the solve's calibrated sigma with the selection-aware rescale).
Rows = disjoint 6-frame windows inside the field [0, W) (= metrics.frc_set, the rows of the real-data FRC). Same rule as the
real data's base set.
Per method (smlm, sf, decode, v2; internal extra rows: sf_gtp = SIMFLUX with the GT pattern, v1 = the preprint SIMCODE when run):
  n, gt_first / gt_sust (GT-FRC first / sustained 1/7 crossing, frcopt.Scorer: 5 nm renders, full field, 5-shell smoothing),
  hs_first / hs_sust (half-split FRC), fp (no emitter ON within 3 own sigma), rms_tp_nm, classes (2D error <= 20 nm / 20-50 nm /
  > 50 nm = false positive), ttf (GT-FRC under the shared rule gtfrc.RULE (sustained) on the first N frames; frames to
  40 / 30 / 20 nm), curves.
SIMCODE also at its previous 12 nm cut ('v2_12nm', the production SIGMA_MAX_NM) for the cost of the common rule.
usage: python analysis/papersim/common_rule.py --set Dexp1_v2psf_wsm3d[,Dexp1_v2psf_wsm,...] [--model m1809v2]
output: out/papersim/common_rule/<set>.json (+ <set>_rows.npz: the selected rows per method with GT error / false-positive flag)"""
import os, sys, json, argparse, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis', 'paper'))
import common as C, metrics as M, gtfrc as G
import frcopt as FO
from gtfrc_run import emitter_windows

OUTD = os.path.join(C.OUT, 'common_rule')
SIG_MAX = M.SIGMA_MAX                    # 0.15 px = 16.25 nm
GRID = (250, 500, 1000, 2000, 4000, 8000, 15000, 30000, 45000, 60000)
R_PREC, R_FP = 20.0, 50.0


def classes(B, m, r_prec=R_PREC, r_fp=R_FP):
    e = np.hypot(B['ex'][m], B['ey'][m]) * C.PX_NM; e = np.where(np.isfinite(e), e, np.inf)
    return [int((e <= r_prec).sum()), int(((e > r_prec) & (e <= r_fp)).sum()), int((e > r_fp).sum())]


def ttf(S, B, m, nf, targets=(40, 30, 20)):
    """GT-FRC (sustained crossing) on the rows with frame < N, N in GRID; frames to each target (log-interpolated)"""
    grid = [n for n in GRID if n < nf] + [nf]; res = []
    for N in grid:
        mm = m & (B['f'] < N)
        g = G.gt_frc(np.c_[B['x'][mm], B['y'][mm]], None, S.lo, S.hi, S.pix, C.PX_NM, gt_img=S.gimg)
        res.append(G.resolution(g['freq'], g['curve']) if len(g['freq']) else float('nan'))     # the shared rule G.RULE
    reach = {}
    for t in targets:
        r = np.array(res); ok = np.isfinite(r) & (r <= t)
        if not ok.any(): reach[str(t)] = None; continue
        i = int(np.argmax(ok))
        if i == 0 or not np.isfinite(r[i - 1]): reach[str(t)] = float(grid[i]); continue
        u = (r[i - 1] - t) / (r[i - 1] - r[i])
        reach[str(t)] = float(np.exp(np.log(grid[i - 1]) + u * (np.log(grid[i]) - np.log(grid[i - 1]))))
    return dict(frames=grid, res_nm=res, frames_to=reach, rule=G.RULE)


def base_sets(name, model):
    """the rows of the common rule per method -> (dict of row dicts, W, nframes, gt, EW)"""
    md = os.path.join(C.OUT, name); run = os.path.join(md, 'full')
    g = json.load(open(os.path.join(md, 'gen.json'))); W = int(g['W']); nf = int(g['nframes'])
    gt = dict(np.load(os.path.join(md, 'gt.npz'))); EW = emitter_windows(gt, nf)
    P = M.method_paths(run, model); out = {}
    for k in ('smlm', 'sf', 'decode', 'sf_gtp', 'v1'):
        if k not in P: continue
        T = M.load_method(k, P[k]); _, _, base = M.frc_set(T, W, nf); i = np.nonzero(base)[0]
        B = {c: T[c][i] for c in ('x', 'y', 'f', 'sx', 'sy')}
        B['key'] = np.maximum(B['sx'], B['sy']); B['ex'], B['ey'], B['fp'] = FO.truth(B['x'], B['y'], B['f'], B['sx'], B['sy'], EW)
        out[k] = B
    d = os.path.dirname(P['v2sig']); T = FO.v2_rows(d, W)
    inside = (T['x'] >= 0) & (T['x'] < W) & (T['y'] >= 0) & (T['y'] < W)
    disj = inside & (T['f'] % 6 == 0) & (T['f'] + 6 <= nf); pp = T['p'] >= 0.63
    sx, sy = T['sigma'](pp & inside); key = np.maximum(sx, sy)
    for k, lim in (('v2', SIG_MAX), ('v2_12nm', 12.0 / C.PX_NM)):
        i = np.nonzero(disj & pp & ((key < lim) if k == 'v2' else (key <= lim)))[0]
        B = dict(x=T['x'][i], y=T['y'][i], f=T['f'][i], sx=sx[i], sy=sy[i], key=key[i], p=T['p'][i])
        B['ex'], B['ey'], B['fp'] = FO.truth(B['x'], B['y'], B['f'], B['sx'], B['sy'], EW)
        out[k] = B
    return out, W, nf, gt, md


def run_set(name, model, out=OUTD):
    t0 = time.time()
    Bs, W, nf, gt, md = base_sets(name, model)
    S = FO.Scorer(W, gt['pts'][:, :2])
    g = json.load(open(os.path.join(md, 'gen.json')))
    res = dict(set=name, model=model, W=W, nframes=nf, rule='disjoint 6-frame windows in [0, W), own quality filters, max(sx, sy) < 0.15 px '
               '(SIMCODE: p >= 0.63, calibrated sigma; v2_12nm: <= 12 nm)', sig_max_px=SIG_MAX, sig_max_nm=SIG_MAX * C.PX_NM,
               gt=g.get('gt'), z_dist_um=g.get('z_dist_um'), z_rule=g.get('z_rule'), methods={})
    for k, B in Bs.items():
        m = np.ones(len(B['x']), bool)
        r = S(B, m, curves=True)
        r['classes'] = classes(B, m); r['ttf'] = ttf(S, B, m, nf)
        r['med_sig_nm'] = float(np.median(B['key']) * C.PX_NM) if len(B['key']) else float('nan')
        res['methods'][k] = r
        print(f"{name} {k:8s} n {r['n']:7d}  GT-FRC first {r.get('gt_first', np.nan):5.1f} sust {r.get('gt_sust', np.nan):5.1f}  "
              f"rms_tp {r.get('rms_tp_nm', np.nan):5.1f}  fp {r.get('fp', np.nan):.3f}  ttf {r['ttf']['frames_to']}  ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f'{name}.json'); json.dump(res, open(fn, 'w'), indent=1, default=float)
    np.savez_compressed(os.path.join(out, f'{name}_rows.npz'), **{f'{k}__{c}': v for k, B in Bs.items() for c, v in B.items()})
    print('written', fn, f'{time.time() - t0:.0f} s', flush=True)
    return res


def load(name):
    R = json.load(open(os.path.join(OUTD, f'{name}.json'))); Z = np.load(os.path.join(OUTD, f'{name}_rows.npz'))
    B = {}
    for c in Z.files:
        k, col = c.split('__', 1); B.setdefault(k, {})[col] = Z[c]
    return R, B


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--set', required=True); ap.add_argument('--model', default='m1809v2')
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('psim-rule')
    for s in a.set.split(','): run_set(s, a.model)


if __name__ == '__main__':
    main()
