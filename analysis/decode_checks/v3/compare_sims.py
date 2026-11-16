"""Old vs new DECODE on the paper simulations: dec_sim1809v2 (<run>/decode/locs.npz via decode_table = the frame-label-fixed
table) vs the v3 model (<run>/decode/locs_v3.npz, dec_m1809v3, published training scale).
Per density movie (gt.npz with structure points): the gtfrc_run.py metrics on the FRC set of metrics.py = the paper's common rule
(disjoint windows, max(sigma) < 0.15 px on DECODE's predicted sigma; GT-FRC with the shared crossing rule gtfrc.RULE, frames to
40 / 30 / 20 nm, sigma-gated Jaccard, false-positive ('ghost') fraction, coverage CV) + pixel bias as pixlock.py (|c1| x / y
of all rows inside [0.5, W - 0.5), GT bias A_bias x / y of isolated GT-matched rows) + rows. Moving movies: pixel bias and rows
only (their GT moves; their metrics come from the moving scripts).
usage: python analysis/decode_checks/v3/compare_sims.py [movie names] -> out/decode_checks/v3/compare_sims.{json,md}"""
import os, sys, json, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAIN = os.path.expanduser('~/simcode2')
for p in (REPO, os.path.join(REPO, 'analysis', 'papersim'), os.path.join(REPO, 'analysis', 'paper'), os.path.join(REPO, 'analysis', 'pixbias')):
    sys.path.insert(0, p)
import common as C, metrics as M, gtfrc as G, gtfrc_run as GR
import pblib as B, sim as PS
from simcode.dataset import decode_table

PS_DIR = os.path.join(MAIN, 'out', 'papersim')
OUT = os.path.join(MAIN, 'out', 'decode_checks', 'v3')
DENS = [f'{p}{d}_v2psf{s}' for s in ('', '_wsm', '_wsm3d', '_wsmaxis') for p in ('D', 'Dexp') for d in ('0.5', '1', '2', '4')]
MOVING = ['moving_exp_v2psf', 'moving_exp_v2psf_wsm']


def pixbias(T, gt, nf, W):
    m = np.isfinite(T['x']) & (T['x'] >= 0.5) & (T['x'] < W - 0.5) & (T['y'] >= 0.5) & (T['y'] < W - 0.5) & (T['f'] + 6 <= nf)
    idx = np.nonzero(m)[0]
    if len(idx) > 400000: idx = np.sort(np.random.default_rng(0).choice(idx, 400000, replace=False))
    x, y, f = T['x'][idx], T['y'][idx], T['f'][idx]
    r = dict(c1_x=B.ustats(B.sub(x))['c1_abs'], c1_y=B.ustats(B.sub(y))['c1_abs'])
    tx, ty, _ = PS.gt_rows(gt, nf, f, x, y); g = np.isfinite(tx)
    bx, by = PS.bias_curve(x[g] - tx[g], B.sub(tx[g])), PS.bias_curve(y[g] - ty[g], B.sub(ty[g]))
    r.update(A_bias_x_nm=bx['A_nm'] if bx else np.nan, A_bias_y_nm=by['A_nm'] if by else np.nan,
             rms_matched_nm=float(np.sqrt(np.mean(np.r_[x[g] - tx[g], y[g] - ty[g]] ** 2)) * B.PX) if g.sum() else np.nan, n_matched=int(g.sum()))
    return r


def one(mv):
    md = os.path.join(PS_DIR, mv); run = os.path.join(md, 'full')
    g = json.load(open(os.path.join(md, 'gen.json'))); W = float(g['W']); nf = int(g['nframes'])
    gt = dict(np.load(os.path.join(md, 'gt.npz')))
    paths = dict(old=decode_table(os.path.join(run, 'decode', 'locs.npz')), new=os.path.join(run, 'decode', 'locs_v3.npz'))
    R = dict(movie=mv, paths=paths, methods={})
    dens = mv not in MOVING and 'pts' in gt
    if dens:
        pts = gt['pts'][:, :2]; EW = GR.emitter_windows(gt, nf)
        seg, seg_len = G.segments(pts, GR.SEG_NM / C.PX_NM, 10 / C.PX_NM); wt = GR.event_weight(gt, nf, len(pts))
        pix = GR.PIXEL_NM / C.PX_NM; lo, hi = (0.0, 0.0), (W, W); gimg = G.render(pts, lo, hi, pix)
        grid = [n for n in (250, 500, 1000, 2000, 4000, 8000, 15000, 30000, 45000, 60000) if n < nf] + [nf]
    for k, p in paths.items():
        if not os.path.exists(p): continue
        T = M.load_method('decode', p)
        r = dict(rows=int(len(T['x'])))
        if dens:
            _, _, f = M.frc_set(T, W, nf)
            pos = np.c_[T['x'][f], T['y'][f]]; sig = np.c_[T['sx'][f], T['sy'][f]]; fr = T['f'][f]
            r['n_frc'] = int(f.sum())
            gf = G.gt_frc(pos, None, lo, hi, pix, C.PX_NM, gt_img=gimg)
            r['gt_frc_nm'] = gf['res_nm']; r['gt_frc_first_nm'] = gf.get('res_nm_first'); r['frc_rule'] = gf.get('rule')
            r['frames_to'] = G.time_to_fidelity(pos, fr, pts, lo, hi, pix, C.PX_NM, grid=grid, targets=(40, 30, 20))['frames_to']
            r['sigma_jaccard'] = G.sigma_jaccard(pos, sig, pts, GR.K)['jaccard']
            r['false_pos_frac'] = G.ghost_fraction(pos, sig, fr, EW, GR.K)['ghost_fraction']
            r['coverage_cv'] = G.coverage_cv(pos, sig, pts, seg, seg_len, wt, GR.SEG_NM / C.PX_NM, GR.K)['cv']
        r.update(pixbias(T, gt, nf, int(W)))
        R['methods'][k] = r
        print(mv, k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
    return R


def fmt(v, d=2):
    return '-' if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f'{v:.{d}f}' if isinstance(v, float) else str(v))


def main():
    if sys.argv[1:2] == ['--merge']:
        import glob
        R = {}
        for f in sorted(glob.glob(os.path.join(OUT, 'compare_sims_*.json'))): R.update(json.load(open(f)))
        R = {m: R[m] for m in DENS + MOVING if m in R}
    else:
        mvs = sys.argv[1:] or DENS + MOVING
        mvs = [m for m in mvs if os.path.exists(os.path.join(PS_DIR, m, 'gt.npz'))]
        R = {}
        for mv in mvs: R[mv] = one(mv)
    tag = os.environ.get('TAG', '')
    if tag:      # partial run (one group of movies): its own json, merged by --merge
        json.dump(R, open(os.path.join(OUT, f'compare_sims_{tag}.json'), 'w'), indent=1, default=float); return
    json.dump(R, open(os.path.join(OUT, 'compare_sims.json'), 'w'), indent=1, default=float)
    L = ['| movie | model | rows | n FRC | GT-FRC nm | frames to 30 nm | sigma-Jaccard | false-pos. frac | |c1| x / y | A_bias x / y nm | RMS matched nm |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for mv, r in R.items():
        for k, lab in (('old', 'DECODE dec_sim1809v2'), ('new', 'DECODE dec_m1809v3')):
            q = r['methods'].get(k)
            if not q: continue
            ft = (q.get('frames_to') or {}).get('30', (q.get('frames_to') or {}).get(30))
            L.append(f"| {mv} | {lab} | {q['rows']} | {q.get('n_frc', '-')} | {fmt(q.get('gt_frc_nm'), 1)} | {fmt(ft, 0) if ft is not None else '-'} | "
                     f"{fmt(q.get('sigma_jaccard'), 3)} | {fmt(q.get('false_pos_frac'), 3)} | {fmt(q['c1_x'], 3)} / {fmt(q['c1_y'], 3)} | "
                     f"{fmt(q['A_bias_x_nm'])} / {fmt(q['A_bias_y_nm'])} | {fmt(q['rms_matched_nm'], 1)} |")
    open(os.path.join(OUT, 'compare_sims.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
