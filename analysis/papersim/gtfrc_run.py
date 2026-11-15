"""GT-FRC, time-to-fidelity and the fidelity diagnostics (gtfrc.py) for every method of a papersim density run.

Rows per method = its rendered reconstruction = the FRC set of metrics.py (disjoint 6-frame windows, max(sx, sy) < 0.15 px,
inside the field [0, W)), with the method's own reported sigma. Ground truth = all labelling sites of the structure (gt.npz
pts). Field [0, W) px, 5 nm pixels.
  gt_frc          FRC(reconstruction, ground truth), 1/7 crossing, curve (60 000 frames)
  ttf             GT-FRC on the first N frames, N = 250 ... 60k; frames to reach 40 / 30 / 20 nm
  sigma_jaccard   Jaccard with the 3-sigma Mahalanobis gate against the structure points
  ghost           share of rows with no emitter ON in their own window within 3 of their own sigma
  coverage        per 250 nm filament segment: recovered (sigma-gated, nearest structure point) / expected (the segment's
                  share of ground-truth emitter-windows: each blink event counts the disjoint windows it overlaps) -> CV,
                  Poisson floor, excess
Methods: those of metrics.method_paths(run, --model) plus, with --ref <model>, that model's v2 table as 'v2sig_ref' (e.g. the
13 Nov v2 next to the rerun).
usage: python analysis/papersim/gtfrc_run.py <run dir> --movie <movie dir> [--model m1809_3d] [--ref m1809] [--nframes N]
output: <run>/gtfrc_<model>.json"""
import os, sys, json, argparse, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M, gtfrc as G

PIXEL_NM = 5.0
SEG_NM = 250.0
K = 3.0


def emitter_windows(gt, nframes):
    starts = np.arange(0, nframes - 5, 6)
    GW = M.gt_windows(gt, nframes, starts)
    return {int(f): np.c_[v[0], v[1]] for f, v in GW.items()}


def event_weight(gt, nframes, npts):
    """per structure point: number of (event, disjoint window) overlaps of its blink events in [0, nframes)"""
    t0, t1 = gt['ev_t0'], np.minimum(gt['ev_t1'], (nframes // 6) * 6)
    nw = np.maximum(np.ceil(t1 / 6) - np.floor(t0 / 6), 0)
    return np.bincount(gt['ev_id'].astype(int), nw, minlength=npts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--movie', required=True); ap.add_argument('--model', default='m1809')
    ap.add_argument('--ref', default=''); ap.add_argument('--nframes', type=int, default=0)
    a = ap.parse_args()
    run, md = os.path.abspath(a.run), os.path.abspath(a.movie)
    g = json.load(open(os.path.join(md, 'gen.json'))); W = float(g['W']); nf = a.nframes or g['nframes']
    gt = dict(np.load(os.path.join(md, 'gt.npz'))); pts = gt['pts'][:, :2]
    t0 = time.time()
    EW = emitter_windows(gt, nf)
    seg, seg_len = G.segments(pts, SEG_NM / C.PX_NM, 10 / C.PX_NM)
    wt = event_weight(gt, nf, len(pts))
    pix = PIXEL_NM / C.PX_NM; lo, hi = (0.0, 0.0), (W, W)
    grid = [n for n in (250, 500, 1000, 2000, 4000, 8000, 15000, 30000, 45000, 60000) if n < nf] + [nf]
    P = M.method_paths(run, a.model)
    if a.ref:
        rp = f'{run}/v2_{a.ref}/locs_p2_sim.npz'
        if os.path.exists(rp): P['v2sig_ref'] = rp
    R = dict(run=run, movie=md, nframes=nf, model=a.model, ref=a.ref, pixel_nm=PIXEL_NM, segment_nm=SEG_NM, k_sigma=K,
             rows='FRC set of metrics.py (disjoint windows, max sigma < 0.15 px, inside [0, W))', methods={},
             n_gt_points=int(len(pts)), n_segments=int(len(seg_len)))
    gimg = G.render(pts, lo, hi, pix)
    for name, path in P.items():
        T = M.load_method('v2sig' if name == 'v2sig_ref' else name, path)
        _, _, f = M.frc_set(T, W, nf)
        pos = np.c_[T['x'][f], T['y'][f]]; sig = np.c_[T['sx'][f], T['sy'][f]]; fr = T['f'][f]
        r = dict(path=path, n=int(f.sum()))
        r['gt_frc'] = G.gt_frc(pos, None, lo, hi, pix, C.PX_NM, gt_img=gimg)
        r['ttf'] = G.time_to_fidelity(pos, fr, pts, lo, hi, pix, C.PX_NM, grid=grid, targets=(40, 30, 20))
        r['sigma_jaccard'] = G.sigma_jaccard(pos, sig, pts, K)
        r['ghost'] = G.ghost_fraction(pos, sig, fr, EW, K)
        r['coverage'] = G.coverage_cv(pos, sig, pts, seg, seg_len, wt, SEG_NM / C.PX_NM, K)
        R['methods'][name] = r
        print(f"{name:12s} n {r['n']:7d} GT-FRC {r['gt_frc']['res_nm']:6.1f} nm  ttf {r['ttf']['frames_to']}  sigJ {r['sigma_jaccard']['jaccard']:.3f}  "
              f"ghost {r['ghost']['ghost_fraction']:.3f}  cov CV {r['coverage']['cv']:.3f} (Poisson {r['coverage']['poisson_cv']:.3f}, excess {r['coverage']['excess_cv']:.3f})", flush=True)
    fn = os.path.join(run, f'gtfrc_{a.model}.json'); json.dump(R, open(fn, 'w'), indent=1)
    print('written', fn, round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
