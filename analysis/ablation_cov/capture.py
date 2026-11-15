"""Covariance ablation (2026-11-12): pass-2 head outputs of a finished pipeline run, WITH the Cholesky factor of the intensity
covariance (the pipeline's pass2_est2_<f0>.npz files do not store it), so the explicit solve can be re-run offline per variant.
= the pass-2 stage of simcode.run (processing.front_fast.Pass2FrontFast.run_chunk, return_head=True; same networks, pattern_est.pt,
candidate cut, crop, batches), no package code changed. Per chunk: <out>/head_<f0>.npz with the solve inputs (I, L, x, y, lsxy,
x_det, y_det; region coordinates), bw (window index into PW), PW [nW,6,5] (pattern per window), and the production output rows
(wstart, frame, p, phot, X0, Y0, and the production solve x_hy0, y_hy0, sx_cal, sy_cal, q_dof, kin_hy as the reference for variant A).
usage: CUDA_VISIBLE_DEVICES=2 python analysis/ablation_cov/capture.py --run <pipeline out dir> --movie <tif> --calib <json> --out <dir>
          --p2_cal <calib json> [--crop 5,5,110] [--chunks 0,6000] [--model m1809]
(--model: a name of simcode/models.json or a model folder; its pass-2 detector and head are used)"""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from simcode import config
from simcode.util import proctitle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--movie', required=True); ap.add_argument('--calib', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--p2_cal', required=True)
    ap.add_argument('--crop', default='5,5,110'); ap.add_argument('--chunks', default=''); ap.add_argument('--chunk', type=int, default=6000)
    ap.add_argument('--model', default='m1809'); ap.add_argument('--cand_pmin', type=float, default=0.3)
    ap.add_argument('--pattern', default='')
    a = ap.parse_args()
    proctitle.set_title('covabl')
    N = config.model_set(a.model)
    from simcode.processing import front as _scf, front_fast as _scff
    from simcode.processing import pass2_front as _p2f
    _scf.CAND_PMIN = _scff.CAND_PMIN = _p2f.CAND_PMIN = a.cand_pmin
    from simcode.psf import TiffMovie
    from simcode.processing.common import load_calib
    os.makedirs(a.out, exist_ok=True)
    calib = config.resolve(a.calib); movie = TiffMovie(config.resolve(a.movie, 'data')); cal = load_calib(calib)
    crop = tuple(int(v) for v in a.crop.split(','))
    nf = movie.n
    chunks = [int(v) for v in a.chunks.split(',')] if a.chunks else list(range(0, nf, a.chunk))
    p2 = _scff.Pass2FrontFast(calib, head=config.resolve(N['p2_head']), det=config.resolve(N['p2_det']), compile='auto')
    Pt = torch.load(a.pattern or f'{a.run}/pattern_est.pt')
    scal = config.resolve(a.p2_cal)
    for f0 in chunks:
        fn = f'{a.out}/head_{f0}.npz'
        if os.path.exists(fn): continue
        t0 = time.time(); tm = {}
        P1 = dict(np.load(f'{a.run}/pass1_{f0}.npz'))       # pass-1 rows: the z of the pass-2 rows (not stored here)
        R, H = p2.run_chunk(movie, cal, f0, min(f0 + a.chunk, nf), Pt, scal, P1, config.resolve(N['cal_real']), crop=crop, margin=5, timing=tm,
                            return_head=True)
        o = {k: v.float().cpu().numpy() for k, v in H['o'].items() if k in ('I', 'L', 'x', 'y', 'lsxy', 'x_det', 'y_det', 'bg')}
        bw = H['bw'].cpu().numpy(); uw, inv = np.unique(bw, return_inverse=True)
        PW = H['PWall'][torch.tensor(uw, device=H['PWall'].device)].float().cpu().numpy()
        X0 = float(R['x_prior'][0] - o['x'][0]) if len(bw) else 0.0; Y0 = float(R['y_prior'][0] - o['y'][0]) if len(bw) else 0.0
        np.savez(fn, **{'o_' + k: v for k, v in o.items()}, bw=inv.astype(np.int32), PW=PW, X0=round(X0), Y0=round(Y0),
                 **{k: R[k] for k in ('wstart', 'frame', 'p', 'phot', 'x_hy0', 'y_hy0', 'sx_cal', 'sy_cal', 'q_dof', 'kin_hy', 'x_prior', 'y_prior',
                                      'x_det', 'y_det', 'x_dir', 'y_dir', 'sx_dir', 'sy_dir', 'I_net')})
        del H, R; torch.cuda.empty_cache()
        print(f'chunk {f0}: {len(bw)} rows, {time.time() - t0:.1f}s', json.dumps({k: round(v, 1) for k, v in tm.items()}), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
