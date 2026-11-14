"""Nanoruler ablation: pass-2 head outputs of a pipeline run WITH the Cholesky factor of the intensity covariance (the pipeline's
pass2_est2_<f0>.npz do not store it), so that the explicit solve can be re-run offline per variant (abl_solve.py).
= analysis/ablation_cov/capture.py for this setup: the pass-2 stage of simcode.run (pipeline.sc_front_fast.Pass2P2Fast.run_chunk with
return_head=True; the run's pattern_est.pt, candidate cut 0.3, the run's crop and chunking), detector batch 8 (a 400 x 400 px field
needs 12-20 GB at the default 64; peak ~5 GB at 8), no package code changed. Networks and the production pass-2 calibration: from
out/nanoruler/models/netset_nr80.json (p2_det, p2_head, p2_cal_sim) unless given.
Per chunk <out>/head_<f0>.npz: the solve inputs o_I, o_L, o_x, o_y, o_lsxy, o_x_det, o_y_det (region coordinates), X0, Y0, crop, and the
production rows (wstart, frame, p, phot, x_prior, y_prior, I_net; x_hy0, y_hy0, sx_cal, sy_cal, q_dof, kin_hy = the production solve,
the reference for variant A).
usage: CUDA_VISIBLE_DEVICES=g python analysis/nanoruler/abl_capture.py <set p2|sima_p2|simc0_p2> [--p2_det f --p2_head f --p2_cal f] [--det_batch 8]"""
import glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A


def main():
    a = sys.argv[1:]; g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    setn = a[0]; S = A.SETS[setn]; A.title()
    mv = A.MOVIES['real' if S['kind'] == 'real' else os.path.basename(S['sim'])]
    nsf = os.path.join(A.NR, 'models', 'netset_nr80.json'); N = json.load(open(nsf))['nr80'] if os.path.exists(nsf) else {}
    md = os.path.join(A.NR, 'models')
    p2_det = g('--p2_det', N.get('p2_det', f'{md}/p2det_nr80.pt')); p2_head = g('--p2_head', N.get('p2_head', f'{md}/p2headL_nr80.pt'))
    p2_cal = g('--p2_cal', N.get('p2_cal_sim', S['prod'])); det_batch = int(g('--det_batch', 8))
    from simcode import config
    from simcode.pipeline import sc_front as _scf, sc_front_fast as _scff
    from simcode.p2 import front as _p2f
    _scf.CAND_PMIN = _scff.CAND_PMIN = _p2f.CAND_PMIN = 0.3
    from simcode.pipeline.common import Movie, load_calib
    out = g('--out', S['head']); os.makedirs(out, exist_ok=True)       # --out, --chunks: smoke tests only
    movie = Movie(mv['movie']); cal = load_calib(mv['calib']); crop = tuple(int(v) for v in mv['crop'].split(','))
    chunks = sorted(int(f.rsplit('_', 1)[1][:-4]) for f in glob.glob(f'{S["run"]}/pass1_*.npz')); step = mv['chunk']
    p2 = _scff.Pass2P2Fast(mv['calib'], head=config.resolve(p2_head), det=config.resolve(p2_det), compile='auto')
    Pt = torch.load(f'{S["run"]}/pattern_est.pt', weights_only=False); scal = config.resolve(p2_cal)
    json.dump(dict(set=setn, movie=mv['movie'], calib=mv['calib'], crop=crop, p2_det=p2_det, p2_head=p2_head, p2_cal=p2_cal, det_batch=det_batch, pattern=f'{S["run"]}/pattern_est.pt'),
              open(f'{out}/capture.json', 'w'), indent=1)
    if g('--chunks'): chunks = [int(v) for v in g('--chunks').split(',')]
    for f0 in chunks:
        fn = f'{out}/head_{f0}.npz'
        if os.path.exists(fn): continue
        t0 = time.time(); tm = {}
        for attempt in range(40):       # the GPUs are shared: another process filling the card gives an OOM here although this one needs < 5 GB; wait and redo the chunk
            try:
                R, H = p2.run_chunk(movie, cal, f0, min(f0 + step, movie.n), Pt, scal, crop=crop, margin=5, timing=tm, det_batch=det_batch, return_head=True)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); print(f'chunk {f0}: CUDA out of memory (attempt {attempt + 1}, own peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB); waiting 60 s', flush=True); time.sleep(60)
        else: raise SystemExit(f'chunk {f0}: out of memory 40 times')
        o = {k: v.float().cpu().numpy() for k, v in H['o'].items() if k in ('I', 'L', 'x', 'y', 'lsxy', 'x_det', 'y_det')}
        n = len(R['wstart']); X0 = float(R['x_prior'][0] - o['x'][0]) if n else 0.0; Y0 = float(R['y_prior'][0] - o['y'][0]) if n else 0.0
        np.savez(fn + '.tmp.npz', **{'o_' + k: v for k, v in o.items()}, X0=round(X0), Y0=round(Y0), crop=np.array(crop),
                 **{k: R[k] for k in ('wstart', 'frame', 'p', 'phot', 'x_hy0', 'y_hy0', 'sx_cal', 'sy_cal', 'q_dof', 'kin_hy', 'x_prior', 'y_prior', 'I_net')})
        os.replace(fn + '.tmp.npz', fn)
        del H, R; torch.cuda.empty_cache()
        print(f'chunk {f0}: {n} rows, {time.time() - t0:.1f}s, peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB', flush=True)
    print('capture done', flush=True)


if __name__ == '__main__':
    main()
