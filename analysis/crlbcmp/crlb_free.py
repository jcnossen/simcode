"""CRLB comparison (2026-11-13): the gate scenes' CRLB table extended by the free-model target block (the bound of a
PATTERN-FREE position estimator on the modulated data: x, y, z, I_1..I_T free per emitter, bg per angle, multi-emitter) and the
ideal two-stage bound (crlb.two_stage_bounds: free-model (x, y, I_t) of the target combined with its SIMFLUX model, full covariance).
Same generator / seeds / iteration loop as simcode/benchmark/gate.main (GT checked against out/psfmatch/gate_m1809a), PSF of the m1809 set.
Output out/crlbcmp/crlbfree/<scen>__s<seed>.npz: gt_x, gt_phot, gt_target, fr_xy [n,2] (free-model x / y CRLB, px),
cov_or [n,8,8], ts_full [n,2], sm_m, sf_m (check).
usage: CUDA_VISIBLE_DEVICES=1 python analysis/crlbcmp/crlb_free.py iso 301,302,303 --calib models/insitu/calib_1809_2_psf3d_avg.json"""
import argparse, os, sys, time
import numpy as np, torch
from simcode.simulate import gate_layouts as ST
from simcode.simulate.gate_layouts import CD, S
from simcode.simulate import crlb as CR
from simcode.benchmark import gate_scenes as GA

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'out', 'crlbcmp', 'crlbfree')
GD = os.path.join(REPO, 'out', 'psfmatch', 'gate_m1809a')


def main(scen, seeds, ntarget=3500, calib=''):
    os.makedirs(OUT, exist_ok=True)
    CD.a.calib = GA.gate_calib(calib)
    T = 6
    sim, rpsf = CD.models(T)
    for sc in scen:
        for seed in seeds:
            fo = f'{OUT}/{sc}__s{seed}.npz'
            if os.path.exists(fo): print('exists', fo); continue
            t0 = time.time(); L = []; need = ntarget; it = 0
            while need > 0 and it < 400:
                it += 1
                o = GA.gen(sc, it, seed)
                G = o['gt']; act = G['onf'].sum(1) > 0
                G = {k: v[act] for k, v in G.items()}
                Gx = G['x'] + o['shift_est'][G['fov']][:, 0]
                ph = G['phot'].sum(1); on = G['onf'].sum(1)
                inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
                vis = (on >= 1) & (ph >= 100)
                tg = torch.nonzero(inter & vis)[:, 0]
                if len(tg) == 0: continue
                fovs = torch.unique(G['fov'][tg]); ai = torch.nonzero(torch.isin(G['fov'], fovs))[:, 0]
                psS = torch.zeros(len(G['x']), CD.psf_channels(rpsf), CD.R, CD.R, device='cuda')
                ck = int(os.environ.get('GATE_PSF_CHUNK', '0')) or len(ai)
                for i in range(0, len(ai), ck):
                    a_ = ai[i:i + ck]; psS[a_] = CD.psf_stack(rpsf, G['x'][a_], G['y'][a_], G['z'][a_])
                Cb = CD.crlb_all(G, o, psS, tg, T, sim.focus); del psS
                full, _ = CR.two_stage_bounds(Cb['cov_or'], G, tg, o, T)
                n = len(G['x'])
                fill = lambda v, sh: torch.full((n,) + sh, float('nan'), device='cuda').index_copy_(0, tg, v.float())
                cov = Cb['cov_or']
                L.append({k: v.detach().cpu().numpy() for k, v in dict(
                    x=Gx, phot=ph, target=(inter & vis).float(), fr_xy=fill(cov[:, [0, 1], [0, 1]].clamp(min=0).sqrt(), (2,)),
                    cov_or=fill(cov, (8, 8)), ts_full=fill(full, (2,)), sm_m=fill(Cb['sm_m'], (2,)), sf_m=fill(Cb['sf_m'], (2,)),
                    sm_i=fill(Cb['sm_i'], (2,)), sf_i=fill(Cb['sf_i'], (2,))).items()})
                need -= len(tg)
                del Cb; torch.cuda.empty_cache()
            D = {'gt_' + k if k in ('x', 'phot', 'target') else k: np.concatenate([r[k] for r in L]) for k in L[0]}
            R = np.load(f'{GD}/{sc}__s{seed}_hd_cur.npz')
            ok = len(R['gt_x']) == len(D['gt_x']) and np.allclose(R['gt_x'], D['gt_x'], atol=1e-4)
            dsm = np.nanmax(np.abs(R['gt_sm_m'] - D['sm_m']) / R['gt_sm_m']) if ok else np.nan
            print(f'{sc} s{seed}: GT identical {ok}, max rel diff sm_m {dsm:.2e}, {time.time() - t0:.0f}s', flush=True)
            np.savez_compressed(fo, **D)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('scen'); ap.add_argument('seeds')
    ap.add_argument('--calib', default='', help="scene PSF calibration (GA.gate_calib; '' = the default model's PSF)")
    a = ap.parse_args()
    main(a.scen.split(','), [int(s) for s in a.seeds.split(',')], calib=a.calib)
