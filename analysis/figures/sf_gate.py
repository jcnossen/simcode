"""Joint SIMFLUX MLE reference (analysis/figures/lmref.SFRef; EVALUATION ONLY, critic 34 #4) on the gate scenes, in the gate npz
format (front 'sc') so plot_crlb_sc3.curves reads it: the same scenes / seeds / target loop / det_n1 candidates (p >= 0.5)
as analysis/figures/lm_gate.py (GT copied from the _hd_sc5nbb15pm npz after checking that the regenerated GT is identical).
  x_lm / y_lm  pattern-free LM (LMRef, z free) from the detector point (= SFRef's start)
  x_sc / y_sc  joint SIMFLUX MLE (known pattern = the scene's patt_est; exact = true pattern on iso / rs05q), 9 fringe starts
kinetics: off (one B per window) on all scenes by default; SF_KIN=1 fits the on-interval [t0, t1] (the CRLB_SF nuisance; tested, worse).
usage: CUDA_VISIBLE_DEVICES=1 python sf_gate.py iso,rs05q,rs05qm 301,302,303 [P=11] [bg=angle] [maxit]
-> <out>/sc7/gate_lm/<scene>__s<seed>_sf<P><bgtag>.npz (bgtag '' = angle, 'f' = frame, 'o' = one)"""
import sys, os, time
import numpy as np, torch
ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]
if not ARGS or ARGS[0] in ('-h', '--help'): print(__doc__); sys.exit(0)
from simcode.simulate.gate_layouts import CD, dev, S
from simcode.benchmark import gate as GA
from simcode.emitter import model as SCM
from simcode.detector import model as DM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lmref import SFRef
from simcode.detector.select import select_subpixel
from simcode import config
OUTR = config.root('out')

GDIR = os.path.join(OUTR, 'gate')
OUT = os.path.join(OUTR, 'sc7', 'gate_lm')


def main():
    scenes = ARGS[0].split(','); seeds = [int(s) for s in ARGS[1].split(',')]
    P = int(ARGS[2]) if len(ARGS) > 2 else 11
    bgm = ARGS[3] if len(ARGS) > 3 else 'angle'
    maxit = int(ARGS[4]) if len(ARGS) > 4 else 400
    bgt = {'angle': '', 'frame': 'f', 'one': 'o'}[bgm]
    os.makedirs(OUT, exist_ok=True)
    CD.a.calib = config.model(config.INSITU_05)   # = the development calib_inputs/calib_0.5nM.json
    T = 6
    sim, rpsf, lut = CD.models(T)
    det, _ = DM.load(config.DETECTOR); det.eval()
    for sc in scenes:
        # kinetics: default OFF on every scene (one B per window). The on-interval model (SF_KIN=1, tag k1) was tested and is
        # worse: iso 1.2x vs 1.0x CRLB (t0 / t1 start inside the window on always-on emitters), rs05q nn > 6 px robust 1.3-1.5 vs 1.1-1.15
        kenv = os.environ.get('SF_KIN', 'auto')
        kin = False if kenv == 'auto' else kenv == '1'
        ktag = '' if kenv == 'auto' else f'k{kenv}'
        sf = SFRef(sim, P, zfree=True, bg=bgm, kin=kin)
        for seed in seeds:
            ref = dict(np.load(f'{GDIR}/{sc}__s{seed}_hd_sc5nbb15pm.npz'))
            t0 = time.time(); GX = []; OUTS = []; need = 3500; it = 0; fov0 = 0; pdiff = 0.0
            while need > 0 and it < maxit:
                it += 1
                o = GA.gen(sc, it, seed)
                nb = o['nb']
                G = o['gt']; act = G['onf'].sum(1) > 0
                G = {k_: v[act] for k_, v in G.items()}
                Gx = G['x'] + o['shift_est'][G['fov']][:, 0]
                ph = G['phot'].sum(1); on = G['onf'].sum(1)
                inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
                tg = torch.nonzero(inter & (on >= 1) & (ph >= 100))[:, 0]
                if len(tg) == 0: continue
                GX.append(Gx.cpu().numpy())
                pdiff = max(pdiff, float((o['patt_est'] - o['patt']).abs().max()))
                with torch.no_grad():
                    fr = o['frames']
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        dd = det(fr)
                    c = select_subpixel(dd['logit'].float(), dd['dxy'].float(), 640, p_min=GA.THR0, radius=0.7)
                    keep = torch.isfinite(c['logit'])
                    b, x, y, lg = (c[k][keep] for k in ('b', 'x', 'y', 'logit'))
                    p = torch.sigmoid(lg); s5 = torch.nonzero(p >= 0.5)[:, 0]
                    n = len(b); nan = lambda w=None: torch.full((n,) if w is None else (n, w), float('nan'), device=dev)
                    r_ = dict(b=b, x_det=x, y_det=y, score=p, x_sc=nan(), y_sc=nan(), x_lm=nan(), y_lm=nan(), s_lm=nan(2), s_sc=nan(2),
                              B_sf=nan(), z_sf=nan(), ishift=nan(), edge=nan(), nll=nan())
                    for c0 in range(0, len(s5), 2048):
                        j = s5[c0:c0 + 2048]
                        cx, cy = torch.round(x[j]).long(), torch.round(y[j]).long()
                        r = sf.fit(fr, o['rn'], o['patt_est'], b[j], cx, cy, x[j], y[j])
                        r_['x_lm'][j] = r['x_lm']; r_['y_lm'][j] = r['y_lm']; r_['s_lm'][j] = r['s_lm']
                        r_['x_sc'][j] = r['x']; r_['y_sc'][j] = r['y']
                        r_['s_sc'][j] = torch.stack([r['cov'][:, 0, 0], r['cov'][:, 1, 1]], 1).clamp(min=0).sqrt()
                        r_['B_sf'][j] = r['B']; r_['z_sf'][j] = r['z']; r_['ishift'][j] = r['ishift'].float()
                        r_['edge'][j] = r['edge'].float(); r_['nll'][j] = r['nll']
                    r_['b'] = r_['b'] + fov0
                    OUTS.append({k_: v.detach().float().cpu().numpy() for k_, v in r_.items()})
                fov0 += nb; need -= len(tg)
                torch.cuda.empty_cache()
            gx = np.concatenate(GX)
            D = {k_: v for k_, v in ref.items() if k_.startswith('gt_') or k_ == 'n_fov'}
            if maxit < 400:     # test run: truncated
                keep = D['gt_fov'] < fov0; D = {k_: (v[keep] if k_.startswith('gt_') else v) for k_, v in D.items()}; D['n_fov'] = np.array(fov0)
            assert len(gx) == len(D['gt_x']) and np.allclose(gx, D['gt_x'], atol=1e-4), f'GT mismatch {sc} {seed}'
            D.update({f'sc_{k_}': np.concatenate([q[k_] for q in OUTS]) for k_ in OUTS[0]})
            tagf = '_test' if maxit < 400 else ''
            np.savez_compressed(f'{OUT}/{sc}__s{seed}_sf{P}{bgt}{ktag}{tagf}.npz', **D)
            print(f'{sc} seed {seed}: GT identical, {int(D["gt_target"].sum())} targets, {fov0} FOVs, max|patt_est - patt| {pdiff:.2e}, '
                  f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
