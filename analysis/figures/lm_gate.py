"""Basic single-emitter LM reference (analysis/figures/lmref.py; EVALUATION REFERENCE ONLY, not the method) on the gate scenes,
in the gate npz format so plot_crlb_sc3.curves can read it (front 'sc'): the same scenes / seeds / target loop as
benchmark/gate.py (GT copied from the existing _hd_sc5nbb15pm npz after checking that the regenerated GT is identical), the same
det_n1 candidates (sub-pixel NMS, p >= THR0; the LM is run on the p >= 0.5 ones, as the curves use score >= 0.5):
  LM on the 11 x 11 ROI at the rounded detector point, start = detector point -> x_lm / y_lm (pattern-free),
  then the explicit pattern solve (solve.solve_emitters, PM, box 1.5, LM Fisher covariance unscaled) on the LM I_t -> x_sc / y_sc.
usage: [LM_ZFREE=1] [LM_BG=angle] CUDA_VISIBLE_DEVICES=1 python lm_gate.py iso,rs05q,rs05qm 301,302,303 [P]
-> <out>/sc7/gate_lm/<scene>__s<seed>_lm<P>.npz"""
import sys, os, time
import numpy as np, torch
ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]
if not ARGS or ARGS[0] in ('-h', '--help'): print(__doc__); sys.exit(0)
from simcode.simulate.gate_layouts import CD, dev, S
from simcode.benchmark import gate as GA
from simcode.emitter import model as SCM
from simcode.solve import solve as SV
from simcode.detector import model as DM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lmref import LMRef, solve_input, LM_CAL
from simcode.detector.select import select_subpixel
from simcode import config
OUTR = config.root('out')

GDIR = os.path.join(OUTR, 'gate')
OUT = os.path.join(OUTR, 'sc7', 'gate_lm')


def main():
    scenes = ARGS[0].split(','); seeds = [int(s) for s in ARGS[1].split(',')]
    P = int(ARGS[2]) if len(ARGS) > 2 else 11
    os.makedirs(OUT, exist_ok=True)
    CD.a.calib = config.model(config.INSITU_05)   # = the development calib_inputs/calib_0.5nM.json
    T = 6
    sim, rpsf, lut = CD.models(T)
    det, _ = DM.load(config.DETECTOR); det.eval()
    ZF = os.environ.get('LM_ZFREE', '0') == '1'; zt = 'z' if ZF else ''      # z free (tag suffix 'z') or fixed 50 nm
    BG = os.environ.get('LM_BG', 'frame'); zt += 'a' if BG == 'angle' else ''     # critic 34: LM_BG=angle -> one bg per pattern angle (tag suffix 'a')
    lm = LMRef(sim, P, zfree=ZF, bg=BG)
    for sc in scenes:
        for seed in seeds:
            ref = dict(np.load(f'{GDIR}/{sc}__s{seed}_hd_sc5nbb15pm.npz'))
            t0 = time.time(); GX = []; OUTS = []; need = 3500; it = 0; fov0 = 0
            while need > 0 and it < 400:
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
                              I_lm=nan(T), bg_lm=nan(T), edge=nan())
                    for c0 in range(0, len(s5), 2048):
                        j = s5[c0:c0 + 2048]
                        cx, cy = torch.round(x[j]).long(), torch.round(y[j]).long()
                        pe = o['patt_est'][b[j]]
                        r = lm.fit(fr, o['rn'], b[j], cx, cy, x[j], y[j], ang=(pe[..., 0].abs() < pe[..., 1].abs()).long())
                        so = SV.solve_emitters(solve_input(r, x[j], y[j]), o['patt_est'], b[j], LM_CAL)
                        r_['x_lm'][j] = r['x']; r_['y_lm'][j] = r['y']; r_['I_lm'][j] = r['I']; r_['bg_lm'][j] = r['bg']
                        r_['s_lm'][j] = torch.stack([r['cov'][:, 0, 0], r['cov'][:, 1, 1]], 1).clamp(min=0).sqrt()
                        r_['x_sc'][j] = so['r'][:, 0]; r_['y_sc'][j] = so['r'][:, 1]; r_['s_sc'][j] = so['s']
                        r_['edge'][j] = r['edge'].float()
                    r_['b'] = r_['b'] + fov0
                    OUTS.append({k_: v.detach().float().cpu().numpy() for k_, v in r_.items()})
                fov0 += nb; need -= len(tg)
                torch.cuda.empty_cache()
            gx = np.concatenate(GX)
            assert len(gx) == len(ref['gt_x']) and np.allclose(gx, ref['gt_x'], atol=1e-4), f'GT mismatch {sc} {seed}'
            D = {k_: v for k_, v in ref.items() if k_.startswith('gt_') or k_ == 'n_fov'}
            D.update({f'sc_{k_}': np.concatenate([q[k_] for q in OUTS]) for k_ in OUTS[0]})
            np.savez_compressed(f'{OUT}/{sc}__s{seed}_lm{P}{zt}.npz', **D)
            print(f'{sc} seed {seed}: GT identical, {int(D["gt_target"].sum())} targets, {fov0} FOVs, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
