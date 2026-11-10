"""Basic single-emitter LM reference (analysis/figures/lmref.py; EVALUATION REFERENCE ONLY) for the per-frame I_t / bg rows:
the same scenes, seeds, targets and det_n1 candidate matching as icrlb.py (the target list is checked against the existing
icrlb_sc5nb_<scene>.npz), LM on a P x P ROI:
  det    = ROI at the rounded matched detector point, start = detector point
  forced = ROI at the rounded true position, start = the true position
Also stores the LM positions (pattern-free) and the explicit pattern solve on the LM I_t (PM, box 1.5) for both.
usage: [LM_ZFREE=1] CUDA_VISIBLE_DEVICES=1 python lm_icrlb.py <P> [scenes] [seeds] -> <out>/sc7/icrlb_lm<P>_<scene>.npz"""
import sys, os
import numpy as np, torch
ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]
if not ARGS or ARGS[0] in ('-h', '--help'): print(__doc__); sys.exit(0)
from simcode2.sim.simtools import CD, dev, S
from simcode2.hd import gate as GA, sc as SCM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lmref import LMRef, solve_input, LM_CAL
from simcode2.det.model import select_subpixel
from simcode2 import config
OUTR = config.root('out')

OUT = os.path.join(OUTR, 'sc7')
T = 6


def run(P, scenes, seeds, ntarget=3500):
    CD.a.calib = config.model(config.INSITU_05)   # = the development calib_inputs/calib_0.5nM.json
    sim, rpsf, lut = CD.models(T)
    det, _ = SCM.load_det_front(config.DET); det.eval()
    ZF = os.environ.get('LM_ZFREE', '0') == '1'; zt = 'z' if ZF else ''      # z free (tag suffix 'z') or fixed 50 nm
    lm = LMRef(sim, P, zfree=ZF)
    for sc in scenes:
        rows = []
        for seed in seeds:
            need = ntarget; it = 0
            while need > 0 and it < 400:
                it += 1
                o = GA.gen(sc, it, seed)
                Gf = o['gt']; act = Gf['onf'].sum(1) > 0
                G = {k: v[act] for k, v in Gf.items()}
                ph = G['phot'].sum(1); on = G['onf'].sum(1)
                inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
                tg = torch.nonzero(inter & (on >= 1) & (ph >= 100))[:, 0]
                if len(tg) == 0: continue
                with torch.no_grad():
                    fr = o['frames']
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        dd = det(fr)
                    lgm = dd['logit'].float()
                    c = select_subpixel(lgm, dd['dxy'].float(), 640, p_min=GA.THR0, radius=0.7)
                    keep = torch.isfinite(c['logit'])
                    b, x, y, lg = (c[k][keep] for k in ('b', 'x', 'y', 'logit'))
                    sel = torch.sigmoid(lg) >= 0.5
                    bd, xd, yd = b[sel], x[sel], y[sel]
                    D = torch.cdist(torch.stack([G['x'][tg], G['y'][tg]], 1), torch.stack([xd, yd], 1))
                    D = torch.where(G['fov'][tg][:, None] == bd[None], D, torch.full_like(D, 1e9)).cpu().numpy()
                    mi = np.full(len(tg), -1)
                    if D.size:
                        ii, jj = np.nonzero(D < 1.0); o_ = np.argsort(D[ii, jj]); ua, ub = set(), set()
                        for k in o_:
                            if ii[k] in ua or jj[k] in ub: continue
                            ua.add(ii[k]); ub.add(jj[k]); mi[ii[k]] = jj[k]
                    mt = torch.tensor(mi, device=dev); mm = mt >= 0
                    n = len(tg)
                    nanT = torch.full((n, T), float('nan'), device=dev); nan2 = torch.full((n, 2), float('nan'), device=dev)
                    r = dict(ph=ph[tg], I_gt=G['phot'][tg], tx=G['x'][tg], ty=G['y'][tg], matched=mm.float(),
                             bg_t=o['bg'][G['fov'][tg], :, torch.round(G['y'][tg]).long(), torch.round(G['x'][tg]).long()])
                    for lab, (bb, xx, yy, m_) in dict(
                            det=(bd[mt[mm]] if mm.any() else None, xd[mt[mm]] if mm.any() else None, yd[mt[mm]] if mm.any() else None, mm),
                            f=(G['fov'][tg], G['x'][tg], G['y'][tg], torch.ones_like(mm))).items():
                        I_, bg_, p_, pm_ = nanT.clone(), nanT.clone(), nan2.clone(), nan2.clone()
                        if bb is not None and len(bb):
                            cx, cy = torch.round(xx).long(), torch.round(yy).long()
                            q = lm.fit(fr, o['rn'], bb, cx, cy, xx, yy)
                            so = SCM.sc_solve(solve_input(q, xx, yy), o['patt_est'], bb, LM_CAL)
                            I_[m_] = q['I']; bg_[m_] = q['bg']; p_[m_] = torch.stack([q['x'], q['y']], 1); pm_[m_] = so['r']
                        r.update({f'I_{lab}': I_, f'bg_{lab}': bg_, f'pos_{lab}': p_, f'pospm_{lab}': pm_})
                rows.append({k: v.detach().float().cpu().numpy() for k, v in r.items()})
                need -= len(tg); torch.cuda.empty_cache()
            print(f'lm{P} {sc} seed {seed}: done ({it} it)', flush=True)
        Dd = {k: np.concatenate([r_[k] for r_ in rows]) for k in rows[0]}
        ref = np.load(f'{OUT}/icrlb_sc5nb_{sc}.npz')
        assert len(ref['ph']) == len(Dd['ph']) and np.allclose(ref['ph'], Dd['ph'], rtol=1e-4), 'target list mismatch'
        assert np.allclose(ref['matched'], Dd['matched']), 'det matching mismatch'
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(f'{OUT}/icrlb_lm{P}{zt}_{sc}.npz', **Dd)
        print(f'lm{P}{zt} {sc}: target list identical to icrlb_sc5nb ({len(Dd["ph"])})', flush=True)


if __name__ == '__main__':
    P = int(ARGS[0])
    scenes = (ARGS[1] if len(ARGS) > 1 else 'iso,rs05q,rs05qm').split(',')
    seeds = [int(s) for s in (ARGS[2] if len(ARGS) > 2 else '301,302').split(',')]
    run(P, scenes, seeds)
