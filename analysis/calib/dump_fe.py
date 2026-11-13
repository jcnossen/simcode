"""Dump the sc head outputs (I_t, full Cholesky L, prior, detector point, score) + per-FOV patterns (estimate / true) + GT
on the gate scenes (simcode/hd/gate.gen, identical seeds / iteration loop as gate.main, no CRLB), so pattern-solve
variants (Sigma models, per-bin calibration, prior offset) can be re-run offline. The CRLB (gt_sf_m) is copied from the
existing gate npz of the same scene / seed after checking that the GT is identical.
usage: CUDA_VISIBLE_DEVICES=1 python dump_fe.py --scen iso,rs05q,rs05qm --seeds 301,302,303 [--ref_tag _hd_sc5nbb15pm]"""
import argparse, os, sys, time
from simcode import config as _cfg
import numpy as np, torch
from simcode.sim import simtools as ST
from simcode.sim.simtools import CD, dev, S
from simcode.hd import gate as GA
from simcode.hd import sc as SCM
from simcode.det.model import select_subpixel

OUT = os.path.join(_cfg.root('out'), 'bright', 'fe')


@torch.no_grad()
def head_out(det, head, col, o):
    fr = o['frames']
    with torch.autocast('cuda', dtype=torch.bfloat16):
        dd = det(fr)
    c = select_subpixel(dd['logit'].float(), dd['dxy'].float(), 640, p_min=GA.THR0, radius=0.7)
    keep = torch.isfinite(c['logit'])
    b, iy, ix, x, y, lg = (c[k][keep] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit'))
    if len(b) == 0: return None
    nbr = SCM.nbr_for(head, dd['logit'], b, iy, ix, x, y, lg)
    outs = []
    for c0 in range(0, len(b), 4096):
        sl = slice(c0, c0 + 4096)
        oh = SCM.apply_head(head, col, fr, None, b[sl], iy[sl], ix[sl], x[sl], y[sl], nbr=nbr, p=torch.sigmoid(lg[sl]))
        outs.append({k: oh[k].float() for k in ('I', 'L', 'x', 'y', 'lsxy', 'bg')})
    out = {k: torch.cat([q[k] for q in outs]) for k in outs[0]}
    out.update(b=b, x_det=x, y_det=y, score=torch.sigmoid(lg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', default='iso,rs05q,rs05qm')
    ap.add_argument('--seeds', default='301,302,303')
    ap.add_argument('--ntarget', type=int, default=3500)
    ap.add_argument('--head', default='hd_sc5nb.pt')
    ap.add_argument('--name', default='sc5nb')
    ap.add_argument('--ref_tag', default='_hd_sc5nbb15pm')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    CD.a.calib = GA.gate_calib()
    sim, _, _ = CD.models(6)
    det, _ = SCM.load_det_front(_cfg.model(_cfg.DET)); det.eval()   # as the gate runs (SC_DET) and the pipeline
    head = SCM.load(a.head); col = SCM.PSFCol(sim)
    for sc in a.scen.split(','):
        for seed in [int(s) for s in a.seeds.split(',')]:
            fn = f'{OUT}/{sc}__s{seed}_{a.name}.npz'
            if os.path.exists(fn): print('exists', fn); continue
            t0 = time.time(); GT, OU, PE, PT = [], [], [], []; need = a.ntarget; it = 0; fov0 = 0
            while need > 0 and it < 400:
                it += 1
                o = GA.gen(sc, it, seed)
                nb = o['nb']
                G = o['gt']; act = G['onf'].sum(1) > 0
                G = {k: v[act] for k, v in G.items()}
                Gx = G['x'] + o['shift_est'][G['fov']][:, 0]; Gy = G['y'] + o['shift_est'][G['fov']][:, 1]
                ph = G['phot'].sum(1); on = G['onf'].sum(1)
                inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
                vis = (on >= 1) & (ph >= 100)
                tg = torch.nonzero(inter & vis)[:, 0]
                if len(tg) == 0: continue
                nn = ST.nn_dist(G, tg)
                n = len(G['x'])
                nnf = torch.full((n,), float('nan'), device=dev).index_copy_(0, tg, nn.float())
                GT.append({k: v.detach().float().cpu().numpy() for k, v in dict(
                    fov=(G['fov'] + fov0).float(), x=Gx, y=Gy, phot=ph, onfr=on, target=(inter & vis).float(), nn=nnf, I=G['phot']).items()})
                r = head_out(det, head, col, o)
                if r is not None:
                    r['b'] = r['b'] + fov0
                    OU.append({k: v.detach().float().cpu().numpy() for k, v in r.items()})
                PE.append(o['patt_est'].float().cpu().numpy()); PT.append(o['patt'].float().cpu().numpy())
                fov0 += nb; need -= len(tg)
                torch.cuda.empty_cache()
            cat = lambda L: {k: np.concatenate([r[k] for r in L]) for k in L[0]}
            D = {'gt_' + k: v for k, v in cat(GT).items()}
            D.update({'sc_' + k: v for k, v in cat(OU).items()})
            D['patt_est'] = np.concatenate(PE); D['patt_true'] = np.concatenate(PT); D['n_fov'] = np.array(fov0)
            ref = f'{GA.OUT}/{sc}__s{seed}{a.ref_tag}.npz'
            if os.path.exists(ref):
                R = np.load(ref)
                same = len(R['gt_x']) == len(D['gt_x']) and np.allclose(R['gt_x'], D['gt_x']) and np.allclose(R['gt_phot'], D['gt_phot'])
                print(f'  GT identical to {os.path.basename(ref)}: {same}', flush=True)
                if same: D['gt_sf_m'] = R['gt_sf_m']
            np.savez_compressed(fn, **D)
            print(f'{sc} seed {seed}: {int(D["gt_target"].sum())} targets, {fov0} FOVs, {len(D["sc_b"])} candidates, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
