"""
Gate-scene dumps for the pass-2 benchmark (as <out>/bright/dump_fe.py; simcode/benchmark/gate_scenes.gen, identical seeds
and iteration loop as gate_scenes.main). Per scene x seed x front-end: GT table, per-FOV patterns (estimate fed to the networks /
the solve, and the truth), and the raw head outputs (I_t, Cholesky L, prior x, y, log sigma, bg, detector point, score; z,
log s_z|xy, b of a 3D head),
so the explicit pattern solve, the calibration and all reports run offline. CRLB gt_sf_m copied from an existing gate npz
of the same scene / seed (--gate_dir) after checking that the GT is identical.
Fronts:
  pass1  pass-1 detector + emitter network of the default model (config.DETECTOR / EMITTER; pattern-free)
  pass2  pass-2 detector + emitter network of the default model (config.DETECTOR_PASS2 / EMITTER_PASS2)
  (other detector / network pairs: --fronts name=det.pt:head.pt)
Pattern fed (both passes, networks and solve): the scene's patt_est (gate default; exact for iso / pairs, the acquisition simulator errors for
*qm), or with --pert 1 a perturbed estimate of the truth (pattern/illumination.perturb, the training error model, own seed) with the
GT moved into its gauge (shift = K^-1 e_angle, as the acquisition simulator shift_est).
usage: CUDA_VISIBLE_DEVICES=1 python -m simcode.processing.gate_dump --scen iso,pair1 --fronts pass1,pass2 --seeds 301,302,303 \
         [--calib <scene PSF calib>] [--gate_dir <simcode.benchmark.gate_scenes --out>] [--out <dir>]
"""
import argparse, glob, os, sys, time
import numpy as np, torch
from simcode.simulate import gate_layouts as ST
from simcode.simulate.gate_layouts import CD, dev, S
from simcode.benchmark import gate_scenes as GA
from simcode.pattern import illumination as IL
from simcode.emitter import model as EM
from simcode.detector import model as DM

from simcode import config as _cfg

OUT = os.path.join('simcode_out', 'gate_dumps')         # default output folder
GATE = os.path.join('simcode_out', 'gate')              # default --gate_dir (simcode.benchmark.gate_scenes --out)
FRONTS = dict(pass1=(_cfg.model(_cfg.DETECTOR), _cfg.model(_cfg.EMITTER)),            # pass 1 of the default model
              pass2=(_cfg.model(_cfg.DETECTOR_PASS2), _cfg.model(_cfg.EMITTER_PASS2)))  # pass 2 of the default model


@torch.no_grad()
def front_out(F, o, il, col, pf_head=None):
    """F = (detector, emitter network, illumination on): the network outputs on the gate scene o with the pattern maps il"""
    det, head, il_on = F
    fr = o['frames']
    ilx = il if il_on else torch.zeros_like(il)
    c, nbr, lgm = EM.detect_candidates(det, fr, ilx)
    if len(c['b']) == 0: return None
    p = torch.sigmoid(c['logit'])
    outs = []
    for c0 in range(0, len(p), 4096):
        sl = slice(c0, c0 + 4096)
        args = (fr, ilx, c['b'][sl], c['iy'][sl], c['ix'][sl], c['x'][sl], c['y'][sl], nbr, p[sl])
        oh = EM.apply_to_candidates(head, col, *args)
        r = {k: oh[k].float() for k in ('I', 'L', 'C', 'x', 'y', 'lsxy', 'bg')}
        r.update(EM.emitter_z(head, oh))           # 3D networks: z, lsz, bz (NaN for 2D checkpoints)
        if pf_head is not None:
            oa = EM.apply_to_candidates(pf_head, col, *args)
            r.update(pf_x=oa['x'].float(), pf_y=oa['y'].float(), pf_lsxy=oa['lsxy'].float(), pf_I=oa['I'].float(), pf_L=oa['L'].float())
        outs.append(r)
    out = {k: torch.cat([q[k] for q in outs]) for k in outs[0]}
    out.update(b=c['b'], x_det=c['x'], y_det=c['y'], score=p)
    return out


def gauge_shift(pe, pt):
    """common per-angle phase error of the estimate (at the window centre) -> translation K^-1 e [nb,2]"""
    from simcode.solve import pattern as C
    ang = IL.angle_of(pt)
    cx = cy = (S - 1) / 2
    ph = lambda q: q[..., 2] - q[..., 0] * cx - q[..., 1] * cy           # phase offset at the centre
    d = torch.remainder(ph(pe) - ph(pt) + np.pi, 2 * np.pi) - np.pi      # [nb,T]
    e = torch.stack([(d * (ang == a)).sum(1) / (ang == a).sum(1).clamp(min=1) for a in (0, 1)], 1)
    K = torch.stack([torch.stack([(pt[..., j] * (ang == a)).sum(1) / (ang == a).sum(1).clamp(min=1) for j in (0, 1)], 1) for a in (0, 1)], 1)
    return torch.linalg.solve(K, e[..., None])[..., 0]


def find_ref(gate_dir, sc, seed, gx0, gph):
    for f in sorted(glob.glob(f'{gate_dir}/{sc}__s{seed}_hd_*.npz')):
        R = np.load(f)
        if 'gt_sf_m' in R.files and len(R['gt_x']) == len(gx0) and np.allclose(R['gt_x'], gx0, atol=1e-4) and np.allclose(R['gt_phot'], gph):
            return R['gt_sf_m'], os.path.basename(f)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', default='iso')
    ap.add_argument('--seeds', default='301,302,303')
    ap.add_argument('--fronts', default='pass1,pass2')     # names in FRONTS, or name=det.pt:head.pt
    ap.add_argument('--ntarget', type=int, default=3500)
    ap.add_argument('--pert', type=int, default=0)
    ap.add_argument('--pert_model', default='v1')  # illumination.MODELS ('real' = the training error model of the earlier default pass-2 network)
    ap.add_argument('--tag', default='')
    ap.add_argument('--p2', default='')           # override 'det,head' of the p2 front
    ap.add_argument('--calib', default='', help="PSF of the scenes (in-situ calibration json); default the default model's PSF")
    ap.add_argument('--gate_dir', default=GATE, help='folder of the simcode.benchmark.gate_scenes npz files of the same scenes / seeds (their CRLB is copied)')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    CD.a.calib = GA.gate_calib(a.calib)
    sim, _ = CD.models(6)
    col = EM.PSFCol(sim)
    if a.p2: FRONTS['p2'] = tuple(a.p2.split(','))
    FR = {}
    pf_head = None
    fronts = []
    for f in a.fronts.split(','):
        if '=' in f:
            f, dh = f.split('='); FRONTS[f] = tuple(dh.split(':'))
        fronts.append(f)
    for f in fronts:
        dp, hp = FRONTS[f]
        det, _ = DM.load(dp)
        FR[f] = (det, EM.load(hp), f != 'ctl')
        if f == 'p2': pf_head = EM.load(_cfg.model(_cfg.EMITTER))
    for sc in a.scen.split(','):
        for seed in [int(s) for s in a.seeds.split(',')]:
            fns = {f: f'{a.out}/{sc}__s{seed}_{f}{a.tag}.npz' for f in FR}
            todo = [f for f in FR if not os.path.exists(fns[f])]
            if not todo: print('exists', sc, seed); continue
            t0 = time.time(); GT, PE, PT = [], [], []; OU = {f: [] for f in todo}; need = a.ntarget; it = 0; fov0 = 0
            gx0 = []
            while need > 0 and it < 400:
                it += 1
                o = GA.gen(sc, it, seed)
                nb = o['nb']
                G = o['gt']; act = G['onf'].sum(1) > 0
                G = {k: v[act] for k, v in G.items()}
                pe = o['patt_est']; shift = o['shift_est']
                if a.pert:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(seed * 31 + it)
                        pe = IL.perturb(o['patt'], S, S, model=a.pert_model)
                    shift = gauge_shift(pe, o['patt'])
                Gx = G['x'] + shift[G['fov']][:, 0]; Gy = G['y'] + shift[G['fov']][:, 1]
                gx0.append((G['x'] + o['shift_est'][G['fov']][:, 0]).cpu().numpy())
                ph = G['phot'].sum(1); on = G['onf'].sum(1)
                inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
                vis = (on >= 1) & (ph >= 100)
                tg = torch.nonzero(inter & vis)[:, 0]
                if len(tg) == 0: continue
                nn = ST.nn_dist(G, tg)
                n = len(G['x'])
                nnf = torch.full((n,), float('nan'), device=dev).index_copy_(0, tg, nn.float())
                GT.append({k: v.detach().float().cpu().numpy() for k, v in dict(
                    fov=(G['fov'] + fov0).float(), x=Gx, y=Gy, xt=G['x'], yt=G['y'], z=G['z'], phot=ph, onfr=on,
                    target=(inter & vis).float(), nn=nnf, I=G['phot']).items()})
                il = IL.illum_maps(pe, S, S)
                for f in todo:
                    r = front_out(FR[f], o, il, col, pf_head if f == 'p2' else None)
                    if r is not None:
                        r['b'] = r['b'] + fov0
                        OU[f].append({k: v.detach().float().cpu().numpy() for k, v in r.items()})
                PE.append(pe.float().cpu().numpy()); PT.append(o['patt'].float().cpu().numpy())
                fov0 += nb; need -= len(tg)
                torch.cuda.empty_cache()
            cat = lambda L: {k: np.concatenate([r[k] for r in L]) for k in L[0]}
            D = {'gt_' + k: v for k, v in cat(GT).items()}
            D['patt_est'] = np.concatenate(PE); D['patt_true'] = np.concatenate(PT); D['n_fov'] = np.array(fov0)
            sf, rn = find_ref(a.gate_dir, sc, seed, np.concatenate(gx0), D['gt_phot'])
            print(f'  CRLB from {rn}', flush=True)
            D['gt_sf_m'] = sf if sf is not None else np.full((len(D['gt_x']), 2), np.nan, np.float32)
            for f in todo:
                Df = dict(D); Df.update({'sc_' + k: v for k, v in cat(OU[f]).items()})
                np.savez_compressed(fns[f], **Df)
            print(f'{sc} seed {seed}: {int(D["gt_target"].sum())} targets, {fov0} FOVs, fronts {todo}, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
