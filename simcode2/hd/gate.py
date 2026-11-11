"""
Gate benchmark suite (critic review 26 #2), simulation only, experiment-matched simulator (simcode2/hd/realsim.py).
Scenes (64 x 64 FOVs, T = 6):
  iso         isolated always-on emitters (18 px grid), window photons log-uniform 100-20000, 0.5 nM preset EXACT model
  pair<s>     1:1 pairs at s = 1 / 1.5 / 2 / 3 px, random orientation, totals 300-5000, kinetics (>= 3 on-frames), exact
  rs05x/rs25x presets with the EXACT model (no emitter intensity variation, no PSF / halo jitter, no camera error,
              exact pattern, static structured bg) -> the multi-emitter CRLB is the right bound
  rs05/rs25   full mismatch (fluct bg, emitter variation, PSF + halo jitter, pattern-estimate errors, camera error);
              judged as degradation relative to the x version (same emitters / seeds)
  rs25s       stress: 2.5 nM preset at 2x density with 3x the fluctuation
  rs05p/rs25p paper photons (data.paper_bright), exact model; rs05pm/rs25pm the same emitters + photon draw (same
              seeds) with the full rs05 / rs25 mismatch (iteration 31)
Each scene x eval seed (301, 302, 303; tuning used 202 / 5202) stores the GT table (all active emitters: visible
targets = interior, >= 1 on-frame, >= 100 photons, with their multi-emitter / isolated CRLBs; dim emitters are
'don't care') and every front-end's FULL output list (position, sigma, chi2-aware sigma, detection score, q / dof) at
a low detection threshold (0.2), so precision / recall curves are swept offline (simcode2/hd/gate_report.py).
Front-end: sc = detector (det_n1, env SC_DET to override) + head (--sc, default models/hd_sc5nb.pt) + the explicit pattern
solve with the calibration --sccal (default models/calib/calib_sc_hd_sc5nbb15_pm_bright.json, the sim-sigma gate calibration).
Other front-ends (SIMCODE: simcode2/eff/simcode_gate.py; pass 2: simcode2/p2/dump.py) write the same npz format.
Reports: python -m simcode2.hd.gate_report_xdet --dir <out> --tag <tag> --scen iso,...
usage: python -m simcode2.hd.gate --scen iso,pair1.5 --fronts sc --ntarget 3500 [--out <dir>] [--tag _sc5nb]
"""
import argparse, json, math, os, time
import numpy as np, torch
from simcode2.sim import simtools as ST
from simcode2.sim.simtools import CD, S
from simcode2.hd.realsim import RealSim
from simcode2 import config

dev = 'cuda'
OUT = os.path.join(config.root('out'), 'gate')
RSI = {}
THR0 = 0.2


def rsim(pre):
    if pre not in RSI: RSI[pre] = RealSim(pre, S, S, 6)
    return RSI[pre]


POVER, SOVER = {}, {}     # iteration 32 (critic 31 #2) mismatch ablation: RealSim preset / scene() overrides for the *pm / *qm scenes
Q_MED = {'0.5nM': 1260.0, '2.5nM': 1890.0}    # iteration 32: label-unit paper brightness (rs05q / rs25q) = SIMCODE fit x kappa, sc7/kappa.txt
Q_SD = 0.45
# (2026-11-10, retrain on realistic sims) realistic gate scenes rs05r / rs25r = the m05r / m25rc movie recipes
# (<out>/simreal2/run_m05r.sh, run_m25rc.sh) on 64 x 64 FOVs, full mismatch (as rs05pm), but with a RANDOM static
# pattern phase / depth field per FOV (realsim pf_*; mean rms 0.12 / 0.14 rad ~ 4.3 / 5 nm, the measured own10 amplitude)
# instead of the measured map (critic 35: the measured map is partly anchor-specific). dim = (density, window ph, tau).
RREC = {'rs05r': dict(pre='0.5nM', med=700.0, sd=0.45, dim=None,
                      p=dict(dens=0.0105, fil_on=0.97, fil_resample=True, fil_density=0.6, frame_sd=0.09, step_sd=0.14,
                             pf_rad=0.24, pf_sig=(8.0, 12.0), pf_dep=0.14)),
        'rs25r': dict(pre='2.5nM', med=750.0, sd=0.45, dim=(0.02, 300.0, 20.0),
                      p=dict(fil_on=0.97, fil_resample=True, fil_density=0.8, frame_sd=0.09, step_sd=0.10,
                             pf_rad=0.28, pf_sig=(8.0, 12.0), pf_dep=0.10))}
# bright variants (calibration support above 2000 ph): the same recipes with 2.5x the window-photon median
RREC.update({k + 'b': dict(v, med=2.5 * v['med']) for k, v in list(RREC.items())})


def gen(sc, it, seed, T=6):
    # rs05pm / rs25pm (iteration 31, critic 30 #2): seeded as rs05p / rs25p -> the same emitters and paper_bright draw
    # rs05qm / rs25qm (iteration 32): seeded as rs05q / rs25q (label-unit brightness Q_MED / Q_SD)
    sk = sc[:-1] if sc in ('rs05pm', 'rs25pm', 'rs05qm', 'rs25qm') else sc
    torch.manual_seed(seed * 1000003 + it * 7919 + sum(ord(c) * (i + 1) for i, c in enumerate(sk)))
    exact = dict(emitter_var=False, psf_var=False, cam_err=False, exact_est=True, bgmode='struct')
    if sc in RREC:
        from simcode2.hd.data import paper_bright
        RC = RREC[sc]; pre = RC['pre']; rs = rsim(pre); nb = 12
        save = dict(rs.p); rs.p.update(RC['p'])
        try:
            em = rs.emitters(nb, struct_frac=1.0)
            em['bright'] = paper_bright(pre, nb, em['fov'], em['z'], rs, jitter=None, med0=RC['med'], sd=RC['sd'])
            if RC['dim'] is not None:
                e2 = rs.emitters(nb, dens=RC['dim'][0], tau=RC['dim'][2], struct_frac=1.0, fil_pts_in=em['fil_pts'])
                e2['bright'] = paper_bright(pre, nb, e2['fov'], e2['z'], rs, jitter=None, med0=RC['dim'][1], sd=0.5)
                for k_ in ('fov', 'x', 'y', 'z', 'bright', 't0', 't1'): em[k_] = torch.cat([em[k_], e2[k_]])
            o = rs.scene(nb, em=em)
        finally:
            rs.p.clear(); rs.p.update(save)
        return o
    if sc == 'iso' or sc.startswith('pair'):
        rs = rsim('0.5nM'); nb = 24
        if sc == 'iso':
            em = ST.iso_emitters(nb, T, rs.zmax, tau=50.0, min_on=3.0)
            em['t0'] = torch.full_like(em['t0'], -1e4); em['t1'] = torch.full_like(em['t1'], 1e4)
            ST.wide_brightness(em, T)
        else:
            em, _ = ST.pair_emitters(nb, float(sc[4:]), T, rs.zmax)
            n = len(em['x']) // 2
            tot = torch.exp(math.log(300.0) + (math.log(5000.0) - math.log(300.0)) * torch.rand(n, device=dev))
            em['bright'] = torch.cat([tot, tot]) / T
        em['z'] = (rs.p['z_med'] * torch.exp(rs.p['z_sd'] * torch.randn(len(em['x']), device=dev))).clamp(0, rs.zmax)
        o = rs.scene(nb, em=em, **dict(exact, bgmode='flat'))
    else:
        pre = {'rs05': '0.5nM', 'rs25': '2.5nM'}[sc[:4]]
        rs = rsim(pre)
        dens = rs.p['dens'] * (2.0 if sc == 'rs25s' else 1.0)
        nb = max(2, min(24, int(2.4e4 // max(1.0, dens * (T + 50) / 50 * (S + 12) ** 2 * 10))))
        if sc == 'rs25s': nb = min(nb, 4)          # CRLB memory (many neighbours per target)
        if sc in ('rs25q', 'rs25qm'): nb = min(nb, 12)    # iteration 32: CRLB memory (brighter -> more visible neighbours)
        em = rs.emitters(nb, dens=dens)
        if sc[4:5] == 'q':       # iteration 32: label-unit paper brightness; 'qm' = + full mismatch (with the ablation overrides)
            from simcode2.hd.data import paper_bright
            em['bright'] = paper_bright(pre, nb, em['fov'], em['z'], rs, jitter=None, med0=Q_MED[pre], sd=Q_SD)
            if sc.endswith('qm'):
                save = dict(rs.p); rs.p.update(POVER)
                try:
                    o = rs.scene(nb, em=em, **SOVER)
                finally:
                    rs.p.clear(); rs.p.update(save)
            else:
                o = rs.scene(nb, em=em, **exact)
        elif sc.endswith('pm'):    # iteration 31: paper photons (same draw as rs..p) + the full mismatch of rs05 / rs25
            from simcode2.hd.data import paper_bright
            em['bright'] = paper_bright(pre, nb, em['fov'], em['z'], rs, jitter=None)
            save = dict(rs.p); rs.p.update(POVER)
            try:
                o = rs.scene(nb, em=em, **SOVER)
            finally:
                rs.p.clear(); rs.p.update(save)
        elif sc.endswith('p'):     # iteration 30b: paper-matched photons (data.paper_bright, no per-window jitter), exact model
            from simcode2.hd.data import paper_bright
            em['bright'] = paper_bright(pre, nb, em['fov'], em['z'], rs, jitter=None)
            o = rs.scene(nb, em=em, **exact)
        elif sc.endswith('x'):
            o = rs.scene(nb, em=em, **exact)
        elif sc == 'rs25s':
            o = rs.scene(nb, em=em, fluct=3 * rs.p['fluct'])
        else:
            o = rs.scene(nb, em=em)
    return o


@torch.no_grad()
def out_sc(SC, o, zmax):
    """sc front-end: detector (sub-pixel NMS + top-k, p > THR0) -> SCHead (network I_t, full covariance, bg_t,
    position prior) -> explicit intensity-domain pattern solve (fringe hypotheses, kinetic selection, chi2-aware sigma)."""
    from simcode2.det.model import select_subpixel
    from simcode2.hd import sc as SCM
    det, head, col, cal = SC
    fr = o['frames']; nb = fr.shape[0]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        dd = det(fr)
    c = select_subpixel(dd['logit'].float(), dd['dxy'].float(), 640, p_min=THR0, radius=0.7)
    keep = torch.isfinite(c['logit'])
    b, iy, ix, x, y, lg = (c[k][keep] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit'))
    if len(b) == 0: return None
    nbr = SCM.nbr_for(head, dd['logit'], b, iy, ix, x, y, lg)
    outs = []
    for c0 in range(0, len(b), 4096):
        sl = slice(c0, c0 + 4096)
        oh = SCM.apply_head(head, col, fr, None, b[sl], iy[sl], ix[sl], x[sl], y[sl], nbr=nbr, p=torch.sigmoid(lg[sl]))
        so = SCM.sc_solve(oh, o['patt_est'], b[sl], cal)
        outs.append(dict(x_sc=so['r'][:, 0], y_sc=so['r'][:, 1], s_sc=so['s'], scal_sc=so['s_cal'], q=so['q'],
                         x_prior=oh['x'], y_prior=oh['y'], s_prior=oh['lsxy'].exp(), I_net=oh['I'],
                         sI_net=oh['L'].pow(2).sum(-1).sqrt(), bg_net=oh['bg'], kin=so['kin'].float()))
    out = {k: torch.cat([q[k] for q in outs]) for k in outs[0]}
    out.update(b=b, x_det=x, y_det=y, score=torch.sigmoid(lg), phot=out['I_net'].sum(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', default='iso')
    ap.add_argument('--fronts', default='sc')
    ap.add_argument('--seeds', default='301,302,303')
    ap.add_argument('--ntarget', type=int, default=3500)     # visible interior targets per seed
    ap.add_argument('--tag', default='')
    ap.add_argument('--sc', default=config.HEAD)
    ap.add_argument('--sccal', default=config.CAL_P1_SIM)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--pover', default='')     # e.g. 'frame_sd=0,step_sd=0' (RealSim preset values, *pm / *qm scenes only)
    ap.add_argument('--sover', default='')     # e.g. 'psf_var=0,bgmode=struct,cam_err=0,exact_est=1' (scene() flags, *pm / *qm)
    a = ap.parse_args()
    for kv in filter(None, a.pover.split(',')):
        k_, v_ = kv.split('='); POVER[k_] = float(v_)
    for kv in filter(None, a.sover.split(',')):
        k_, v_ = kv.split('='); SOVER[k_] = v_ if k_ == 'bgmode' else bool(int(v_))
    if POVER or SOVER: print('ablation overrides', POVER, SOVER, flush=True)
    os.makedirs(a.out, exist_ok=True)
    CD.a.calib = config.model(config.INSITU_05)
    T = 6
    sim, rpsf, lut = CD.models(T)
    fronts = a.fronts.split(',')
    assert fronts == ['sc'], 'only the sc front-end is available (SIMCODE: simcode2.eff.simcode_gate; pass 2: simcode2.p2.dump)'
    from simcode2.hd import sc as SCM
    det_, _ = SCM.load_det_front()
    SC = (det_, SCM.load(config.resolve(a.sc)), SCM.PSFCol(sim), json.load(open(config.resolve(a.sccal))) if a.sccal else {})
    for sc in a.scen.split(','):
        for seed in [int(s_) for s_ in a.seeds.split(',')]:
            t0 = time.time(); GT, OUTS = [], {f: [] for f in fronts}; need = a.ntarget; it = 0; fov0 = 0
            while need > 0 and it < 400:
                it += 1
                o = gen(sc, it, seed)
                nb = o['nb']
                G = o['gt']; act = G['onf'].sum(1) > 0
                G = {k_: v[act] for k_, v in G.items()}
                Gx = G['x'] + o['shift_est'][G['fov']][:, 0]; Gy = G['y'] + o['shift_est'][G['fov']][:, 1]
                ph = G['phot'].sum(1); on = G['onf'].sum(1)
                inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
                vis = (on >= 1) & (ph >= 100)
                tg = torch.nonzero(inter & vis)[:, 0]
                if len(tg) == 0: continue
                nn = ST.nn_dist(G, tg)
                fovs = torch.unique(G['fov'][tg]); ai = torch.nonzero(torch.isin(G['fov'], fovs))[:, 0]
                psS = torch.zeros(len(G['x']), CD.psf_channels(rpsf), CD.R, CD.R, device=dev)
                _ck = int(os.environ.get('GATE_PSF_CHUNK', '0'))    # 2026-11-09 simcode_base2: optional chunking (memory cap); default unchanged
                if _ck:
                    for _i in range(0, len(ai), _ck):
                        _a = ai[_i:_i + _ck]; psS[_a] = CD.psf_stack(rpsf, G['x'][_a], G['y'][_a], G['z'][_a], sim.focus)
                else:
                    psS[ai] = CD.psf_stack(rpsf, G['x'][ai], G['y'][ai], G['z'][ai], sim.focus)
                Cb = CD.crlb_all(G, o, psS, tg, T, sim.focus); del psS
                n = len(G['x'])
                full = lambda v, w=2: torch.full((n, w), float('nan'), device=dev).index_copy_(0, tg, v.float())
                nnf = torch.full((n,), float('nan'), device=dev).index_copy_(0, tg, nn.float())
                GT.append({k_: v.detach().float().cpu().numpy() for k_, v in dict(
                    fov=(G['fov'] + fov0).float(), x=Gx, y=Gy, phot=ph, onfr=on, z=G['z'], target=(inter & vis).float(),
                    dontcare=(inter & ~vis).float(), nn=nnf, sf_m=full(Cb['sf_m']), sf_i=full(Cb['sf_i']), sm_m=full(Cb['sm_m']),
                    sm_i=full(Cb['sm_i']), I=G['phot'], fr_m=full(Cb['fr_m'], T)).items()})
                for f in fronts:
                    try:
                        r_ = out_sc(SC, o, sim.zmax)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache(); print('OOM', sc, f, it, flush=True); r_ = None
                    if r_ is None: continue
                    r_['b'] = r_['b'] + fov0
                    OUTS[f].append({k_: (v.detach().float().cpu().numpy()) for k_, v in r_.items()})
                fov0 += nb; need -= len(tg)
                torch.cuda.empty_cache()
            cat = lambda L: {k_: np.concatenate([r[k_] for r in L]) for k_ in L[0]} if L else {}
            D = {'gt_' + k_: v for k_, v in cat(GT).items()}
            D['n_fov'] = np.array(fov0)
            for f in fronts:
                D.update({f'{f}_{k_}': v for k_, v in cat(OUTS[f]).items()})
            np.savez_compressed(f'{a.out}/{sc}__s{seed}{a.tag}.npz', **D)
            print(f'{sc} seed {seed}: {int(D["gt_target"].sum())} targets, {fov0} FOVs, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
