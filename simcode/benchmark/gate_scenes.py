"""
Gate benchmark suite (critic review 26 #2), simulation only, experiment-matched simulator (simcode/simulate/acquisition.py).
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
'don't care') and every front-end's FULL output list (position, sigma, posterior sigma, detection score, q / dof) at
a low detection threshold (0.2), so precision / recall curves are swept offline (simcode/benchmark/gate_scenes_report.py).
Front-end: sc = detector (--detector) + emitter network (--emitter) + the explicit pattern solve with the calibration --cal (defaults: the default
model of simcode/models.json, its sim-sigma calibration cal_sim). PSF of the scenes, the CRLB and the head's PSF template: --calib
(an in-situ calibration json), default the PSF the default model was trained on (models.json 'psf'); the scenes are rendered
with simcode.simulate.crlb.a.calib, which main() (and every other user of gen()) sets first.
Other front-ends (SIMCODE: archive-2026-11-15:simcode/benchmark/preprint_gate.py; pass 2: simcode/processing/gate_dump.py) write the same npz format.
Reports: python -m simcode.benchmark.gate_scenes_report_detector_matched --dir <out> --tag <tag> --scen iso,...
usage: python -m simcode.benchmark.gate_scenes --scen iso,pair1.5 --fronts sc --ntarget 3500 [--calib <in-situ calib>] [--out <dir>] [--tag _pretrain]
"""
import argparse, json, math, os, time
import numpy as np, torch
from simcode.simulate import gate_layouts as ST
from simcode.simulate.gate_layouts import CD, S
from simcode.simulate.acquisition import AcquisitionSim
from simcode import config

dev = 'cuda'
OUT = os.path.join('simcode_out', 'gate')          # default output folder
RSI = {}
THR0 = 0.2


def gate_calib(calib=''):
    """PSF calib of the gate scenes (renders the scenes, the CRLB and the head's PSF template): `calib` (an in-situ calibration
    json, name or path), default the PSF of the default model (models.json 'psf'; until 2026-11-14 the default was the 0.5 nM
    in-situ calib insitu/calib_0.5nM.json). Set simcode.simulate.crlb.a.calib to it before gen()."""
    return config.resolve(calib) if calib else config.model(config.model_set()['psf'])


def rsim(pre):
    if pre not in RSI:
        RSI[pre] = AcquisitionSim(pre, S, S, 6, calib=CD.a.calib)
    return RSI[pre]


POVER, SOVER = {}, {}     # mismatch ablation (analysis/ablation_cov/gate_dump.py): AcquisitionSim preset / scene() overrides for the *pm / *qm scenes
Q_MED = {'0.5nM': 1260.0, '2.5nM': 1890.0}    # iteration 32: label-unit paper brightness (rs05q / rs25q) = SIMCODE fit x kappa, sc7/kappa.txt
Q_SD = 0.45
def gen(sc, it, seed, T=6):
    # rs05pm / rs25pm (iteration 31, critic 30 #2): seeded as rs05p / rs25p -> the same emitters and paper_bright draw
    # rs05qm / rs25qm (iteration 32): seeded as rs05q / rs25q (label-unit brightness Q_MED / Q_SD)
    sk = sc[:-1] if sc in ('rs05pm', 'rs25pm', 'rs05qm', 'rs25qm') else sc
    torch.manual_seed(seed * 1000003 + it * 7919 + sum(ord(c) * (i + 1) for i, c in enumerate(sk)))
    exact = dict(emitter_var=False, psf_var=False, cam_err=False, exact_est=True, bgmode='struct')
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
        em['z'] = rs.draw_z(len(em['x']))     # the presets' lognormal (same draws as before), or the calibration's heights
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
            from simcode.simulate.batches import paper_bright
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
            from simcode.simulate.batches import paper_bright
            em['bright'] = paper_bright(pre, nb, em['fov'], em['z'], rs, jitter=None)
            save = dict(rs.p); rs.p.update(POVER)
            try:
                o = rs.scene(nb, em=em, **SOVER)
            finally:
                rs.p.clear(); rs.p.update(save)
        elif sc.endswith('p'):     # iteration 30b: paper-matched photons (data.paper_bright, no per-window jitter), exact model
            from simcode.simulate.batches import paper_bright
            em['bright'] = paper_bright(pre, nb, em['fov'], em['z'], rs, jitter=None)
            o = rs.scene(nb, em=em, **exact)
        elif sc.endswith('x'):
            o = rs.scene(nb, em=em, **exact)
        elif sc == 'rs25s':
            o = rs.scene(nb, em=em, fluct=3 * rs.p['fluct'])
        else:
            o = rs.scene(nb, em=em)
    return o


def psf_rows(psS, rpsf, G, ai):
    """psS[ai] = CD.psf_stack of the emitters ai, in chunks that fit the free GPU memory (halved after an out-of-memory error;
    per-emitter values, the same for any chunk size). Replaces the env GATE_PSF_CHUNK (2026-11-14)."""
    ck = len(ai)
    while True:
        try:
            for i in range(0, len(ai), max(ck, 1)):
                a_ = ai[i:i + ck]; psS[a_] = CD.psf_stack(rpsf, G['x'][a_], G['y'][a_], G['z'][a_])
            return
        except torch.OutOfMemoryError:
            if ck <= 16: raise
            torch.cuda.empty_cache(); ck = (ck + 1) // 2


@torch.no_grad()
def out_sc(SC, o, zmax):
    """the SIMCODE v2 front end: detector (sub-pixel NMS + top-k, p > THR0) -> emitter network (I_t, full covariance, bg_t,
    position prior) -> explicit intensity-domain pattern solve (fringe hypotheses, kinetic selection, posterior sigma)."""
    from simcode.detector.select import select_subpixel
    from simcode.emitter import model as EM
    from simcode.solve import solve as SV
    det, head, col, cal = SC
    fr = o['frames']; nb = fr.shape[0]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        dd = det(fr)
    c = select_subpixel(dd['logit'].float(), dd['dxy'].float(), 640, p_min=THR0, radius=0.7)
    keep = torch.isfinite(c['logit'])
    b, iy, ix, x, y, lg = (c[k][keep] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit'))
    if len(b) == 0: return None
    nbr = EM.nbr_map(dd['logit'].float(), b, iy, ix, x, y, lg)
    outs = []
    for c0 in range(0, len(b), 4096):
        sl = slice(c0, c0 + 4096)
        oh = EM.apply_to_candidates(head, col, fr, None, b[sl], iy[sl], ix[sl], x[sl], y[sl], nbr=nbr, p=torch.sigmoid(lg[sl]))
        so = SV.solve_emitters(oh, o['patt_est'], b[sl], cal)
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
    ap.add_argument('--detector', default=config.DETECTOR)
    ap.add_argument('--emitter', default=config.EMITTER)
    ap.add_argument('--cal', default=config.CAL_SIM)
    ap.add_argument('--calib', default='', help="PSF of the scenes (in-situ calibration json); default the default model's PSF")
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    CD.a.calib = gate_calib(a.calib)
    T = 6
    sim, rpsf = CD.models(T)
    fronts = a.fronts.split(',')
    assert fronts == ['sc'], 'only the sc front-end is available (SIMCODE preprint: archive-2026-11-15:simcode/benchmark/preprint_gate.py; pass 2: simcode.processing.gate_dump)'
    from simcode.emitter import model as EM
    from simcode.detector import model as DM
    det_, _ = DM.load(a.detector)
    from simcode.solve.solve import solve_cal
    from simcode.simulate.crlb import BRIGHTNESS_NOISE        # the gate scenes' constant emitter brightness: c = 0
    SC = (det_, EM.load(config.resolve(a.emitter)), EM.PSFCol(sim), solve_cal(config.resolve(a.cal), BRIGHTNESS_NOISE) if a.cal else {})
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
                psf_rows(psS, rpsf, G, ai)
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
