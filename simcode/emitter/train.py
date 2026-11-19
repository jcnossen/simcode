"""
Trainer of the emitter network (emitter/model.EmitterNet), pass 1 and pass 2 (one trainer; the pass follows from the
checkpoints, see --illum).

Common: candidates of a given detector (--detector) on simulated batches (simulate/batches.TrainingBatches), matched one-to-one to
the ground truth (simulate/targets.match_gt, < 1 px, detector p >= 0.5); losses with the nn-bin weights
(simulate/targets.nn_weights: < 1.5 / 1.5-2.5 / 2.5-4 / > 4 px carry 0.15 / 0.25 / 0.25 / 0.35 of each window's weight):
  NLL of the GT per-frame photons given the prior position error (nll_joint: the joint (x, y, I_1..6) covariance of the
  network, 2026-11-17; = nll_full when C = 0) / 6 + the position-prior NLL vs the TRUE position (x, y) + the
  z | x, y NLL (nll_z; 3D prior, 2026-11-14) + 0.1 bg + w_mr x the anti-shrinkage Huber of the mean-removed intensities +
  w_slope x (1 - slope of the mean-removed intensities on the isolated targets)^2. bf16 autocast for the CNN trunk.
  Training emitters over the TIRF height range (simulate/batches.Z_RANGE_UM); a 2D checkpoint as --init gets the zero z rows
  (model.pad_out), so it starts at exactly its 2D outputs.
Pass 1 (illum 0): emitter_batch with the paper-matched photons and the bright tail (BRIGHT_TAIL), density --dens, per-window
  median --med, pair fraction --pair_frac (pair curriculum), on-time --tau; the neighbour map from the detector's p >= 0.2
  candidates; held-out = the stored gate-type patches of emitter/heldout.py (--hold), evaluated every --eval_every steps.
Pass 2 (illum 1): illumination_batch ('paper' mix, pair curriculum, the pattern-error model --pert_model of the input
  estimate, generic patterns in half the FOVs: simulate/batches.P_STD, fixed recipe), the illumination patches as extra input (zero-initialised
  from a pass-1 --init, so it starts exactly at that network); the detector gets the same estimated pattern when it is a
  pass-2 detector. Held-out: fixed seeded batches (make_pattern_hold): nll_I, |I - I_gt| / I_gt, prior RMSE / z per nn bin
  and photon bin, with the default model's pass-1 network on its pass-1 detector as reference (eval_ref.json).
--illum: default = 1 if --detector is a pass-2 detector, else 1 if --init is a pass-2 network, else 0. Checkpoints:
  {'model', 'args' (arch tag model.ARCH, illum), 'step'} -> <out>/ckpt_<step>.pt, <out>/final.pt.
usage: CUDA_VISIBLE_DEVICES=1 python -m simcode.emitter.train --out runs/emitter --detector runs/detector/final.pt --calib <psf.json> ...
       CUDA_VISIBLE_DEVICES=1 python -m simcode.emitter.train --out runs/emitter_pass2 --detector runs/detector_pass2/final.pt --init runs/emitter/final.pt ...
       python -m simcode.emitter.train --check 1        (zero-init check: a pass-2 network from a pass-1 one == that network)
(models/README.md: the recipe of a model.)
"""
import argparse, json, os, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from simcode.util.checkpoint import save as ckpt_save     # atomic checkpoint write with a disk guard
from simcode import config
from simcode.layers.blocks import extract_patches
from simcode.emitter.model import (EmitterNet, ARCH, PP, PSFCol, NBR_PMIN, nbr_map, nbr_patches, nll_full, nll_joint, nll_z, pad_out,
                                   mod_slope, detect_candidates, apply_to_candidates, is_pass2)
from simcode.simulate.targets import nn_weights, match_gt, emitter_targets

dev = 'cuda'
# defaults that differ between the passes (None in the parser)
DEFAULTS = {0: dict(detector=config.PRETRAIN_DETECTOR, init='', steps=100000, lr=5e-4, pct=0.05, seed=0, eval_every=2000, ckpt_every=10000,
                    tau='15,80', dens='0.0005,0.04'),
            1: dict(detector=None, init=config.model(config.EMITTER), steps=40000, lr=1.5e-4, pct=0.1, seed=21, eval_every=4000,
                    ckpt_every=10000, tau='', dens='')}


def bnoise_of(a):
    """the --bnoise flag as TrainingBatches bnoise ((c, c_a) or None)"""
    from simcode.simulate.batches import parse_bnoise
    return parse_bnoise(getattr(a, 'bnoise', ''))


@torch.no_grad()
def z_metrics(out, tx, ty, tz, ph):
    """running z metrics of a training batch (fresh simulated data): z RMSE (px) of the prior z and of z | true x, y, their
    z-score RMS, the median predicted sd; the same for window photons >= 500"""
    zc = out['z'] + out['bz'][:, 0] * (tx - out['x']) + out['bz'][:, 1] * (ty - out['y'])
    sz = out['lsz'].exp()
    var_m = sz ** 2 + (out['bz'][:, 0] * out['lsxy'][:, 0].exp()) ** 2 + (out['bz'][:, 1] * out['lsxy'][:, 1].exp()) ** 2
    r = dict(z_rmse=float((out['z'] - tz).pow(2).mean().sqrt()), zc_rmse=float((zc - tz).pow(2).mean().sqrt()),
             z_zrms=float(((out['z'] - tz) ** 2 / var_m).mean().sqrt()), zc_zrms=float(((zc - tz) / sz).pow(2).mean().sqrt()),
             z_sd_med=float(var_m.sqrt().median()), z_sd_true=float(tz.std()))
    b = ph >= 500
    if b.sum() > 10:
        r.update(z_rmse_500=float((out['z'][b] - tz[b]).pow(2).mean().sqrt()), z_zrms_500=float(((out['z'] - tz)[b] ** 2 / var_m[b]).mean().sqrt()))
    return r


def load_for_training(ck, illum, ch=96, bf16=True):
    """network to train from checkpoint ck (2D checkpoints get the zero z rows). illum: a pass-2 network; from a pass-1
    checkpoint its illumination weights are zero (it starts exactly at that network), from a pass-2 one it continues"""
    from simcode.config import resolve
    h = EmitterNet(6, ch, P=PP, bf16=bf16, illum=illum).to(dev)
    sd = pad_out(torch.load(resolve(ck), map_location=dev, weights_only=False)['model'], h)
    if illum:
        own = h.state_dict(); own.update(sd); h.load_state_dict(own)
        if 'cin_il.weight' not in sd:
            nn.init.zeros_(h.cin_il.weight)
    else:
        if is_pass2(sd): raise ValueError(f'{ck}: a pass-2 network cannot be trained as a pass-1 network')
        h.load_state_dict(sd)
    return h


def loss_terms(out, t, m, wt, tz, w_mr, w_slope):
    """the training loss (module doc) and its parts"""
    Ig, tx, ty = t['Ig'], t['tx'], t['ty']
    iso_w = (m['nn'] >= 2.0).float()
    ex = tx - out['x']; ey = ty - out['y']; ls = out['lsxy']
    zp = torch.stack([ex / ls[:, 0].exp(), ey / ls[:, 1].exp()], 1)
    nll_i, z = nll_joint(Ig, out['I'], out['L'], out['C'], zp)          # intensities given the prior error (joint covariance)
    nll_I = (wt * nll_i).mean()
    nll_p = (wt * (0.5 * (ex / ls[:, 0].exp()) ** 2 + ls[:, 0] + 0.5 * (ey / ls[:, 1].exp()) ** 2 + ls[:, 1])).mean()
    l_z = (wt * nll_z(out, tx, ty, tz)).mean()
    cn = (Ig.clamp(min=0).sum(1, keepdim=True) / 6 + 30).sqrt()
    mr = ((out['I'] - out['I'].mean(1, keepdim=True)) - (Ig - Ig.mean(1, keepdim=True))) / cn
    l_mr = (wt[:, None] * F.huber_loss(mr, torch.zeros_like(mr), delta=3.0, reduction='none')).mean()
    am = out['I'] - out['I'].mean(1, keepdim=True); gm = Ig - Ig.mean(1, keepdim=True)
    slope_b = (am * gm * iso_w[:, None]).sum() / (gm * gm * iso_w[:, None]).sum().clamp(min=1e-6)
    l_sl = (1 - slope_b) ** 2
    bgt = t['bgt_px']
    l_bg = (wt[:, None] * ((out['bg'] - bgt) ** 2) / (bgt + 1.0)).mean()
    loss = nll_I / 6 + nll_p + l_z + 0.1 * l_bg + w_mr * l_mr + w_slope * l_sl
    rec = dict(slope_iso=slope_b.item(), mr=l_mr.item(), nll_I=nll_I.item(), nll_p=nll_p.item(), bg=l_bg.item(), n=len(Ig),
               zI=float((z ** 2).mean().sqrt()), pos_rmse=float(torch.sqrt((ex ** 2 + ey ** 2).mean() / 2)),
               zp_rms=float((zp ** 2).mean().sqrt()), C_rms=float((out['C'] ** 2).mean().sqrt()))
    return loss, rec


# ------------------------------------------------------------------------------------------------ pass-2 held-out
def make_pattern_hold(gen, n=30, nb=16, seed=909, pert_model='v1'):
    H = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        for i in range(n):
            o = gen.illumination_batch(nb, mix='paper', pair_curr=True, pert_model=pert_model)
            H.append(o)
    return H


@torch.no_grad()
def eval_pattern_hold(det, head, col, H, il_on=True):
    was = head.training; head.eval()
    acc = {}
    for o in H:
        il = o['illum_input'] if il_on else torch.zeros_like(o['illum_input'])
        c, nbr, lgm = detect_candidates(det, o['frames'], il)
        c5 = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))
        m = match_gt(o, c5)
        if m is None: continue
        p_own = torch.sigmoid(lgm[m['b'], m['iy'], m['ix']])
        out = apply_to_candidates(head, col[o['calib_i']] if isinstance(col, list) else col, o['frames'], il, m['b'], m['iy'], m['ix'], m['x'], m['y'], nbr, p_own)
        G = o['gt']; g = m['g']; Ig = G['phot'][g]
        ep = torch.stack([out['x'] - G['x'][g], out['y'] - G['y'][g]], 1)
        z = ep / out['lsxy'].exp()
        nll, _ = nll_joint(Ig, out['I'].float(), out['L'].float(), out['C'].float(), -z)
        ph = Ig.sum(1); nnd = m['nn']
        eI = ((out['I'] - Ig).abs().sum(1) / Ig.sum(1).clamp(min=1))
        for nm, (a, b) in (('nn<1.5', (0, 1.5)), ('nn1.5-2.5', (1.5, 2.5)), ('nn>2.5', (2.5, 1e9))):
            for pl, phh in ((200, 700), (700, 2000), (2000, 1e9)):
                mm = (nnd >= a) & (nnd < b) & (ph >= pl) & (ph < phh)
                if not mm.any(): continue
                k = f'{nm}|{pl}-{phh:.0f}'
                r = acc.setdefault(k, dict(n=0, nll=0.0, eI=[], e2=0.0, z2=0.0))
                r['n'] += int(mm.sum()); r['nll'] += float(nll[mm].sum()); r['eI'].append(eI[mm].cpu())
                r['e2'] += float((ep[mm] ** 2).sum()); r['z2'] += float((z[mm] ** 2).sum())
    if was: head.train()
    return {k: dict(n=r['n'], nll_I=r['nll'] / r['n'], eI_med=float(torch.cat(r['eI']).median()),
                    prior_rmse_nm=float(np.sqrt(r['e2'] / (2 * r['n']))) * 108.33, prior_zrms=float(np.sqrt(r['z2'] / (2 * r['n']))))
            for k, r in sorted(acc.items())}


# ------------------------------------------------------------------------------------------------ training
def parser():
    ap = argparse.ArgumentParser(description='trainer of the emitter network (pass 1 and pass 2)')
    ap.add_argument('--out', default='')
    ap.add_argument('--illum', type=int, default=None, help='1: pass-2 network (illumination inputs); default: from --detector / --init')
    ap.add_argument('--detector', default=None, help="detector of the candidates (pass 1 default 'pretrain_detector_pass1.pt'; required for pass 2)")
    ap.add_argument('--init', default=None, help='start checkpoint (pass 1 default: none; pass 2 default: the default model\'s pass-1 network)')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--pct', type=float, default=None)          # OneCycle pct_start
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--eval_every', type=int, default=None)
    ap.add_argument('--ckpt_every', type=int, default=None)
    ap.add_argument('--w_mr', type=float, default=2.0)
    ap.add_argument('--w_slope', type=float, default=50.0)
    ap.add_argument('--ch', type=int, default=96)
    ap.add_argument('--bf16', type=int, default=1)
    # the PSF of the simulator (in-situ calibration json; a comma list = a mixture, simulate/batches.TrainingBatches calib; the
    # PSF-template column follows the batch's calib)
    ap.add_argument('--calib', default='')
    # emitter brightness noise of the simulator 'c,c_a' or 'c_lo:c_hi,c_a' (c drawn uniformly per batch) (a dataset's measured
    # values; simulate/batches.TrainingBatches bnoise);
    # '' = the presets' (pass 1) / pass-2 defaults
    ap.add_argument('--bnoise', default='')
    # training ranges 'lo,hi': ON-emitter density (/px^2) and mean on-time (frames); pass 2 '' = the batch defaults (g13: 3,80)
    ap.add_argument('--dens', default=None)
    ap.add_argument('--tau', default=None)
    # pass 1
    ap.add_argument('--hold', default=None, help='held-out patches (python -m simcode.emitter.heldout; default simcode_out/emitter_heldout/heldout4.pt if it exists, else none)')
    ap.add_argument('--pair_frac', type=float, default=0.25)
    ap.add_argument('--med', default='40,2500')
    ap.add_argument('--nnw', type=int, default=1)
    # pass 2
    ap.add_argument('--no_illum', type=int, default=0)       # control: same data / schedule, illumination inputs zeroed
    ap.add_argument('--pert_model', default='real')         # pattern-error model of the input estimate (illumination.MODELS; the
                                                            # recipe's first pass-2 stage: v1)
    ap.add_argument('--hold_v1', type=int, default=0)       # also a held-out set with the v1 error model (n batches)
    ap.add_argument('--check', type=int, default=0, help='1: zero-init check only')
    return ap


def infer_illum(a):
    """--illum default: a pass-2 detector (--detector) or a pass-2 init checkpoint -> 1"""
    from simcode.detector.model import is_pass2 as det_is_pass2
    if a.detector and det_is_pass2(torch.load(config.resolve(a.detector), map_location='cpu', weights_only=False)['model']): return 1
    if a.init and is_pass2(torch.load(config.resolve(a.init), map_location='cpu', weights_only=False)['model']): return 1
    return 0


def main(argv=None):
    a = parser().parse_args(argv)
    if a.check: return check()
    if not a.out or not a.calib: raise SystemExit('--out and --calib are required')
    if a.illum is None: a.illum = infer_illum(a)
    for k, v in DEFAULTS[a.illum].items():
        if getattr(a, k) is None: setattr(a, k, v)
    if a.illum and not a.detector: raise SystemExit('--detector is required for a pass-2 network')
    from simcode.util.proctitle import set_title; set_title('train-p2head' if a.illum else 'train-head')     # name in nvidia-smi / ps
    (train_pass2 if a.illum else train_pass1)(a)


def train_pass1(a):
    from simcode.simulate.batches import TrainingBatches, Z_RANGE_UM, Z_UNIF, BRIGHT_TAIL, z_target_px
    from simcode.emitter.heldout import HOLD, HOLD5_NAME, eval_hold
    from simcode.detector.model import load as load_det
    from simcode.detector.select import select_subpixel
    if a.hold is None and os.path.isfile(HOLD): a.hold = HOLD     # the default set if it exists; else no held-out evaluation
    args = dict(vars(a), arch=ARCH, paper=1, bright_tail=','.join(f'{v:g}' for v in BRIGHT_TAIL))
    args.update(z3d=1, z_range_um=list(Z_RANGE_UM), z_unif=Z_UNIF)
    torch.manual_seed(a.seed)
    det, _ = load_det(a.detector)
    gen = TrainingBatches(64, 64, 6, seed=a.seed, calib=[config.resolve(c) for c in a.calib.split(',')], bnoise=bnoise_of(a))
    args.update(heights=gen.heights())
    os.makedirs(a.out, exist_ok=True); json.dump(args, open(a.out + '/args.json', 'w'), indent=1)
    head = EmitterNet(6, a.ch, P=PP, bf16=bool(a.bf16)).to(dev)
    if a.init:      # a 2D init (e.g. pretrain_emitter_pass1.pt) gets the zero z rows (pad_out): it starts at exactly its 2D outputs
        head.load_state_dict(pad_out(torch.load(config.resolve(a.init), map_location=dev, weights_only=False)['model'], head))
    col = PSFCol(gen.rs['0.5nM'].sim)
    cols = [PSFCol(r['0.5nM'].sim) for r in gen.rsm] if getattr(gen, 'calibs', None) else None
    Hd = torch.load(a.hold, weights_only=False) if a.hold else None
    sc5 = bool(a.hold) and os.path.basename(a.hold) == HOLD5_NAME
    tau_r = tuple(float(v) for v in a.tau.split(','))
    dens = tuple(float(v) for v in a.dens.split(',')); med = tuple(float(v) for v in a.med.split(','))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a'); elog = open(a.out + '/eval.jsonl', 'a')
    t0 = time.time(); acc = {}; IS, GS = [], []
    for step in range(a.steps + 1):
        with torch.no_grad():
            o = gen.emitter_batch(a.batch, dens=dens, med_photons=med, pair_frac=a.pair_frac, pair_curr=True, paper=True, tau_r=tau_r,
                                  bright_tail=BRIGHT_TAIL)
            if cols is not None: col = cols[o['calib_i']]
            fr = o['frames']
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dd = det(fr)
            lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
            c = select_subpixel(lgm, dd['dxy'].float(), 640, p_min=NBR_PMIN, radius=0.7)
            kp = torch.isfinite(c['logit'])
            nbr = nbr_map(lgm, *(c[k][kp] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit')))
            c = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))   # targets p >= 0.5
            m = match_gt(o, c)
        if m is None:
            sched.step(); continue
        with torch.no_grad():
            t = emitter_targets(o, m, col)
            fov = o['gt']['fov'][m['g']]
            wt = nn_weights(fov, m['nn']) if a.nnw else torch.ones_like(m['nn'])
            off = torch.stack([m['x'] - m['ix'].float(), m['y'] - m['iy'].float()], 1)
            pt = extract_patches(fr, m['b'], m['iy'], m['ix'])
            psf = col(off)
            # own p for the stamp removal: the candidate's logit at its NMS pixel
            p_own = torch.sigmoid(lgm[m['b'], m['iy'], m['ix']])
            nbp = nbr_patches(nbr, m['b'], m['iy'], m['ix'], m['x'], m['y'], p_own)
        out = head(pt, None, off, psf, nbp)
        out['x'] = m['x'] + out['dxy'][:, 0]; out['y'] = m['y'] + out['dxy'][:, 1]
        tz = z_target_px(o['gt']['z'][m['g']], gen.rsm[o['calib_i']]['0.5nM'].sim if cols is not None else gen.rs['0.5nM'].sim)
        loss, rec = loss_terms(out, t, m, wt, tz, a.w_mr, a.w_slope)
        step_log(head, opt, sched, loss, rec, out, t, tz, acc, IS, GS)
        if step % 200 == 0:
            r = log_record(step, t0, sched, acc, IS, GS)
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if Hd is not None and step % a.eval_every == 0:
            ev = eval_hold(head, Hd, band=(200, 2000), extra=True) if sc5 else eval_hold(head, Hd)
            elog.write(json.dumps(dict(step=step, t=round(time.time() - t0), **ev)) + '\n'); elog.flush()
            print('EVAL', step, json.dumps({s: ev[s].get('all', ev[s].get('nn1.5-2.5')) for s in ev}), flush=True)
        if step % a.ckpt_every == 0 and step > 0:
            ckpt_save(dict(model=head.state_dict(), args=args, step=step), a.out + f'/ckpt_{step}.pt')
    ckpt_save(dict(model=head.state_dict(), args=args, step=a.steps), a.out + '/final.pt')


def step_log(head, opt, sched, loss, rec, out, t, tz, acc, IS, GS):
    """optimizer step (skipped for a non-finite loss) + the running log accumulation"""
    opt.zero_grad(set_to_none=True)
    if torch.isfinite(loss):
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    else:
        acc['nonfinite'] = acc.get('nonfinite', 0) + 1
    sched.step()
    with torch.no_grad():
        Ig = t['Ig']
        IS.append(out['I'].detach()); GS.append(Ig)
        for k, v in dict(rec, **z_metrics(out, t['tx'], t['ty'], tz, Ig.sum(1))).items():
            acc[k] = acc.get(k, 0) + v
        acc['_n'] = acc.get('_n', 0) + 1


def log_record(step, t0, sched, acc, IS, GS, **extra):
    nn_ = acc.pop('_n', 1)
    Ia, Ga = torch.cat(IS), torch.cat(GS); IS.clear(); GS.clear()
    r = dict(step=step, t=round(time.time() - t0), lr=sched.get_last_lr()[0], slope=mod_slope(Ia, Ga), **extra,
             **{k: v / nn_ for k, v in acc.items()})
    acc.clear()
    return r


def train_pass2(a):
    from simcode.simulate.batches import TrainingBatches, z_target_px, P_STD
    from simcode.detector.model import load as load_det
    from simcode.emitter.model import load as load_net
    a.p_std = P_STD          # share of SIMCODE-pattern FOVs (fixed recipe, models/README.md; stored in args)
    args = dict(vars(a), arch=ARCH)
    os.makedirs(a.out, exist_ok=True); json.dump(args, open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    gen = TrainingBatches(64, 64, 6, seed=a.seed, calib=[config.resolve(c) for c in a.calib.split(',')], bnoise=bnoise_of(a), p_std=a.p_std)
    psub = {}
    if a.dens: psub['dens'] = tuple(float(v) for v in a.dens.split(','))
    if a.tau: psub['tau_r'] = tuple(float(v) for v in a.tau.split(','))
    det, _ = load_det(a.detector)
    head = load_for_training(a.init, True, a.ch, bool(a.bf16))
    col = PSFCol(gen.rs['0.5nM'].sim)
    if getattr(gen, 'calibs', None): col = [PSFCol(r['0.5nM'].sim) for r in gen.rsm]    # per-calib template columns
    cols = col
    Hd = make_pattern_hold(gen, pert_model=a.pert_model)
    Hv1 = make_pattern_hold(gen, n=a.hold_v1, seed=919) if a.hold_v1 else None
    ilon = not a.no_illum
    # references: the default model's pass-1 network on its pass-1 detector's candidates (config.EMITTER / config.DETECTOR; evaluation
    # only), and the initial network on this detector
    det1, _ = load_det(config.model(config.DETECTOR))
    ref = dict(p1_ref=eval_pattern_hold(det1, load_net(config.model(config.EMITTER)), col, Hd), init=eval_pattern_hold(det, head, col, Hd, ilon))
    if Hv1: ref['init_v1'] = eval_pattern_hold(det, head, col, Hv1, ilon)
    del det1
    json.dump(ref, open(a.out + '/eval_ref.json', 'w'), indent=1)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a'); elog = open(a.out + '/eval.jsonl', 'a')
    t0 = time.time(); acc = {}; IS, GS = [], []
    for step in range(a.steps + 1):
        with torch.no_grad():
            o = gen.illumination_batch(a.batch, mix='paper', pair_curr=True, pert_model=a.pert_model, **psub)
            col = cols[o['calib_i']] if isinstance(cols, list) else cols
            fr = o['frames']; il = o['illum_input'] if ilon else torch.zeros_like(o['illum_input'])
            c, nbr, lgm = detect_candidates(det, fr, il)
            c5 = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))
            m = match_gt(o, c5)
        if m is None:
            sched.step(); continue
        with torch.no_grad():
            t = emitter_targets(o, m, col)
            wt = nn_weights(o['gt']['fov'][m['g']], m['nn'])
            p_own = torch.sigmoid(lgm[m['b'], m['iy'], m['ix']])
        out = apply_to_candidates(head, col, fr, il, m['b'], m['iy'], m['ix'], m['x'], m['y'], nbr, p_own)
        tz = z_target_px(o['gt']['z'][m['g']], gen.rsm[o['calib_i']]['0.5nM'].sim if getattr(gen, 'calibs', None) else gen.rs['0.5nM'].sim)
        loss, rec = loss_terms(out, t, m, wt, tz, a.w_mr, a.w_slope)
        step_log(head, opt, sched, loss, rec, out, t, tz, acc, IS, GS)
        if step % 200 == 0:
            r = log_record(step, t0, sched, acc, IS, GS, w_il=float(head.cin_il.weight.abs().mean()))
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if step % a.eval_every == 0 and step > 0:
            ev = eval_pattern_hold(det, head, cols, Hd, ilon)
            if Hv1: ev['v1'] = eval_pattern_hold(det, head, cols, Hv1, ilon)
            elog.write(json.dumps(dict(step=step, t=round(time.time() - t0), **ev)) + '\n'); elog.flush()
            print('EVAL', step, json.dumps(ev), flush=True)
        if step % a.ckpt_every == 0 and step > 0:
            ckpt_save(dict(model=head.state_dict(), args=args, step=step), a.out + f'/ckpt_{step}.pt')
    ckpt_save(dict(model=head.state_dict(), args=args, step=a.steps), a.out + '/final.pt')


def check():
    """zero-init check: a pass-2 network from the default model's pass-1 network == that network (pass-1 detector candidates)"""
    from simcode.simulate.batches import TrainingBatches, P_STD
    from simcode.detector.model import load as load_det
    from simcode.emitter.model import load as load_net
    torch.manual_seed(0)
    gen = TrainingBatches(64, 64, 6, seed=0, p_std=P_STD)
    o = gen.illumination_batch(4, pair_curr=True)
    det, _ = load_det(config.model(config.DETECTOR))
    col = PSFCol(gen.rs['0.5nM'].sim)
    c, nbr, lgm = detect_candidates(det, o['frames'], o['illum_input'])
    p = torch.sigmoid(c['logit'])
    h0 = load_net(config.model(config.EMITTER)); h1 = load_for_training(config.model(config.EMITTER), True).eval()
    with torch.no_grad():
        a = apply_to_candidates(h0, col, o['frames'], o['illum_input'], c['b'], c['iy'], c['ix'], c['x'], c['y'], nbr, p)
        b = apply_to_candidates(h1, col, o['frames'], o['illum_input'], c['b'], c['iy'], c['ix'], c['x'], c['y'], nbr, p)
    print(len(p), 'cands; max |dI|', float((a['I'] - b['I']).abs().max()), 'max |dx|', float((a['x'] - b['x']).abs().max()))


if __name__ == '__main__':
    main()
