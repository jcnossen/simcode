"""
Trainer of the detector (detector/model.Detector), pass 1 and pass 2 (one trainer; the pass follows from the checkpoint,
see --illum).

Targets (simulate/targets.detector_targets) on the native grid: positives need >= 1 on-frame and >= 60 window photons, others
'don't care'; a colliding emitter takes the nearest free 8-neighbour cell. Loss: focal BCE (focal_w) with per-positive nn-bin
weights (simulate/targets.nn_weights: < 1.5 / 1.5-2.5 / 2.5-4 / > 4 px carry 0.15 / 0.25 / 0.25 / 0.35 of each window's
positive weight; nn over all active GT emitters) + the per-cell Gaussian NLL of the offset (same weights), x --w_off.
Pass 1 (illum 0): TrainingBatches.detector_batch (the acquisition simulator: presets 0.5 / 2.5 nM per batch, density
  log-uniform 0.0005-0.04 /px, per-window median per-frame brightness log-uniform 40-2500 / per-emitter EM_PHOTONS, pairs
  0.3-3.0 px). --nnw 0: unit weights.
Pass 2 (illum 1): the pass-1 detector + illumination inputs, zero-initialised from a pass-1 --init (starts exactly at it) or
  continued from a pass-2 --init; batches alternate TrainingBatches.illumination_batch mix 'paper' (label-unit brightness +
  bright tail) and 'det' (the pass-1 photon mix), pairs 0.3-3.0 px, perturbed pattern estimates as input. Pattern draw (fixed
  recipe, simulate/batches.P_STD_DEDICATED / P_STD): from a pass-1 --init the SIMCODE pattern only (the dedicated pass-2
  detector), from a pass-2 --init (fine-tuning) generic patterns in half the FOVs. Held-out: fixed seeded batches (make_pattern_hold), recall at p >= 0.5 per nn bin and FP per
  window, with the pass-1 detector --pass1_ref on the same batches as reference (eval_ref.json).
--illum: default = that of --init (0 without --init). Pass 2 without --init starts from the default model's pass-1 detector.
Checkpoints {'model', 'args' (arch = model.ARCH, ch, stem_ch, up = 1, illum), 'step'} -> <out>/ckpt_<step>.pt, <out>/final.pt.
usage: CUDA_VISIBLE_DEVICES=1 python -m simcode.detector.train --out runs/det --calib <psf.json> ...
       CUDA_VISIBLE_DEVICES=1 python -m simcode.detector.train --out runs/p2det --illum 1 --init runs/det/final.pt --calib <psf.json> ...
       python -m simcode.detector.train --check 1          (zero-init check: a pass-2 detector from a pass-1 one == that detector)
       python -m simcode.detector.train --stats 40         (pass-1 training-mix statistics: target photons, nn bins)
(models/README.md: the recipe of a model.)
"""
import argparse, json, os, time
import numpy as np, torch
import torch.nn.functional as F
from simcode.util.ckpt import save as ckpt_save     # atomic checkpoint write with a disk guard
from simcode import config
from simcode.detector.model import ARCH, Detector, build, is_pass2, n_params
from simcode.detector.select import select_points
from simcode.simulate.targets import detector_targets, gt_nn, nn_weights

dev = 'cuda'
# defaults that differ between the passes (None in the parser)
DEFAULTS = {0: dict(lr=1e-3, seed=0), 1: dict(lr=3e-4, seed=11)}


def focal_w(logits, label, ign, wpos, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
    pt = p * label + (1 - p) * (1 - label)
    w = (alpha * label * wpos + (1 - alpha) * (1 - label)) * (~ign).float()
    return (w * (1 - pt) ** gamma * ce).sum() / label.sum().clamp(min=1)


def load_for_training(ck, illum):
    """detector to train from checkpoint ck: illum = a pass-2 detector (from a pass-1 checkpoint: the illumination weights
    zero, it starts exactly at that detector; from a pass-2 one: continued)"""
    c = torch.load(config.resolve(ck), map_location=dev, weights_only=False)
    p2 = is_pass2(c['model'])
    if p2 and not illum: raise ValueError(f'{ck}: a pass-2 detector cannot be trained as a pass-1 detector')
    if illum:
        net = Detector(illum=True).to(dev)
        if p2: net.load_state_dict(c['model'])
        else: net.load_pass1(c['model'])
        return net
    net = build(c.get('args')).to(dev); net.load_state_dict(c['model'])
    return net


# ------------------------------------------------------------------------------------------------ pass-2 held-out
def make_pattern_hold(gen, n=24, nb=16, seed=777):
    H = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        for i in range(n):
            o = gen.illumination_batch(nb, mix='paper', pair_sep=(0.3, 3.0))
            H.append(dict(frames=o['frames'], illum=o['illum_input'], gt={k: v for k, v in o['gt'].items()}, nb=nb))
    return H


@torch.no_grad()
def eval_pattern_hold(net, H, thr=0.5):
    """recall (1 px, one-to-one, targets >= 1 on-frame, >= 200 window photons, interior) per nn bin x photon bin, FP / window"""
    from simcode.benchmark.gate_report import match
    was = net.training; net.eval()
    rec = {}; nfp = 0; nw = 0
    for h in H:
        fr = h['frames']; S = fr.shape[-1]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(fr, h['illum']) if net.illum else net(fr)
        c = select_points(out, k=640, p_min=thr)
        keep = torch.isfinite(c['logit'])
        b, x, y = c['b'][keep].cpu().numpy(), c['x'][keep].cpu().numpy(), c['y'][keep].cpu().numpy()
        G = h['gt']; act = G['onf'].sum(1) > 0
        gx, gy, gf = G['x'][act].cpu().numpy(), G['y'][act].cpu().numpy(), G['fov'][act].cpu().numpy()
        ph = G['phot'][act].sum(1).cpu().numpy()
        nnd = gt_nn({k: v[act] for k, v in G.items()}).cpu().numpy()
        og = match(gf, gx, gy, b, x, y, 1.0)
        hit = np.zeros(len(gx), bool); hit[og[og >= 0]] = True
        inter = (gx >= 4) & (gx < S - 4) & (gy >= 4) & (gy < S - 4)
        tgt = inter & (ph >= 200)
        oin = (x >= 4) & (x < S - 4) & (y >= 4) & (y < S - 4)
        fp = (og < 0) & oin
        nfp += int(fp.sum()); nw += h['nb']
        for nm, (a, bb) in (('nn<1', (0, 1)), ('nn1-1.5', (1, 1.5)), ('nn1.5-2.5', (1.5, 2.5)), ('nn>2.5', (2.5, 1e9))):
            for pl, phh in ((200, 700), (700, 2000), (2000, 1e9)):
                m = tgt & (nnd >= a) & (nnd < bb) & (ph >= pl) & (ph < phh)
                k = f'{nm}|{pl}-{phh:.0f}'
                r = rec.setdefault(k, [0, 0]); r[0] += int(hit[m].sum()); r[1] += int(m.sum())
    if was: net.train()
    return dict(recall={k: (v[0] / max(v[1], 1), v[1]) for k, v in rec.items()}, fp_per_win=nfp / max(nw, 1))


# ------------------------------------------------------------------------------------------------ training
def parser():
    ap = argparse.ArgumentParser(description='trainer of the detector (pass 1 and pass 2)')
    ap.add_argument('--out', default='')
    ap.add_argument('--illum', type=int, default=None, help='1: pass-2 detector (illumination inputs); default: that of --init')
    ap.add_argument('--init', default='', help='start checkpoint (pass-1 or pass-2 detector; pass 2 default: the default model\'s detector)')
    ap.add_argument('--pass1_ref', default='', help='pass 2: pass-1 detector of the held-out reference (default: --init if it is a '
                    'pass-1 detector, else the default model\'s detector)')
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--w_off', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--pct', type=float, default=0.05)
    ap.add_argument('--ckpt_every', type=int, default=5000)
    # the PSF of the simulator (in-situ calibration json; a comma list = a mixture, simulate/batches.TrainingBatches calib)
    ap.add_argument('--calib', default='')
    # ('' = the batch defaults) training ranges 'lo,hi': ON-emitter density (/px^2); pass 2 also the mean on-time (frames)
    ap.add_argument('--dens', default='')
    ap.add_argument('--tau', default='')
    # pass 1
    ap.add_argument('--nnw', type=int, default=1)
    # pass 2
    ap.add_argument('--eval_every', type=int, default=2500)
    ap.add_argument('--no_illum', type=int, default=0)     # control: same data / schedule, illumination inputs zeroed
    ap.add_argument('--check', type=int, default=0, help='1: zero-init check only')
    ap.add_argument('--stats', type=int, default=0, help='n > 0: training-mix statistics of n pass-1 batches only')
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.check: return check()
    if a.stats: return stats(a.stats)
    if not a.out or not a.calib: raise SystemExit('--out and --calib are required')
    if a.illum is None:
        a.illum = int(bool(a.init) and is_pass2(torch.load(config.resolve(a.init), map_location='cpu', weights_only=False)['model']))
    for k, v in DEFAULTS[a.illum].items():
        if getattr(a, k) is None: setattr(a, k, v)
    from simcode.util.proctitle import set_title; set_title('train-p2det' if a.illum else 'train-det')     # name in nvidia-smi / ps
    (train_pass2 if a.illum else train_pass1)(a)


def weights(o, label, tid, nnw=True):
    """per-positive nn-bin weights (unit weights with nnw False) and their map"""
    pm = label > 0
    w = torch.ones(int(pm.sum()), device=dev)
    if nnw and pm.any():
        gi = tid[pm]
        w = nn_weights(o['gt']['fov'][gi], gt_nn(o['gt'])[gi])
    wmap = torch.zeros_like(label); wmap[pm] = w
    return pm, w, wmap


def loss_of(out, label, ign, off, pm, w, wmap, w_off):
    lg = out['logit'].float(); dxy = out['dxy'].float(); ls = out['log_sig'].float()
    l_det = focal_w(lg, label, ign, wmap)
    e = None
    if pm.any():
        e = (dxy - off).permute(0, 2, 3, 1)[pm]
        s = ls[pm][:, None]
        l_off = ((0.5 * (e / s.exp()) ** 2 + s).sum(1) * w).sum() / w.sum().clamp(min=1e-6)
    else:
        l_off = torch.zeros((), device=dev)
    return l_det + w_off * l_off, l_det, l_off, e, lg


def train_pass1(a):
    from simcode.simulate.batches import TrainingBatches, EM_PHOTONS
    os.makedirs(a.out, exist_ok=True)
    args = vars(a); args.update(arch=ARCH, up=1, ch=[48, 96, 160], stem_ch=16, fch=32)
    json.dump(args, open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    gen = TrainingBatches(a.size, a.size, 6, seed=a.seed, calib=[config.resolve(c) for c in a.calib.split(',')])
    net = build(args).to(dev)
    if a.init:
        net.load_state_dict(torch.load(config.resolve(a.init), map_location=dev, weights_only=False)['model'])
    print('params', n_params(net), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a')
    t0 = time.time(); S = a.size
    for step in range(a.steps + 1):
        with torch.no_grad():
            o = gen.detector_batch(a.batch, em_photons=EM_PHOTONS, **({'dens': tuple(float(v) for v in a.dens.split(','))} if a.dens else {}))
            label, ign, tid, off = detector_targets(o, 1, S, S)
            pm, w, wmap = weights(o, label, tid, a.nnw)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(o['frames'])
        loss, l_det, l_off, e, lg = loss_of(out, label, ign, off, pm, w, wmap, a.w_off)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 250 == 0:
            with torch.no_grad():
                rm = float(e.pow(2).mean().sqrt()) * 1.0 if pm.any() else float('nan')
                rec = float((torch.sigmoid(lg)[pm] > 0.5).float().mean()) if pm.any() else float('nan')
            r = dict(step=step, t=round(time.time() - t0), loss=loss.item(), det=l_det.item(), off=l_off.item(),
                     cell_recall=rec, n_pos=int(pm.sum()), off_rmse_px=rm, lr=sched.get_last_lr()[0],
                     mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if step % a.ckpt_every == 0 and step > 0:
            ckpt_save(dict(model=net.state_dict(), args=args, step=step), a.out + f'/ckpt_{step}.pt')
    ckpt_save(dict(model=net.state_dict(), args=args, step=a.steps), a.out + '/final.pt')


def train_pass2(a):
    from simcode.simulate.batches import TrainingBatches, P_STD, P_STD_DEDICATED
    from simcode.detector.model import load as load_det
    init = a.init or config.DETECTOR
    cont = is_pass2(torch.load(config.resolve(init), map_location='cpu', weights_only=False)['model'])     # fine-tuning a pass-2 detector
    if not a.pass1_ref:
        a.pass1_ref = config.DETECTOR if cont else init
    # share of SIMCODE-pattern FOVs (fixed recipe, models/README.md): dedicated first stage, then the 50 / 50 mix (stored in args)
    a.p_std = P_STD if cont else P_STD_DEDICATED
    os.makedirs(a.out, exist_ok=True)
    args = vars(a); args.update(arch=ARCH, up=1, ch=[48, 96, 160], stem_ch=16)
    json.dump(args, open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    gen = TrainingBatches(a.size, a.size, 6, seed=a.seed, calib=[config.resolve(c) for c in a.calib.split(',')], p_std=a.p_std)
    psub = {}
    if a.dens: psub['dens'] = tuple(float(v) for v in a.dens.split(','))
    if a.tau: psub['tau_r'] = tuple(float(v) for v in a.tau.split(','))
    net = load_for_training(init, True)
    print('params', n_params(net), flush=True)
    Hd = make_pattern_hold(gen)
    if a.no_illum:
        for h in Hd: h['illum'] = torch.zeros_like(h['illum'])
    n1, _ = load_det(a.pass1_ref)
    ref = eval_pattern_hold(n1, Hd); del n1
    ev0 = eval_pattern_hold(net, Hd)
    json.dump(dict(pass1_ref=ref, pass2_init=ev0), open(a.out + '/eval_ref.json', 'w'), indent=1)
    print('REF pass-1 detector', json.dumps(ref), '\nINIT pass 2', json.dumps(ev0), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a'); elog = open(a.out + '/eval.jsonl', 'a')
    t0 = time.time(); S = a.size
    for step in range(a.steps + 1):
        with torch.no_grad():
            mix = 'paper' if step % 2 == 0 else 'det'
            o = gen.illumination_batch(a.batch, mix=mix, pair_sep=(0.3, 3.0), **psub)
            label, ign, tid, off = detector_targets(o, 1, S, S)
            pm, w, wmap = weights(o, label, tid)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(o['frames'], o['illum_input'] * (0.0 if a.no_illum else 1.0))
        loss, l_det, l_off, e, lg = loss_of(out, label, ign, off, pm, w, wmap, a.w_off)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        sched.step()
        if step % 250 == 0:
            r = dict(step=step, t=round(time.time() - t0), loss=loss.item(), det=l_det.item(), off=l_off.item(), n_pos=int(pm.sum()),
                     lr=sched.get_last_lr()[0], mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                     w_il=float(net.cin_il.weight.abs().mean()), w_st=float(net.stem[0].weight[:, 1:].abs().mean()))
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if step % a.eval_every == 0 and step > 0:
            ev = eval_pattern_hold(net, Hd)
            elog.write(json.dumps(dict(step=step, t=round(time.time() - t0), **ev)) + '\n'); elog.flush()
            print('EVAL', step, json.dumps(ev), flush=True)
        if step % a.ckpt_every == 0 and step > 0:
            ckpt_save(dict(model=net.state_dict(), args=args, step=step), a.out + f'/ckpt_{step}.pt')
    ckpt_save(dict(model=net.state_dict(), args=args, step=a.steps), a.out + '/final.pt')


def stats(n):
    """pass-1 training-mix statistics: target photon fraction in 500-5k, nn-bin fractions"""
    from simcode.simulate.batches import TrainingBatches, EM_PHOTONS
    torch.manual_seed(7)
    gen = TrainingBatches(64, 64, 6, seed=0)
    P, NN = [], []
    for i in range(n):
        o = gen.detector_batch(16, em_photons=EM_PHOTONS)
        label, ign, tid, off = detector_targets(o, 1, 64, 64)
        gi = tid[label > 0]
        P.append(o['gt']['phot'].sum(1)[gi]); NN.append(gt_nn(o['gt'])[gi])
    P = torch.cat(P); NN = torch.cat(NN)
    print('targets', len(P), 'frac 500-5k', float(((P >= 500) & (P < 5000)).float().mean()),
          '<500', float((P < 500).float().mean()), '>5k', float((P >= 5000).float().mean()))
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 4), (4, 1e9)):
        print(f'nn {lo}-{hi}', float(((NN >= lo) & (NN < hi)).float().mean()))


def check():
    """zero-init check: a pass-2 detector from the default model's pass-1 detector == that detector on a realistic batch"""
    from simcode.simulate.batches import TrainingBatches, P_STD
    from simcode.detector.model import load as load_det
    torch.manual_seed(0)
    gen = TrainingBatches(64, 64, 6, seed=0, p_std=P_STD)
    o = gen.illumination_batch(4)
    n1, _ = load_det(config.DETECTOR); p2 = load_for_training(config.DETECTOR, True).eval()
    with torch.no_grad():
        a = n1(o['frames']); b = p2(o['frames'], o['illum_input'])
    print('max |dlogit|', float((a['logit'] - b['logit']).abs().max()), 'max |ddxy|', float((a['dxy'] - b['dxy']).abs().max()))
    print('illum maps', o['illum_input'].shape, float(o['illum_input'][:, :6].std()), float(o['illum_input'][:, 6:].std()))
    print('patt err phase', float((o['patt_input'][..., 2] - o['patt'][..., 2]).std()), 'depth', float((o['patt_input'][..., 3] - o['patt'][..., 3]).mean()))


if __name__ == '__main__':
    main()
