"""Train a pattern-free detector variant (simcode2/det/model.py) on simulated windows (simcode2/det/data.py).
Loss: focal BCE on the per-pixel logits (ignore mask) + Gaussian NLL of the sub-pixel offset on the positives.
The production detector det_n1 is trained with simcode2/det/up2.py (--up 1); this trainer is the earlier pattern-free
detector of the det_stem lineage (DetNet input variants, optional PSF-perturbation augmentation), kept for ablations.
usage: python -m simcode2.det.train --inp frames_sum --out runs/det_frames_sum --steps 15000
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
from simcode2 import config
from simcode2.sim.simulator import Simulator
from simcode2.det import data as D
from simcode2.det.model import DetNet, select_subpixel, n_params

ap = argparse.ArgumentParser()
ap.add_argument('--calib', default=config.INSITU_05)
ap.add_argument('--out', required=True)
ap.add_argument('--inp', default='frames_sum')
ap.add_argument('--ch', default='48,96,160')
ap.add_argument('--stem_ch', type=int, default=16)
ap.add_argument('--steps', type=int, default=15000)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--size', type=int, default=64)
ap.add_argument('--T', type=int, default=6)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--pair_frac', type=float, default=0.25)
ap.add_argument('--w_off', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--init', default='')                   # fine-tune from this checkpoint
ap.add_argument('--pert', type=int, default=0)          # PSF-perturbation augmentation (simcode2/det/psfpert.py)
ap.add_argument('--pert_w', default='0.9,1.35')         # PSF width factor range (log-uniform)
ap.add_argument('--pert_focus', type=float, default=0.2)   # +- focus shift (um)
ap.add_argument('--pert_zmax', default='0.15,0.5')      # emitter z range upper bound (um), uniform per LUT
ap.add_argument('--pert_zern', type=float, default=0.025)  # Zernike jitter sd (waves rms)
ap.add_argument('--pert_nominal', type=float, default=0.25)  # fraction of batches with the calibrated PSF
ap.add_argument('--pert_every', type=int, default=4)    # new random PSF every n steps
ap.add_argument('--pert_gain', type=float, default=0.1)
ap.add_argument('--pert_offset', type=float, default=2.0)
ap.add_argument('--carpet', type=float, default=0.0)    # max density of a dim flickering emitter carpet (data.emitters)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
args = vars(a); args['ch'] = [int(v) for v in a.ch.split(',')]
json.dump(args, open(a.out + '/args.json', 'w'), indent=1)
dev = 'cuda'
torch.manual_seed(a.seed)
sim = Simulator(a.calib, H=a.size, W=a.size, T=a.T, device=dev)
net = DetNet(a.T, a.inp, tuple(args['ch']), a.stem_ch).to(dev)
if a.init:
    net.load_state_dict(torch.load(config.resolve(a.init), map_location=dev, weights_only=False)['model'])
print('params', n_params(net), flush=True)
if a.pert:
    import math as _m
    from simcode2.det.psfpert import build_lut, camera_error
    base_lut, base_zmax = sim.lut, sim.zmax
    w_lo, w_hi = (float(v) for v in a.pert_w.split(',')); z_lo, z_hi = (float(v) for v in a.pert_zmax.split(','))
    nz = len(sim.c['zernike_waves'])
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=0.05)
log = open(a.out + '/log.jsonl', 'a')


def focal(logits, label, ign, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
    pt = p * label + (1 - p) * (1 - label)
    w = (alpha * label + (1 - alpha) * (1 - label)) * (~ign).float()
    return (w * (1 - pt) ** gamma * ce).sum() / label.sum().clamp(min=1)


t0 = time.time()
for step in range(a.steps + 1):
    with torch.no_grad():
        if a.pert and step % a.pert_every == 0:
            if torch.rand(1).item() < a.pert_nominal:
                sim.lut, sim.zmax = base_lut, base_zmax
            else:
                wi = _m.exp(_m.log(w_lo) + (_m.log(w_hi) - _m.log(w_lo)) * torch.rand(1).item())
                zm = z_lo + (z_hi - z_lo) * torch.rand(1).item()
                sim.lut, sim.zmax = build_lut(sim, wi, torch.randn(nz, device=dev) * a.pert_zern,
                                              (torch.rand(1).item() - 0.5) * 2 * a.pert_focus, zm)
        o = D.sample(sim, a.batch, pair_frac=a.pair_frac, carpet=a.carpet)
        if a.pert:
            o = camera_error(o, sim, a.pert_gain, a.pert_offset)
        label, ign, tid, off = D.assign(o['gt'], a.batch, a.size, a.size, dontcare=o.get('carpet'))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = net(o['frames'])
    lg = out['logit'].float(); dxy = out['dxy'].float(); ls = out['log_sig'].float()
    l_det = focal(lg, label, ign)
    pm = label > 0
    if pm.any():
        e = (dxy - off).permute(0, 2, 3, 1)[pm]                       # [n,2]
        s = ls[pm][:, None]
        l_off = (0.5 * (e / s.exp()) ** 2 + s).sum(1).mean()
    else:
        l_off = torch.zeros((), device=dev)
    loss = l_det + a.w_off * l_off
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 250 == 0:
        with torch.no_grad():
            c = select_subpixel(lg, dxy, 320, p_min=0.5)
            ok = torch.isfinite(c['logit'])
            n_det = int(ok.sum()); n_pos = int(pm.sum())
            rm = float(e.pow(2).mean().sqrt()) if pm.any() else float('nan')
            # quick pixel-level recall: positives whose own pixel gets p > 0.5
            rec = float((torch.sigmoid(lg)[pm] > 0.5).float().mean()) if pm.any() else float('nan')
        r = dict(step=step, t=round(time.time() - t0), loss=loss.item(), det=l_det.item(), off=l_off.item(),
                 pix_recall=rec, n_det=n_det, n_pos=n_pos, off_rmse_px=rm, lr=sched.get_last_lr()[0])
        log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
    if step % 5000 == 0 and step > 0:
        torch.save(dict(model=net.state_dict(), args=args, step=step), a.out + f'/ckpt_{step}.pt')
torch.save(dict(model=net.state_dict(), args=args, step=a.steps), a.out + '/final.pt')
