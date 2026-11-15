"""
det_up2 / det_n1 (2026-11-09): pattern-free detector with an optional 2x-upsampled output grid, trained on the
experiment-matched simulator (simcode/hd/realsim.py via a copy of hd/data.Gen.batch) with the critic-29 mix.

Model (DetUp): the det_stem backbone (simcode/det/model.DetNet, inp 'stem', ch 48/96/160) unchanged; output head
  up = 1 : the DetNet head (native 108 nm grid) -> det_n1 (native control on the same data)
  up = 2 : last decoder features (48 ch, native) -> conv 3x3 48 -> 4*32 + PixelShuffle(2) -> 32 ch on the 2H x 2W grid
           (54 nm cells) -> concat a bilinear 2x copy of the decoder features (48 ch) -> conv 3x3 + GELU -> 1x1 -> 4 maps
  per cell: logit, offset (dx, dy) in NATIVE px relative to the cell centre (tanh range 1.5 / up), log-sigma (native px).
  Fine cell j (per axis) covers native coordinates [j/up - 0.5, (j+1)/up - 0.5): centre c_j = (j + 0.5)/up - 0.5.
Selection (select): sub-pixel greedy NMS (model.select_subpixel) on the cell grid with radius given in NATIVE px
  (default 0.7 for up 1, 0.4 for up 2), window win cells; returns native-px points; iy / ix = the native pixel
  containing the point (round), so the downstream SCHead patch gather is unchanged.
Targets: model.data.assign on the cell grid (GT coordinates mapped to cell units; a colliding emitter takes the nearest
  free 8-neighbour cell); positives need >= 1 on-frame and >= 60 window photons, others 'don't care'.
Loss: focal BCE (as det.train) with per-positive nn-bin weights (sc4.nn_weights: < 1.5 / 1.5-2.5 / 2.5-4 / > 4 px carry
  0.15 / 0.25 / 0.25 / 0.35 of each window's positive weight; nn over all active GT emitters) + the same per-cell
  Gaussian NLL offset loss as det_stem (weighted by the same nn weights), w_off 1.
Data (batch): realsim presets 0.5 / 2.5 nM per batch (as hd/data.Gen), density log-uniform 0.0005-0.04 /px, per-window
  median per-frame brightness log-uniform 40-2500 (half the windows) / per-emitter log-uniform em_photons (other half),
  pair_frac 0.25 with pair separations uniform 0.3-3.0 px (random direction), bg / bgmode / depth as Gen.batch.
usage: CUDA_VISIBLE_DEVICES=2 python -m simcode.det.up2 train --up 2 --out runs/det_up2
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode.util.ckpt import save as ckpt_save     # atomic checkpoint write with a disk guard
from simcode.det.model import DetNet, select_subpixel, n_params

dev = 'cuda'


class DetUp(DetNet):
    def __init__(self, up=2, T=6, ch=(48, 96, 160), stem_ch=16, fch=32):
        super().__init__(T, 'stem', ch, stem_ch, dxy_range=1.5 / up)
        self.up = up
        c0 = ch[0]
        if up > 1:
            del self.head
            self.shuf = nn.Sequential(nn.Conv2d(c0, fch * up * up, 3, padding=1), nn.PixelShuffle(up))
            self.fhead = nn.Sequential(nn.Conv2d(fch + c0, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 4, 1))
            nn.init.constant_(self.fhead[-1].bias, 0.0)
            with torch.no_grad():
                self.fhead[-1].bias[0] = -5.0

    def features(self, frames):
        x = F.gelu(self.cin(self.make_input(frames)))
        s0 = self.e0(x)
        s1 = self.e1(self.d1(F.gelu(s0)))
        x = self.e2(self.d2(F.gelu(s1)))
        x = F.interpolate(self.u1(F.gelu(x)), scale_factor=2, mode='bilinear', align_corners=False) + s1
        x = self.r1(x)
        x = F.interpolate(self.u0(F.gelu(x)), scale_factor=2, mode='bilinear', align_corners=False) + s0
        return self.r0(x)

    def forward(self, frames):
        if self.up == 1:
            return super().forward(frames)
        B, T, H, W = frames.shape
        ph, pw = (-H) % 4, (-W) % 4
        if ph or pw:
            frames = F.pad(frames, (0, pw, 0, ph), mode='replicate')
        x = F.gelu(self.features(frames))
        u = self.up
        f = torch.cat([F.gelu(self.shuf(x)), F.interpolate(x, scale_factor=u, mode='bilinear', align_corners=False)], 1)
        o = self.fhead(f)[..., :H * u, :W * u]
        return dict(logit=o[:, 0], dxy=torch.tanh(o[:, 1:3]) * self.dxy_range,
                    log_sig=-5.0 + 6.0 * torch.sigmoid(o[:, 3]))


@torch.no_grad()
def select(out, up, k=640, p_min=0.2, radius=None, win=3, H=None, W=None):
    """-> dict b, iy, ix (native pixel of the point), x, y (native px), logit, iyf, ixf (cell); entries with
    logit -inf are below p_min."""
    if radius is None: radius = 0.7 if up == 1 else 0.4
    lg = out['logit'].float(); dxy = out['dxy'].float() * up           # offsets in cell units for the NMS
    c = select_subpixel(lg, dxy, k, p_min=p_min, radius=radius * up, win=win)
    x = (c['x'] + 0.5) / up - 0.5; y = (c['y'] + 0.5) / up - 0.5
    Hn = lg.shape[1] // up if H is None else H; Wn = lg.shape[2] // up if W is None else W
    ix = torch.round(x).long().clamp(0, Wn - 1); iy = torch.round(y).long().clamp(0, Hn - 1)
    if up == 1:
        ix, iy = c['ix'], c['iy']        # identical to the old pipeline (NMS pixel)
    return dict(b=c['b'], iy=iy, ix=ix, x=x, y=y, logit=c['logit'], iyf=c['iy'], ixf=c['ix'])


def build(args):
    net = DetUp(args.get('up', 2), 6, tuple(args.get('ch', (48, 96, 160))), args.get('stem_ch', 16), args.get('fch', 32))
    return net


def load(ck, dev='cuda'):
    from simcode.config import resolve
    c = torch.load(resolve(ck), map_location=dev, weights_only=False)
    net = build(c['args']).to(dev); net.load_state_dict(c['model']); net.eval()
    return net, c['args']


# ------------------------------------------------------------------------------------------------ data
EM_PHOTONS = (40.0, 2500.0)     # per-emitter photons of the per-emitter half of the windows (the trainer's flag value of every model)


def batch(gen, nb, dens=(0.0005, 0.04), photons=(40.0, 6000.0), med_photons=(40.0, 2500.0), em_photons=None,
          pair_frac=0.25, pair_sep=(0.3, 3.0), bgr=(1.0, 30.0)):
    """copy of simcode/hd/data.Gen.batch with a pair_sep argument (critic-29 mix)"""
    from simcode.hd.realsim import lu
    pre = '0.5nM' if torch.rand(1).item() < 0.5 else '2.5nM'
    rs = gen.pick(pre)
    g = lambda *s: torch.rand(*s, device=dev)
    d = lu(*dens, nb); tau = lu(2.0, 80.0, nb)
    med = lu(*med_photons, nb)
    em = rs.emitters(nb, dens=d, tau=tau, bright=med, pair_frac=pair_frac, pair_sep=pair_sep)
    N = len(em['fov'])
    per_em = (g(nb) < 0.5)[em['fov']]
    b_em = lu(*(em_photons or photons), N) * torch.exp(-(em['z'] - rs.p['z_med']) / rs.sim.tirf)
    em['bright'] = torch.where(per_em, b_em, em['bright']).clamp(photons[0] * 0.3, photons[1] * 3)
    u = torch.rand(1).item()
    mode = 'flat' if u < 0.2 else ('struct' if u < 0.5 else 'fluct')
    b0 = lu(*bgr, nb)
    sdf = 0.4 * g(nb)
    dep = 0.5 + 0.5 * torch.rand(1).item()
    save = dict(rs.p)
    rs.p['depth'] = (dep, min(1.0, dep + 0.03))
    try:
        o = rs.scene(nb, em=em, bgmode=mode, bg=b0, bg_sd=sdf * b0, bg_hp=0.05 * g(nb) * b0,
                     fluct=1.5 * torch.rand(1).item() * (b0 / 11.7).sqrt())
    finally:
        rs.p.clear(); rs.p.update(save)
    o['preset'] = pre
    return o


def gt_nn(gt):
    """nearest-neighbour distance (px) among the active GT emitters of the same window"""
    act = gt['onf'].sum(1) > 0
    x, y, f = gt['x'], gt['y'], gt['fov']
    nn_ = torch.full_like(x, 1e3)
    D = (x[:, None] - x[None]) ** 2 + (y[:, None] - y[None]) ** 2
    ok = (f[:, None] == f[None]) & act[None] & ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    D = torch.where(ok, D, torch.full_like(D, 1e6))
    if len(x): nn_ = D.min(1).values.sqrt()
    return nn_


def targets(o, up, H, W):
    from simcode.det import data as D
    gt = o['gt']
    g2 = dict(gt, x=(gt['x'] + 0.5) * up - 0.5, y=(gt['y'] + 0.5) * up - 0.5)
    label, ign, tid, off = D.assign(g2, o['nb'], H * up, W * up)
    return label, ign, tid, off / up                                    # offsets in native px


def focal_w(logits, label, ign, wpos, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
    pt = p * label + (1 - p) * (1 - label)
    w = (alpha * label * wpos + (1 - alpha) * (1 - label)) * (~ign).float()
    return (w * (1 - pt) ** gamma * ce).sum() / label.sum().clamp(min=1)


def _cfg_resolve(c):
    from simcode.config import resolve
    return resolve(c)


def train(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--w_off', type=float, default=1.0)
    ap.add_argument('--nnw', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', default='')
    ap.add_argument('--pct', type=float, default=0.05)
    ap.add_argument('--ckpt_every', type=int, default=5000)
    # the PSF of the simulator (in-situ calibration json; a comma list = a mixture, hd/data.Gen calib)
    ap.add_argument('--calib', required=True)
    # (2026-11-12 g13 retrain; '' = the default range) ON-emitter density range 'lo,hi' (/px^2) of the batches
    ap.add_argument('--dens', default='')
    a = ap.parse_args(argv)
    from simcode.util.proctitle import auto; auto('train-det')     # recognizable name in nvidia-smi / ps
    from simcode.hd import data as HD
    from simcode.hd.sc4 import nn_weights
    os.makedirs(a.out, exist_ok=True)
    args = vars(a); args.update(up=1, ch=[48, 96, 160], stem_ch=16, fch=32)
    json.dump(args, open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    gen = HD.Gen(a.size, a.size, 6, seed=a.seed, calib=[_cfg_resolve(c) for c in a.calib.split(',')])
    net = build(args).to(dev)
    if a.init:
        net.load_state_dict(torch.load(__import__('simcode.config', fromlist=['x']).resolve(a.init), map_location=dev, weights_only=False)['model'])
    print('params', n_params(net), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a')
    t0 = time.time(); S = a.size; up = 1
    for step in range(a.steps + 1):
        with torch.no_grad():
            o = batch(gen, a.batch, em_photons=EM_PHOTONS, **({'dens': tuple(float(v) for v in a.dens.split(','))} if a.dens else {}))
            label, ign, tid, off = targets(o, up, S, S)
            pm = label > 0
            w = torch.ones(int(pm.sum()), device=dev)
            if a.nnw and pm.any():
                gi = tid[pm]
                nnd = gt_nn(o['gt'])[gi]
                w = nn_weights(o['gt']['fov'][gi], nnd)
            wmap = torch.zeros_like(label); wmap[pm] = w
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(o['frames'])
        lg = out['logit'].float(); dxy = out['dxy'].float(); ls = out['log_sig'].float()
        l_det = focal_w(lg, label, ign, wmap)
        if pm.any():
            e = (dxy - off).permute(0, 2, 3, 1)[pm]
            s = ls[pm][:, None]
            l_off = ((0.5 * (e / s.exp()) ** 2 + s).sum(1) * w).sum() / w.sum().clamp(min=1e-6)
        else:
            l_off = torch.zeros((), device=dev)
        loss = l_det + a.w_off * l_off
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


def stats(argv):
    """training-mix statistics: target photon fraction in 500-5k, nn-bin fractions"""
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=40); ap.add_argument('--em_photons', default='40,6000')
    a = ap.parse_args(argv)
    from simcode.util.proctitle import auto; auto('train-det')     # recognizable name in nvidia-smi / ps
    from simcode.hd import data as HD
    torch.manual_seed(7)
    gen = HD.Gen(64, 64, 6, seed=0)
    emp = tuple(float(v) for v in a.em_photons.split(','))
    P, NN = [], []
    for i in range(a.n):
        o = batch(gen, 16, em_photons=emp)
        label, ign, tid, off = targets(o, 1, 64, 64)
        gi = tid[label > 0]
        P.append(o['gt']['phot'].sum(1)[gi]); NN.append(gt_nn(o['gt'])[gi])
    P = torch.cat(P); NN = torch.cat(NN)
    print('targets', len(P), 'frac 500-5k', float(((P >= 500) & (P < 5000)).float().mean()),
          '<500', float((P < 500).float().mean()), '>5k', float((P >= 5000).float().mean()))
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 4), (4, 1e9)):
        print(f'nn {lo}-{hi}', float(((NN >= lo) & (NN < hi)).float().mean()))


if __name__ == '__main__':
    {'train': train, 'stats': stats}[sys.argv[1]](sys.argv[2:])
