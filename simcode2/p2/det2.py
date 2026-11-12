"""
DetP2: pattern-aware pass-2 detector = det_n1 (simcode2/det/up2.DetUp, up = 1, native 108 nm grid) + illumination inputs.
  per-frame stem input (frame_t) -> (frame_t, cos_t, sin_t): the stem's first conv gets 2 extra input channels per frame
       (zero-initialised), so it can demodulate frame t with its own illumination
  fusion conv cin: + the 2T illumination maps (zero-initialised)
With all extra weights zero the network is exactly det_n1 (checked in `check`). Outputs / selection as det_n1
(simcode2/det/up2.select, model.select_subpixel).
Training (train): init det_n1, realistic batches (p2/data.Gen2: half 'paper' label-unit brightness + bright tail, half the
det_n1 photon mix; pairs 0.3-3.0 px), perturbed pattern estimates as input; loss = det_n1's (focal BCE with nn-bin weights +
offset Gaussian NLL). Held-out: fixed seeded batches (paper mix, perturbed patterns), recall at p >= 0.5 per nn bin and
FP per window, det_n1 reference on the same batches.
usage: CUDA_VISIBLE_DEVICES=1 python -m simcode2.p2.det2 train --out runs/p2_det
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from simcode2.det.up2 import DetUp, targets, focal_w, gt_nn, select, band_extra, band_off_loss
from simcode2.det.model import n_params

dev = 'cuda'
from simcode2 import config as _cfg
DET_N1 = _cfg.model(_cfg.DET)


class DetP2(DetUp):
    def __init__(self, T=6, ch=(48, 96, 160), stem_ch=16):
        super().__init__(1, T, ch, stem_ch)
        self.stem[0] = nn.Conv2d(3, stem_ch, 3, padding=1)
        self.cin_il = nn.Conv2d(2 * T, ch[0], 3, padding=1, bias=False)
        nn.init.zeros_(self.cin_il.weight)

    def load_n1(self, sd):
        """det_n1 state dict -> self, extra illumination weights zero"""
        own = self.state_dict()
        for k, v in sd.items():
            if k == 'stem.0.weight':
                w = torch.zeros_like(own[k]); w[:, :1] = v; own[k] = w
            else:
                own[k] = v
        nn.init.zeros_(self.cin_il.weight)
        own['cin_il.weight'] = torch.zeros_like(own['cin_il.weight'])
        self.load_state_dict(own)

    def forward(self, frames, illum):
        """frames [B,T,H,W] photons, illum [B,2T,H,W] (illum.illum_maps) -> as DetNet.forward"""
        B, T, H, W = frames.shape
        ph, pw = (-H) % 4, (-W) % 4
        if ph or pw:
            frames = F.pad(frames, (0, pw, 0, ph), mode='replicate'); illum = F.pad(illum, (0, pw, 0, ph), mode='replicate')
        from simcode2.det.model import norm_photons
        m = norm_photons(frames.mean(1, keepdim=True))
        f = norm_photons(frames)
        Hp, Wp = f.shape[-2:]
        sin = torch.stack([f, illum[:, :T], illum[:, T:]], 2).reshape(B * T, 3, Hp, Wp)
        s = self.stem(sin).reshape(B, -1, Hp, Wp)
        x = F.gelu(self.cin(torch.cat([s, m], 1)) + self.cin_il(illum))
        s0 = self.e0(x)
        s1 = self.e1(self.d1(F.gelu(s0)))
        x = self.e2(self.d2(F.gelu(s1)))
        x = F.interpolate(self.u1(F.gelu(x)), scale_factor=2, mode='bilinear', align_corners=False) + s1
        x = self.r1(x)
        x = F.interpolate(self.u0(F.gelu(x)), scale_factor=2, mode='bilinear', align_corners=False) + s0
        x = self.r0(x)
        o = self.head(F.gelu(x))[..., :H, :W]
        out = dict(logit=o[:, 0], dxy=torch.tanh(o[:, 1:3]) * self.dxy_range, log_sig=-5.0 + 6.0 * torch.sigmoid(o[:, 3]))
        if getattr(self, 'psum', 0.0) != 0 and not self.training:     # (2026-11-10, opt-in via args['psum'])
            from simcode2.det.model import psum_out
            out = psum_out(out, self.psum)
        return out


def build(args=None):
    net = DetP2()
    net.psum = float((args or {}).get('psum', 0.0) or 0.0)
    return net


def load(ck, dev='cuda'):
    from simcode2.config import resolve
    c = torch.load(resolve(ck), map_location=dev, weights_only=False)
    net = build(c.get('args')).to(dev); net.load_state_dict(c['model']); net.eval()
    return net, c.get('args', {})


def load_init(ck=DET_N1):
    net = build().to(dev)
    net.load_n1(torch.load(ck, map_location=dev, weights_only=False)['model'])
    return net


# ------------------------------------------------------------------------------------------------ held-out
def make_hold(gen, n=24, nb=16, seed=777):
    H = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        for i in range(n):
            o = gen.batch2(nb, mix='paper', pair_sep=(0.3, 3.0))
            H.append(dict(frames=o['frames'], illum=o['illum_p2'], gt={k: v for k, v in o['gt'].items()}, nb=nb))
    return H


@torch.no_grad()
def eval_hold(net, H, p2=True, thr=0.5):
    """recall (1 px, one-to-one, targets >= 1 on-frame, >= 200 window photons, interior) per nn bin x photon bin, FP / window"""
    from simcode2.hd.gate_report import match
    was = net.training; net.eval()
    rec = {}; nfp = 0; nw = 0
    for h in H:
        fr = h['frames']; S = fr.shape[-1]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(fr, h['illum']) if p2 else net(fr)
        c = select(out, 1, k=640, p_min=thr)
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
        vis = ph >= 60
        oin = (x >= 4) & (x < S - 4) & (y >= 4) & (y < S - 4)
        fp = (og < 0) & oin
        # outputs matched to non-visible GT are not FP
        nfp += int(fp.sum()); nw += h['nb']
        for nm, (a, bb) in (('nn<1', (0, 1)), ('nn1-1.5', (1, 1.5)), ('nn1.5-2.5', (1.5, 2.5)), ('nn>2.5', (2.5, 1e9))):
            for pl, phh in ((200, 700), (700, 2000), (2000, 1e9)):
                m = tgt & (nnd >= a) & (nnd < bb) & (ph >= pl) & (ph < phh)
                k = f'{nm}|{pl}-{phh:.0f}'
                r = rec.setdefault(k, [0, 0]); r[0] += int(hit[m].sum()); r[1] += int(m.sum())
    if was: net.train()
    return dict(recall={k: (v[0] / max(v[1], 1), v[1]) for k, v in rec.items()}, fp_per_win=nfp / max(nw, 1))


# ------------------------------------------------------------------------------------------------ training
def train(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--pct', type=float, default=0.05)
    ap.add_argument('--w_off', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--eval_every', type=int, default=2500)
    ap.add_argument('--ckpt_every', type=int, default=5000)
    ap.add_argument('--no_illum', type=int, default=0)     # control: same data / schedule, illumination inputs zeroed
    ap.add_argument('--gen', default='')                   # (2026-11-10) 'r1' = simcode2/hd/realgen.RGen (held-out too)
    ap.add_argument('--init', default='')                  # fine-tune from a DetP2 checkpoint (default: det_n1 zero-init)
    ap.add_argument('--psum', type=float, default=0.0)     # stored in args: summed-p output radius at inference (eval too)
    ap.add_argument('--band', type=float, default=0.0)     # sub-pixel-invariant target band (det/up2.band_extra), 0 = off
    # (2026-11-11 psf3d retrain, opt-in; defaults unchanged) PSF calib mixture (comma list, hd/data.Gen calib), close-pair
    # enrichment 'frac,lo,hi' (realsim.emitters pair_sub)
    ap.add_argument('--calib', default='')
    ap.add_argument('--pair_sub', default='')
    # (2026-11-12 g13 retrain, opt-in; '' = unchanged) training ranges 'lo,hi': ON-emitter density (/px^2) and mean on-time (frames)
    ap.add_argument('--dens', default='')
    ap.add_argument('--tau', default='')
    ap.add_argument('--init_n1', default='')               # (opt-in) det_n1-type checkpoint for the zero-init (default det_n1.pt)
    a = ap.parse_args(argv)
    from simcode2.util.proctitle import auto; auto('train-p2det')     # recognizable name in nvidia-smi / ps
    from simcode2.p2.data import Gen2
    from simcode2.hd.sc4 import nn_weights
    from simcode2.det.up2 import load as load_n1
    os.makedirs(a.out, exist_ok=True)
    args = vars(a); json.dump(args, open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    if a.gen == 'r1':
        from simcode2.hd.realgen import RGen
        gen = RGen(a.size, a.size, 6, seed=a.seed)
    else:
        gen = Gen2(a.size, a.size, 6, seed=a.seed, calib=[_cfg.resolve(c) for c in a.calib.split(',')] if a.calib else None)
    psub = {'pair_sub': tuple(float(v) for v in a.pair_sub.split(','))} if a.pair_sub else {}
    if a.dens: psub['dens'] = tuple(float(v) for v in a.dens.split(','))
    if a.tau: psub['tau_r'] = tuple(float(v) for v in a.tau.split(','))
    if a.init:
        net = build().to(dev); net.load_state_dict(torch.load(__import__('simcode2.config', fromlist=['x']).resolve(a.init), map_location=dev, weights_only=False)['model'])
    else:
        net = load_init(_cfg.resolve(a.init_n1) if a.init_n1 else DET_N1)
    net.psum = a.psum
    print('params', n_params(net), flush=True)
    Hd = make_hold(gen)
    if a.no_illum:
        for h in Hd: h['illum'] = torch.zeros_like(h['illum'])
    n1, _ = load_n1(_cfg.resolve(a.init_n1) if a.init_n1 else DET_N1)
    ref = eval_hold(n1, Hd, p2=False); del n1
    ev0 = eval_hold(net, Hd)
    json.dump(dict(det_n1=ref, p2_init=ev0), open(a.out + '/eval_ref.json', 'w'), indent=1)
    print('REF det_n1', json.dumps(ref), '\nINIT p2', json.dumps(ev0), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a'); elog = open(a.out + '/eval.jsonl', 'a')
    t0 = time.time(); S = a.size
    for step in range(a.steps + 1):
        with torch.no_grad():
            mix = 'paper' if step % 2 == 0 else 'det'
            o = gen.batch2(a.batch, mix=mix, pair_sep=(0.3, 3.0), **psub)
            label, ign, tid, off = targets(o, 1, S, S)
            pm = label > 0
            w = torch.ones(int(pm.sum()), device=dev)
            if pm.any():
                gi = tid[pm]
                w = nn_weights(o['gt']['fov'][gi], gt_nn(o['gt'])[gi])
            wmap = torch.zeros_like(label); wmap[pm] = w
            ign, ex = band_extra(o['gt'], label, ign, tid, a.band)
            if ex is not None:
                wg = torch.zeros(len(o['gt']['x']), device=dev); wg[tid[pm]] = w; wg = wg[ex['g']]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(o['frames'], o['illum_p2'] * (0.0 if a.no_illum else 1.0))
        lg = out['logit'].float(); dxy = out['dxy'].float(); ls = out['log_sig'].float()
        l_det = focal_w(lg, label, ign, wmap)
        if pm.any():
            e = (dxy - off).permute(0, 2, 3, 1)[pm]
            s = ls[pm][:, None]
            if ex is not None and len(ex['b']):
                lb, wb = band_off_loss(dxy, ls, ex, wg)
                l_off = (((0.5 * (e / s.exp()) ** 2 + s).sum(1) * w).sum() + lb) / (w.sum() + wb).clamp(min=1e-6)
            else:
                l_off = ((0.5 * (e / s.exp()) ** 2 + s).sum(1) * w).sum() / w.sum().clamp(min=1e-6)
        else:
            l_off = torch.zeros((), device=dev)
        loss = l_det + a.w_off * l_off
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
            ev = eval_hold(net, Hd)
            elog.write(json.dumps(dict(step=step, t=round(time.time() - t0), **ev)) + '\n'); elog.flush()
            print('EVAL', step, json.dumps(ev), flush=True)
        if step % a.ckpt_every == 0 and step > 0:
            torch.save(dict(model=net.state_dict(), args=args, step=step), a.out + f'/ckpt_{step}.pt')
    torch.save(dict(model=net.state_dict(), args=args, step=a.steps), a.out + '/final.pt')


def check():
    """zero-init check: DetP2 from det_n1 == det_n1 on a realistic batch"""
    from simcode2.p2.data import Gen2
    from simcode2.det.up2 import load as load_n1
    torch.manual_seed(0)
    gen = Gen2(64, 64, 6, seed=0)
    o = gen.batch2(4)
    n1, _ = load_n1(DET_N1); p2 = load_init().eval()
    with torch.no_grad():
        a = n1(o['frames']); b = p2(o['frames'], o['illum_p2'])
    print('max |dlogit|', float((a['logit'] - b['logit']).abs().max()), 'max |ddxy|', float((a['dxy'] - b['dxy']).abs().max()))
    print('illum maps', o['illum_p2'].shape, float(o['illum_p2'][:, :6].std()), float(o['illum_p2'][:, 6:].std()))
    print('patt err phase', float((o['patt_p2'][..., 2] - o['patt'][..., 2]).std()), 'depth', float((o['patt_p2'][..., 3] - o['patt'][..., 3]).mean()))


if __name__ == '__main__':
    {'train': train, 'check': lambda a: check()}[sys.argv[1]](sys.argv[2:])
