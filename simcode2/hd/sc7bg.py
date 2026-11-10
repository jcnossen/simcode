"""
sc7bg (user 2026-11-09: 'background is bad'). Diagnosis (<out>/sc7/bg/bg_diag.txt): the SCHead4 bg_t =
21 x 21 border median b0 + an MLP correction, and that bg also enters the learned linear readout I_lin = w_t . (y - bg_t)
without a stop-gradient, so the intensity loss shapes it into a readout nuisance (on iso the head's bg is WORSE than its
own starting statistic b0: rmsd 0.57 vs 0.48 ph/px, CRLB 0.14). Fix, network-side and pattern-free:
  BgNet  a small dense U-Net on the full 64 x 64 window (each frame + window mean / min as context; 4 downsamplings ->
         receptive field >> 21 px) predicting the per-frame bg map, trained ONLY against the true bg map o['bg'] with a
         dense per-pixel loss (pattern-free: frames only, no pattern / illumination input).
  SCHead7 = SCHead4 (sc5nb weights) whose bg is the BgNet map sampled at the candidate (detached, no head correction):
         bg_t = BgNet(t) at the candidate pixel; the readout subtracts the full BgNet patch, I_lin,t = w_t . (y_t - B_t)
         (spatially varying bg); the input 'lin' channels use the BgNet patch instead of the scalar border level.
No pixel fit anywhere (the head is unchanged otherwise; the pattern solve is unchanged).
Training: stage 1 BgNet only (--bg_pre steps), stage 2 head fine-tune from sc5nb with the sc6n mix while BgNet keeps
training on its own dense loss (separate optimiser; the head sees a detached bg).
usage: CUDA_VISIBLE_DEVICES=2 python -m simcode2.hd.sc7bg train --out runs/hd_sc7bg ...
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode2.hd.sc import PSFCol, patches, mod_slope, PP, norm_photons
from simcode2.hd.sc4 import SCHead4, nbr_map, nbr_patches, nn_weights, nll_full, NBR_PMIN

dev = 'cuda'


def _blk(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(), nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())


class BgNet(nn.Module):
    """dense per-frame bg map from the whole window (pattern-free). frames [B,T,H,W] photons -> bg [B,T,H,W]"""
    def __init__(self, ch=(32, 48, 64, 96, 128)):
        super().__init__()
        self.enc = nn.ModuleList()
        ci = 6
        for c in ch:
            self.enc.append(_blk(ci, c)); ci = c
        self.dec = nn.ModuleList()
        for c_skip, c in zip(reversed(ch[:-1]), reversed(ch[:-1])):
            self.dec.append(_blk(ci + c_skip, c)); ci = c
        self.out = nn.Conv2d(ci, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, fr):
        B, T, H, W = fr.shape
        m = fr.flatten(2).median(-1).values.clamp(min=0.1)                     # per-frame level [B,T]
        s = (m + 1).sqrt()
        lin = (fr - m[..., None, None]) / s[..., None, None]
        wm = fr.mean(1, keepdim=True).expand(B, T, H, W); wn = fr.min(1, keepdim=True).values.expand(B, T, H, W)
        mm = m.mean(1)[:, None, None, None]; sm = (mm + 1).sqrt()
        x = torch.stack([lin, norm_photons(fr), (wm - mm) / sm, (wn - mm) / sm, norm_photons(m)[..., None, None].expand(B, T, H, W),
                         ((m - mm[:, :, 0, 0]) / sm[:, :, 0, 0])[..., None, None].expand(B, T, H, W)], 2).reshape(B * T, 6, H, W)
        hs = []
        h = x
        for i, e in enumerate(self.enc):
            if i > 0: h = F.avg_pool2d(h, 2)
            h = e(h); hs.append(h)
        for d, sk in zip(self.dec, reversed(hs[:-1])):
            h = F.interpolate(h, size=sk.shape[-2:], mode='bilinear', align_corners=False)
            h = d(torch.cat([h, sk], 1))
        o = self.out(h).reshape(B, T, H, W)
        return m[..., None, None] + s[..., None, None] * o


def bg_loss(bgp, bgt):
    return (((bgp - bgt) ** 2) / (bgt + 1.0)).mean()


class SCHead7(SCHead4):
    """SCHead4 with the BgNet bg (see module doc). forward(..., bgp [n,T,P,P] BgNet patch at the candidate pixel)."""
    bgmap = True

    def __init__(self, T=6, ch=96, nblk=4, P=21, bf16=True):
        super().__init__(T, ch, nblk, P, bf16)
        self.bgnet = BgNet()

    def forward(self, pt, rn2p, off, psf=None, nbrp=None, bgp=None):
        n, T, P, _ = pt.shape
        h_ = P // 2
        bgp = bgp.detach()
        flat = pt.reshape(n, T, -1)
        b0 = bgp[:, :, h_, h_]                                                  # BgNet bg at the candidate pixel
        o = torch.arange(P, device=pt.device, dtype=torch.float32) - P // 2
        dx = o[None, None, :] - off[:, 0, None, None]; dy = o[None, :, None] - off[:, 1, None, None]
        cx = dx.expand(n, P, P) / (P // 2); cy = dy.expand(n, P, P) / (P // 2)
        lin = (pt - bgp) / 100.0
        x = torch.cat([norm_photons(pt), norm_photons(pt.mean(1, keepdim=True)), lin, lin.mean(1, keepdim=True),
                       cx[:, None], cy[:, None], nbrp], 1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            hmap = self.blk(F.gelu(self.cin(x)))
            m1 = F.gelu(self.wmap[0](hmap)); d1 = F.gelu(self.dmap[0](hmap))
            h = self.down(hmap).flatten(1)
            sc = torch.cat([norm_photons(b0), off, torch.zeros(n, T, device=pt.device)], 1)
            hm = self.mlp(torch.cat([h, sc], 1))
        m = self.wmap[2](m1.float())[:, 0]; d = self.dmap[2](d1.float())
        q = self.out(hm.float())
        g = psf
        gg = (g * g).sum((1, 2), keepdim=True)
        w0 = g / gg
        w = w0 * torch.exp(m.clamp(-5, 5))
        w = w / (w * g).sum((1, 2), keepdim=True)
        D = d * w0.amax((1, 2), keepdim=True)[:, None]
        D = D - ((D * g[:, None]).sum((2, 3), keepdim=True) / gg[:, None]) * g[:, None]
        wt = w[:, None] + D
        nt = T * (T + 1) // 2
        bg = b0
        Ilin = (wt.reshape(n, T, -1) * (flat - bgp.reshape(n, T, -1))).sum(-1)
        c = (Ilin.clamp(min=0).sum(1, keepdim=True) / T + bg.clamp(min=0).mean(1, keepdim=True) * 9).sqrt().clamp(min=3.0)
        dI = q[:, :T]
        ld = q[:, T:2 * T]; od = q[:, 2 * T:T + nt]
        L = torch.zeros(n, T, T, device=pt.device)
        L[:, torch.arange(T), torch.arange(T)] = c * torch.exp(ld.clamp(-4, 4)) * 1.5
        L[:, self.tril[0], self.tril[1]] = c * od
        I = Ilin + c * dI
        dxy = 0.5 * torch.tanh(q[:, T + nt:T + nt + 2])
        lsxy = -4.5 + 5.0 * torch.sigmoid(q[:, T + nt + 2:T + nt + 4])
        return dict(I=I, L=L, dxy=dxy, lsxy=lsxy, bg=bg, I_lin=Ilin, w=w, wt=wt, dI=dI)


def train(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', default='hd_sc5nb.pt')
    ap.add_argument('--steps', type=int, default=24000)
    ap.add_argument('--bg_pre', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--lr_bg', type=float, default=1e-3)
    ap.add_argument('--pct', type=float, default=0.1)
    ap.add_argument('--det', default='det_n1.pt')
    ap.add_argument('--tau', default='15,80')
    ap.add_argument('--pair_frac', type=float, default=0.25)
    ap.add_argument('--dens', default='0.0005,0.04')
    ap.add_argument('--med', default='40,2500')
    ap.add_argument('--bright_tail', default='0.1,2000,20000')
    ap.add_argument('--paper_med', default='1260,1890')
    ap.add_argument('--paper_sd', type=float, default=0.45)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--w_mr', type=float, default=2.0)
    ap.add_argument('--w_slope', type=float, default=50.0)
    ap.add_argument('--ckpt_every', type=int, default=8000)
    a = ap.parse_args(argv)
    a.arch = 'sc7bg'; a.ch = 96; a.bf16 = 1
    from simcode2.hd.sc4 import match_gt, targets
    from simcode2.hd import data as D
    from simcode2.det.model import select_subpixel
    from simcode2.hd.sc import load_det_front
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    det, _ = load_det_front(a.det)
    gen = D.Gen(64, 64, 6, seed=a.seed)
    head = SCHead7(6, 96, P=PP, bf16=True).to(dev)
    sd = torch.load(__import__('simcode2.config', fromlist=['x']).resolve(a.init), map_location=dev, weights_only=False)['model']
    miss, unexp = head.load_state_dict(sd, strict=False)
    assert all(k.startswith('bgnet.') for k in miss) and not unexp, (miss, unexp)
    btail = tuple(float(v) for v in a.bright_tail.split(',')) if a.bright_tail else None
    pmed = dict(zip(('0.5nM', '2.5nM'), (float(v) for v in a.paper_med.split(','))))
    col = PSFCol(gen.rs['0.5nM'].sim)
    dens = tuple(float(v) for v in a.dens.split(',')); med = tuple(float(v) for v in a.med.split(','))
    tau_r = tuple(float(v) for v in a.tau.split(','))
    hp = [p for n_, p in head.named_parameters() if not n_.startswith('bgnet.')]
    opt = torch.optim.AdamW(hp, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    ntot = a.bg_pre + a.steps + 1
    optb = torch.optim.AdamW(head.bgnet.parameters(), lr=a.lr_bg, weight_decay=1e-4)
    schedb = torch.optim.lr_scheduler.OneCycleLR(optb, max_lr=a.lr_bg, total_steps=ntot, pct_start=0.05)
    log = open(a.out + '/log.jsonl', 'a')
    t0 = time.time(); acc = {}; IS, GS = [], []
    for gstep in range(ntot):
        stage2 = gstep >= a.bg_pre; step = gstep - a.bg_pre
        with torch.no_grad():
            o = gen.batch(a.batch, dens=dens, med_photons=med, pair_frac=a.pair_frac, pair_curr=True, paper=True, tau_r=tau_r,
                          bright_tail=btail, paper_med=pmed, paper_sd=a.paper_sd)
            fr = o['frames']
        # ---- BgNet: dense loss against the true bg map
        bgm = head.bgnet(fr)
        lb = bg_loss(bgm, o['bg'])
        optb.zero_grad(set_to_none=True)
        if torch.isfinite(lb):
            lb.backward(); torch.nn.utils.clip_grad_norm_(head.bgnet.parameters(), 1.0); optb.step()
        schedb.step()
        rec = dict(bgmap=lb.item(), bgmap_rmsd=float((bgm.detach() - o['bg']).pow(2).mean().sqrt()))
        if stage2:
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    dd = det(fr)
                lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
                c = select_subpixel(lgm, dd['dxy'].float(), 640, p_min=NBR_PMIN, radius=0.7)
                kp = torch.isfinite(c['logit'])
                nbr = nbr_map(lgm, *(c[k][kp] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit')))
                c = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))
                m = match_gt(o, c)
            if m is None:
                sched.step(); continue
            with torch.no_grad():
                t = targets(o, m, col)
                iso_w = (m['nn'] >= 2.0).float()
                wt = nn_weights(o['gt']['fov'][m['g']], m['nn'])
                off = torch.stack([m['x'] - m['ix'].float(), m['y'] - m['iy'].float()], 1)
                pt = patches(fr, m['b'], m['iy'], m['ix'])
                psf = col(off)
                p_own = torch.sigmoid(lgm[m['b'], m['iy'], m['ix']])
                nbp = nbr_patches(nbr, m['b'], m['iy'], m['ix'], m['x'], m['y'], p_own)
                bgp = patches(bgm.detach(), m['b'], m['iy'], m['ix'])
            out = head(pt, None, off, psf, nbp, bgp)
            out['x'] = m['x'] + out['dxy'][:, 0]; out['y'] = m['y'] + out['dxy'][:, 1]
            Ig, tx, ty = t['Ig'], t['tx'], t['ty']
            nll_i, z = nll_full(Ig, out['I'], out['L'])
            nll_I = (wt * nll_i).mean()
            ex = tx - out['x']; ey = ty - out['y']; ls = out['lsxy']
            nll_p = (wt * (0.5 * (ex / ls[:, 0].exp()) ** 2 + ls[:, 0] + 0.5 * (ey / ls[:, 1].exp()) ** 2 + ls[:, 1])).mean()
            cn = (Ig.clamp(min=0).sum(1, keepdim=True) / 6 + 30).sqrt()
            mr = ((out['I'] - out['I'].mean(1, keepdim=True)) - (Ig - Ig.mean(1, keepdim=True))) / cn
            l_mr = (wt[:, None] * F.huber_loss(mr, torch.zeros_like(mr), delta=3.0, reduction='none')).mean()
            am = out['I'] - out['I'].mean(1, keepdim=True); gm = Ig - Ig.mean(1, keepdim=True)
            slope_b = (am * gm * iso_w[:, None]).sum() / (gm * gm * iso_w[:, None]).sum().clamp(min=1e-6)
            l_sl = (1 - slope_b) ** 2
            loss = nll_I / 6 + nll_p + a.w_mr * l_mr + a.w_slope * l_sl
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(hp, 1.0); opt.step()
            else:
                acc['nonfinite'] = acc.get('nonfinite', 0) + 1
            sched.step()
            with torch.no_grad():
                IS.append(out['I'].detach()); GS.append(Ig)
                bgt = t['bgt_px']
                rec.update(slope_iso=slope_b.item(), mr=l_mr.item(), nll_I=nll_I.item(), nll_p=nll_p.item(), n=len(Ig),
                           zI=float((z ** 2).mean().sqrt()), pos_rmse=float(torch.sqrt((ex ** 2 + ey ** 2).mean() / 2)),
                           bg_rmsd=float((out['bg'] - bgt).pow(2).mean().sqrt()))
        for k, v in rec.items():
            acc[k] = acc.get(k, 0) + v
        acc['_n'] = acc.get('_n', 0) + 1
        if gstep % 200 == 0:
            nn_ = acc.pop('_n', 1)
            r = dict(step=gstep, stage=2 if stage2 else 1, t=round(time.time() - t0), lr=sched.get_last_lr()[0], lr_bg=schedb.get_last_lr()[0],
                     **{k: v / nn_ for k, v in acc.items()})
            if IS:
                r['slope'] = mod_slope(torch.cat(IS), torch.cat(GS)); IS, GS = [], []
            acc = {}
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if stage2 and step % a.ckpt_every == 0 and step > 0:
            torch.save(dict(model=head.state_dict(), args=vars(a), step=step), a.out + f'/ckpt_{step}.pt')
        if gstep == a.bg_pre - 1:
            torch.save(dict(model=head.state_dict(), args=vars(a), step=0), a.out + '/bgpre.pt')
    torch.save(dict(model=head.state_dict(), args=vars(a), step=a.steps), a.out + '/final.pt')


if __name__ == '__main__':
    st = sys.argv[1]; rest = sys.argv[2:]; sys.argv = sys.argv[:1]
    if st == 'train':
        train(rest)
