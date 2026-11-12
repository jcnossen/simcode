"""Structured 4Pi head (v8, 2026-11-12; review out/4pi/REVIEW_NET.md). Same trunk, inputs and outputs as Head4Pi
(head4pi.py); the read-out construction follows the review's hand-built linear read-out, which reaches 1.10-1.12 x the
pixel CRLB laterally at all photon levels with pass-1 knowledge only:
  * background: ONE value per channel, shared by the 6 frames (as in the pixel MLE and its CRLB), anchored on the border
    median of the frame-mean image plus a learned correction, supervised. (v3: 24 free per-(frame, channel) values on the
    per-image border median: frame-specific background errors are pattern-phase noise.)
  * T read-out: matched filter on the phase-averaged 4Pi channel PSF (LUT) at the head's own prior (x, y, z), with
    variance weights g / (T g + bg + V) (uniform-over-the-PSF weighting for bright emitters), times a learned map, the
    same weights for the 6 frames of a channel; plus the learned additive correction.
  * C read-out: Gaussian window at the prior position times a learned map, plus the correction.
  * a learned gain per channel for T and for C, shared by the 6 frames (template mismatch: 4Pi phase, z, position).
At initialisation (zero maps / corrections) T is exactly that hand-built read-out.
Outputs as Head4Pi: T, C [n,6,4], L [n,6,4,2,2], bg [n,6,4] (frame-shared values expanded), dxy, z, ls.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode2.nn.blocks import ResBlock
from simcode2.fourpi.head4pi import norm_photons

VREAD = 1.14 ** 2 + 0.2 ** 2


class Head4PiS(nn.Module):
    def __init__(self, P=15, ch=96, nblk=4, wsd=1.0, bf16=True, bg_frame=False):
        super().__init__()
        self.P, self.wsd, self.bf16, self.bg_frame = P, wsd, bf16, bg_frame
        cin = 24 + 24 + 4 + 8
        self.cin = nn.Conv2d(cin, ch, 3, padding=1)
        self.blk = nn.Sequential(*[ResBlock(ch, 1 + (i % 2)) for i in range(nblk)])
        self.wmap = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 8, 1))
        nn.init.zeros_(self.wmap[-1].weight); nn.init.zeros_(self.wmap[-1].bias)
        self.down = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4))
        self.mlp = nn.Sequential(nn.Linear(ch * 16 + 24 + 8, 384), nn.GELU(), nn.Linear(384, 384), nn.GELU())
        self.nout = 24 + 24 + 72 + 4 + 6 + 8      # ... + per-channel log gains of T and C (appended last)
        self.out = nn.Linear(384, self.nout)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.tlut = None        # LUT4Pi(psf, P=15, sub=1), set by the caller (not a module: not in the state dict)

    def load_v3(self, state):
        """initialise from a Head4Pi (v3-type) state dict: trunk and prior rows; the weight maps, the T / C / background
        corrections and the covariance factors start at zero (= the hand-built read-out with its analytic covariance)"""
        own = self.state_dict()
        for k, v in state.items():
            if k in own and own[k].shape == v.shape and not k.startswith('wmap.2') and not k.startswith('out.'):
                own[k] = v.clone()
        W, b = state['out.weight'], state['out.bias']
        own['out.weight'].zero_(); own['out.bias'].zero_()
        own['out.weight'][124:130] = W[144:150]; own['out.bias'][124:130] = b[144:150]      # prior
        self.load_state_dict(own)

    def forward(self, img, doff):
        n, T, C, P, _ = img.shape
        dev = img.device
        border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
        b0 = border.median(-1).values                                            # [n,6,4] (trunk input, as Head4Pi)
        bm = border.mean(1).median(-1).values                                    # [n,4] frame-mean border median
        o = torch.arange(P, device=dev, dtype=torch.float32) - P // 2
        dx = (o[None, None, None, :] - doff[:, :, 0, None, None]).expand(n, 4, P, P)
        dy = (o[None, None, :, None] - doff[:, :, 1, None, None]).expand(n, 4, P, P)
        cx = dx / (P // 2); cy = dy / (P // 2)
        lin = (img - b0[..., None, None]) / 50.0
        x = torch.cat([norm_photons(img).reshape(n, 24, P, P), lin.reshape(n, 24, P, P),
                       norm_photons(img.mean(1)), cx, cy], 1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            h = self.blk(F.gelu(self.cin(x)))
            m1 = F.gelu(self.wmap[0](h))
            hd = self.down(h).flatten(1)
            hm = self.mlp(torch.cat([hd, norm_photons(b0).reshape(n, 24), doff.reshape(n, 8)], 1))
        m = self.wmap[2](m1.float())                                             # [n,8,P,P]
        q = self.out(hm.float())
        i = 0
        dT = q[:, i:i + 24].reshape(n, 6, 4); i += 24
        dC = q[:, i:i + 24].reshape(n, 6, 4); i += 24
        ch_ = q[:, i:i + 72].reshape(n, 6, 4, 3); i += 72
        dbg = q[:, i:i + 4]; i += 4
        pr = q[:, i:i + 6]; i += 6
        gT = torch.exp(q[:, i:i + 4].clamp(-1, 1))[:, None]; gC = torch.exp(q[:, i + 4:i + 8].clamp(-1, 1))[:, None]   # [n,1,4]
        dxy = 1.5 * torch.tanh(pr[:, :2])
        z = 0.6 * torch.tanh(pr[:, 2])
        ls = -6.0 + 6.0 * torch.sigmoid(pr[:, 3:6])
        bgc = bm + 0.3 * dbg * (bm.abs() + 1).sqrt()                              # [n,4]
        bg = bgc[:, None].expand(n, 6, 4)
        # template at the head's own prior (detached: the prior is trained by its own loss only)
        px_, py_, pz_ = dxy[:, 0].detach(), dxy[:, 1].detach(), z.detach()
        tl = self.tlut
        g = torch.stack([tl.luts[0][c].evaluate(torch.stack([px_ + doff[:, c, 0], py_ + doff[:, c, 1], pz_], 1))
                         for c in range(4)], 1).clamp(min=1e-7)
        g = g / g.sum((-1, -2), keepdim=True)
        y = img - bg[..., None, None]
        with torch.no_grad():   # first-pass photons per channel (frame mean) for the variance weights
            T0 = ((g[:, None] * y).sum((-1, -2)) / (g * g).sum((-1, -2))[:, None]).mean(1).clamp(min=0)    # [n,4]
            v = T0[..., None, None] * g + bgc.clamp(min=0.05)[..., None, None] + VREAD
        w1 = g / v * torch.exp(m[:, :4].clamp(-5, 5))
        w1 = w1 / (w1 * g).sum((-1, -2), keepdim=True)                           # w1 . g = 1
        r2c = (o[None, None, None, :] - (px_[:, None, None, None] + doff[:, :, 0, None, None])) ** 2 \
            + (o[None, None, :, None] - (py_[:, None, None, None] + doff[:, :, 1, None, None])) ** 2
        w2 = torch.exp(-0.5 * r2c / self.wsd ** 2) * torch.exp(m[:, 4:].clamp(-5, 5))
        Tl = torch.einsum('ncuv,ntcuv->ntc', w1, y)
        Cl = torch.einsum('ncuv,ntcuv->ntc', w2, y)
        # analytic covariance of the two linear read-outs (pixel variance from the read-out itself), times learned factors
        with torch.no_grad():
            vp = Tl.clamp(min=0)[..., None, None] * g[:, None] + bg.clamp(min=0.05)[..., None, None] + VREAD   # [n,6,4,P,P]
        a2 = torch.einsum('ncuv,ntcuv->ntc', w1 * w1, vp).clamp(min=1e-4)
        c22 = torch.einsum('ncuv,ntcuv->ntc', w2 * w2, vp).clamp(min=1e-4)
        c12 = torch.einsum('ncuv,ntcuv->ntc', w1 * w2, vp)
        # per-channel gains (shared by the 6 frames): the template is phase-averaged and sits at the prior, so the linear
        # read-outs have a channel gain error that depends on the 4Pi phase and on z; frame-shared = no pattern-phase error
        s1 = a2.sqrt() * gT
        s2 = (c22 - c12 ** 2 / a2).clamp(min=1e-4 * c22).sqrt() * gC
        c12 = c12 * gT * gC
        Tl = Tl * gT; Cl = Cl * gC
        Tq = Tl + s1 * dT
        Cq = Cl + s2 * dC
        L = torch.zeros(n, 6, 4, 2, 2, device=dev)
        L[..., 0, 0] = s1 * torch.exp(ch_[..., 0].clamp(-4, 4))
        L[..., 1, 1] = s2 * torch.exp(ch_[..., 1].clamp(-4, 4))
        L[..., 1, 0] = c12 / s1 + s2 * ch_[..., 2]
        return dict(T=Tq, C=Cq, L=L, bg=bg, bgc=bgc, dxy=dxy, z=z, ls=ls, T_lin=Tl, C_lin=Cl)
