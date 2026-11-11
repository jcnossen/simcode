"""4Pi-SIMFLUX per-candidate head: 24 unwarped channel patches (6 frames x 4 channels, P x P) -> per (frame, channel)
two linear read-out intensities with a 2 x 2 covariance, per-(t, c) background, and a pattern-free, Phi-free 3D prior.

Outputs per candidate (no pixel fit anywhere; the geometry is solved explicitly afterwards, simcode2/fourpi/solve4pi.py):
  T [n,6,4]  expected photons of the candidate in the channel patch (read-out 1: learned weight map per channel,
             initialised at a Gaussian matched filter at the channel's sub-pixel position, normalised to be unbiased for
             the template, plus a learned correction)
  C [n,6,4]  photons weighted by a Gaussian window (sd wsd) at the true position (read-out 2: learned map initialised
             at that window at the candidate point, plus a correction). T and C together keep the 4Pi information
             (the centre carries ~2x the interference contrast of the total; stage-2 information check).
  L [n,6,4,2,2] Cholesky factor of the (T, C) covariance per (t, c)
  bg [n,6,4], prior dxy [n,2] px, z [n] um, log sigma [n,3]
The 4Pi phase field Phi0 is not an input: the prior cannot use it, so its z is the astigmatic / shape z.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode2.nn.blocks import ResBlock


def norm_photons(x, scale=30.0):
    return torch.sign(x) * torch.log1p(x.abs() / scale)


class Head4Pi(nn.Module):
    def __init__(self, P=15, ch=96, nblk=4, wsd=1.0, bf16=True):
        super().__init__()
        self.P, self.wsd, self.bf16 = P, wsd, bf16
        cin = 24 + 24 + 4 + 8
        self.cin = nn.Conv2d(cin, ch, 3, padding=1)
        self.blk = nn.Sequential(*[ResBlock(ch, 1 + (i % 2)) for i in range(nblk)])
        self.wmap = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 8, 1))
        nn.init.zeros_(self.wmap[-1].weight); nn.init.zeros_(self.wmap[-1].bias)
        self.down = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4))
        self.mlp = nn.Sequential(nn.Linear(ch * 16 + 24 + 8, 384), nn.GELU(), nn.Linear(384, 384), nn.GELU())
        self.nout = 24 + 24 + 72 + 24 + 6
        self.out = nn.Linear(384, self.nout)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, img, doff):
        """img [n,6,4,P,P] e (offset-corrected), doff [n,4,2] sub-pixel (x, y) offset of each channel's patch"""
        n, T, C, P, _ = img.shape
        dev = img.device
        border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
        b0 = border.median(-1).values                                            # [n,6,4]
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
        r2 = dx ** 2 + dy ** 2                                                    # [n,4,P,P]
        g = torch.exp(-0.5 * r2 / 1.3 ** 2); g = g / g.sum((-1, -2), keepdim=True)
        w1 = g / (g * g).sum((-1, -2), keepdim=True) * torch.exp(m[:, :4].clamp(-5, 5))
        w1 = w1 / (w1 * g).sum((-1, -2), keepdim=True)                           # w1 . g = 1
        w2 = torch.exp(-0.5 * r2 / self.wsd ** 2) * torch.exp(m[:, 4:].clamp(-5, 5))
        i = 0
        dT = q[:, i:i + 24].reshape(n, 6, 4); i += 24
        dC = q[:, i:i + 24].reshape(n, 6, 4); i += 24
        ch_ = q[:, i:i + 72].reshape(n, 6, 4, 3); i += 72
        dbg = q[:, i:i + 24].reshape(n, 6, 4); i += 24
        pr = q[:, i:i + 6]
        bg = b0 + dbg * (b0.abs() + 1).sqrt()
        y = img - bg[..., None, None]
        Tl = torch.einsum('ncuv,ntcuv->ntc', w1, y)
        Cl = torch.einsum('ncuv,ntcuv->ntc', w2, y)
        s1 = (Tl.clamp(min=0) + bg.clamp(min=0) * 12 + 2).sqrt()
        s2 = (Cl.clamp(min=0) + bg.clamp(min=0) * 3 + 1).sqrt()
        Tq = Tl + s1 * dT
        Cq = Cl + s2 * dC
        L = torch.zeros(n, 6, 4, 2, 2, device=dev)
        L[..., 0, 0] = s1 * torch.exp(ch_[..., 0].clamp(-4, 4)) * 1.5
        L[..., 1, 1] = s2 * torch.exp(ch_[..., 1].clamp(-4, 4)) * 1.5
        L[..., 1, 0] = s2 * ch_[..., 2]
        dxy = 1.5 * torch.tanh(pr[:, :2])
        z = 0.6 * torch.tanh(pr[:, 2])
        ls = -6.0 + 6.0 * torch.sigmoid(pr[:, 3:6])                              # log sigma: x, y px; z um
        return dict(T=Tq, C=Cq, L=L, bg=bg, dxy=dxy, z=z, ls=ls, T_lin=Tl, C_lin=Cl)


def nll_2x2(Tg, Cg, o):
    """Gaussian NLL of (T, C) per (t, c) with the predicted Cholesky factor"""
    r1 = Tg - o['T']; r2 = Cg - o['C']
    L = o['L']
    a = L[..., 0, 0]; b = L[..., 1, 0]; d = L[..., 1, 1]
    u1 = r1 / a; u2 = (r2 - b * u1) / d
    return 0.5 * (u1 ** 2 + u2 ** 2) + torch.log(a) + torch.log(d)


def nll_prior(x, y, z, o):
    s = torch.exp(o['ls'])
    e = torch.stack([(x - o['dxy'][:, 0]) / s[:, 0], (y - o['dxy'][:, 1]) / s[:, 1], (z - o['z']) / s[:, 2]], 1)
    return 0.5 * (e ** 2).sum(1) + o['ls'].sum(1)
