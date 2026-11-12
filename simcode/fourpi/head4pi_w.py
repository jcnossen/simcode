"""Learned-window 4Pi head (v10, 2026-11-12): the read-out weights are learned, with an architectural guard against the
frame / channel mixing that the review measured (out/4pi/REVIEW_NET.md).

  Y[t, c, k] = sum_px W'[c, k, px] (img[t, c, px] - bg[c]),      W' = W + D
  * W: the K canonical windows at the prior p0 (windows4pi; the fixed read-out = teacher, D = 0 at initialisation);
  * D [4, K, P, P]: learned window change, produced by a CNN that sees only FRAME-SYMMETRIC context (the frame-mean
    channel images, the template at p0, coordinates): the same weights for the 6 frames, and nothing in D can depend on
    which frame is bright, so the read-out of frame t is linear in that frame's own pixels;
  * D is projected orthogonal to the emitter's own image components at p0 (incoherent PSF and the two 4Pi cross terms),
    so the response to the emitter is that of W for any 4Pi phase: the learned freedom can only change how noise,
    neighbours and background structure enter (null a neighbour, avoid a bright pixel), it cannot rescale the signal;
  * bounded: |D| <= bound x the window amplitude;
  * covariance: analytic for the windows actually used (pixel variance incl. the frame-mean excess over the model, i.e.
    neighbours), times a learned scale >= 1; one supervised background per channel.
The prior (x, y, z, sigma) comes from a frozen Head4Pi (v3L) inside this module. The explicit solve (solvek4pi) models W'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode.nn.blocks import ResBlock
from simcode.fourpi.head4pi import Head4Pi, norm_photons
from simcode.fourpi.windows4pi import window, WINS_K

VREAD = 1.14 ** 2 + 0.2 ** 2


class Head4PiW(nn.Module):
    def __init__(self, P=15, ch=64, nblk=4, wins=WINS_K, bound=1.0, prior_ch=96, prior_nblk=4):
        super().__init__()
        self.P, self.wins, self.K, self.bound = P, tuple(wins), len(wins), bound
        self.prior = Head4Pi(ch=prior_ch, nblk=prior_nblk)
        for p_ in self.prior.parameters(): p_.requires_grad_(False)
        cin = 4 + 4 + 4 + 8
        self.cin = nn.Conv2d(cin, ch, 3, padding=1)
        self.blk = nn.Sequential(*[ResBlock(ch, 1 + (i % 2)) for i in range(nblk)])
        self.dmap = nn.Sequential(nn.Conv2d(ch, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 4 * self.K, 1))
        nn.init.zeros_(self.dmap[-1].weight); nn.init.zeros_(self.dmap[-1].bias)
        self.down = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4))
        self.mlp = nn.Sequential(nn.Linear(ch * 16 + 4, 128), nn.GELU(), nn.Linear(128, 5))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.lut = None

    def train(self, mode=True):
        super().train(mode); self.prior.eval(); return self

    def forward(self, img, doff, learned=True):
        n, T, C, P, _ = img.shape; dev = img.device; K = self.K; lut = self.lut
        with torch.no_grad():
            op = self.prior(img, doff)
            dxy, z, ls = op['dxy'], op['z'], op['ls']
            p0 = torch.cat([dxy, z[:, None]], 1)
            border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
            bm = border.mean(1).median(-1).values                                # [n,4]
            fm = img.mean(1)                                                      # [n,4,P,P] frame-symmetric
            o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
            oy, ox = torch.meshgrid(o, o, indexing='ij')
            Ws, gs, Bs = [], [], []
            for c in range(4):
                pos = torch.stack([dxy[:, 0] + doff[:, c, 0], dxy[:, 1] + doff[:, c, 1], z], 1)
                inc_, cr, ci = (lut.luts[j][c].evaluate(pos) for j in range(3))
                inc = inc_.clamp(min=1e-7)
                ex = ox[None] - pos[:, 0, None, None]; ey = oy[None] - pos[:, 1, None, None]
                Ws.append(torch.stack([window(k, par, inc, ex, ey, (cr, ci)) for k, par in self.wins], 1))
                gs.append(inc / inc.sum((-1, -2), keepdim=True)); Bs.append(torch.stack([inc_, cr, ci], -1).reshape(n, P * P, 3))
            W = torch.stack(Ws, 1); g = torch.stack(gs, 1)                        # [n,4,K,P,P], [n,4,P,P]
            Q = torch.linalg.qr(torch.stack(Bs, 1))[0]                            # [n,4,P*P,3] orthonormal basis of the emitter's image
            cx = (ox[None, None] - doff[:, :, 0, None, None]) / (P // 2); cy = (oy[None, None] - doff[:, :, 1, None, None]) / (P // 2)
            x = torch.cat([norm_photons(fm), (fm - bm[..., None, None]) / 50.0, g / g.amax((-1, -2), keepdim=True), cx, cy], 1)
        h = self.blk(F.gelu(self.cin(x)))
        q = self.mlp(torch.cat([self.down(h).flatten(1), norm_photons(bm)], 1))
        dbg, cq = q[:, :4], q[:, 4]
        if learned:
            amp = W.abs().amax((-1, -2), keepdim=True)                            # [n,4,K,1,1]
            D = (self.bound * torch.tanh(self.dmap(h).reshape(n, 4, K, P, P)) * amp).reshape(n, 4, K, P * P)
            D = D - torch.einsum('nckj,ncpj->nckp', torch.einsum('nckp,ncpj->nckj', D, Q), Q)   # no response to the emitter itself
            Wp = W + D.reshape(n, 4, K, P, P)
            bgc = bm + 0.3 * dbg * (bm.abs() + 1).sqrt()
            cs = 1.0 + F.softplus(cq - 3.0)
        else:
            Wp = W; bgc = bm; cs = torch.ones(n, device=dev)
        y = img - bgc[:, None, :, None, None]
        Y = torch.einsum('nckuv,ntcuv->ntck', Wp, y)
        with torch.no_grad():
            Y0 = torch.einsum('ncuv,ntcuv->ntc', W[:, :, 0], y).clamp(min=0)      # photons per (t, c) through the canonical window
            mod = Y0[..., None, None] * g[:, None] + bgc[:, None, :, None, None].clamp(min=0.05)
            vp = mod + VREAD + (fm[:, None] - mod.mean(1, keepdim=True)).clamp(min=0)   # + frame-mean excess (neighbours)
        S = torch.einsum('nckuv,ncluv,ntcuv->ntckl', Wp, Wp, vp)
        S = S + 1e-6 * torch.diagonal(S, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(K, device=dev)
        Lb0 = torch.linalg.cholesky(S)
        return dict(Y=Y, Lb0=Lb0, cs=cs, Lb=Lb0 * cs[:, None, None, None, None], Wsum=Wp.sum((-1, -2)), W=Wp, p0=p0,
                    bgc=bgc, bg=bgc[:, None].expand(n, 6, 4), dxy=dxy, z=z, ls=ls)
