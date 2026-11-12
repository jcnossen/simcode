"""Learned K-read-out 4Pi head (v9, 2026-11-12; user decision 13:00: the intensities must come from the network).

Outputs per candidate: Y [n,6,4,K] = the network's estimate of the main emitter's expected signal through K fixed windows
placed at the head's own prior p0 = (x, y, z) (windows4pi; target  Y*[t,c,k] = sum_px w_k(px; p0) mu_main[t,c,px], i.e.
without background, neighbours and noise), a K x K covariance per (frame, channel), one background per channel and the
pattern-free 3D prior. The explicit solve (solvek4pi.SolveK) models these windows.
Construction (review findings, out/4pi/REVIEW_NET.md):
  * anchor = the linear read-out of the image through the windows with the head's background (linread4pi): at
    initialisation the head IS the fixed read-out (the measured reference, 1.02-1.11 x CRLB on isolated emitters);
  * learned part = a bounded additive correction per read-out (+-3 sd of the read-out noise) and a covariance scale
    (>= the analytic pixel-noise covariance): the freedom is for what the linear read-out cannot do (neighbours,
    background structure), not for re-weighting the emitter's own pixels;
  * one background per channel, shared by the 6 frames.
Training is solve-aware (analysis/fourpi/train_head4pi_k.py), not per-read-out NLL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode2.nn.blocks import ResBlock
from simcode2.fourpi.head4pi import norm_photons
from simcode2.fourpi.linread4pi import linear_readout
from simcode2.fourpi.windows4pi import DEFAULT

WINS_K = (('mf', 0.003), ('cr', 0.003), ('ci', 0.003), ('flat', 0.0))


class Head4PiK(nn.Module):
    def __init__(self, P=15, ch=96, nblk=4, wins=WINS_K, bf16=True, bound=3.0):
        super().__init__()
        self.P, self.bf16, self.wins, self.K, self.bound = P, bf16, tuple(wins), len(wins), bound
        cin = 24 + 24 + 4 + 8
        self.cin = nn.Conv2d(cin, ch, 3, padding=1)
        self.blk = nn.Sequential(*[ResBlock(ch, 1 + (i % 2)) for i in range(nblk)])
        self.down = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4))
        self.mlp = nn.Sequential(nn.Linear(ch * 16 + 24 + 8, 384), nn.GELU(), nn.Linear(384, 384), nn.GELU())
        self.out = nn.Linear(384, 24 * self.K + 4 + 6 + 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.lut = None          # LUT4Pi (set by the loader; not in the state dict)

    def load_v3(self, state):
        """trunk and prior rows of a Head4Pi (v3-type) checkpoint; corrections, background and covariance scale at zero"""
        own = self.state_dict()
        for k, v in state.items():
            if k in own and own[k].shape == v.shape and not k.startswith('out.'):
                own[k] = v.clone()
        n0 = 24 * self.K + 4
        own['out.weight'].zero_(); own['out.bias'].zero_()
        own['out.weight'][n0:n0 + 6] = state['out.weight'][144:150]; own['out.bias'][n0:n0 + 6] = state['out.bias'][144:150]
        self.load_state_dict(own)

    def forward(self, img, doff, learned=True):
        n, T, C, P, _ = img.shape; dev = img.device; K = self.K
        border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
        b0 = border.median(-1).values
        bm = border.mean(1).median(-1).values                                    # [n,4]
        o = torch.arange(P, device=dev, dtype=torch.float32) - P // 2
        dx = (o[None, None, None, :] - doff[:, :, 0, None, None]).expand(n, 4, P, P)
        dy = (o[None, None, :, None] - doff[:, :, 1, None, None]).expand(n, 4, P, P)
        lin = (img - b0[..., None, None]) / 50.0
        x = torch.cat([norm_photons(img).reshape(n, 24, P, P), lin.reshape(n, 24, P, P),
                       norm_photons(img.mean(1)), dx / (P // 2), dy / (P // 2)], 1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            h = self.blk(F.gelu(self.cin(x)))
            hd = self.down(h).flatten(1)
            hm = self.mlp(torch.cat([hd, norm_photons(b0).reshape(n, 24), doff.reshape(n, 8)], 1))
        q = self.out(hm.float())
        i = 24 * K
        dY = q[:, :i].reshape(n, 6, 4, K); dbg = q[:, i:i + 4]; pr = q[:, i + 4:i + 10]; cq = q[:, i + 10]
        dxy = 1.5 * torch.tanh(pr[:, :2]); z = 0.6 * torch.tanh(pr[:, 2]); ls = -6.0 + 6.0 * torch.sigmoid(pr[:, 3:6])
        if not learned:
            dY = dY * 0; dbg = dbg * 0; cq = cq * 0
        bgc = bm + 0.3 * dbg * (bm.abs() + 1).sqrt()
        p0 = torch.cat([dxy, z[:, None]], 1).detach()
        Yl, Lb, Ws, W = linear_readout(img, doff, self.lut, self.wins, p0[:, :2], p0[:, 2], bgc, return_w=True)
        sd = torch.linalg.norm(Lb, dim=-1).detach()                               # sqrt of the covariance diagonal [n,6,4,K]
        Y = Yl + sd * self.bound * torch.tanh(dY)
        cs = torch.exp(F.softplus(cq) - 0.6931)                                   # covariance scale (sd), 1 at initialisation
        return dict(Y=Y, Y_lin=Yl, Lb0=Lb.detach(), cs=cs, Lb=Lb.detach() * cs[:, None, None, None, None], Wsum=Ws, W=W, p0=p0,
                    bgc=bgc, bg=bgc[:, None].expand(n, 6, 4), dxy=dxy, z=z, ls=ls)
