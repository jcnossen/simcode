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
from simcode.layers.blocks import ResBlock
from simcode.fourpi.head4pi import Head4Pi, norm_photons
from simcode.fourpi.windows4pi import window, WINS_K

VREAD = 1.14 ** 2 + 0.2 ** 2


class Head4PiW(nn.Module):
    def __init__(self, P=15, ch=64, nblk=4, wins=WINS_K, bound=1.0, prior_ch=96, prior_nblk=4, nbr_in=False, train_prior=False):
        super().__init__()
        self.P, self.wins, self.K, self.bound = P, tuple(wins), len(wins), bound
        self.nbr_in, self.train_prior = nbr_in, train_prior   # v11: neighbour-candidate map input (4 planes); prior trained jointly
        self.prior = Head4Pi(ch=prior_ch, nblk=prior_nblk)
        for p_ in self.prior.parameters(): p_.requires_grad_(train_prior)
        cin = 4 + 4 + 4 + 8 + (4 if nbr_in else 0)
        self.cin = nn.Conv2d(cin, ch, 3, padding=1)
        self.blk = nn.Sequential(*[ResBlock(ch, 1 + (i % 2)) for i in range(nblk)])
        self.dmap = nn.Sequential(nn.Conv2d(ch, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 4 * self.K, 1))
        nn.init.zeros_(self.dmap[-1].weight); nn.init.zeros_(self.dmap[-1].bias)
        self.down = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4))
        self.mlp = nn.Sequential(nn.Linear(ch * 16 + 4, 128), nn.GELU(), nn.Linear(128, 5))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.lut = None
        self.vread = VREAD      # read variance per pixel (e^2); K x for K summed frames (set by the caller)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_prior: self.prior.eval()
        return self

    def forward(self, img, doff, learned=True, nbr=None):
        """nbr [n,4,P,P]: map of the OTHER candidates of the window in each channel patch (Gaussian sd 0.7 px stamps), frame-symmetric;
        inputs on another device than the network (patches kept in host memory) are moved to the network's device"""
        dev = self.cin.weight.device
        img, doff = img.to(dev), doff.to(dev)
        if nbr is not None: nbr = nbr.to(dev)
        n, T, C, P, _ = img.shape; K = self.K; lut = self.lut
        with torch.set_grad_enabled(self.train_prior and self.training):
            op = self.prior(img, doff)
        dxy_o, z_o, ls_o = op['dxy'], op['z'], op['ls']
        with torch.no_grad():
            dxy, z, ls = dxy_o.detach(), z_o.detach(), ls_o.detach()
            p0 = torch.cat([dxy, z[:, None]], 1)
            border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
            bm = border.mean(1).median(-1).values                                # [n,4]
            fm = img.mean(1)                                                      # [n,4,P,P] frame-symmetric
            o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
            oy, ox = torch.meshgrid(o, o, indexing='ij')
            Ws, gs, Bs = [], [], []
            for c in range(4):
                pos = torch.stack([dxy[:, 0] + doff[:, c, 0], dxy[:, 1] + doff[:, c, 1], z], 1)
                inc_, cr, ci = (lut.luts[j][c].render(pos) for j in range(3))
                inc = inc_.clamp(min=1e-7)
                ex = ox[None] - pos[:, 0, None, None]; ey = oy[None] - pos[:, 1, None, None]
                Ws.append(torch.stack([window(k, par, inc, ex, ey, (cr, ci)) for k, par in self.wins], 1))
                gs.append(inc / inc.sum((-1, -2), keepdim=True)); Bs.append(torch.stack([inc_, cr, ci], -1).reshape(n, P * P, 3))
            W = torch.stack(Ws, 1); g = torch.stack(gs, 1)                        # [n,4,K,P,P], [n,4,P,P]
            Q = torch.linalg.qr(torch.stack(Bs, 1))[0]                            # [n,4,P*P,3] orthonormal basis of the emitter's image
            cx = (ox[None, None] - doff[:, :, 0, None, None]) / (P // 2); cy = (oy[None, None] - doff[:, :, 1, None, None]) / (P // 2)
            x = torch.cat([norm_photons(fm), (fm - bm[..., None, None]) / 50.0, g / g.amax((-1, -2), keepdim=True), cx, cy], 1)
            if self.nbr_in:
                x = torch.cat([x, nbr if nbr is not None else torch.zeros(n, 4, P, P, device=dev)], 1)
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
            vp = mod + self.vread + (fm[:, None] - mod.mean(1, keepdim=True)).clamp(min=0)   # + frame-mean excess (neighbours)
        S = torch.einsum('nckuv,ncluv,ntcuv->ntckl', Wp, Wp, vp)
        S = S + 1e-6 * torch.diagonal(S, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(K, device=dev)
        Lb0 = torch.linalg.cholesky(S)
        return dict(Y=Y, Lb0=Lb0, cs=cs, Lb=Lb0 * cs[:, None, None, None, None], Wsum=Wp.sum((-1, -2)), W=Wp, p0=p0,
                    bgc=bgc, bg=bgc[:, None].expand(n, 6, 4), dxy=dxy_o, z=z_o, ls=ls_o)


def nbr_map(dx, dy, valid, doff, P=15, sd=0.7):
    """neighbour-candidate map: dx, dy [n,m] offsets (px, ch0 frame) of up to m other candidates relative to the candidate
    pixel, valid [n,m] bool, doff [n,4,2] -> [n,4,P,P] sum of Gaussian stamps at (dx, dy) + doff_c in each channel patch"""
    o = torch.arange(P, device=dx.device, dtype=torch.float32) - (P - 1) / 2
    px = dx[:, None, :] + doff[:, :, 0, None]; py = dy[:, None, :] + doff[:, :, 1, None]            # [n,4,m]
    gx = torch.exp(-0.5 * ((o[None, None, None, :] - px[..., None]) / sd) ** 2)                      # [n,4,m,P]
    gy = torch.exp(-0.5 * ((o[None, None, None, :] - py[..., None]) / sd) ** 2)
    return torch.einsum('ncmv,ncmu,nm->ncvu', gy, gx, valid.float())


def load_v10(net, state):
    """initialise a neighbour-input head (v11) from a v10 state dict: the extra input planes start at zero weight"""
    own = net.state_dict()
    for k, v in state.items():
        if k not in own: continue
        if own[k].shape == v.shape: own[k] = v.clone()
        elif k == 'cin.weight':
            own[k].zero_(); own[k][:, :v.shape[1]] = v
    net.load_state_dict(own)


def candidate_neighbours(w, x, y, m=10, r=9.0):
    """for every candidate (window w, position x, y in ch0 px; numpy arrays) the offsets to up to m other candidates of the same window
    within r px (nearest first) -> dx, dy [n,m] float32 (relative to the candidate's own position), valid [n,m] bool"""
    import numpy as np
    from scipy.spatial import cKDTree
    n = len(w); dx = np.zeros((n, m), np.float32); dy = np.zeros((n, m), np.float32); val = np.zeros((n, m), bool)
    order = np.argsort(w, kind='stable'); u, st = np.unique(w[order], return_index=True); st = np.r_[st, n]
    for i in range(len(u)):
        a = order[st[i]:st[i + 1]]
        if len(a) < 2: continue
        k = min(m + 1, len(a))
        d, j = cKDTree(np.c_[x[a], y[a]]).query(np.c_[x[a], y[a]], k=k, distance_upper_bound=r)
        d = d[:, 1:]; j = j[:, 1:]; ok = np.isfinite(d)
        jj = np.where(ok, j, 0)
        dx[a[:, None].repeat(k - 1, 1), np.arange(k - 1)[None]] = np.where(ok, x[a][jj] - x[a][:, None], 0)
        dy[a[:, None].repeat(k - 1, 1), np.arange(k - 1)[None]] = np.where(ok, y[a][jj] - y[a][:, None], 0)
        val[a[:, None].repeat(k - 1, 1), np.arange(k - 1)[None]] = ok
    return dx, dy, val
