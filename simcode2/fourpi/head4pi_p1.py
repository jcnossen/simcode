"""4Pi pass-1 head with the SIMCODE v2 (SCHead4 / sc5nb) additions, on top of head4pi (v3: pattern-phase-free prior):
  - neighbour-aware: a splat of the other candidates (p-weighted Gaussian blobs, sd 1 px, in ch0 patch coordinates) and
    the detector-probability map around the candidate as 2 extra input planes (zero-initialised: starts as v3);
  - covariance of the 48 read-outs (T, C per frame and channel): the per-(t, c) 2 x 2 Cholesky blocks of v3 plus a
    rank-4 term U U^T over all 48 (frame-shared and channel-shared errors, e.g. position and background errors);
  - training losses: Gaussian NLL with that covariance, the 3D prior NLL, and the sc4 mean-removed anti-shrinkage term
    on the frame profile of T per channel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode2.fourpi.head4pi import Head4Pi, norm_photons

RANK = 4


class Head4PiP1(Head4Pi):
    def __init__(self, P=15, ch=96, nblk=4, wsd=1.0, bf16=True, rank=RANK, bg_mode='frame'):
        super().__init__(P=P, ch=ch, nblk=nblk, wsd=wsd, bf16=bf16)
        self.rank = rank
        # bg_mode 'angle' (v4, CRLB decomposition 00:10): background per (angle, channel) = border median pooled over the 3
        # frames of the angle (2-px ring) + a learned correction shared by those frames; 'frame' = per (t, c) as v3
        self.bg_mode = bg_mode
        # template (v5, 2026-11-12): None = Gaussian sd 1.3 px at the candidate (v3 / v4). A LUT4Pi object = the 4Pi channel
        # PSF template (phase-averaged component) at the head's own prior (x, y, z), detached, as in sc4 where the learned
        # linear read-out is anchored on the calibrated PSF template; the C window is centred at the prior position.
        # Not part of the state dict: set head.tlut after loading.
        self.tlut = None
        old = self.cin
        self.cin = nn.Conv2d(old.in_channels + 2, old.out_channels, 3, padding=1)
        with torch.no_grad():
            self.cin.weight.zero_(); self.cin.weight[:, :old.in_channels] = old.weight; self.cin.bias.copy_(old.bias)
        oo = self.out
        self.out = nn.Linear(oo.in_features, oo.out_features + 48 * rank)
        with torch.no_grad():
            self.out.weight.zero_(); self.out.bias.zero_()
            self.out.weight[:oo.out_features] = oo.weight; self.out.bias[:oo.out_features] = oo.bias

    @classmethod
    def from_v3(cls, ck, **kw):
        a = ck['args']
        base = Head4Pi(ch=a['ch'], nblk=a['nblk']); base.load_state_dict(ck['state'])
        net = cls(ch=a['ch'], nblk=a['nblk'], **kw)
        sd = net.state_dict()
        for k, v in base.state_dict().items():
            if k in ('cin.weight', 'out.weight', 'out.bias'): continue
            sd[k] = v
        sd['cin.weight'][:, :base.cin.in_channels] = base.cin.weight; sd['cin.bias'] = base.cin.bias
        no = base.out.out_features
        sd['out.weight'][:no] = base.out.weight; sd['out.bias'][:no] = base.out.bias
        net.load_state_dict(sd)
        return net

    def forward(self, img, doff, extra=None):
        """img [n,6,4,P,P] e, doff [n,4,2], extra [n,2,P,P] (neighbour splat, detector probability) or None"""
        n, T, C, P, _ = img.shape
        dev = img.device
        if extra is None:
            extra = torch.zeros(n, 2, P, P, device=dev)
        border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
        if getattr(self, 'bg_mode', 'frame') == 'angle':
            ring = torch.cat([img[..., :2, :].flatten(-2), img[..., -2:, :].flatten(-2), img[..., 2:-2, :2].flatten(-2), img[..., 2:-2, -2:].flatten(-2)], -1)
            ba = ring.reshape(n, 2, 3, C, -1).permute(0, 1, 3, 2, 4).reshape(n, 2, C, -1).median(-1).values    # [n,2,C]
            b0 = ba[:, [0, 0, 0, 1, 1, 1]]                                                                    # [n,6,C]
        else:
            b0 = border.median(-1).values
        o = torch.arange(P, device=dev, dtype=torch.float32) - P // 2
        dx = (o[None, None, None, :] - doff[:, :, 0, None, None]).expand(n, 4, P, P)
        dy = (o[None, None, :, None] - doff[:, :, 1, None, None]).expand(n, 4, P, P)
        cx = dx / (P // 2); cy = dy / (P // 2)
        lin = (img - b0[..., None, None]) / 50.0
        x = torch.cat([norm_photons(img).reshape(n, 24, P, P), lin.reshape(n, 24, P, P),
                       norm_photons(img.mean(1)), cx, cy, extra], 1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            h = self.blk(F.gelu(self.cin(x)))
            m1 = F.gelu(self.wmap[0](h))
            hd = self.down(h).flatten(1)
            hm = self.mlp(torch.cat([hd, norm_photons(b0).reshape(n, 24), doff.reshape(n, 8)], 1))
        m = self.wmap[2](m1.float())
        q = self.out(hm.float())
        if getattr(self, 'tlut', None) is None:
            r2 = dx ** 2 + dy ** 2
            g = torch.exp(-0.5 * r2 / 1.3 ** 2); g = g / g.sum((-1, -2), keepdim=True)
            r2c = r2
        else:
            # prior (x, y, z) from this forward (detached) -> per-channel 4Pi template (phase-averaged LUT component)
            ip = q[:, 24 * 3 + 72:24 * 3 + 72 + 6].detach()      # the prior outputs (dT 24, dC 24, chol 72, dbg 24 come first)
            px_, py_ = 1.5 * torch.tanh(ip[:, 0]), 1.5 * torch.tanh(ip[:, 1]); pz_ = 0.6 * torch.tanh(ip[:, 2])
            gl = []
            for c in range(4):
                pos = torch.stack([px_ + doff[:, c, 0], py_ + doff[:, c, 1], pz_], 1)
                gl.append(self.tlut.luts[0][c].evaluate(pos))
            g = torch.stack(gl, 1).clamp(min=0); g = g / g.sum((-1, -2), keepdim=True).clamp(min=1e-6)
            r2c = (o[None, None, None, :] - (px_[:, None, None, None] + doff[:, :, 0, None, None])) ** 2 \
                + (o[None, None, :, None] - (py_[:, None, None, None] + doff[:, :, 1, None, None])) ** 2
        w1 = g / (g * g).sum((-1, -2), keepdim=True) * torch.exp(m[:, :4].clamp(-5, 5))
        w1 = w1 / (w1 * g).sum((-1, -2), keepdim=True)
        w2 = torch.exp(-0.5 * r2c / self.wsd ** 2) * torch.exp(m[:, 4:].clamp(-5, 5))
        i = 0
        dT = q[:, i:i + 24].reshape(n, 6, 4); i += 24
        dC = q[:, i:i + 24].reshape(n, 6, 4); i += 24
        ch_ = q[:, i:i + 72].reshape(n, 6, 4, 3); i += 72
        dbg = q[:, i:i + 24].reshape(n, 6, 4); i += 24
        pr = q[:, i:i + 6]; i += 6
        U = q[:, i:i + 48 * self.rank].reshape(n, 48, self.rank)
        if getattr(self, 'bg_mode', 'frame') == 'angle':
            dbg = dbg.reshape(n, 2, 3, C).mean(2)[:, [0, 0, 0, 1, 1, 1]]
        bg = b0 + dbg * (b0.abs() + 1).sqrt()
        y = img - bg[..., None, None]
        Tl = torch.einsum('ncuv,ntcuv->ntc', w1, y)
        Cl = torch.einsum('ncuv,ntcuv->ntc', w2, y)
        s1 = (Tl.clamp(min=0) + bg.clamp(min=0) * 12 + 2).sqrt()
        s2 = (Cl.clamp(min=0) + bg.clamp(min=0) * 3 + 1).sqrt()
        L = torch.zeros(n, 6, 4, 2, 2, device=dev)
        L[..., 0, 0] = s1 * torch.exp(ch_[..., 0].clamp(-4, 4)) * 1.5
        L[..., 1, 1] = s2 * torch.exp(ch_[..., 1].clamp(-4, 4)) * 1.5
        L[..., 1, 0] = s2 * ch_[..., 2]
        scale = torch.stack([s1, s2], -1).reshape(n, 48)                      # read-out order: (t, c, [T, C])
        U = U * scale[..., None] * 0.5
        return dict(T=Tl + s1 * dT, C=Cl + s2 * dC, L=L, U=U, bg=bg, dxy=1.5 * torch.tanh(pr[:, :2]),
                    z=0.6 * torch.tanh(pr[:, 2]), ls=-6.0 + 6.0 * torch.sigmoid(pr[:, 3:6]), T_lin=Tl, C_lin=Cl)


def full_cov(o):
    """Sigma [n,48,48] = blockdiag of the per-(t, c) 2 x 2 blocks + U U^T (read-out order (t, c, [T, C]))"""
    L = o['L']; n = L.shape[0]
    B = L @ L.transpose(-1, -2)                                               # [n,6,4,2,2]
    S = torch.zeros(n, 48, 48, device=L.device)
    idx = torch.arange(24, device=L.device) * 2
    Bf = B.reshape(n, 24, 2, 2)
    for i in range(2):
        for j in range(2):
            S[:, idx + i, idx + j] = Bf[..., i, j]
    if 'U' in o and o['U'] is not None:
        S = S + o['U'] @ o['U'].transpose(-1, -2)
    return S


def nll_lowrank(Tg, Cg, o):
    """Gaussian NLL of the 48 read-outs with the full covariance (per candidate)"""
    y = torch.stack([Tg, Cg], -1).reshape(Tg.shape[0], 48)
    mu = torch.stack([o['T'], o['C']], -1).reshape(Tg.shape[0], 48)
    S = full_cov(o)
    Lc = torch.linalg.cholesky(S + 1e-3 * torch.eye(48, device=S.device))
    z = torch.linalg.solve_triangular(Lc, (y - mu)[..., None], upper=False)[..., 0]
    return 0.5 * (z ** 2).sum(1) + torch.log(torch.diagonal(Lc, dim1=1, dim2=2)).sum(1)


def mean_removed(Tg, o):
    """sc4 anti-shrinkage term: frame profile of T per channel (mean removed), normalised by sqrt(photons)"""
    cn = (Tg.clamp(min=0).mean(1, keepdim=True) + 10).sqrt()
    mr = ((o['T'] - o['T'].mean(1, keepdim=True)) - (Tg - Tg.mean(1, keepdim=True))) / cn
    return F.huber_loss(mr, torch.zeros_like(mr), delta=3.0, reduction='none').mean((1, 2))


def slope_penalty(Tg, o):
    """sc4 modulation-slope penalty: the frame profile of T (mean removed per channel) regressed on the GT profile must
    have slope 1 (no shrinkage of the pattern modulation)"""
    am = o['T'] - o['T'].mean(1, keepdim=True); gm = Tg - Tg.mean(1, keepdim=True)
    return (1 - (am * gm).sum() / (gm * gm).sum().clamp(min=1e-6)) ** 2
