"""The 4Pi emitter network: per candidate the T x 4 unwarped channel patches (T frames x 4 interference channels, P x P
px, one patch per channel at the mapped candidate position; 4Pi-SIMFLUX T = 6 = 2 angles x 3 steps, plain 4Pi T = 1)
and the per-channel sub-pixel offsets doff [n, 4, 2] of the patches -> K window read-outs per (frame, channel) with
their covariance, a background per channel and a pattern-free, Phi-free 3D prior. Network intensities -> explicit solve
(simcode/solve/fourpi.py); no pixel fit.

  FourPiEmitterNet   the learned-window network (v10, checkpoint arch tag 'w'):
      Y[t, c, k] = sum_px W'[c, k, px] (img[t, c, px] - bg[c]),      W' = W + D
      * W: the K canonical windows at the prior p0 (window(); the fixed read-out = teacher, D = 0 at initialisation);
      * D [4, K, P, P]: learned window change from a CNN that sees only FRAME-SYMMETRIC context (frame-mean channel
        images, the template at p0, coordinates): the read-out of frame t is linear in that frame's own pixels;
      * D is projected orthogonal to the emitter's own image components at p0 (incoherent PSF and the two 4Pi cross
        terms), so the response to the emitter is that of W for any 4Pi phase: the learned freedom only changes how
        noise, neighbours and background structure enter;
      * bounded: |D| <= bound x the window amplitude; with T = 1 the frame mean is the frame itself, so the read-out is
        no longer linear in the pixels (the solve-aware loss still trains the reported covariance);
      * covariance: analytic for the windows used (pixel variance incl. the frame-mean excess over the model), times a
        learned scale >= 1; one supervised background per channel.
      The prior (x, y, z, sigma) comes from a frozen FourPiPrior (v3L) inside it. Optional neighbour-candidate map
      input (v11: nbr_in, neighbour_map / candidate_neighbours).
  FourPiPrior        the per-candidate network with two read-outs per (frame, channel) (T = template-matched photons,
      C = Gaussian-window photons) with a 2 x 2 covariance, per-(t, c) background and the 3D prior (checkpoints without
      arch tag: v2 .. v8, v3L = the frozen prior of FourPiEmitterNet, also the candidate front of the simulated movies)
  window(kind, ...)  the canonical read-out windows ('mf', 'inc', 'gauss', 'flat', 'cr', 'ci', 'dx', 'dy', 'dz'); WINDOWS_K =
      the K = 4 windows of FourPiEmitterNet (4Pi-SIMFLUX: the lateral position comes from the SIM pattern),
      WINDOWS_PLAIN = WINDOWS_K + the three shift windows 'dx', 'dy', 'dz' (plain 4Pi: without a pattern the four
      symmetric windows carry almost no lateral information), WINDOWS_TC = ('mf', 'gauss') of the T / C solve
  linear_readout     fixed-window read-out with analytic covariance (the teacher; reference of the learned network)
  losses: nll_2x2, nll_prior (FourPiPrior training), solve_loss (solve-aware loss of FourPiEmitterNet: the explicit solve
      linearised at the truth, Gaussian NLL of its error under its reported covariance)
  load(tag_or_path) -> (net in eval mode on cuda, checkpoint dict); tags: <4Pi work dir>/models/head4pi_<tag>.pt.
      Checkpoint format {'state', 'args', 'psf'}, architecture by args['arch'] (ARCH_TAGS):
        'fourpi_emitter'  FourPiEmitterNet, T / zmax / vread from the args (since 2026-11-15; plain 4Pi T = 1)
        'fourpi_prior'    FourPiPrior, T / zmax from the args (since 2026-11-15)
      the checkpoints trained before (v10, v11: FourPiEmitterNet; v2 .. v8, v3L: FourPiPrior; T = 6) carry the earlier tags of
      util/legacy_names.FOURPI_ARCH_TAGS.
      net.lut = the 4Pi PSF spline table of the checkpoint's PSF state (z range args['zlut'], default +-0.6 um).
"""
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode.layers.blocks import ResBlock, norm_photons
from simcode.util.legacy_names import FOURPI_ARCH_TAGS

VREAD = 1.14 ** 2 + 0.2 ** 2     # read variance per pixel (e^2) of the 4Pi-SIMFLUX camera (1.14 e read noise + 0.2 e offset error)
ARCH_TAGS = {'fourpi_emitter': 'FourPiEmitterNet', 'fourpi_prior': 'FourPiPrior'}


# -------------------------------------------------------------------------------------------------------------- read-out windows
WINDOWS_TC = (('mf', 0.003), ('gauss', 1.0))
WINDOWS_K = (('mf', 0.003), ('cr', 0.003), ('ci', 0.003), ('flat', 0.0))   # the K = 4 read-outs of FourPiEmitterNet
WINDOWS_PLAIN = WINDOWS_K + (('dx', 0.003), ('dy', 0.003), ('dz', 0.003))   # plain 4Pi: + the shift read-outs (K = 7)


def needs_coh(wins):
    return any(k in ('cr', 'ci') for k, _ in wins)


def needs_grad(wins):
    return any(k in ('dx', 'dy', 'dz') for k, _ in wins)


def window(kind, par, inc, ex, ey, coh=None, grad=None):
    """inc [n,P,P] incoherent channel PSF at p (clamped positive); ex, ey [n,P,P] pixel - p offsets; coh = (cr, ci) the
    coherent cross-term components at p (only for 'cr' / 'ci'); grad = (d/dx, d/dy, d/dz) of the incoherent PSF at p
    (only for 'dx' / 'dy' / 'dz': the Poisson-weighted shift read-outs, the linear estimators of a small displacement) -> w [n,P,P]"""
    if kind in ('dx', 'dy', 'dz'):
        return grad['xyz'.index(kind[1])] / (inc + par)
    if kind in ('cr', 'ci'):   # quadrature windows on the 4Pi cross term: the image is inc + cos(Phi) cr - sin(Phi) ci, so
        # these two read the interference amplitude and phase without knowing Phi (Poisson weights 1 / (inc + beta))
        return coh[0 if kind == 'cr' else 1] / (inc + par)
    if kind == 'mf':
        u = inc / (inc + par)
        return u / (u * inc).sum((-1, -2), keepdim=True) * inc.sum((-1, -2), keepdim=True)
    if kind == 'inc':
        return inc / (inc * inc).sum((-1, -2), keepdim=True) * inc.sum((-1, -2), keepdim=True)
    if kind == 'flat':      # all pixels of the patch (with the background nuisance in the solve: the background estimator)
        return torch.ones_like(inc)
    if kind == 'gauss':
        return torch.exp(-(ex ** 2 + ey ** 2) / (2 * par ** 2))
    raise ValueError(kind)


def windows_at(lut, c, pos, wins, ox, oy):
    """all K windows of channel c at positions pos [n,3] -> [n,K,P,P], and the normalised incoherent template [n,P,P]"""
    if needs_grad(wins):
        v, gx, gy, gz = lut.luts[0][c].render(pos, True); inc = v.clamp(min=1e-7); grad = (gx, gy, gz)
    else:
        inc = lut.luts[0][c].render(pos).clamp(min=1e-7); grad = None
    coh = (lut.luts[1][c].render(pos), lut.luts[2][c].render(pos)) if needs_coh(wins) else None
    ex = ox[None] - pos[:, 0, None, None]; ey = oy[None] - pos[:, 1, None, None]
    return torch.stack([window(k, p, inc, ex, ey, coh, grad) for k, p in wins], 1), inc / inc.sum((-1, -2), keepdim=True)

# -------------------------------------------------------------------------------------------------------------- FourPiPrior
class FourPiPrior(nn.Module):
    def __init__(self, P=15, ch=96, nblk=4, wsd=1.0, bf16=True, T=6, zmax=0.6):
        """T frames per window; zmax: the prior z lies in (-zmax, zmax) um"""
        super().__init__()
        self.P, self.wsd, self.bf16, self.T, self.zmax = P, wsd, bf16, T, zmax
        TC = 4 * T
        cin = TC + TC + 4 + 8
        self.cin = nn.Conv2d(cin, ch, 3, padding=1)
        self.blk = nn.Sequential(*[ResBlock(ch, 1 + (i % 2)) for i in range(nblk)])
        self.wmap = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 8, 1))
        nn.init.zeros_(self.wmap[-1].weight); nn.init.zeros_(self.wmap[-1].bias)
        self.down = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4))
        self.mlp = nn.Sequential(nn.Linear(ch * 16 + TC + 8, 384), nn.GELU(), nn.Linear(384, 384), nn.GELU())
        self.nout = TC + TC + 3 * TC + TC + 6
        self.out = nn.Linear(384, self.nout)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, img, doff):
        """img [n,T,4,P,P] e (offset-corrected), doff [n,4,2] sub-pixel (x, y) offset of each channel's patch; inputs on another
        device than the network (patches kept in host memory) are moved to the network's device"""
        dev = self.cin.weight.device
        img, doff = img.to(dev), doff.to(dev)
        n, T, C, P, _ = img.shape
        border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
        b0 = border.median(-1).values                                            # [n,T,4]
        TC = T * C
        o = torch.arange(P, device=dev, dtype=torch.float32) - P // 2
        dx = (o[None, None, None, :] - doff[:, :, 0, None, None]).expand(n, 4, P, P)
        dy = (o[None, None, :, None] - doff[:, :, 1, None, None]).expand(n, 4, P, P)
        cx = dx / (P // 2); cy = dy / (P // 2)
        lin = (img - b0[..., None, None]) / 50.0
        x = torch.cat([norm_photons(img).reshape(n, TC, P, P), lin.reshape(n, TC, P, P),
                       norm_photons(img.mean(1)), cx, cy], 1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            h = self.blk(F.gelu(self.cin(x)))
            m1 = F.gelu(self.wmap[0](h))
            hd = self.down(h).flatten(1)
            hm = self.mlp(torch.cat([hd, norm_photons(b0).reshape(n, TC), doff.reshape(n, 8)], 1))
        m = self.wmap[2](m1.float())                                             # [n,8,P,P]
        q = self.out(hm.float())
        r2 = dx ** 2 + dy ** 2                                                    # [n,4,P,P]
        tl = getattr(self, 'tlut', None)
        if tl is None:
            g = torch.exp(-0.5 * r2 / 1.3 ** 2); g = g / g.sum((-1, -2), keepdim=True)
            r2c = r2
        else:   # (2026-11-12, opt-in, v7) 4Pi channel PSF template (phase-averaged LUT component) at the head's own prior
            ip = q[:, 6 * TC:6 * TC + 6].detach()
            px_, py_ = 1.5 * torch.tanh(ip[:, 0]), 1.5 * torch.tanh(ip[:, 1]); pz_ = self.zmax * torch.tanh(ip[:, 2])
            g = torch.stack([tl.luts[0][c].render(torch.stack([px_ + doff[:, c, 0], py_ + doff[:, c, 1], pz_], 1))
                             for c in range(4)], 1).clamp(min=0)
            g = g / g.sum((-1, -2), keepdim=True).clamp(min=1e-6)
            r2c = (o[None, None, None, :] - (px_[:, None, None, None] + doff[:, :, 0, None, None])) ** 2 \
                + (o[None, None, :, None] - (py_[:, None, None, None] + doff[:, :, 1, None, None])) ** 2
        w1 = g / (g * g).sum((-1, -2), keepdim=True) * torch.exp(m[:, :4].clamp(-5, 5))
        w1 = w1 / (w1 * g).sum((-1, -2), keepdim=True)                           # w1 . g = 1
        w2 = torch.exp(-0.5 * r2c / self.wsd ** 2) * torch.exp(m[:, 4:].clamp(-5, 5))
        i = 0
        dT = q[:, i:i + TC].reshape(n, T, 4); i += TC
        dC = q[:, i:i + TC].reshape(n, T, 4); i += TC
        ch_ = q[:, i:i + 3 * TC].reshape(n, T, 4, 3); i += 3 * TC
        dbg = q[:, i:i + TC].reshape(n, T, 4); i += TC
        pr = q[:, i:i + 6]
        bg = b0 + dbg * (b0.abs() + 1).sqrt()
        y = img - bg[..., None, None]
        Tl = torch.einsum('ncuv,ntcuv->ntc', w1, y)
        Cl = torch.einsum('ncuv,ntcuv->ntc', w2, y)
        s1 = (Tl.clamp(min=0) + bg.clamp(min=0) * 12 + 2).sqrt()
        s2 = (Cl.clamp(min=0) + bg.clamp(min=0) * 3 + 1).sqrt()
        Tq = Tl + s1 * dT
        Cq = Cl + s2 * dC
        L = torch.zeros(n, T, 4, 2, 2, device=dev)
        L[..., 0, 0] = s1 * torch.exp(ch_[..., 0].clamp(-4, 4)) * 1.5
        L[..., 1, 1] = s2 * torch.exp(ch_[..., 1].clamp(-4, 4)) * 1.5
        L[..., 1, 0] = s2 * ch_[..., 2]
        dxy = 1.5 * torch.tanh(pr[:, :2])
        z = self.zmax * torch.tanh(pr[:, 2])
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

# -------------------------------------------------------------------------------------------------------------- FourPiEmitterNet
class FourPiEmitterNet(nn.Module):
    def __init__(self, P=15, ch=64, nblk=4, wins=WINDOWS_K, bound=1.0, prior_ch=96, prior_nblk=4, nbr_in=False, train_prior=False,
                 T=6, zmax=0.6, vread=VREAD, excess=True):
        """T frames per window (the prior's input; the window network itself sees only frame-symmetric context);
        zmax: prior z range (um); vread: read variance per pixel (e^2) of the camera; excess: the pixel variance includes
        the excess of the frame-mean image over the model at the prior (neighbours; False for single frames, where that
        excess is the emitter's own mismatch at the prior position and the noise itself)"""
        super().__init__()
        self.P, self.wins, self.K, self.bound, self.T = P, tuple(wins), len(wins), bound, T
        self.nbr_in, self.train_prior = nbr_in, train_prior   # v11: neighbour-candidate map input (4 planes); prior trained jointly
        self.prior = FourPiPrior(ch=prior_ch, nblk=prior_nblk, T=T, zmax=zmax)
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
        self.vread = vread      # read variance per pixel (e^2); K x for K summed frames (set by the caller)
        self.excess = excess
        self.use_cs = True      # False: the learned window change and background with the analytic covariance (scale 1)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_prior: self.prior.eval()
        return self

    def forward(self, img, doff, learned=True, nbr=None, at=None):
        """nbr [n,4,P,P]: map of the OTHER candidates of the window in each channel patch (Gaussian sd 0.7 px stamps), frame-symmetric;
        at [n,3] (x, y px, z um): place the windows there instead of at the prior (a refinement pass at a solved position;
        the prior outputs are returned unchanged); inputs on another device than the network (patches kept in host memory)
        are moved to the network's device"""
        dev = self.cin.weight.device
        img, doff = img.to(dev), doff.to(dev)
        if nbr is not None: nbr = nbr.to(dev)
        n, T, C, P, _ = img.shape; K = self.K; lut = self.lut
        with torch.set_grad_enabled(self.train_prior and self.training):
            op = self.prior(img, doff)
        dxy_o, z_o, ls_o = op['dxy'], op['z'], op['ls']
        with torch.no_grad():
            dxy, z, ls = dxy_o.detach(), z_o.detach(), ls_o.detach()
            if at is not None: dxy, z = at[:, :2].to(dev), at[:, 2].to(dev)
            p0 = torch.cat([dxy, z[:, None]], 1)
            border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
            bm = border.mean(1).median(-1).values                                # [n,4]
            fm = img.mean(1)                                                      # [n,4,P,P] frame-symmetric
            o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
            oy, ox = torch.meshgrid(o, o, indexing='ij')
            Ws, gs, Bs = [], [], []
            for c in range(4):
                pos = torch.stack([dxy[:, 0] + doff[:, c, 0], dxy[:, 1] + doff[:, c, 1], z], 1)
                if needs_grad(self.wins):
                    inc_, *grad = lut.luts[0][c].render(pos, True); cr, ci = (lut.luts[j][c].render(pos) for j in (1, 2))
                else:
                    inc_, cr, ci = (lut.luts[j][c].render(pos) for j in range(3)); grad = None
                inc = inc_.clamp(min=1e-7)
                ex = ox[None] - pos[:, 0, None, None]; ey = oy[None] - pos[:, 1, None, None]
                Ws.append(torch.stack([window(k, par, inc, ex, ey, (cr, ci), grad) for k, par in self.wins], 1))
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
            cs = 1.0 + F.softplus(cq - 3.0) if self.use_cs else torch.ones(n, device=dev)
        else:
            Wp = W; bgc = bm; cs = torch.ones(n, device=dev)
        y = img - bgc[:, None, :, None, None]
        Y = torch.einsum('nckuv,ntcuv->ntck', Wp, y)
        with torch.no_grad():
            Y0 = torch.einsum('ncuv,ntcuv->ntc', W[:, :, 0], y).clamp(min=0)      # photons per (t, c) through the canonical window
            mod = Y0[..., None, None] * g[:, None] + bgc[:, None, :, None, None].clamp(min=0.05)
            vp = mod + self.vread
            if self.excess: vp = vp + (fm[:, None] - mod.mean(1, keepdim=True)).clamp(min=0)   # + frame-mean excess (neighbours)
        S = torch.einsum('nckuv,ncluv,ntcuv->ntckl', Wp, Wp, vp)
        S = S + 1e-6 * torch.diagonal(S, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(K, device=dev)
        Lb0 = torch.linalg.cholesky(S)
        return dict(Y=Y, Lb0=Lb0, cs=cs, Lb=Lb0 * cs[:, None, None, None, None], Wsum=Wp.sum((-1, -2)), W=Wp, p0=p0,
                    bgc=bgc, bg=bgc[:, None].expand(n, T, 4), dxy=dxy_o, z=z_o, ls=ls_o)


def neighbour_map(dx, dy, valid, doff, P=15, sd=0.7):
    """neighbour-candidate map: dx, dy [n,m] offsets (px, ch0 frame) of up to m other candidates relative to the candidate
    pixel, valid [n,m] bool, doff [n,4,2] -> [n,4,P,P] sum of Gaussian stamps at (dx, dy) + doff_c in each channel patch"""
    o = torch.arange(P, device=dx.device, dtype=torch.float32) - (P - 1) / 2
    px = dx[:, None, :] + doff[:, :, 0, None]; py = dy[:, None, :] + doff[:, :, 1, None]            # [n,4,m]
    gx = torch.exp(-0.5 * ((o[None, None, None, :] - px[..., None]) / sd) ** 2)                      # [n,4,m,P]
    gy = torch.exp(-0.5 * ((o[None, None, None, :] - py[..., None]) / sd) ** 2)
    return torch.einsum('ncmv,ncmu,nm->ncvu', gy, gx, valid.float())


def init_from_v10(net, state):
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

# -------------------------------------------------------------------------------------------------------------- fixed-window read-out (teacher / reference)
def linear_readout(img, doff, lut, wins, xy, z, bgc, gain=None, vread=VREAD, return_w=False):
    """img [n,T,4,P,P] e; doff [n,4,2]; xy [n,2] px, z [n] um (window position); bgc [n,4] e/px; gain [n,4,K] or None
    -> Y [n,T,4,K], Lb [n,T,4,K,K] (lower Cholesky), Wsum [n,4,K] (window sums: a residual background b = true - bgc adds b Wsum to Y)"""
    n, T, C, P, _ = img.shape; dev = img.device; K = len(wins)
    o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
    oy, ox = torch.meshgrid(o, o, indexing='ij')
    Y = torch.zeros(n, T, C, K, device=dev); S = torch.zeros(n, T, C, K, K, device=dev); Ws = torch.zeros(n, C, K, device=dev)
    Wall = []
    for c in range(C):
        pos = torch.stack([xy[:, 0] + doff[:, c, 0], xy[:, 1] + doff[:, c, 1], z], 1)
        w, g = windows_at(lut, c, pos, wins, ox, oy)                               # [n,K,P,P], [n,P,P]
        if gain is not None:
            w = w * gain[:, c, :, None, None]
        y = img[:, :, c] - bgc[:, None, c, None, None]                             # [n,T,P,P]
        Yc = torch.einsum('nkuv,ntuv->ntk', w, y)
        vp = Yc[..., 0].clamp(min=0)[..., None, None] * g[:, None] + bgc[:, None, c, None, None].clamp(min=0.05) + vread
        Y[:, :, c] = Yc; Ws[:, c] = w.sum((-1, -2))
        if return_w: Wall.append(w)
        S[:, :, c] = torch.einsum('nkuv,nluv,ntuv->ntkl', w, w, vp)
    S = S + 1e-6 * torch.diagonal(S, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(K, device=dev)
    if return_w:
        return Y, torch.linalg.cholesky(S), Ws, torch.stack(Wall, 1)
    return Y, torch.linalg.cholesky(S), Ws


def border_bg(img):
    """[n,4]: median over the border ring of the frame-mean image per channel (one background per channel)"""
    border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
    return border.mean(1).median(-1).values

# -------------------------------------------------------------------------------------------------------------- solve-aware loss
def solve_loss(sv, o, S, Ystar, sb=0.2, clamp=50.0, grad_model=False):
    """sv: FourPiSolve (pattern k of the simulator; plain 4Pi: k None, frame angles sv.ang); o: FourPiEmitterNet output;
    S: FourPiPatchSim sample (or MultiChannelAcquisitionSim.candidate_patches); Ystar [n,T,4,K] targets.
    -> nll [n], dict(u [n,3] = dtheta / reported sigma, r2 [n] = mean whitened read-out residual^2)"""
    from simcode.solve.fourpi import psi_from_eps
    n, T = Ystar.shape[:2]; dev = Ystar.device; K = Ystar.shape[-1]
    with torch.set_grad_enabled(grad_model):   # grad_model: learned windows (FourPiEmitterNet) -> the Jacobian and covariance carry gradients
        sv._ctx = dict(W=o['W'], p0=None, Wsum=None)
        A, _ = sv.readout_k(S['x'], S['y'], S['z'], S['Phi0'], S['doff'])
        A = A[:, torch.tensor(sv.ang, device=dev)]                                # [n,T,4,K,5]
        Pt, dPx, dPy = sv.pattern(S['x'], S['y'], S['m'], psi_from_eps(S['eps']) if sv.k is not None else None)
        Fq = (S['F'][:, None] * S['q'])                                           # [n,T]
        A0 = A[..., 0]
        J = torch.zeros(n, T, 4, K, 8, device=dev)
        J[..., 0] = (Fq * dPx)[:, :, None, None] * A0 + (Fq * Pt)[:, :, None, None] * A[..., 1]
        J[..., 1] = (Fq * dPy)[:, :, None, None] * A0 + (Fq * Pt)[:, :, None, None] * A[..., 2]
        J[..., 2] = (Fq * Pt)[:, :, None, None] * A[..., 3]
        J[..., 3] = (S['q'] * Pt)[:, :, None, None] * A0
        for c in range(4):
            J[:, :, c, :, 4 + c] = o['Wsum'][:, None, c, :]
        Ju = torch.linalg.solve_triangular(o['Lb0'], J, upper=False).reshape(n, -1, 8)
        JJ = torch.einsum('nmp,nmq->npq', Ju, Ju)
        sp = torch.exp(o['ls'])
        bt = S['bg'][:, 0].mean((-1, -2))
    bstar = bt - o['bgc']                                                         # true residual background [n,4]
    r0 = o['Y'] - Ystar - (bstar[:, :, None] * o['Wsum'])[:, None]
    ru = torch.linalg.solve_triangular(o['Lb0'], r0[..., None], upper=False)[..., 0].reshape(n, -1)
    w = 1.0 / o['cs'] ** 2
    Pp = torch.zeros(n, 8, device=dev); Pp[:, :3] = 1.0 / sp.detach() ** 2; Pp[:, 4:] = 1.0 / sb ** 2
    H = JJ * w[:, None, None] + torch.diag_embed(Pp) + 1e-8 * torch.eye(8, device=dev)
    g = torch.einsum('nmp,nm->np', Ju, ru) * w[:, None]
    ep = torch.stack([o['dxy'][:, 0] - S['x'], o['dxy'][:, 1] - S['y'], o['z'] - S['z']], 1).detach()
    g = g + torch.cat([ep * Pp[:, :3], torch.zeros(n, 1, device=dev), -bstar * Pp[:, 4:]], 1)
    Hi = torch.linalg.inv(H)
    dth = torch.einsum('npq,nq->np', Hi, g)[:, :3]
    var = torch.diagonal(Hi, dim1=1, dim2=2)[:, :3].clamp(min=1e-12)
    u2 = (dth ** 2 / var)
    nll = 0.5 * (u2.clamp(max=clamp) + torch.log(var)).sum(1)
    return nll, dict(u=(dth / var.sqrt()).detach(), r2=(ru ** 2).mean(1) * w, dth=dth.detach(), var=var.detach())

# -------------------------------------------------------------------------------------------------------------- checkpoints
def checkpoint_path(tag_or_path):
    from simcode.config import root
    return tag_or_path if os.path.sep in tag_or_path else f"{root('fourpi')}/models/head4pi_{tag_or_path}.pt"


def load(tag_or_path, dev='cuda'):
    """4Pi emitter-network checkpoint (path, or a tag of <4Pi work dir>/models/head4pi_<tag>.pt) -> (net in eval mode,
    checkpoint dict), by the architecture tag args['arch']"""
    from simcode.fourpi.spline_table import LUT4Pi
    f = checkpoint_path(tag_or_path)
    ck = torch.load(f, weights_only=False)
    a = ck['args']
    cls = ARCH_TAGS.get(a.get('arch')) or FOURPI_ARCH_TAGS.get(a.get('arch'))
    if cls is None:
        raise ValueError(f"{f}: 4Pi emitter network arch {a.get('arch')!r} is not supported ({list(ARCH_TAGS)})")
    tg = dict(T=a.get('T', 6), zmax=a.get('zmax', 0.6)) if a.get('arch') in ('fourpi_emitter', 'fourpi_prior') else {}   # T = 6 tags: the defaults
    zl = dict(zip(('zmin', 'zmax'), a['zlut'])) if a.get('zlut') else {}
    if cls == 'FourPiEmitterNet':
        if a.get('arch') == 'fourpi_emitter': tg.update(vread=a['vread'], excess=a.get('excess', True))
        net = FourPiEmitterNet(ch=a['ch'], nblk=a['nblk'], wins=tuple(tuple(w) for w in a['wins']), bound=a.get('bound', 1.0), nbr_in=a.get('nbr_in', False), **tg).to(dev).eval()
        net.load_state_dict(ck['state']); net.lut = LUT4Pi(ck['psf'], P=15, dev=dev, **zl)
        return net, ck
    net = FourPiPrior(ch=a['ch'], nblk=a['nblk'], **tg).to(dev).eval()
    net.load_state_dict(ck['state'])
    if a.get('template') == 'lut':
        net.tlut = LUT4Pi(ck['psf'], P=15, sub=1, dev=dev, **zl)
    return net, ck
