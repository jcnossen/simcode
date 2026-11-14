"""
sc4d / sc4n (critic review 29 #1, #3, #6, #7). Same generator, detector matching and 100k OneCycle schedule as sc2L
(sc3.train --arch sc2); changes:
  DATA (both arms): density log-uniform 0.0005-0.04 /px, per-window median brightness log-uniform 40-2500 (per-emitter
        draws unchanged), pair_frac 0.25, per-target loss weights so the nn bins < 1.5 / 1.5-2.5 / 2.5-4 / > 4 px carry
        0.15 / 0.25 / 0.25 / 0.35 of each window's weight (window weights sum to its target count; empty bins are
        renormalised away).
  --arch sc2  (sc4d): sc2 head + losses unchanged.
  --arch sc4n (sc4n): SCHead4 = sc2 head with
        (a) +2 input channels: a splat of the OTHER detected candidates (detector p x Gaussian sigma 0.5 px at their
            sub-pixel points; candidates p >= 0.2 after NMS, rendered once per window, cropped with the patch gather,
            the candidate's own stamp removed exactly) and the cropped detector-probability map (pattern-free);
        (b) signed per-frame readout w_t = w0 exp(m) / (.. g) + D_t, D_t = 6 signed maps (zero-initialised conv) with
            the g-component projected out, so w_t . g = 1 for every frame;
        bg per frame as sc2 (not detached), sc2 losses; no scale term, no dI L2. bf16 autocast for the CNN trunk, fp32
        for the output layers, the readout and the Cholesky / NLL.
3D (2026-11-14): the position prior is x, y, z with a 3 x 3 covariance (SCHead4 doc); loss + nll_z (z | x, y; the x, y prior
loss is unchanged); training emitters over the TIRF height range (hd/data.Z_RANGE_UM); 2D checkpoints load with zero z rows
(pad_out), so --init hd_sc5nb.pt starts at exactly its 2D outputs.
Held-out (critic 29 #6): gate-type scenes with their own seeds (iso / pair2 / rs25x, seed 401), multi-emitter cov_or
(crlb_all) per target -> projected efficiency (median of (w.e)^2 / w C_or w over chi2_1 median) and Sigma_net
calibration along the B-profiled phase-gradient directions, nll_I, slope, I_lin / I_gt and |I - I_lin| / I_gt, prior z,
per nn bin; prepared once (stage 'prep'), evaluated every 2k steps from stored patches.
usage: python -m simcode.hd.sc4 prep
       CUDA_VISIBLE_DEVICES=1 python -m simcode.hd.sc4 train --arch sc2 --out runs/hd_sc4d
       CUDA_VISIBLE_DEVICES=2 python -m simcode.hd.sc4 train --arch sc4n --out runs/hd_sc4n
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from simcode.util.ckpt import save as ckpt_save     # atomic checkpoint write with a disk guard
from simcode.nn.blocks import ResBlock
from simcode.hd.sc import PSFCol, patches, mod_slope, PP, norm_photons
from simcode import config

dev = 'cuda'
HOLD = os.path.join(config.root('out'), 'sc4', 'heldout4.pt')      # stage 'prep' output (not shipped: 0.2-0.3 GB)
HOLD5 = os.path.join(config.root('out'), 'sc4', 'heldout5.pt')      # iteration 30b: det_n1 candidates, + rs05p / rs25p, >= 150 ph
SC5_SCEN = (('iso', 13000), ('pair2', 5000), ('rs25x', 5000), ('rs05p', 6000), ('rs25p', 6000))
PHB = ((200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 1e9))
NBR_PMIN = 0.2          # candidates in the neighbour splat (= the gate detection threshold THR0)
SIG_SPLAT = 0.5
NN_EDGES = (1.5, 2.5, 4.0)
NN_W = (0.15, 0.25, 0.25, 0.35)
OFFS = torch.arange(-3, 4)


# ------------------------------------------------------------------------------------------------ neighbour inputs
def nbr_map(logit, b, iy, ix, x, y, lg):
    """logit [B,H,W] detector map; kept candidates (b, iy, ix, x, y, lg) -> [B,2,H,W]: (splat of all candidates,
    detector probability). Each candidate's stamp: p exp(-d^2 / 2 sig^2) on the 7 x 7 pixels around its NMS pixel."""
    B, H, W = logit.shape
    p = torch.sigmoid(lg.float())
    o = OFFS.to(logit.device)
    py = iy[:, None, None] + o[None, :, None]; px = ix[:, None, None] + o[None, None, :]
    v = p[:, None, None] * torch.exp(-((px - x[:, None, None]) ** 2 + (py - y[:, None, None]) ** 2) / (2 * SIG_SPLAT ** 2))
    ok = (py >= 0) & (py < H) & (px >= 0) & (px < W)
    bb = b[:, None, None].expand_as(py)
    sp = torch.zeros(B * H * W, device=logit.device)
    sp.index_put_((((bb * H + py) * W + px)[ok],), v[ok], accumulate=True)
    return torch.stack([sp.reshape(B, H, W), torch.sigmoid(logit.float())], 1)


def own_stamp(off, p, iy, ix, H, W, P=PP):
    """the candidate's own splat stamp in patch coordinates [n,P,P] (exactly as rendered by nbr_map)"""
    n = len(p); h = P // 2
    o = OFFS.to(off.device)
    v = p[:, None, None] * torch.exp(-((o[None, None, :] - off[:, 0, None, None]) ** 2 + (o[None, :, None] - off[:, 1, None, None]) ** 2)
                                     / (2 * SIG_SPLAT ** 2))
    ok = ((iy[:, None, None] + o[None, :, None]) >= 0) & ((iy[:, None, None] + o[None, :, None]) < H) & \
         ((ix[:, None, None] + o[None, None, :]) >= 0) & ((ix[:, None, None] + o[None, None, :]) < W)
    st = torch.zeros(n, P, P, device=off.device)
    st[:, h - 3:h + 4, h - 3:h + 4] = v * ok
    return st


def nbr_patches(nbr, b, iy, ix, x, y, p):
    """crop the neighbour maps at the candidates and remove the candidate's own stamp from the splat channel"""
    q = patches(nbr, b, iy, ix)
    off = torch.stack([x - ix.float(), y - iy.float()], 1)
    st = own_stamp(off, p.float(), iy, ix, nbr.shape[2], nbr.shape[3])
    return torch.stack([(q[:, 0] - st).clamp(min=0), q[:, 1]], 1)


# ------------------------------------------------------------------------------------------------ head
class SCHead4(nn.Module):
    """SCHead + neighbour channels + signed per-frame readout residual (see module doc).
    3D position prior (2026-11-14): x, y, z with a 3 x 3 covariance. Output layer q = [dI T | log diag L_I T | off-diag L_I |
    dx, dy | log sx, log sy | bg T | z, log s_z|xy, b_x, b_y] (the last 4 rows appended: a 2D checkpoint loads with zero rows,
    pad_out, and gives exactly its 2D outputs). Covariance in Cholesky form with the order (x, y, z):
        L_p = [[sx, 0, 0], [0, sy, 0], [b_x sx, b_y sy, s_z|xy]]      Sigma_p = L_p L_p^T
    i.e. x and y uncorrelated (as the 2D prior and the solve's prior), z | x, y = z + b_x (x' - x) + b_y (y' - y) with sd
    s_z|xy. z in px of the lateral pixel size relative to the focal plane of the PSF calibration (hd/data.z_target_px).
    The NLL of the 3D Gaussian factorizes into the 2D x, y prior NLL (unchanged) + the NLL of z | x, y (nll_z)."""
    nbr = True
    NZ = 4      # appended z outputs

    def __init__(self, T=6, ch=96, nblk=4, P=21, bf16=True):
        super().__init__()
        self.T, self.P, self.bf16 = T, P, bf16
        cin = 2 * T + 4 + 2
        self.cin = nn.Conv2d(cin, ch, 3, padding=1)
        self.blk = nn.Sequential(*[ResBlock(ch, 1 + (i % 2)) for i in range(nblk)])
        self.wmap = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.wmap[-1].weight); nn.init.zeros_(self.wmap[-1].bias)
        self.dmap = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, T, 1))
        nn.init.zeros_(self.dmap[-1].weight); nn.init.zeros_(self.dmap[-1].bias)
        self.down = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(),
                                  nn.AdaptiveAvgPool2d(4))
        nsc = 2 * T + 2
        self.mlp = nn.Sequential(nn.Linear(ch * 16 + nsc, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        nt = T * (T + 1) // 2
        self.out = nn.Linear(256, T + nt + 4 + T + self.NZ)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.tril = torch.tril_indices(T, T, -1)

    def trunk(self, x, b0, off, illp=None):
        """bf16 CNN trunk + MLP (the autocast region of forward) -> m1, d1 (bf16 maps before the fp32 1x1 read-outs), hm.
        Separate method (2026-11-11, speed branch; identical ops) so that the fast path can torch.compile this part only."""
        n, T = x.shape[0], self.T
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            hmap = self.blk(F.gelu(self.cin(x) if illp is None else self.cin(x) + self.cin_il(illp)))
            m1 = F.gelu(self.wmap[0](hmap)); d1 = F.gelu(self.dmap[0](hmap))
            h = self.down(hmap).flatten(1)
            sc = torch.cat([norm_photons(b0), off, torch.zeros(n, T, device=x.device)], 1)
            hm = self.mlp(torch.cat([h, sc], 1))
        return m1, d1, hm

    def _diag_idx(self, dev):
        """(2026-11-11) device copies of the Cholesky index tensors: CPU index tensors in the index_put below forced a
        blocking host-to-device copy (a host sync) in every forward; same values -> identical output"""
        c = getattr(self, '_idx_cache', None)
        if c is None or c[0] != dev:
            c = (dev, torch.arange(self.T, device=dev), self.tril[0].to(dev), self.tril[1].to(dev)); self._idx_cache = c
        return c[1:]

    def forward(self, pt, rn2p, off, psf=None, nbrp=None, illp=None, trunk_fn=None):
        """illp: the pass-2 illumination patches (SCHeadP2; None for pass 1)"""
        n, T, P, _ = pt.shape
        flat = pt.reshape(n, T, -1)
        border = torch.cat([pt[:, :, 0], pt[:, :, -1], pt[:, :, 1:-1, 0], pt[:, :, 1:-1, -1]], -1)
        b0 = border.median(-1).values
        lvl = b0.mean(1)[:, None, None, None]
        o = torch.arange(P, device=pt.device, dtype=torch.float32) - P // 2
        dx = o[None, None, :] - off[:, 0, None, None]; dy = o[None, :, None] - off[:, 1, None, None]
        cx = dx.expand(n, P, P) / (P // 2); cy = dy.expand(n, P, P) / (P // 2)
        lin = (pt - lvl) / 100.0
        x = torch.cat([norm_photons(pt), norm_photons(pt.mean(1, keepdim=True)), lin, lin.mean(1, keepdim=True),
                       cx[:, None], cy[:, None], nbrp], 1)
        m1, d1, hm = (trunk_fn or self.trunk)(x, b0, off, illp)
        m = self.wmap[2](m1.float())[:, 0]; d = self.dmap[2](d1.float())
        q = self.out(hm.float())
        if psf is not None:
            g = psf
        else:
            g = torch.exp(-0.5 * (dx ** 2 + dy ** 2) / 1.3 ** 2).expand(n, P, P)
            g = 0.7 * g / g.sum((1, 2), keepdim=True)
        gg = (g * g).sum((1, 2), keepdim=True)
        w0 = g / gg
        w = w0 * torch.exp(m.clamp(-5, 5))
        w = w / (w * g).sum((1, 2), keepdim=True)
        # signed per-frame residual, g-component projected out -> w_t . g = 1
        D = d * w0.amax((1, 2), keepdim=True)[:, None]                         # [n,T,P,P] in units of the template peak
        D = D - ((D * g[:, None]).sum((2, 3), keepdim=True) / gg[:, None]) * g[:, None]
        wt = w[:, None] + D
        nt = T * (T + 1) // 2
        k = T + nt + 4
        bg = b0 + q[:, k:k + T] * (b0.abs() + 1).sqrt()
        Ilin = (wt.reshape(n, T, -1) * (flat - bg[..., None])).sum(-1)
        c = (Ilin.clamp(min=0).sum(1, keepdim=True) / T + bg.clamp(min=0).mean(1, keepdim=True) * 9).sqrt().clamp(min=3.0)
        dI = q[:, :T]
        ld = q[:, T:2 * T]; od = q[:, 2 * T:T + nt]
        L = torch.zeros(n, T, T, device=pt.device)
        ar, t0, t1 = self._diag_idx(pt.device)
        L[:, ar, ar] = c * torch.exp(ld.clamp(-4, 4)) * 1.5
        L[:, t0, t1] = c * od
        I = Ilin + c * dI
        dxy = 0.5 * torch.tanh(q[:, T + nt:T + nt + 2])
        lsxy = -4.5 + 5.0 * torch.sigmoid(q[:, T + nt + 2:T + nt + 4])
        qz = q[:, k + T:k + T + self.NZ]
        z = qz[:, 0]                                                    # px, relative to the calibration's focal plane
        lsz = -4.0 + 5.5 * torch.sigmoid(qz[:, 1])                      # log sd of z | x, y (px): 0.018 .. 4.5 px
        bz = qz[:, 2:4].clamp(-5.0, 5.0)                                # dz / dx, dz / dy of z | x, y
        return dict(I=I, L=L, dxy=dxy, lsxy=lsxy, bg=bg, I_lin=Ilin, w=w, wt=wt, dI=dI, z=z, lsz=lsz, bz=bz)


def pad_out(sd, head):
    """state dict of a 2D head (out layer without the z rows) -> the 3D layout: zero rows appended (exactly the 2D outputs,
    z = 0 at the focal plane, s_z|xy = exp(-1.25) px, b = 0); a 3D state dict is returned unchanged"""
    n = head.out.weight.shape[0]
    if sd['out.weight'].shape[0] == n: return sd
    sd = dict(sd); k = n - sd['out.weight'].shape[0]
    assert k == head.NZ, f"out layer {sd['out.weight'].shape[0]} rows, expected {n} or {n - head.NZ}"
    sd['out.weight'] = torch.cat([sd['out.weight'], sd['out.weight'].new_zeros(k, sd['out.weight'].shape[1])])
    sd['out.bias'] = torch.cat([sd['out.bias'], sd['out.bias'].new_zeros(k)])
    return sd


def is_3d(sd):
    """True for a state dict of a 3D head (the z rows of the out layer exist)"""
    T = 6; nt = T * (T + 1) // 2
    return sd['out.weight'].shape[0] == T + nt + 4 + T + SCHead4.NZ


def nll_z(out, tx, ty, tz):
    """NLL of the true z given the true x, y under the head's 3D prior (the z | x, y factor of the 3D Gaussian NLL; the x, y
    factor is the 2D prior NLL). x, y of the prior are detached here: the x, y loss stays the 2D one. -> per target [n]"""
    zc = out['z'] + out['bz'][:, 0] * (tx - out['x'].detach()) + out['bz'][:, 1] * (ty - out['y'].detach())
    return 0.5 * ((tz - zc) / out['lsz'].exp()) ** 2 + out['lsz']


def prior_cov(out):
    """3 x 3 covariance of the head's position prior (x, y, z; px^2) [n,3,3] = L_p L_p^T (see SCHead4)"""
    sx, sy, sz = out['lsxy'][:, 0].exp(), out['lsxy'][:, 1].exp(), out['lsz'].exp()
    L = torch.zeros(len(sx), 3, 3, device=sx.device)
    L[:, 0, 0] = sx; L[:, 1, 1] = sy; L[:, 2, 2] = sz
    L[:, 2, 0] = out['bz'][:, 0] * sx; L[:, 2, 1] = out['bz'][:, 1] * sy
    return L @ L.transpose(1, 2)


def head_call(head, pt, off, psf, nbrp=None):
    if getattr(head, 'nbr', False):
        return head(pt, None, off, psf, nbrp)
    return head(pt, None, off, psf)


# ------------------------------------------------------------------------------------------------ weights
def nn_bin(nnd):
    b = torch.zeros_like(nnd, dtype=torch.long)
    for e in NN_EDGES: b = b + (nnd >= e).long()
    return b


def nn_weights(fov, nnd):
    """per-target weights: within each window the nn bins get NN_W of the weight (empty bins renormalised), split
    equally among the bin's targets; each window's weights sum to its target count"""
    bn = nn_bin(nnd)
    key = fov * 4 + bn
    cnt = torch.zeros(int(key.max()) + 1, device=fov.device).index_add_(0, key, torch.ones_like(nnd))
    wb = torch.tensor(NN_W, device=fov.device)
    w = wb[bn] / cnt[key]
    nf = int(fov.max()) + 1
    tot = torch.zeros(nf, device=fov.device).index_add_(0, fov, w)
    nw = torch.zeros(nf, device=fov.device).index_add_(0, fov, torch.ones_like(w))
    return w / tot[fov] * nw[fov]


# ------------------------------------------------------------------------------------------------ GT matching (from the sc3 trainer)
def match_gt(o, c, dev=dev):
    """one-to-one matching (< 1 px) of detector candidates to the active GT (as sc.train)"""
    from scipy.optimize import linear_sum_assignment
    nb = o['frames'].shape[0]
    keep = torch.isfinite(c['logit'])
    cb, cy_, cx_, xx, yy = c['b'][keep], c['iy'][keep], c['ix'][keep], c['x'][keep], c['y'][keep]
    G = o['gt']; act = (G['onf'].sum(1) > 0) & (G['phot'].sum(1) >= 50)
    gi = torch.nonzero(act)[:, 0]
    mi, mg = [], []
    gb, gx, gy = G['fov'][gi].cpu().numpy(), G['x'][gi].cpu().numpy(), G['y'][gi].cpu().numpy()
    pb, px, py = cb.cpu().numpy(), xx.cpu().numpy(), yy.cpu().numpy()
    for f in range(nb):
        ip = np.nonzero(pb == f)[0]; ig = np.nonzero(gb == f)[0]
        if len(ip) == 0 or len(ig) == 0: continue
        Dm = (px[ip, None] - gx[None, ig]) ** 2 + (py[ip, None] - gy[None, ig]) ** 2
        Cc = np.where(Dm < 1, Dm, 1e6); r_, c_ = linear_sum_assignment(Cc); ok = Cc[r_, c_] < 1
        mi.append(ip[r_[ok]]); mg.append(ig[c_[ok]])
    if not mi: return None
    mi = torch.tensor(np.concatenate(mi), device=dev); mg = gi[torch.tensor(np.concatenate(mg), device=dev)]
    tx, ty = G['x'][mg], G['y'][mg]
    Dg = torch.cdist(torch.stack([tx, ty], 1), torch.stack([G['x'][gi], G['y'][gi]], 1))
    Dg = torch.where(G['fov'][mg][:, None] == G['fov'][gi][None], Dg, torch.full_like(Dg, 1e9))
    nnd = Dg.sort(1).values[:, 1] if Dg.shape[1] > 1 else torch.full_like(tx, 1e9)
    return dict(b=cb[mi], iy=cy_[mi], ix=cx_[mi], x=xx[mi], y=yy[mi], g=mg, nn=nnd)


def ref_var(Ig, g, bgp, rn2p):
    """per-frame intensity Fisher variance with known PSF template g [n,P,P] (true offset), bg [n,T,P,P], rn^2 [n,1,P,P]:
    1 / sum_q g^2 / (I g + bg + rn^2)"""
    mu = Ig.clamp(min=0)[..., None, None] * g[:, None] + bgp.clamp(min=0.3) + rn2p
    return 1.0 / ((g[:, None] ** 2) / mu).sum((2, 3)).clamp(min=1e-9)


def targets(o, m, col=None):
    """GT quantities for matched candidates: Ig, tx, ty, pt, t0, t1, bg patch [n,T,P,P], rn2 at the target pixel,
    and the bg at the rounded true pixel (the sc2 bg target)"""
    G = o['gt']; g = m['g']; nb = o['frames'].shape[0]
    tx, ty = G['x'][g], G['y'][g]
    bgp = patches(o['bg'], m['b'], m['iy'], m['ix'])
    rn = o['rn'].expand(nb, 1, *o['rn'].shape[-2:])
    rn2 = (rn ** 2)[G['fov'][g], 0, torch.round(ty).long().clamp(0, 63), torch.round(tx).long().clamp(0, 63)]
    bgt_px = o['bg'][G['fov'][g], :, torch.round(ty).long().clamp(0, 63), torch.round(tx).long().clamp(0, 63)]
    rn2p = patches((rn ** 2).float(), m['b'], m['iy'], m['ix'])
    C = ref_var(G['phot'][g], col(torch.stack([tx - m['ix'].float(), ty - m['iy'].float()], 1)), bgp, rn2p)
    return dict(C=C, Ig=G['phot'][g], tx=tx, ty=ty, pt=o['patt'][G['fov'][g]], t0=G['t0'][g], t1=G['t1'][g], bgp=bgp,
                rn2=rn2[:, None], bgt_px=bgt_px)


# ------------------------------------------------------------------------------------------------ held-out (cov_or)
def prep(scen=(('iso', 13000), ('pair2', 5000), ('rs25x', 5000)), seed=401, out=HOLD, minph=300):
    """gate-type scenes (own seed), det candidates (p >= 0.2 splat; score >= 0.5 matched to targets on x_det, 1 px),
    stored per matched target: frame patch, offset, PSF column, neighbour patch, GT I, cov_or I-block, the B-profiled
    phase-gradient projection vectors (cov_or metric) and their reference variances, nn, photons, kinetics."""
    from simcode.sim import simtools as ST
    from simcode.sim.simtools import CD, S
    from simcode.hd import gate as GA
    from simcode.hd.gate_report import match
    from simcode.det.model import load as load_det, select_subpixel
    from simcode.sim.crlb import illum_at
    T = 6
    CD.a.calib = config.model(config.INSITU_05)
    os.environ.setdefault('SIMCODE_GATE_CALIB', CD.a.calib)    # the scenes with the same PSF (the gate default is the model's PSF since 2026-11-14)
    sim, rpsf, _ = CD.models(T)
    from simcode.hd.sc import load_det_front
    det, _ = load_det_front()
    col = PSFCol(sim)
    H = {}
    for sc, ntarget in scen:
        rec = []; need = ntarget; it = 0; t0 = time.time()
        while need > 0 and it < 400:
            it += 1
            o = GA.gen(sc, it, seed); nb = o['nb']
            G = o['gt']; act = G['onf'].sum(1) > 0
            G = {k: v[act] for k, v in G.items()}
            ph = G['phot'].sum(1); on = G['onf'].sum(1)
            inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
            tg = torch.nonzero(inter & (on >= 1) & (ph >= minph))[:, 0]
            if len(tg) == 0: continue
            nnd = ST.nn_dist(G, tg)
            fovs = torch.unique(G['fov'][tg]); ai = torch.nonzero(torch.isin(G['fov'], fovs))[:, 0]
            psS = torch.zeros(len(G['x']), CD.psf_channels(rpsf), CD.R, CD.R, device=dev)
            psS[ai] = CD.psf_stack(rpsf, G['x'][ai], G['y'][ai], G['z'][ai], sim.focus)
            Cb = CD.crlb_all(G, o, psS, tg, T, sim.focus); del psS
            cov = Cb['cov_or'].double(); ok = torch.isfinite(cov).all(2).all(1) & torch.isfinite(Cb['sf_m']).all(1)
            tg, nnd, cov = tg[ok], nnd[ok], cov[ok]; sfm = Cb['sf_m'][ok]
            with torch.no_grad():
                fr = o['frames']
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    dd = det(fr)
                c = select_subpixel(dd['logit'].float(), dd['dxy'].float(), 640, p_min=NBR_PMIN, radius=0.7)
                keep = torch.isfinite(c['logit'])
                b, iy, ix, x, y, lg = (c[k][keep] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit'))
                nbr = nbr_map(dd['logit'].float()[:, 0] if dd['logit'].dim() == 4 else dd['logit'].float(), b, iy, ix, x, y, lg)
                Gx = G['x'] + o['shift_est'][G['fov']][:, 0]; Gy = G['y'] + o['shift_est'][G['fov']][:, 1]
                s5 = torch.sigmoid(lg) >= 0.5
                ci = torch.nonzero(s5)[:, 0]
                og = match(G['fov'][tg].cpu().numpy(), Gx[tg].cpu().numpy(), Gy[tg].cpu().numpy(),
                           b[ci].cpu().numpy(), x[ci].cpu().numpy(), y[ci].cpu().numpy(), 1.0)
                sel = og >= 0
                if sel.sum() == 0: continue
                cj = ci[torch.tensor(np.nonzero(sel)[0], device=dev)]; tj = torch.tensor(og[sel], device=dev)
                gi = tg[tj]
                pt = patches(fr, b[cj], iy[cj], ix[cj])
                off = torch.stack([x[cj] - ix[cj].float(), y[cj] - iy[cj].float()], 1)
                nbp = nbr_patches(nbr, b[cj], iy[cj], ix[cj], x[cj], y[cj], torch.sigmoid(lg[cj]))
                # projection vectors (as decomp3.int_diag): C_or^-1 (g - a il), B-profiled
                CI = cov[tj][:, 2:, 2:]
                CI = 0.5 * (CI + CI.transpose(1, 2))
                Ci = torch.linalg.inv(CI + 1e-9 * CI.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(T, device=dev, dtype=torch.float64))
                ptr = o['patt'][G['fov'][gi]].double()
                il, gdx, gdy = illum_at(ptr, G['x'][gi].double(), G['y'][gi].double())
                Bv = G['bright'][gi].double()[:, None]
                Wv = []
                for gx in (Bv * gdx, Bv * gdy):
                    a_ = (il[:, None] @ Ci @ gx[..., None])[:, 0, 0] / (il[:, None] @ Ci @ il[..., None])[:, 0, 0]
                    Wv.append((Ci @ (gx - a_[:, None] * il)[..., None])[..., 0])
                Wv = torch.stack(Wv, 1)                                                     # [n,2,T]
                vo = torch.einsum('nat,nts,nas->na', Wv, CI, Wv)
                rec.append(dict(pt=pt.half().cpu(), off=off.cpu(), psf=col(off).cpu(), nbr=nbp.half().cpu(), x=x[cj].cpu(), y=y[cj].cpu(),
                                Ig=G['phot'][gi].cpu(), tx=Gx[gi].cpu(), ty=Gy[gi].cpu(), W=Wv.float().cpu(), vo=vo.float().cpu(),
                                CI=CI.float().cpu(), nn=nnd[tj].cpu(), ph=ph[gi].cpu(),
                                on=((G['t0'][gi] < 0) & (G['t1'][gi] > T)).cpu(), sfm=sfm[tj].cpu(), fov=(G['fov'][gi] + 1000 * it).cpu()))
                need -= len(gi)
            torch.cuda.empty_cache()
        H[sc] = {k: torch.cat([r[k] for r in rec]) for k in rec[0]}
        print(f'heldout {sc}: {len(H[sc]["Ig"])} matched targets ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(H, out)


def nll_full(Ig, I, L):
    e = (Ig - I)[..., None]
    z = torch.linalg.solve_triangular(L, e, upper=False)[..., 0]
    nll = (0.5 * (z ** 2).sum(1) + torch.log(L.diagonal(dim1=1, dim2=2).abs().clamp(min=1e-6)).sum(1)
           - torch.log((Ig.clamp(min=0).sum(1) / 6 + 30).sqrt()) * 6)
    return nll, z


@torch.no_grad()
def eval_hold(head, H, chunk=4096, band=(500, 5000), extra=False):
    """metrics per scene x nn bin in the photon band; extra: + 's5|' nn bins at 500-5k and 'ph<lo>-<hi>' photon bins (all nn)"""
    was = head.training; head.eval()
    R = {}
    for sc, D in H.items():
        outs = {k: [] for k in ('I', 'Ilin', 'Sn', 'px', 'py', 'sp', 'nll')}
        n = len(D['Ig'])
        for c0 in range(0, n, chunk):
            sl = slice(c0, c0 + chunk)
            pt = D['pt'][sl].to(dev).float(); off = D['off'][sl].to(dev); psf = D['psf'][sl].to(dev)
            o = head_call(head, pt, off, psf, D['nbr'][sl].to(dev).float())
            nll, _ = nll_full(D['Ig'][sl].to(dev), o['I'].float(), o['L'].float())
            outs['I'].append(o['I'].float()); outs['Ilin'].append(o['I_lin'].float()); outs['Sn'].append(o['L'].float() @ o['L'].float().transpose(1, 2))
            outs['px'].append(D['x'][sl].to(dev) + o['dxy'][:, 0]); outs['py'].append(D['y'][sl].to(dev) + o['dxy'][:, 1])
            outs['sp'].append(o['lsxy'].exp()); outs['nll'].append(nll)
        A = {k: torch.cat(v) for k, v in outs.items()}
        Ig = D['Ig'].to(dev); W = D['W'].to(dev); vo = D['vo'].to(dev); nnd = D['nn'].to(dev); ph = D['ph'].to(dev); on = D['on'].to(dev)
        e = A['I'] - Ig
        pe = (W * e[:, None]).sum(-1) ** 2 / vo.clamp(min=1e-9)
        vn = torch.einsum('nat,nts,nas->na', W, A['Sn'], W)
        cal = (W * e[:, None]).sum(-1) ** 2 / vn.clamp(min=1e-9)
        ez = torch.stack([(A['px'] - D['tx'].to(dev)) / A['sp'][:, 0], (A['py'] - D['ty'].to(dev)) / A['sp'][:, 1]], 1)
        ep = torch.stack([A['px'] - D['tx'].to(dev), A['py'] - D['ty'].to(dev)], 1)
        R[sc] = {}
        bins = {'all': (0, 1e9)} if sc in ('iso', 'rs05p', 'rs25p') else {}
        bins.update({'nn<1.5': (0, 1.5), 'nn1.5-2.5': (1.5, 2.5), 'nn2.5-4': (2.5, 4), 'nn>4': (4, 1e9)})
        masks = {bn: (ph >= band[0]) & (ph < band[1]) & (nnd >= a_) & (nnd < b_) for bn, (a_, b_) in bins.items()}
        if extra:
            masks.update({'s5|' + bn: (ph >= 500) & (ph < 5000) & (nnd >= a_) & (nnd < b_) for bn, (a_, b_) in bins.items()})
            masks.update({f'ph{lo}-{hi:.0f}': (ph >= lo) & (ph < hi) for lo, hi in PHB})
        for bn, m in masks.items():
            if m.sum() < 30: continue
            mo = m & on
            r = dict(n=int(m.sum()), nll_I=float(A['nll'][m].mean()), slope=mod_slope(A['I'][m], Ig[m]),
                     ilin_ratio=float((A['Ilin'][m].sum(1) / Ig[m].sum(1)).median()),
                     dI_rel=float(((A['I'][m] - A['Ilin'][m]).abs().sum(1) / Ig[m].sum(1)).median()),
                     prior_rmse=float(ep[m].pow(2).mean().sqrt()), prior_zrms=float(ez[m].pow(2).mean().sqrt()),
                     prior_z3=float((ez[m].abs() > 3).any(1).float().mean()))
            if mo.sum() >= 30:
                b = mod_slope(A['I'][mo], Ig[mo])
                I = A['I'][mo]; Ic = I.mean(1, keepdim=True) + (I - I.mean(1, keepdim=True)) / max(b, 0.3)
                pec = (W[mo] * (Ic - Ig[mo])[:, None]).sum(-1) ** 2 / vo[mo].clamp(min=1e-9)
                r.update(n_on=int(mo.sum()), peff=float(pe[mo].median() / 0.4549), peff_sc=float(pec.median() / 0.4549),
                         calib=float(cal[mo].median() / 0.4549))
            R[sc][bn] = r
    if was: head.train()
    return R


@torch.no_grad()
def z_metrics(out, tx, ty, tz, ph):
    """running z metrics of a training batch (fresh simulated data): z RMSE (px) of the prior z and of z | true x, y, their
    z-score RMS, the median predicted sd; the same for window photons >= 500"""
    zc = out['z'] + out['bz'][:, 0] * (tx - out['x']) + out['bz'][:, 1] * (ty - out['y'])
    sz = out['lsz'].exp()
    var_m = sz ** 2 + (out['bz'][:, 0] * out['lsxy'][:, 0].exp()) ** 2 + (out['bz'][:, 1] * out['lsxy'][:, 1].exp()) ** 2
    r = dict(z_rmse=float((out['z'] - tz).pow(2).mean().sqrt()), zc_rmse=float((zc - tz).pow(2).mean().sqrt()),
             z_zrms=float(((out['z'] - tz) ** 2 / var_m).mean().sqrt()), zc_zrms=float(((zc - tz) / sz).pow(2).mean().sqrt()),
             z_sd_med=float(var_m.sqrt().median()), z_sd_true=float(tz.std()))
    b = ph >= 500
    if b.sum() > 10:
        r.update(z_rmse_500=float((out['z'][b] - tz[b]).pow(2).mean().sqrt()), z_zrms_500=float(((out['z'] - tz)[b] ** 2 / var_m[b]).mean().sqrt()))
    return r


# ------------------------------------------------------------------------------------------------ training
BRIGHT_TAIL = (0.1, 2000.0, 20000.0)    # (frac, lo, hi) window photons of the bright-tail draw (hd/data.Gen.batch)


def train(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--ch', type=int, default=96)
    ap.add_argument('--det', default='det_n1.pt')
    ap.add_argument('--tau', default='15,80')                # mean on-time range (frames); g13: 3,80
    ap.add_argument('--hold', default=HOLD)
    ap.add_argument('--pair_frac', type=float, default=0.25)
    ap.add_argument('--dens', default='0.0005,0.04')
    ap.add_argument('--med', default='40,2500')
    ap.add_argument('--nnw', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--w_mr', type=float, default=2.0)
    ap.add_argument('--w_slope', type=float, default=50.0)
    ap.add_argument('--bf16', type=int, default=1)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--init', default='')                     # start from this checkpoint (e.g. hd_sc5nb.pt)
    ap.add_argument('--pct', type=float, default=0.05)        # OneCycle pct_start
    ap.add_argument('--ckpt_every', type=int, default=10000)
    # the PSF of the simulator (in-situ calibration json; a comma list = a mixture, hd/data.Gen calib; the PSF-template column
    # follows the batch's calib)
    ap.add_argument('--calib', required=True)
    a = ap.parse_args(argv)
    from simcode.util.proctitle import auto; auto('train-head')     # recognizable name in nvidia-smi / ps
    from simcode.sim import crlb as crlb_dense  # noqa
    # the fixed recipe (paper-matched photons with the bright tail, neighbour-aware sc4n head) is stored in args as before
    args = dict(vars(a), arch='sc4n', paper=1, bright_tail=','.join(f'{v:g}' for v in BRIGHT_TAIL))
    from simcode.hd.data import Z_RANGE_UM, Z_UNIF
    args.update(z3d=1, z_range_um=list(Z_RANGE_UM), z_unif=Z_UNIF)
    os.makedirs(a.out, exist_ok=True); json.dump(args, open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    from simcode.hd import data as D
    from simcode.det.model import load as load_det, select_subpixel
    from simcode.hd.sc import load_det_front
    det, _ = load_det_front(a.det)
    gen = D.Gen(64, 64, 6, seed=a.seed, calib=[config.resolve(c) for c in a.calib.split(',')])
    head = SCHead4(6, a.ch, P=PP, bf16=bool(a.bf16)).to(dev)
    if a.init:      # a 2D init (e.g. hd_sc5nb.pt) gets the zero z rows (pad_out): it starts at exactly its 2D outputs
        head.load_state_dict(pad_out(torch.load(config.resolve(a.init), map_location=dev, weights_only=False)['model'], head))
    col = PSFCol(gen.rs['0.5nM'].sim)
    cols = [PSFCol(r['0.5nM'].sim) for r in gen.rsm] if getattr(gen, 'calibs', None) else None
    Hd = torch.load(a.hold, weights_only=False)
    sc5 = os.path.abspath(a.hold) == HOLD5
    tau_r = tuple(float(v) for v in a.tau.split(','))
    dens = tuple(float(v) for v in a.dens.split(',')); med = tuple(float(v) for v in a.med.split(','))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a'); elog = open(a.out + '/eval.jsonl', 'a')
    t0 = time.time(); acc = {}; IS, GS = [], []
    for step in range(a.steps + 1):
        with torch.no_grad():
            o = gen.batch(a.batch, dens=dens, med_photons=med, pair_frac=a.pair_frac, pair_curr=True, paper=True, tau_r=tau_r,
                          bright_tail=BRIGHT_TAIL)
            if cols is not None: col = cols[o['calib_i']]
            fr = o['frames']
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dd = det(fr)
            lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
            c = select_subpixel(lgm, dd['dxy'].float(), 640, p_min=NBR_PMIN, radius=0.7)
            kp = torch.isfinite(c['logit'])
            nbr = nbr_map(lgm, *(c[k][kp] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit')))
            c = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))   # head targets p >= 0.5
            m = match_gt(o, c)
        if m is None:
            sched.step(); continue
        with torch.no_grad():
            t = targets(o, m, col)
            iso_w = (m['nn'] >= 2.0).float()
            fov = o['gt']['fov'][m['g']]
            wt = nn_weights(fov, m['nn']) if a.nnw else torch.ones_like(m['nn'])
            off = torch.stack([m['x'] - m['ix'].float(), m['y'] - m['iy'].float()], 1)
            pt = patches(fr, m['b'], m['iy'], m['ix'])
            psf = col(off)
            # own p for the stamp removal: the candidate's logit at its NMS pixel
            p_own = torch.sigmoid(lgm[m['b'], m['iy'], m['ix']])
            nbp = nbr_patches(nbr, m['b'], m['iy'], m['ix'], m['x'], m['y'], p_own)
        out = head_call(head, pt, off, psf, nbp)
        out['x'] = m['x'] + out['dxy'][:, 0]; out['y'] = m['y'] + out['dxy'][:, 1]
        Ig, tx, ty = t['Ig'], t['tx'], t['ty']
        nll_i, z = nll_full(Ig, out['I'], out['L'])
        nll_I = (wt * nll_i).mean()
        ex = tx - out['x']; ey = ty - out['y']; ls = out['lsxy']
        nll_p = (wt * (0.5 * (ex / ls[:, 0].exp()) ** 2 + ls[:, 0] + 0.5 * (ey / ls[:, 1].exp()) ** 2 + ls[:, 1])).mean()
        tz = D.z_target_px(o['gt']['z'][m['g']], gen.rsm[o['calib_i']]['0.5nM'].sim if cols is not None else gen.rs['0.5nM'].sim)
        l_z = (wt * nll_z(out, tx, ty, tz)).mean()
        cn = (Ig.clamp(min=0).sum(1, keepdim=True) / 6 + 30).sqrt()
        mr = ((out['I'] - out['I'].mean(1, keepdim=True)) - (Ig - Ig.mean(1, keepdim=True))) / cn
        l_mr = (wt[:, None] * F.huber_loss(mr, torch.zeros_like(mr), delta=3.0, reduction='none')).mean()
        am = out['I'] - out['I'].mean(1, keepdim=True); gm = Ig - Ig.mean(1, keepdim=True)
        slope_b = (am * gm * iso_w[:, None]).sum() / (gm * gm * iso_w[:, None]).sum().clamp(min=1e-6)
        l_sl = (1 - slope_b) ** 2
        bgt = t['bgt_px']
        l_bg = (wt[:, None] * ((out['bg'] - bgt) ** 2) / (bgt + 1.0)).mean()
        loss = nll_I / 6 + nll_p + l_z + 0.1 * l_bg + a.w_mr * l_mr + a.w_slope * l_sl
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        else:
            acc['nonfinite'] = acc.get('nonfinite', 0) + 1
        sched.step()
        with torch.no_grad():
            IS.append(out['I'].detach()); GS.append(Ig)
            for k, v in dict(slope_iso=slope_b.item(), mr=l_mr.item(), nll_I=nll_I.item(), nll_p=nll_p.item(), bg=l_bg.item(), n=len(Ig),
                             zI=float((z ** 2).mean().sqrt()), pos_rmse=float(torch.sqrt((ex ** 2 + ey ** 2).mean() / 2)),
                             **z_metrics(out, tx, ty, tz, Ig.sum(1))).items():
                acc[k] = acc.get(k, 0) + v
            acc['_n'] = acc.get('_n', 0) + 1
        if step % 200 == 0:
            nn_ = acc.pop('_n', 1)
            Ia, Ga = torch.cat(IS), torch.cat(GS); IS, GS = [], []
            r = dict(step=step, t=round(time.time() - t0), lr=sched.get_last_lr()[0], slope=mod_slope(Ia, Ga),
                     **{k: v / nn_ for k, v in acc.items()})
            acc = {}
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if step % a.eval_every == 0:
            ev = eval_hold(head, Hd, band=(200, 2000), extra=True) if sc5 else eval_hold(head, Hd)
            elog.write(json.dumps(dict(step=step, t=round(time.time() - t0), **ev)) + '\n'); elog.flush()
            print('EVAL', step, json.dumps({s: ev[s].get('all', ev[s].get('nn1.5-2.5')) for s in ev}), flush=True)
        if step % a.ckpt_every == 0 and step > 0:
            ckpt_save(dict(model=head.state_dict(), args=args, step=step), a.out + f'/ckpt_{step}.pt')
    ckpt_save(dict(model=head.state_dict(), args=args, step=a.steps), a.out + '/final.pt')


if __name__ == '__main__':
    st = sys.argv[1]; rest = sys.argv[2:]; sys.argv = sys.argv[:1]
    if st == 'prep':
        prep()
    elif st == 'prep5':
        prep(SC5_SCEN, out=HOLD5, minph=150)
    elif st == 'train':
        train(rest)
