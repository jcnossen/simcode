"""
The emitter network of pass 1 and pass 2 (one class, `EmitterNet`; the pass-2 network is the same network with illumination
inputs), the pieces that feed it on detector candidates, and its checkpoint reader.

Per candidate (detector point, detector/select.py), on a 21 x 21 x T patch at the candidate pixel. NO pixel fit:
  inputs   the T frames (normalised and linear), the window mean, the offset of the detector point (coordinate maps), and two
           neighbour channels: a splat of the OTHER detected candidates (detector p x Gaussian sigma 0.5 px at their sub-pixel
           points; candidates p >= NBR_PMIN after NMS, rendered once per window (nbr_map), cropped with the patch gather, the
           candidate's own stamp removed exactly (nbr_patches)) and the cropped detector-probability map; illum = True (pass 2):
           + the 2T illumination patch channels of the estimated pattern (pattern.illumination.illum_maps cropped with the
           frame patch; zero-initialised input weights cin_il, so a pass-2 network starts exactly at its pass-1 network)
  readout  signed per-frame readout w_t = w0 exp(m) / (.. g) + D_t, D_t = T signed maps (zero-initialised conv) with the
           g-component projected out (g = the PSF template of the candidate offset, PSFCol), so w_t . g = 1 for every frame
  outputs  per-frame intensities I_1..I_T with a FULL T x T covariance (Cholesky factor L), per-frame bg, the position prior
           x, y (+- sigma; pattern-free in pass 1, pattern-informed in pass 2) and z with a 3 x 3 covariance (EmitterNet doc)
bf16 autocast for the CNN trunk, fp32 for the output layers, the readout and the Cholesky.
The final position is always the explicit intensity-domain pattern solve (solve/solve.solve_emitters) on these outputs.

Checkpoints ({'model': state dict, 'args', 'step'}): load() reads every emitter-network checkpoint: the architecture tag
args['arch'] = 'sc4n' (pass-1 checkpoints; the pass-2 ones carry none), pass 1 or pass 2 from the state dict (cin_il), a 2D
checkpoint (before 2026-11-14, no z rows) with zero z rows (pad_out; net.z3d False).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from simcode.layers.blocks import ResBlock, extract_patches, norm_photons

dev = 'cuda'
PP = 21                 # patch size (px)
ARCH = 'sc4n'           # architecture tag of the checkpoints of this network (training args 'arch')
NBR_PMIN = 0.2          # candidates in the neighbour splat (= the gate detection threshold THR0)
SIG_SPLAT = 0.5
OFFS = torch.arange(-3, 4)
HEAD_Z = ('z', 'lsz', 'bz')       # the network's z outputs carried by the pipeline


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
    q = extract_patches(nbr, b, iy, ix)
    off = torch.stack([x - ix.float(), y - iy.float()], 1)
    st = own_stamp(off, p.float(), iy, ix, nbr.shape[2], nbr.shape[3])
    return torch.stack([(q[:, 0] - st).clamp(min=0), q[:, 1]], 1)


# ------------------------------------------------------------------------------------------------ network
class EmitterNet(nn.Module):
    """The emitter network (module doc); illum = True: the pass-2 network with the illumination patch inputs (cin_il).
    3D position prior (2026-11-14): x, y, z with a 3 x 3 covariance. Output layer q = [dI T | log diag L_I T | off-diag L_I |
    dx, dy | log sx, log sy | bg T | z, log s_z|xy, b_x, b_y] (the last 4 rows appended: a 2D checkpoint loads with zero rows,
    pad_out, and gives exactly its 2D outputs). Covariance in Cholesky form with the order (x, y, z):
        L_p = [[sx, 0, 0], [0, sy, 0], [b_x sx, b_y sy, s_z|xy]]      Sigma_p = L_p L_p^T
    i.e. x and y uncorrelated (as the 2D prior and the solve's prior), z | x, y = z + b_x (x' - x) + b_y (y' - y) with sd
    s_z|xy. z in px of the lateral pixel size relative to the focal plane of the PSF calibration (simulate/batches.z_target_px).
    The NLL of the 3D Gaussian factorizes into the 2D x, y prior NLL (unchanged) + the NLL of z | x, y (nll_z)."""
    NZ = 4      # appended z outputs

    def __init__(self, T=6, ch=96, nblk=4, P=21, bf16=True, illum=False):
        super().__init__()
        self.T, self.P, self.bf16, self.illum = T, P, bf16, bool(illum)
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
        if illum:      # pass 2 (created last: the same construction order as the pass-1 layers above)
            self.cin_il = nn.Conv2d(2 * T, ch, 3, padding=1, bias=False)
            nn.init.zeros_(self.cin_il.weight)

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
        """pt [n,T,P,P] photons, rn2p unused (None), off [n,2] detector point - patch centre (px), psf [n,P,P] template (None:
        Gaussian sigma 1.3 px), nbrp [n,2,P,P] (nbr_patches), illp [n,2T,P,P] the illumination patches (pass 2; None for pass 1)"""
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
    return sd['out.weight'].shape[0] == T + nt + 4 + T + EmitterNet.NZ


def nll_full(Ig, I, L):
    e = (Ig - I)[..., None]
    z = torch.linalg.solve_triangular(L, e, upper=False)[..., 0]
    nll = (0.5 * (z ** 2).sum(1) + torch.log(L.diagonal(dim1=1, dim2=2).abs().clamp(min=1e-6)).sum(1)
           - torch.log((Ig.clamp(min=0).sum(1) / 6 + 30).sqrt()) * 6)
    return nll, z


def nll_z(out, tx, ty, tz):
    """NLL of the true z given the true x, y under the head's 3D prior (the z | x, y factor of the 3D Gaussian NLL; the x, y
    factor is the 2D prior NLL). x, y of the prior are detached here: the x, y loss stays the 2D one. -> per target [n]"""
    zc = out['z'] + out['bz'][:, 0] * (tx - out['x'].detach()) + out['bz'][:, 1] * (ty - out['y'].detach())
    return 0.5 * ((tz - zc) / out['lsz'].exp()) ** 2 + out['lsz']


def prior_cov(out):
    """3 x 3 covariance of the head's position prior (x, y, z; px^2) [n,3,3] = L_p L_p^T (see EmitterNet)"""
    sx, sy, sz = out['lsxy'][:, 0].exp(), out['lsxy'][:, 1].exp(), out['lsz'].exp()
    L = torch.zeros(len(sx), 3, 3, device=sx.device)
    L[:, 0, 0] = sx; L[:, 1, 1] = sy; L[:, 2, 2] = sz
    L[:, 2, 0] = out['bz'][:, 0] * sx; L[:, 2, 1] = out['bz'][:, 1] * sy
    return L @ L.transpose(1, 2)


def mod_slope(I, Ig):
    """regression slope of the mean-removed (per emitter) network intensities on the mean-removed GT intensities"""
    a = I - I.mean(1, keepdim=True); g = Ig - Ig.mean(1, keepdim=True)
    return float((a * g).sum() / (g * g).sum().clamp(min=1e-9))


# ------------------------------------------------------------------------------------------------ on detector candidates
class PSFCol:
    """unit-sum-free PSF columns on the PP x PP patch from the calibrated 15 x 15 LUT (z fixed 50 nm)"""
    def __init__(self, sim, z=0.05):
        from simcode.psf import spline
        self.lut = spline(sim.psf, sim.zmax, roisize=PP); self.z = z

    def __call__(self, off):
        return self.lut.render(torch.cat([off, torch.full_like(off[:, :1], self.z)], 1))


@torch.no_grad()
def detect_candidates(det, frames, illum=None, k=640):
    """detector -> kept candidates (p >= NBR_PMIN; dict b, iy, ix, x, y, logit) + neighbour maps [B,2,H,W] + the logit map.
    illum [B,2T,H,W]: the illumination maps, used by a pass-2 detector (det.illum)"""
    from simcode.detector.select import select_subpixel
    with torch.autocast('cuda', dtype=torch.bfloat16):
        dd = det(frames, illum) if det.illum else det(frames)
    lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
    c = select_subpixel(lgm, dd['dxy'].float(), k, p_min=NBR_PMIN, radius=0.7)
    kp = torch.isfinite(c['logit'])
    cc = {kk: c[kk][kp] for kk in ('b', 'iy', 'ix', 'x', 'y', 'logit')}
    nbr = nbr_map(lgm, cc['b'], cc['iy'], cc['ix'], cc['x'], cc['y'], cc['logit'])
    return cc, nbr, lgm


def apply_to_candidates(net, col, frames, illum, b, iy, ix, x, y, nbr, p):
    """the emitter network on candidates (b, pixel iy / ix, detector point x / y, detector p) of the windows frames [B,T,H,W]:
    patches, neighbour patches of nbr (nbr_map), the PSF template col(offset) (col None: Gaussian template), and for a pass-2
    network (net.illum) the illumination patches of illum [B,2T,H,W] (ignored by a pass-1 network). -> the network outputs
    + x, y (prior position, px), x_det, y_det (the detector point)"""
    off = torch.stack([x - ix.float(), y - iy.float()], 1)
    pt = extract_patches(frames, b, iy, ix)
    nbp = nbr_patches(nbr, b, iy, ix, x, y, p)
    psf = col(off) if col is not None else None
    if net.illum:
        o = net(pt, None, off, psf, nbp, extract_patches(illum, b, iy, ix))
    else:
        o = net(pt, None, off, psf, nbp)
    o['x'] = x + o['dxy'][:, 0]; o['y'] = y + o['dxy'][:, 1]
    o['x_det'] = x; o['y_det'] = y          # detector point
    return o


def head_z(net, o):
    """the network's z outputs (z, lsz, bz) of o; NaN for a 2D checkpoint (net.z3d False: zero z rows, not an estimate)"""
    r = {k: o[k].float() for k in HEAD_Z}
    if not getattr(net, 'z3d', True):
        r = {k: torch.full_like(v, float('nan')) for k, v in r.items()}
    return r


# ------------------------------------------------------------------------------------------------ checkpoints
def is_pass2(sd):
    """True for the state dict of a pass-2 network (illumination inputs)"""
    return 'cin_il.weight' in sd


def load(ck):
    """emitter-network checkpoint (path; relative paths resolved in the models directory) -> EmitterNet in eval mode
    (net.illum: pass 2; net.z3d: False for a 2D checkpoint, whose z outputs are not estimates)"""
    from simcode.config import resolve
    c = torch.load(resolve(ck), map_location=dev, weights_only=False)
    a = c.get('args') or {}
    if a.get('arch', ARCH) != ARCH:
        raise ValueError(f"{ck}: emitter network arch {a.get('arch')!r} is not supported (only the neighbour-aware {ARCH!r} network)")
    net = EmitterNet(6, a.get('ch', 96), P=PP, bf16=bool(a.get('bf16', 1)), illum=is_pass2(c['model'])).to(dev)
    net.z3d = is_3d(c['model'])
    net.load_state_dict(pad_out(c['model'], net)); net.eval()
    return net
