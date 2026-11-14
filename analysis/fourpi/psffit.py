"""In-situ vectorial PSF of the registered 4-channel sum (the interference term cancels in the sum, so the summed PSF
is an incoherent PSF). Built on torch_psf.VectorialPSF.

Model (index-matched by default, n_med = n_imm: the emitter z is a pure defocus relative to the focal plane):
  one pupil:  mu = N * PSF(x, y, z; Zernike 5..21, blur) + bg
  two arms:   mu = N * [a PSF(x, y, z; W1) + (1 - a) PSF(x + dx, y + dy, z + dz; W2)] + bg   (incoherent mixture of
              two pupils: the channel sum of a 4Pi = |E_top|^2 + |E_bottom|^2; a, dx, dy, dz, W1, W2 free)
Fit: variable projection. N, bg per sample = weighted LSQ (closed form, 2 x 2) each iteration; x, y, z per sample and
the global pupil parameters by Adam on the Poisson NLL (sCMOS read-noise variance v added).
Bead z-stack mode: z_k = z0 + s * zstage_k (z0 free, s = +-1 chosen by the better fit), x, y per slice free.
"""
import math, os, sys, json
import numpy as np
import torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.psf import VectorialPSF

ZIDX = tuple(range(5, 22))


class SumPSF(torch.nn.Module):
    def __init__(self, R, px=0.128, NA=1.35, n_imm=1.406, n_med=1.406, lam=0.60, two_arm=False, npupil=48, os_=2,
                 depth=0.0):
        super().__init__()
        self.R, self.two_arm, self.depth = R, two_arm, depth
        mk = lambda: VectorialPSF(R, pixel_um=px, NA=NA, n_imm=n_imm, n_med=n_med, wavelength_um=lam, zernike_indices=ZIDX,
                                  npupil=npupil, oversample=os_, device='cpu')
        self.p1 = mk()
        self.logblur = torch.nn.Parameter(torch.tensor(math.log(0.05)))      # isotropic blur variance (px^2)
        if two_arm:
            self.p2 = mk()
            self.a_logit = torch.nn.Parameter(torch.tensor(0.0))
            self.d2 = torch.nn.Parameter(torch.zeros(3))                       # dx, dy (px), dz (um) of arm 2
        self.meta = dict(px=px, NA=NA, n_imm=n_imm, n_med=n_med, lam=lam, two_arm=two_arm, depth=depth)

    def zparams(self):
        ps = [self.p1.zcoef] + ([self.p2.zcoef] if self.two_arm else [])
        return ps

    def forward(self, pos):
        b = torch.exp(self.logblur)
        cov = torch.stack([b, b, torch.zeros_like(b)])
        ze = pos.clone(); ze[:, 2] = ze[:, 2] + self.depth
        f = torch.full((len(pos),), self.depth)
        g = self.p1(ze, f, blur=cov)
        if self.two_arm:
            a = torch.sigmoid(self.a_logit)
            g2 = self.p2(ze + self.d2[None], f, blur=cov)
            g = a * g + (1 - a) * g2
        return g


def varpro_Nbg(g, d, v, iters=3):
    """g [B,R,R] PSF, d data. Weighted LSQ for N, bg (weights 1/(mu+v)), returns N, bg [B]"""
    B = g.shape[0]
    gf, df = g.flatten(1), d.flatten(1)
    w = torch.ones_like(df)
    for _ in range(iters):
        a11 = (w * gf * gf).sum(1); a12 = (w * gf).sum(1); a22 = w.sum(1)
        b1 = (w * gf * df).sum(1); b2 = (w * df).sum(1)
        det = a11 * a22 - a12 ** 2
        N = (a22 * b1 - a12 * b2) / det; bg = (a11 * b2 - a12 * b1) / det
        N = N.clamp(min=1.0); bg = bg.clamp(min=1e-3)
        mu = N[:, None] * gf + bg[:, None]
        w = 1.0 / (mu + v)
    return N, bg


def nll(g, d, v, N, bg):
    mu = N[:, None, None] * g + bg[:, None, None] + v
    dd = (d + v).clamp(min=1e-6)
    return (mu - dd * torch.log(mu)).sum((1, 2)) - (dd - dd * torch.log(dd)).sum((1, 2))   # deviance / 2 >= 0


def z_grid_init(model, d, xy, v, zs=np.arange(-0.8, 0.81, 0.04)):
    best = torch.full((len(d),), float('inf')); bz = torch.zeros(len(d))
    with torch.no_grad():
        for z in zs:
            pos = torch.cat([xy, torch.full((len(d), 1), float(z))], 1)
            g = model(pos); N, bg = varpro_Nbg(g, d, v)
            L = nll(g, d, v, N, bg)
            upd = L < best; best = torch.where(upd, L, best); bz = torch.where(upd, torch.full_like(bz, float(z)), bz)
    return bz


def fit(model, d, xy0, z0, v, iters=300, lr_pos=0.02, lr_glob=0.01, fit_glob=True, zlim=(-1.0, 1.0), bead=None, log=None):
    """d [B,R,R]. bead = dict(zstage [B]) -> z = z0g + s * zstage with global z0g (s fixed by the caller in bead['s'])."""
    xy = xy0.clone().requires_grad_(True)
    if bead is None:
        z = z0.clone().requires_grad_(True); pz = [z]
    else:
        zoff = torch.tensor(float(z0), requires_grad=True); pz = [zoff]
        zst = torch.as_tensor(bead['zstage'], dtype=torch.float32); s = bead['s']
    glob = [p for p in model.zparams()] + [model.logblur] + ([model.a_logit, model.d2] if model.two_arm else [])
    for p in glob: p.requires_grad_(fit_glob)
    opt = torch.optim.Adam([{'params': [xy] + pz, 'lr': lr_pos}] + ([{'params': glob, 'lr': lr_glob}] if fit_glob else []))
    hist = []
    for it in range(iters):
        opt.zero_grad()
        zz = z if bead is None else zoff + s * zst
        pos = torch.cat([xy, zz[:, None]], 1)
        g = model(pos)
        with torch.no_grad(): N, bg = varpro_Nbg(g, d, v)
        L = nll(g, d, v, N, bg)
        loss = L.sum() / len(d)
        loss.backward(); opt.step()
        with torch.no_grad():
            if bead is None: z.clamp_(*zlim)
            xy.clamp_(-3, 3)
        hist.append(loss.item())
        if log and (it % 50 == 0 or it == iters - 1): print(f'   it {it} loss {loss.item():.4f}', flush=True)
    with torch.no_grad():
        zz = z if bead is None else zoff + s * zst
        pos = torch.cat([xy, zz[:, None]], 1); g = model(pos); N, bg = varpro_Nbg(g, d, v); L = nll(g, d, v, N, bg)
        mu = N[:, None, None] * g + bg[:, None, None]
    return dict(xy=xy.detach(), z=zz.detach(), N=N, bg=bg, nll=L, mu=mu, hist=hist,
                zoff=(zoff.item() if bead is not None else None))


def summary(model):
    out = dict(model.meta)
    out['zernike_noll'] = list(ZIDX); out['zcoef1'] = model.p1.zcoef.detach().tolist()
    out['blur_var_px2'] = float(torch.exp(model.logblur))
    if model.two_arm:
        out['zcoef2'] = model.p2.zcoef.detach().tolist(); out['a'] = float(torch.sigmoid(model.a_logit))
        out['d2'] = model.d2.detach().tolist()
    return out


def render_knots(model, zmin, zmax, hz=0.01, S=4, P=15, xmax=2.0, chunk=512):
    """oversampled knot stack g(u, v, z) for zpsf.knot_spline (same layout as torch_psf.ConvCubicSplinePSF): returns
    zstack [nz, Pl*S, Pl*S], u0, z0. Pl = big ROI covering P + 2 xmax."""
    c = int(math.ceil((P - 1) / 2 + xmax)) + 2; Pl = 2 * c + 1
    big = SumPSF(Pl, **{k: model.meta[k] for k in ('px', 'NA', 'n_imm', 'n_med', 'lam', 'two_arm', 'depth')},
                 npupil=model.p1.npupil, os_=3)
    big.load_state_dict({k: v for k, v in model.state_dict().items() if not k.endswith(('.u', '.px', '.py', '.rho', '.phi', '.mask', '.Z'))}, strict=False)
    zs = torch.arange(zmin, zmax + 1e-9, hz)
    a = torch.arange(S, dtype=torch.float32) / S
    Zg, B, A = torch.meshgrid(zs, a, a, indexing='ij')
    pos = torch.stack([A.reshape(-1), B.reshape(-1), Zg.reshape(-1)], 1).float()
    with torch.no_grad():
        imgs = torch.cat([big(p) for p in pos.split(chunk)])
    imgs = imgs.reshape(len(zs), S, S, Pl, Pl)
    G = imgs.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(len(zs), Pl * S, Pl * S)
    u0 = -(c * S + S - 1) / S
    return G.contiguous(), u0, float(zs[0]), hz
