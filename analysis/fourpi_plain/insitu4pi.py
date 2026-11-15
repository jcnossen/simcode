"""In-situ 4Pi PSF estimation on the spots of the movie itself (plain 4Pi, Cell11), with torch-psf's FourPiPSF /
FourPiSplinePSF (torch-psf master 52c75c5, external/torch-psf-52c75c5; env TPSF).

torch-psf has the 4Pi models but no in-situ path for them (its knot-route / track estimators are single-channel), so
the missing pieces are here, on its public API (FourPiPSF.fields, FourPiPSF.to_spline, FourPiSplinePSF.luts /
CubicSplinePSF.render(knots=...)):

  Model4Pi(psf)          FourPiPSF + per-channel registration offsets reg [C, 2] (px, channel 0 fixed) + beta (rad / um:
                         an extra 4Pi phase beta z, i.e. a fringe period different from the model's) -> images(xy [n, C, 2],
                         z [n], Phi [n]) [n, C, P, P]: every channel rendered at its own mapped position (the
                         vectorial model evaluated once per (spot, channel)).
  SplineModel(spl, reg, beta)   the same images from a FourPiSplinePSF (spline table of Model4Pi.psf, or refined
                         knots), value and d/d(x, y, z, Phi); knots_override = a [3, C, nz, n, n] tensor (requires grad)
  localize(model, D, th, fix_phi, iters)   per-spot Levenberg-Marquardt (Poisson, read-noise shifted) of
                         th = (x, y, z, Phi, N, b_0..b_{C-1}) [n, 5 + C] on a SplineModel; Phi fixed when fix_phi
  nll(mu, d, var)        Poisson NLL with the read-noise shift (d + var ~ Poisson(mu + var)), per pixel
  deviance(mu, d, var)   2 [d' log(d' / mu') - (d' - mu')], per pixel (~1 per pixel for a correct model)
  fit_vectorial(model, D, th, ...)   PSF parameters (Zernikes both arms, phi, tau, a, gamma, d_bot, blur, reg, beta) by
                         Adam on minibatches at fixed spot parameters, alternated with localize (outer iterations)

Spots D (load_spots): d [n, C, P, P] photo-electrons, var (read noise^2), pos0 [n, C, 2] (emitter position in each
channel's patch relative to its centre pixel, px, from the SIMCODE localization and the registration map), and the
table columns.
"""
import math, os, sys, time, copy
import numpy as np
import torch

TPSF = os.environ.get('TPSF', '/home/jc/simcode2/external/torch-psf-52c75c5')
if TPSF not in sys.path: sys.path.insert(0, TPSF)
from torch_psf.fourpi import FourPiPSF, FourPiSplinePSF, _quad          # noqa: E402
from torch_psf.vectorial import gauss_blur_os                          # noqa: E402

dev = 'cuda'
OUT = '/home/jc/simcode2/out/plain4pi/p4psf'
PSF0 = '/home/jc/simcode2/out/plain4pi/psf/psf4pi_Cell11_astig.json'
ZR = (-1.0, 1.0)


# ---------------------------------------------------------------------------------------------------------------- data
def load_spots(path=f'{OUT}/spots_P17.npz', sel=None, device=dev):
    """spots -> dict of tensors (d e-, var, pos0) and numpy columns; sel(Z) -> boolean mask over the file's spots"""
    Z = np.load(path)
    m = np.ones(len(Z['x']), bool) if sel is None else sel(Z)
    ii = np.nonzero(m)[0]
    D = {k: Z[k][ii] for k in Z.files if Z[k].ndim and Z[k].shape[0] == len(m) and k != 'adu'}
    D['idx'] = ii
    D['d'] = ((torch.tensor(Z['adu'][ii].astype(np.float32), device=device) - float(Z['offset'])) * float(Z['gain']))
    D['var'] = float(Z['readnoise']) ** 2
    D['pos0'] = torch.tensor(Z['pos'][ii], device=device, dtype=torch.float32)
    return D


def theta0(D, device=dev):
    """th [n, 5 + C] = (x offset, y offset, z, Phi, N, bg_c) from the SIMCODE table; x, y offsets are added to pos0
    (start 0: pos0 is the SIMCODE position)"""
    n = len(D['x']); C = D['pos0'].shape[1]
    th = torch.zeros(n, 5 + C, device=device)
    th[:, 2] = torch.tensor(D['z'], device=device)
    th[:, 3] = torch.tensor(D['Phi0'], device=device)
    th[:, 4] = torch.tensor(D['N'], device=device)
    th[:, 5:] = torch.tensor(D['bg'], device=device).clamp(min=0.5)
    return th


def nll(mu, d, var):
    mu = mu + var; dd = (d + var).clamp(min=0)
    return mu - dd * torch.log(mu.clamp(min=1e-6))


def deviance(mu, d, var):
    mu = (mu + var).clamp(min=1e-6); dd = (d + var).clamp(min=0)
    return 2 * (torch.where(dd > 0, dd * torch.log(dd.clamp(min=1e-9) / mu), torch.zeros_like(dd)) - (dd - mu))


# ---------------------------------------------------------------------------------------------------------------- models
class Model4Pi(torch.nn.Module):
    def __init__(self, psf, beta=0.0):
        super().__init__()
        self.psf = psf
        C = psf.n_channels
        self.reg = torch.nn.Parameter(torch.zeros(C, 2, device=psf.dev))
        self.beta = torch.nn.Parameter(torch.tensor(float(beta), device=psf.dev))
        self.register_buffer('regmask', (torch.arange(C, device=psf.dev) > 0).float()[:, None])

    @property
    def C(self):
        return self.psf.n_channels

    def offsets(self):
        return self.reg * self.regmask

    def phase(self, z, Phi):
        return Phi + self.beta * z

    def images(self, xy, z, Phi, chunk=600):
        """[n, C, P, P] normalized images; xy [n, C, 2] = emitter position in each channel's patch (before reg)"""
        outs = [self._images(xy[s:s + chunk], z[s:s + chunk], Phi[s:s + chunk]) for s in range(0, len(z), chunk)]
        return torch.cat(outs)

    def _images(self, xy, z, Phi):
        p, C, n = self.psf, self.C, len(z)
        pos = torch.cat([xy + self.offsets()[None], z[:, None, None].expand(n, C, 1)], -1)        # [n, C, 3]
        pos = pos.permute(1, 0, 2).reshape(C * n, 3)
        Et, Eb, energy = p.fields(pos)
        sh = Et.shape[1:]
        Et = Et.reshape(C, n, *sh); Eb = Eb.reshape(C, n, *sh)
        a, g = p.a, p.gamma
        M = (torch.eye(3, device=pos.device) / 3)[None].expand(n, 3, 3)
        ph = self.phase(z, Phi)
        out = []
        for c in range(C):
            q = int(p.pol[c])
            et, eb = Et[c][:, :, q], Eb[c][:, :, q]
            e_ph = torch.exp(1j * (p.phi[c] + ph)).reshape(n, 1, 1, 1)
            out.append(_quad(M, et, et) + a * a * _quad(M, eb, eb) + 2 * a * g * _quad(M, et, eb * e_ph))
        I = torch.stack(out, 1) / (1 + a * a)
        bv = p.blur_px2
        I = gauss_blur_os(I.reshape(n * C, *I.shape[-2:]), torch.stack([bv, bv, torch.zeros_like(bv)]), p.os, p.BLUR_PAD)
        R, o = p.roisize, p.os
        I = I.reshape(n, C, R, o, R, o).mean((3, 5)) * p.pixel_um ** 2 / energy
        return I * p.tau[None, :, None, None] * 2

    def spline(self, P=None, z_range=ZR, oversample=4, dz_um=0.01):
        with torch.no_grad():
            spl = self.psf.to_spline(roisize=P or self.psf.roisize, z_range=z_range, oversample=oversample, dz_um=dz_um)
        return SplineModel(spl, self.offsets().detach(), float(self.beta.detach()))

    ARM_EXTRA = ('NA', 'n_med', 'apod', 'saf', 'pupil_phase', 'pupil_amp')

    def free_pupil(self):
        """make the pupil phase (waves) / amplitude maps of both arms trainable (buffers -> parameters)"""
        for arm in (self.psf.top, self.psf.bot):
            for k in ('pupil_phase', 'pupil_amp'):
                if isinstance(getattr(arm, k), torch.nn.Parameter): continue
                v = getattr(arm, k).detach().clone()
                del arm._buffers[k]
                setattr(arm, k, torch.nn.Parameter(v))
        return self

    def state(self):
        """FourPiPSF.state() + reg, beta and the arm terms FourPiPSF.state does not store (NA, n_med, apod, saf, pupil
        maps; key 'arms')"""
        arms = {nm: {k: (getattr(arm, k).detach().cpu().tolist() if torch.is_tensor(getattr(arm, k)) else getattr(arm, k))
                     for k in self.ARM_EXTRA} for nm, arm in (('top', self.psf.top), ('bot', self.psf.bot))}
        return dict(psf=self.psf.state(), reg=self.offsets().detach().cpu().tolist(), beta=float(self.beta.detach()), arms=arms)

    @classmethod
    def from_state(cls, s, roisize=17, device=dev):
        psf = FourPiPSF.from_state(s['psf'], device=device, roisize=roisize, z_range=ZR)
        m = cls(psf, s.get('beta', 0.0))
        with torch.no_grad():
            m.reg.copy_(torch.tensor(s.get('reg', [[0, 0]] * psf.n_channels), device=device))
            for nm, arm in (('top', psf.top), ('bot', psf.bot)):
                for k, v in (s.get('arms') or {}).get(nm, {}).items():
                    t = getattr(arm, k)
                    if torch.is_tensor(t): t.copy_(torch.as_tensor(v, dtype=t.dtype, device=t.device).reshape(t.shape))
        return m


def load_psf0(roisize=17):
    import json
    st = json.load(open(PSF0))['psf']
    return Model4Pi(FourPiPSF.from_state(st, device=dev, roisize=roisize, z_range=ZR))


class SplineModel:
    """images of a FourPiSplinePSF (spl.luts[comp][c]) with per-channel offsets reg [C, 2] and the phase term beta z"""
    def __init__(self, spl, reg, beta=0.0, knots=None):
        self.spl, self.reg, self.beta = spl, reg, beta
        self.knots = knots                 # optional [3, C, nz, n, n] override (e.g. requires grad)
        self.C = spl.n_channels

    def _K(self, k, c):
        return None if self.knots is None else self.knots[k, c]

    def images(self, xy, z, Phi, grad=False):
        """-> [n, C, P, P] (grad: value, d/dx, d/dy, d/dz, d/dPhi); Phi is the per-spot phase (beta z added here)"""
        n, C = len(z), self.C
        ph = Phi + self.beta * z
        cP, sP = torch.cos(ph)[:, None, None], torch.sin(ph)[:, None, None]
        vals = [[] for _ in range(5)]
        for c in range(C):
            pos = torch.cat([xy[:, c] + self.reg[c][None], z[:, None]], 1)
            r = [self.spl.luts[k][c].render(pos, grad, knots=self._K(k, c)) for k in range(3)]
            if not grad:
                vals[0].append(r[0] + cP * r[1] - sP * r[2]); continue
            for j in range(4):
                vals[j].append(r[0][j] + cP * r[1][j] - sP * r[2][j])
            dphi = -sP * r[1][0] - cP * r[2][0]
            vals[3][-1] = vals[3][-1] + self.beta * dphi          # d/dz includes the beta z phase
            vals[4].append(dphi)
        if not grad:
            return torch.stack(vals[0], 1)
        return tuple(torch.stack(v, 1) for v in vals)


# ---------------------------------------------------------------------------------------------------------------- localization
def expected(model, D, th, idx=None, grad=False):
    """mu [n, C, P, P] of spots idx at th (and the Jacobian [n, C, P, P, 5 + C] when grad, SplineModel only)"""
    pos0 = D['pos0'] if idx is None else D['pos0'][idx]
    xy = pos0 + th[:, None, :2]
    if not grad:
        g = model.images(xy, th[:, 2], th[:, 3])
        return th[:, 4, None, None, None] * g + th[:, 5:, None, None]
    g, gx, gy, gz, gp = model.images(xy, th[:, 2], th[:, 3], grad=True)
    N = th[:, 4, None, None, None]
    mu = N * g + th[:, 5:, None, None]
    C = g.shape[1]
    eye = torch.eye(C, device=g.device)[None, :, None, None, :].expand(len(th), C, g.shape[2], g.shape[3], C)
    J = torch.cat([torch.stack([N * gx, N * gy, N * gz, N * gp, g], -1), eye], -1)
    return mu, J


def localize(model, D, th, fix_phi=True, iters=20, batch=4000, xlim=2.5, zlim=ZR, lam0=1e-2, fix=()):
    """per-spot LM (Poisson, read-noise shifted) -> (th, nll per spot)"""
    var = D['var']
    out = th.clone(); L = torch.zeros(len(th), device=th.device)
    fixed = set(fix) | ({3} if fix_phi else set())
    free = [j for j in range(th.shape[1]) if j not in fixed]
    for s in range(0, len(th), batch):
        idx = torch.arange(s, min(s + batch, len(th)), device=th.device)
        t = th[idx].clone(); d = D['d'][idx]
        lam = torch.full((len(t),), lam0, device=t.device)

        def cost(tt):
            return nll(expected(model, D, tt, idx), d, var).sum((1, 2, 3))
        cur = cost(t)
        for it in range(iters):
            mu, J = expected(model, D, t, idx, grad=True)
            J = J[..., free]
            w = 1.0 / (mu + var).clamp(min=1e-3)
            r = (d + var) - (mu + var)
            Jf = J.reshape(len(t), -1, len(free)); wf = w.reshape(len(t), -1); rf = r.reshape(len(t), -1)
            A = torch.einsum('npi,np,npj->nij', Jf, wf, Jf)
            b = torch.einsum('npi,np->ni', Jf, wf * rf)
            Ad = A + lam[:, None, None] * torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2) + 1e-6)
            step = torch.linalg.solve(Ad, b[..., None])[..., 0]
            tn = t.clone(); tn[:, free] = t[:, free] + step
            tn[:, :2] = tn[:, :2].clamp(-xlim, xlim)
            tn[:, 2] = tn[:, 2].clamp(zlim[0] + 0.02, zlim[1] - 0.02)
            tn[:, 4] = tn[:, 4].clamp(min=10.0); tn[:, 5:] = tn[:, 5:].clamp(min=0.01)
            new = cost(tn)
            acc = new < cur
            t = torch.where(acc[:, None], tn, t); cur = torch.where(acc, new, cur)
            lam = torch.where(acc, lam * 0.3, lam * 10).clamp(1e-6, 1e6)
        out[idx] = t; L[idx] = cur
    return out, L


def crlb(model, D, th, fix_phi=True, batch=4000):
    """sqrt diag of the inverse Fisher matrix (x, y, z, [Phi], N, bg) at th"""
    var = D['var']; fixed = {3} if fix_phi else set()
    free = [j for j in range(th.shape[1]) if j not in fixed]
    out = torch.full_like(th, float('nan'))
    for s in range(0, len(th), batch):
        idx = torch.arange(s, min(s + batch, len(th)), device=th.device)
        mu, J = expected(model, D, th[idx], idx, grad=True)
        J = J[..., free].reshape(len(idx), -1, len(free)); w = (1.0 / (mu + var).clamp(min=1e-3)).reshape(len(idx), -1)
        F = torch.einsum('npi,np,npj->nij', J, w, J)
        out[idx[:, None], torch.tensor(free, device=th.device)[None]] = torch.sqrt(torch.diagonal(torch.linalg.inv(F), dim1=1, dim2=2).clamp(min=0))
    return out


# ---------------------------------------------------------------------------------------------------------------- PSF fit
PSF_GROUPS = dict(zern=('psf.top.zcoef', 'psf.bot.zcoef'), phi=('psf.phi',), tau=('psf.tau_logit',), a=('psf.a_log',),
                  gamma=('psf.gamma_logit',), dbot=('psf.d_bot',), blur=('psf.logblur',), reg=('reg',), beta=('beta',),
                  na=('psf.top.NA', 'psf.bot.NA'), apod=('psf.top.apod', 'psf.bot.apod'),
                  pupil=('psf.top.pupil_phase', 'psf.bot.pupil_phase', 'psf.top.pupil_amp', 'psf.bot.pupil_amp'))
LR = dict(zern=2e-3, phi=1e-2, tau=1e-2, a=1e-2, gamma=1e-2, dbot=5e-3, blur=1e-2, reg=5e-3, beta=2e-2, na=1e-3, apod=1e-2,
          pupil=2e-3)
def pupil_penalty(model):
    """squared Laplacian of the pupil phase / amplitude maps inside the pupil (sum over both arms)"""
    tot = 0.0
    for arm in (model.psf.top, model.psf.bot):
        if not isinstance(arm.pupil_phase, torch.nn.Parameter): continue
        for M in (arm.pupil_phase, arm.pupil_amp):
            L = M[1:-1, 1:-1] * 4 - M[2:, 1:-1] - M[:-2, 1:-1] - M[1:-1, 2:] - M[1:-1, :-2]
            tot = tot + (L * arm.mask[1:-1, 1:-1]).pow(2).sum()
    return tot


def psf_step(model, D, th, idx_all, groups, steps=200, batch=2000, lr_scale=1.0, seed=0, log=None, pupil_smooth=0.0):
    """Adam on the PSF parameter groups at fixed spot parameters th (minibatches of spots idx_all)"""
    var = D['var']
    named = dict(model.named_parameters())
    params = []
    for g in groups:
        ps = [named[k] for k in PSF_GROUPS[g]]
        for q in ps: q.requires_grad_(True)
        params.append(dict(params=ps, lr=LR[g] * lr_scale))
    for k, q in named.items():
        if not any(k in PSF_GROUPS[g] for g in groups): q.requires_grad_(False)
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    hist = []
    for it in range(steps):
        idx = torch.tensor(rng.choice(idx_all.cpu().numpy(), min(batch, len(idx_all)), replace=False), device=th.device)
        t = th[idx]
        xy = D['pos0'][idx] + t[:, None, :2]
        g = model.images(xy, t[:, 2], t[:, 3])
        mu = t[:, 4, None, None, None] * g + t[:, 5:, None, None]
        loss = nll(mu, D['d'][idx], var).sum() / len(idx)
        if pupil_smooth > 0 and 'pupil' in groups: loss = loss + pupil_smooth * pupil_penalty(model)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(float(loss.detach()))
        if log and (it % 50 == 0 or it == steps - 1): log(f'  psf step {it}: nll/spot {np.mean(hist[-50:]):.2f}')
    for q in model.parameters(): q.requires_grad_(False)
    return hist


def evaluate(model_or_spline, D, th, idx=None, central=4):
    """per-spot deviance per px (whole ROI, central (2 central + 1)^2 px of every channel) and per-channel"""
    with torch.no_grad():
        idx = torch.arange(len(th), device=th.device) if idx is None else idx
        out = []
        for s in range(0, len(idx), 2000):
            j = idx[s:s + 2000]
            mu = expected(model_or_spline, D, th[j], j)
            out.append(deviance(mu, D['d'][j], D['var']))
        dv = torch.cat(out)                    # [n, C, P, P]
        P = dv.shape[-1]; h = P // 2; c0, c1 = h - central, h + central + 1
        return dict(all=dv.mean((1, 2, 3)), ctr=dv[:, :, c0:c1, c0:c1].mean((1, 2, 3)),
                    ch=dv[:, :, c0:c1, c0:c1].mean((2, 3)))


def baked_spline(model, P=17, z_range=ZR, oversample=2, dz_um=0.02, zpad=3, xy_limit=2.5, chunk=128):
    """FourPiSplinePSF of model with its registration offsets (channel c tabulated at (x, y) + reg[c]) and the beta z
    phase baked into the knots (cr + i ci rotated by beta z), i.e. the table SIMCODE's LUT4Pi builds from the state
    (simcode/fourpi/lut4pi.py table4pi); evaluate it with SplineModel(spl, reg=0, beta=0)"""
    from torch_psf.spline import CubicSplinePSF
    p = model.psf
    geo = CubicSplinePSF(P, z_range, oversample=oversample, dz_um=dz_um, zpad=zpad, xy_limit=xy_limit, device='cpu')
    big = p.resized(geo.Pl)
    S, Pl, nz, dv = oversample, geo.Pl, geo.nz, p.dev
    zs = geo.z0 + dz_um * torch.arange(nz, device=dv, dtype=torch.float64)
    a = torch.arange(S, device=dv, dtype=torch.float64) / S
    Zg, B, A = torch.meshgrid(zs, a, a, indexing='ij')
    pos = torch.stack([A.reshape(-1), B.reshape(-1), Zg.reshape(-1)], 1).float()
    reg = model.offsets().detach(); C = p.n_channels
    comps = []
    with torch.no_grad():
        for ph in (0.0, math.pi, 0.5 * math.pi):
            out = []
            for q in pos.split(chunk):
                f = torch.full((len(q),), ph, device=dv)
                out.append(torch.stack([big(q + torch.cat([reg[c], torch.zeros(1, device=dv)])[None], phi_extra=f)[:, c] for c in range(C)], 1))
            comps.append(torch.cat(out))
    g0, gpi, g90 = comps
    inc = 0.5 * (g0 + gpi); cr = 0.5 * (g0 - gpi); ci = inc - g90
    b = (float(model.beta.detach()) * pos[:, 2])[:, None, None, None]
    cr, ci = cr * torch.cos(b) - ci * torch.sin(b), cr * torch.sin(b) + ci * torch.cos(b)
    K = torch.stack([comp.reshape(nz, S, S, C, Pl, Pl).flip(1).flip(2).permute(3, 0, 4, 1, 5, 2).reshape(C, nz, Pl * S, Pl * S)
                     for comp in (inc, cr, ci)])
    return FourPiSplinePSF(P, z_range, K, oversample=S, dz_um=dz_um, zpad=zpad, xy_limit=xy_limit, tau=p.tau.detach(),
                           device=dv, pixel_um=p.pixel_um, source_state=p.state())


def simcode_state(model):
    """the SIMCODE PSF4Pi state of a Model4Pi (simcode/fourpi/psf4pi.py, lut4pi.py): the FourPiPSF parameter dict
    (without its 'raw' copy) + 'arms', 'reg', 'beta'"""
    st = model.state()
    s = dict(st['psf']); s.pop('raw', None)
    s['meta'] = {k: s['meta'][k] for k in ('R', 'px', 'NA', 'n', 'lam', 'npupil', 'os', 'mirror', 'zidx', 'pol')}
    s['arms'] = st['arms']; s['reg'] = st['reg']; s['beta'] = st['beta']
    return s
