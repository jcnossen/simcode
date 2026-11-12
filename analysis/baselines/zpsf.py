"""One PSF input for the classical / DECODE baselines of the paper rerun (SMLM, SIMFLUX, DECODE all use the SAME PSF).

spec (one string, the `--psf` argument of paper_pipeline.py and decode_paper.py):
  gauss:<sigma_px>        2D Gaussian (pixel-integrated, erf), e.g. gauss:1.3 = the published paper's PSF
  <calib>.json            in-situ calibration json (VectorialPSF fields + opt-in extras). A per-pattern-angle PSF
                          ('psf_dipole_M_angle' / 'psf_blur_angle_px2') enters as the AVERAGE OF THE TWO ANGLE IMAGES
                          (exact: both angles are rendered and the knot stacks averaged).
  <zstack>.npz            z-stack export (format below), e.g. the PSF worker's averaged per-dataset PSF.

z-stack npz format (written by `python analysis/baselines/zpsf.py export <calib.json> <out.npz>`):
  zstack [nz, nv, nu] float32   g(u_j, v_i, z_k): expected fraction of the emitter's photons in a 1 x 1 px pixel whose
                                centre lies at (u, v) px from the emitter (u = column / x, v = row / y); z = emitter height (um)
  u0     float                  u (= v) of knot 0, px;  u_j = u0 + j / S
  S      int                    knots per px (1 = a plain pixel-sampled stack with the emitter at a pixel centre)
  z0_um, dz_um                  z_k = z0_um + k dz_um
  emitter_z_um (optional)       plane used by the 2D fits (default: the calib's emitter_z_median_um, else 0.055)
  focus_um, source (optional)   metadata
The knots are interpolated with the Catmull-Rom 3D spline of simcode.psf.lut.PSFLUT (value and d/dx, d/dy, d/dz).

API: psf = load(spec); g, gx, gy = psf.eval(xc, yc, P[, z]) with xc, yc [N] = emitter position relative to the ROI centre (px;
pixel j of a P-px ROI has its centre at j - (P-1)/2), returns [N,P,P] tensors (g sums to ~1 over the full plane; gx = dg/dxc).
psf.z_fit = the fit plane (um). psf.desc = a short description (stored in the outputs)."""
import os, sys, json, math
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.psf.lut import PSFLUT, geometry_matrix


class GaussPSF:
    def __init__(self, sigma, device='cuda'):
        self.sigma = float(sigma); self.dev = torch.device(device); self.z_fit = 0.0
        self.desc = f'gauss:{self.sigma}'

    def eval(self, xc, yc, P, z=None):
        s = self.sigma * math.sqrt(2)
        o = torch.arange(P, device=xc.device, dtype=torch.float32) - (P - 1) / 2
        def ax(c):
            a = (o[None] + 0.5 - c[:, None]) / s; b = (o[None] - 0.5 - c[:, None]) / s
            e = 0.5 * (torch.erf(a) - torch.erf(b))
            de = -(torch.exp(-a ** 2) - torch.exp(-b ** 2)) / (math.sqrt(math.pi) * s)      # d e / d c
            return e, de
        ex, dex = ax(xc); ey, dey = ax(yc)
        g = ey[:, :, None] * ex[:, None, :]
        return g, ey[:, :, None] * dex[:, None, :], dey[:, :, None] * ex[:, None, :]


class ZStackLUT(PSFLUT):
    """PSFLUT.evaluate on an arbitrary knot stack (no VectorialPSF needed)."""
    def __init__(self, zstack, S, u0, z0, hz, P, xmax=None):
        self.P = self.roisize = P; self.S = S = int(S); self.dev = zstack.device
        self.z0, self.hz, self.nz = float(z0), float(hz), zstack.shape[0]
        self.u0 = float(u0); self.focus = float('nan')
        self.zstack = zstack.contiguous(); D, H, W = zstack.shape; self.HW, self.W = H * W, W
        assert H == W, 'square knot grid expected'
        self.Lw = Lw = (P - 1) * S + 4
        r = lambda n_: torch.arange(n_, device=self.dev)
        self.pat = (r(4)[:, None, None] * H * W + r(Lw)[None, :, None] * W + r(Lw)[None, None, :]).reshape(-1)
        self.G = geometry_matrix(self.dev)
        e = torch.arange(4, device=self.dev); self.e, self.em1 = e, (e - 1).clamp_min(0)
        o0 = -(P - 1) / 2
        self.oS0 = float((o0 - self.u0) * S)
        # largest |x| whose 4 x Lw x Lw knot block stays inside the grid (both sides)
        xm_lo = o0 - self.u0 - 1.0 / S                                   # x = +xmax: block corner >= 0
        xm_hi = (W - 3) / S - (P - 1) - (o0 - self.u0)                   # x = -xmax: block end <= W
        lim = min(xm_lo, xm_hi) - 1e-3
        assert lim > 0.5, f'knot grid too small for a {P}-px ROI (|x| <= {lim:.2f})'
        self.xmax = float(min(lim, P / 2 + 1) if xmax is None else min(xmax, lim))


class SplinePSF:
    def __init__(self, zstack, S, u0, z0, hz, z_fit, desc, device='cuda', meta=None):
        self.dev = torch.device(device)
        self.knots = (zstack.detach().float().to(self.dev) if torch.is_tensor(zstack) else
                      torch.as_tensor(np.asarray(zstack, np.float32), device=self.dev))
        self.S, self.u0, self.z0, self.hz = int(S), float(u0), float(z0), float(hz)
        self.z_fit = float(z_fit); self.desc = desc; self.meta = meta or {}
        self._luts = {}

    def lut(self, P):
        if P not in self._luts:
            self._luts[P] = ZStackLUT(self.knots, self.S, self.u0, self.z0, self.hz, P)
        return self._luts[P]

    def eval(self, xc, yc, P, z=None, grad=True):
        L = self.lut(P)
        zz = torch.full_like(xc, self.z_fit) if z is None else z
        pos = torch.stack([xc, yc, zz], 1).float()
        if not grad:
            return L.evaluate(pos)
        g, gx, gy, gz = L.evaluate(pos, grad=True)
        return g, gx, gy

    def save(self, fn, **extra):
        np.savez_compressed(fn, zstack=self.knots.cpu().numpy(), u0=self.u0, S=self.S, z0_um=self.z0, dz_um=self.hz,
                            emitter_z_um=self.z_fit, source=self.desc, **{k: v for k, v in {**self.meta, **extra}.items()})


def render_calib(calib, device='cuda', P=41, xmax=1.5, zmax=0.2, sub=6, hz=0.01, angles='avg'):
    """knot stack of a calib json's VectorialPSF (as simcode.pipeline.common.load_calib, larger ROI). angles: 'avg' = mean of
    the two angle images when the calib has a per-angle PSF; 0 / 1 = that angle; None = set_psf_extras(angle=None)."""
    from simcode.config import resolve
    from simcode.psf.vectorial import VectorialPSF, set_psf_extras, has_angle_psf
    path = resolve(calib); c = json.load(open(path))
    psf = VectorialPSF(P, pixelsize=c['pixelsize_um'], NA=c['NA'], n_imm=c['n_imm'], n_med=c['n_med'],
                       wavelength=c['wavelength_um'], oversample=3, device=device)
    with torch.no_grad():
        psf.zcoef.copy_(torch.tensor(c['zernike_waves'], device=device)); psf.apod.copy_(torch.tensor(c['apod'], device=device))
    ang = has_angle_psf(c)
    which = [0, 1] if (angles == 'avg' and ang) else ([angles] if isinstance(angles, int) else [None])
    stacks = []
    for a in which:
        set_psf_extras(psf, c, angle=a)
        L = PSFLUT(psf, c['focus_um'], zmax, sub=sub, hz=hz, xmax=xmax)
        stacks.append(L.zstack)
    zs = torch.stack(stacks).mean(0)
    desc = f'calib:{os.path.basename(path)} angles={"avg(0,1)" if len(which) == 2 else which[0]}'
    return SplinePSF(zs, L.S, L.u0, L.z0, L.hz, c.get('emitter_z_median_um', 0.055), desc, device,
                     meta=dict(focus_um=float(c['focus_um'])))


def load(spec, device='cuda', z_fit=None):
    if spec.startswith('gauss:'):
        return GaussPSF(float(spec.split(':', 1)[1]), device)
    if spec.endswith('.json'):
        p = render_calib(spec, device)
    elif spec.endswith('.npz'):
        d = np.load(spec, allow_pickle=True)
        zf = float(d['emitter_z_um']) if 'emitter_z_um' in d.files else 0.055
        p = SplinePSF(d['zstack'], int(d['S']), float(d['u0']), float(d['z0_um']), float(d['dz_um']), zf,
                      f'zstack:{os.path.basename(spec)}' + (f' ({d["source"]})' if 'source' in d.files else ''), device)
    else:
        raise ValueError(f'unknown PSF spec {spec}')
    if z_fit is not None: p.z_fit = float(z_fit)
    return p


if __name__ == '__main__':
    # export <calib.json> <out.npz>   |   check <spec>  (normalisation, centroid, width of the fit-plane PSF)
    cmd = sys.argv[1]
    if cmd == 'export':
        p = render_calib(sys.argv[2]); p.save(sys.argv[3]); print('saved', sys.argv[3], tuple(p.knots.shape), p.desc)
    elif cmd == 'check':
        p = load(sys.argv[2])
        for P in (8, 15, 31):
            x = torch.tensor([0.0, 0.3, -0.45], device='cuda'); y = torch.tensor([0.0, -0.2, 0.45], device='cuda')
            g, gx, gy = p.eval(x, y, P)
            o = torch.arange(P, device='cuda') - (P - 1) / 2
            s = g.sum((1, 2)); cx = (g.sum(1) * o).sum(1) / s; cy = (g.sum(2) * o).sum(1) / s
            print(P, 'sum', s.cpu().numpy().round(4), 'centroid x', cx.cpu().numpy().round(3), 'y', cy.cpu().numpy().round(3))
        # numerical vs analytic derivative
        e = 1e-3; x = torch.tensor([0.21], device='cuda'); y = torch.tensor([-0.13], device='cuda')
        g, gx, gy = p.eval(x, y, 8)
        nx = (p.eval(x + e, y, 8)[0] - p.eval(x - e, y, 8)[0]) / (2 * e)
        print('dg/dx max abs err', float((nx - gx).abs().max()), 'max |gx|', float(gx.abs().max()))
