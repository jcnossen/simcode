"""S1 shared parts: the calibrated vectorial PSF of the zimflux code as a CubicSplinePSF, the astigmatic and ZIMFLUX models
(analytic Jacobians) for simcode.fourpi.lm.lm_poisson_j, and paths.
Coordinates: x, y in px relative to the ROI centre (index 7.5); absolute px = corner + 7.5 + x (x = column, y = row).
z = emitter height z_e above the coverslip in um (the paper's z = z_e - z_d, z_d = 60 nm). Focal plane at the Strehl maximum
for an emitter at z_d (zimflux config.set_saffocus)."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2'); sys.path.insert(0, '/home/jc/simcode2/analysis/zimflux')
from simcode.psf import VectorialPSF
from simcode.psf import emitter_height_pos_fn
sys.path.insert(0, '/home/jc/simcode2/analysis/baselines')
from zpsf import knot_spline
import crlb_s0 as C
WORK = '/home/jc/simcode2/out/zimflux/s1/' + os.environ.get('ZF_S1_SUB', '')   # variant runs in subfolders
RES = '/mnt/data/dmdsf/simcode2_out/zimflux/surfdrive/Zimflux/3Dnanorulers/simcode_results/'
ZD, P, PX = 0.060, 16, 65.0
ZLO, ZHI = -0.08, 0.45
ab = np.load('/home/jc/simcode2/out/zimflux/ref/zimflux/vectorpsf/delft_beads_feb23_15nm.npy').copy(); ab[6] = -0.2
ORD = [(2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3), (4, 0), (4, -2), (4, 2), (5, -1), (5, 1), (6, 0)]
ZERN = {C.NOLL[o]: float(a) / 715.0 for o, a in zip(ORD, ab)}


def transpose_zern(z):
    """Zernikes of the x <-> y transposed pupil (phi -> pi/2 - phi): m = 1 cos <-> sin, m = 2 cos -> -cos,
    m = 3 cos -> -sin, sin -> -cos; m = 0 unchanged."""
    from simcode.psf import noll_to_nm
    out = {}
    for j, c in z.items():
        n, m = noll_to_nm(j); am = abs(m)
        if am == 0: out[j] = out.get(j, 0) + c; continue
        if am == 1: m2, s = -m, 1
        elif am == 2: m2, s = m, (-1 if m > 0 else 1)
        elif am == 3: m2, s = -m, -1
        else: raise ValueError(j)
        j2 = C.NOLL[(n, m2)]; out[j2] = out.get(j2, 0) + s * c
    return out


def make_lut(dev, astig_sign=1.0, zern=None, variant=None, focus_shift=0.0):
    """variant: None / 'flip' (astigmatism Z6 sign flipped) / 'T' (whole pupil transposed x <-> y)."""
    zern = dict(zern or ZERN); zern[6] = astig_sign * zern[6]
    if variant == 'flip': zern[6] = -zern[6]
    if variant == 'T': zern = transpose_zern(zern)
    C.dev = dev
    f = C.focus_for_depth(ZD) + focus_shift
    psf = VectorialPSF(P, pixel_um=0.065, NA=1.49, n_imm=1.52, n_med=1.33, wavelength_um=0.715, zernike_indices=sorted(zern),
                       npupil=64, oversample=3, device=dev)
    with torch.no_grad():
        psf.zcoef[:] = torch.tensor([zern[j] for j in sorted(zern)], device=dev)
    # knots of torch_psf.CubicSplinePSF for the odd ROI P + 1 (the same knot grid as P: Pl = 39), evaluated on the even P x P ROI
    big = psf.to_spline(P + 1, (0.0, ZHI + 0.05), pos_fn=emitter_height_pos_fn(f), oversample=6, dz_um=0.01, zpad=12,
                        xy_limit=P / 2 + 1, chunk=4)
    lut = knot_spline(big.knots, big.S, big.u0, big.z0, big.hz, P, xmax=P / 2 + 1)
    return lut, f, zern


def astig_model(lut):
    def fj(th):                     # th [n,5] = x, y, z, N, b (b per pixel of the summed image)
        h, dx, dy, dz = lut.render(th[:, :3], grad=True)
        n = th.shape[0]; N = th[:, 3, None]
        mu = N * h.view(n, -1) + th[:, 4, None]
        J = torch.stack([N * dx.view(n, -1), N * dy.view(n, -1), N * dz.view(n, -1), h.view(n, -1),
                         torch.ones_like(mu)], -1)
        return mu, J
    return fj


def pattern_w(pat, xa, ya, z, t):
    """w [n,K], dw/d(x, y, z) [n,K,3]; xa, ya absolute px, z um, t = cycle index. pat: dict q [2] rad/px, kz rad/um,
    m [K], eta [K], psi [K, nb] (rad) at bin centres tb [nb] (linear interpolation in t)."""
    q = pat['q']; kz = pat['kz']; m = pat['m']; eta = pat['eta']
    tb = pat['tb']; psi = pat['psi']                                  # [K,nb]
    tt = t.clamp(tb[0], tb[-1])
    i = torch.searchsorted(tb, tt.contiguous()).clamp(1, len(tb) - 1)
    a = ((tt - tb[i - 1]) / (tb[i] - tb[i - 1]))[:, None]
    # interpolate the phase as a unit vector (no wrap problems)
    c = torch.cos(psi); s = torch.sin(psi)
    ci = c[:, i - 1].T * (1 - a) + c[:, i].T * a; si = s[:, i - 1].T * (1 - a) + s[:, i].T * a
    ps = torch.atan2(si, ci)                                          # [n,K]
    ph = (q[0] * xa + q[1] * ya + kz * z)[:, None] - ps
    w = eta * (1 + m * torch.cos(ph))
    dphi = -eta * m * torch.sin(ph)
    dw = torch.stack([dphi * q[0], dphi * q[1], dphi * kz], -1)
    return w, dw


def zf_model(lut, pat, cx, cy, t):
    """ZIMFLUX model on the K frames: mu_kj = N w_k(r) H_j(r) + b; th = x, y, z, N, b (b per pixel per frame)."""
    def fj(th):
        n = th.shape[0]
        h, dx, dy, dz = lut.render(th[:, :3], grad=True)
        h, dx, dy, dz = [a.view(n, 1, -1) for a in (h, dx, dy, dz)]
        w, dw = pattern_w(pat, cx + 7.5 + th[:, 0], cy + 7.5 + th[:, 1], th[:, 2], t)
        N = th[:, 3, None, None]; w3 = w[..., None]
        mu = N * w3 * h + th[:, 4, None, None]
        J = torch.stack([N * (dw[..., 0, None] * h + w3 * dx), N * (dw[..., 1, None] * h + w3 * dy),
                         N * (dw[..., 2, None] * h + w3 * dz), w3 * h, torch.ones_like(mu)], -1)
        return mu.reshape(n, -1), J.reshape(n, -1, 5)
    return fj


def pat_to_torch(p, dev):
    return {k: torch.tensor(np.asarray(v), device=dev, dtype=torch.float32) for k, v in p.items()}
