"""Spline LUT of the coherent 4Pi PSF (simcode/fourpi/psf4pi.py), with the 4Pi phase offset as an exact analytic
dependence.

For a global interferometer phase offset Phi (psf4pi phi_extra) every channel image is exactly
    g_c(x, y, z; Phi) = inc_c(x, y, z) + cos(Phi) cr_c(x, y, z) - sin(Phi) ci_c(x, y, z)
(inc = incoherent part, cr + i ci = coherent cross term). The three components per channel are tabulated from
g(0), g(pi), g(pi/2) as 12 Catmull-Rom spline z-stacks, reusing simcode.psf.conv_spline.ConvSplinePSF's evaluation (the convolutional 3D spline of torch-spline-3d,
https://gitlab.com/jcnossen/torch-spline-3d) (instances are
created with their knot stacks set directly; ConvSplinePSF itself is imported, not modified).

LUT4Pi(psf, zmin, zmax).evaluate(pos [N,3], Phi [N], grad=False) -> [N,4,P,P] (and d/dx, d/dy, d/dz, d/dPhi).
"""
import math
import torch
from simcode.psf.conv_spline import ConvSplinePSF, geometry_matrix
from simcode.fourpi.psf4pi import PSF4Pi, psf_from_state


def _lut_from_stack(G, P, S, z0, hz, c, dev):
    L = object.__new__(ConvSplinePSF)
    L.P = L.roisize = P; L.focus = 0.0; L.dev = dev; L.S = S
    L.xmax = float(P / 2 + 1)
    nz = G.shape[0]
    L.z0, L.hz, L.nz = z0, hz, nz
    L.u0 = -(c * S + S - 1) / S
    L.zstack = G.contiguous()
    D, H, W = L.zstack.shape; L.HW, L.W = H * W, W
    L.Lw = Lw = (P - 1) * S + 4
    r = lambda n_: torch.arange(n_, device=dev)
    L.pat = (r(4)[:, None, None] * H * W + r(Lw)[None, :, None] * W + r(Lw)[None, None, :]).reshape(-1)
    L.G = geometry_matrix(dev)
    e = torch.arange(4, device=dev)
    L.e, L.em1 = e, (e - 1).clamp_min(0)
    o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
    L.oS0 = float((o[0] - L.u0) * S)
    return L


class LUT4Pi:
    def __init__(self, psf_state, P=15, zmin=-0.6, zmax=0.6, hz=0.01, sub=4, zpad=3, chunk=128, dev='cuda'):
        self.P, self.S = P, sub
        xmax = P / 2 + 1
        c = int(math.ceil((P - 1) / 2 + xmax)) + 2
        Pl = 2 * c + 1
        st = dict(psf_state); st['meta'] = dict(st['meta']); st['meta']['R'] = Pl
        kw = {'zmode': 'emitter'} if st['meta'].get('asym') else {}   # movies: the emitter moves in the medium
        big = psf_from_state(st, device=dev, **kw).eval()     # symmetric (PSF4Pi) or asymmetric-arm (PSF4PiA) state
        self.tau = torch.tensor(st['tau'], device=dev)
        nz = int(round((zmax - zmin) / hz)) + 1 + 2 * zpad
        z0 = zmin - zpad * hz
        zs = z0 + hz * torch.arange(nz, device=dev, dtype=torch.float64)
        a = torch.arange(sub, device=dev, dtype=torch.float64) / sub
        Zg, B, A = torch.meshgrid(zs, a, a, indexing='ij')
        pos = torch.stack([A.reshape(-1), B.reshape(-1), Zg.reshape(-1)], 1).float()
        comps = []
        with torch.no_grad():
            for ph in (0.0, math.pi, 0.5 * math.pi):
                out = [big(p, phi_extra=torch.full((len(p),), ph, device=dev)) for p in pos.split(chunk)]
                comps.append(torch.cat(out))                                    # [M,4,Pl,Pl]
        g0, gpi, g90 = comps
        inc = 0.5 * (g0 + gpi); cr = 0.5 * (g0 - gpi); ci = inc - g90
        self.luts = []
        for comp in (inc, cr, ci):
            row = []
            for ch in range(4):
                im = comp[:, ch].reshape(nz, sub, sub, Pl, Pl)
                G = im.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(nz, Pl * sub, Pl * sub)
                row.append(_lut_from_stack(G, P, sub, float(z0), hz, c, dev))
            self.luts.append(row)
        self.zmin, self.zmax = zmin, zmax

    def components(self, pos, grad=False):
        """-> [3 (inc, cr, ci), N, 4, P, P] (+ the same for d/dx, d/dy, d/dz if grad)"""
        out = [[L.evaluate(pos, grad) for L in row] for row in self.luts]
        if not grad:
            return torch.stack([torch.stack(r, 1) for r in out])
        return [torch.stack([torch.stack([o[k] for o in r], 1) for r in out]) for k in range(4)]

    def evaluate_ch(self, pos, Phi, c):
        """one channel: [N,P,P]"""
        inc, cr, ci = (row[c].evaluate(pos) for row in self.luts)
        return inc + torch.cos(Phi)[:, None, None] * cr - torch.sin(Phi)[:, None, None] * ci

    def evaluate_ch_grad(self, pos, Phi, c):
        """one channel with derivatives: value, d/dx, d/dy, d/dz, d/dPhi, each [N,P,P]"""
        cP, sP = torch.cos(Phi)[:, None, None], torch.sin(Phi)[:, None, None]
        a, b, e = (row[c].evaluate(pos, True) for row in self.luts)
        out = [a[k] + cP * b[k] - sP * e[k] for k in range(4)]
        return out + [-sP * b[0] - cP * e[0]]

    def evaluate(self, pos, Phi, grad=False):
        cP, sP = torch.cos(Phi)[:, None, None, None], torch.sin(Phi)[:, None, None, None]
        if not grad:
            inc, cr, ci = self.components(pos)
            return inc + cP * cr - sP * ci
        v, dx, dy, dz = self.components(pos, True)
        f = lambda c: c[0] + cP * c[1] - sP * c[2]
        dphi = -sP * v[1] - cP * v[2]
        return f(v), f(dx), f(dy), f(dz), dphi
