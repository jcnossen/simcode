"""Spline LUT of the coherent 4Pi PSF (simcode/fourpi/coherent_psf.py), with the 4Pi phase offset as an exact analytic
dependence.

For a global interferometer phase offset Phi (coherent_psf phi_extra) every channel image is exactly
    g_c(x, y, z; Phi) = inc_c(x, y, z) + cos(Phi) cr_c(x, y, z) - sin(Phi) ci_c(x, y, z)
(inc = incoherent part, cr + i ci = coherent cross term). The three components per channel are tabulated from
g(0), g(pi), g(pi/2) as 12 Catmull-Rom spline z-stacks: torch_psf.CubicSplinePSF instances created with their knots set
directly (the convolutional 3D spline of torch-spline-3d).

LUT4Pi(psf, zmin, zmax).evaluate(pos [N,3], Phi [N], grad=False) -> [N,4,P,P] (and d/dx, d/dy, d/dz, d/dPhi).

Optional keys of the state (in-situ PSF, analysis/fourpi_plain/insitu4pi.py; states without them are unchanged):
  'arms'    per-arm NA, n_med, apod, saf, pupil maps (simcode/fourpi/coherent_psf.py set_arm_terms)
  'reg'     per-channel image offsets [C][2] px: channel c is tabulated at (x, y) + reg[c]
  'beta'    extra 4Pi phase beta z (rad / um): cr + i ci rotated by beta z in every plane
  'spline'  path of a refined knot table (analysis/fourpi_plain/p4spline_refine.py: K [3, C, nz, n, n] = the table
            of 'spline_base' (the vectorial state it was refined from) + a knot correction dK, on its own grid).
            The table is K_ref + (table(this state) - table(spline_base)) on the refined grid: exactly K_ref
            for the base state itself, the refined table plus the change of the vectorial model for a
            perturbed state (the simulator's PSF bank). sub / hz / zmin / zmax are then those of the file.
"""
import math
import torch
from simcode.psf import CubicSplinePSF
from simcode.fourpi.coherent_psf import PSF4Pi, psf_from_state


def _lut_from_stack(G, P, S, zmin, zmax, hz, zpad, dev):
    """CubicSplinePSF with the knots G [nz, nv, nu] (offset limit P / 2 + 1 px)"""
    return CubicSplinePSF(P, (zmin, zmax), oversample=S, dz_um=hz, zpad=zpad, xy_limit=P / 2 + 1, device=dev, knots=G)


class LUT4Pi:
    def __init__(self, psf_state, P=15, zmin=-0.6, zmax=0.6, hz=0.01, sub=4, zpad=3, chunk=128, dev='cuda'):
        if psf_state.get('spline'):
            return self._init_spline(psf_state, P, chunk, dev)
        self.P, self.S = P, sub
        xmax = P / 2 + 1
        c = int(math.ceil((P - 1) / 2 + xmax)) + 2
        G3 = table4pi(psf_state, 2 * c + 1, zmin, zmax, hz, sub, zpad, chunk, dev)
        self.tau = torch.tensor(psf_state['tau'], device=dev)
        self.luts = [[_lut_from_stack(G3[k, ch], P, sub, zmin, zmax, hz, zpad, dev) for ch in range(G3.shape[1])] for k in range(3)]
        self.zmin, self.zmax = zmin, zmax

    def _init_spline(self, st, P, chunk, dev):
        f = torch.load(st['spline'], weights_only=False, map_location='cpu')
        S, hz, zpad, (zmin, zmax) = f['S'], f['dz'], f['zpad'], f['z_range']
        K = f['K'].to(dev)
        n = K.shape[-1]; c = (n // S - 1) // 2
        base = dict(f['vector_state'] if st.get('spline_base') is None else st['spline_base'])
        same = json_eq(base, {k: v for k, v in st.items() if k not in ('spline', 'spline_base')})
        if not same:
            K = K + table4pi(st, 2 * c + 1, zmin, zmax, hz, S, zpad, chunk, dev) - table4pi(base, 2 * c + 1, zmin, zmax, hz, S, zpad, chunk, dev)
        self.P, self.S = P, S
        self.tau = torch.tensor(st['tau'], device=dev)
        self.luts = [[CubicSplinePSF(P, (zmin, zmax), oversample=S, dz_um=hz, zpad=zpad, xy_limit=P / 2 + 1, device=dev, knots=K[k, ch])
                      for ch in range(K.shape[1])] for k in range(3)]
        self.zmin, self.zmax = zmin, zmax

    def components(self, pos, grad=False):
        """-> [3 (inc, cr, ci), N, 4, P, P] (+ the same for d/dx, d/dy, d/dz if grad)"""
        out = [[L.render(pos, grad) for L in row] for row in self.luts]
        if not grad:
            return torch.stack([torch.stack(r, 1) for r in out])
        return [torch.stack([torch.stack([o[k] for o in r], 1) for r in out]) for k in range(4)]

    def evaluate_ch(self, pos, Phi, c):
        """one channel: [N,P,P]"""
        inc, cr, ci = (row[c].render(pos) for row in self.luts)
        return inc + torch.cos(Phi)[:, None, None] * cr - torch.sin(Phi)[:, None, None] * ci

    def evaluate_ch_grad(self, pos, Phi, c):
        """one channel with derivatives: value, d/dx, d/dy, d/dz, d/dPhi, each [N,P,P]"""
        cP, sP = torch.cos(Phi)[:, None, None], torch.sin(Phi)[:, None, None]
        a, b, e = (row[c].render(pos, True) for row in self.luts)
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


def json_eq(a, b):
    import json
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def table4pi(psf_state, Pl, zmin, zmax, hz, sub, zpad, chunk, dev):
    """knot table [3 (inc, cr, ci), C, nz, Pl sub, Pl sub] of a 4Pi PSF state (with the optional reg / beta / arms keys)"""
    st = dict(psf_state); st['meta'] = dict(st['meta']); st['meta']['R'] = Pl
    kw = {'zmode': 'emitter'} if st['meta'].get('asym') else {}   # movies: the emitter moves in the medium
    big = psf_from_state(st, device=dev, **kw).eval()     # symmetric (PSF4Pi) or asymmetric-arm (PSF4PiA) state
    C = len(st['tau'])
    nz = int(round((zmax - zmin) / hz)) + 1 + 2 * zpad
    z0 = zmin - zpad * hz
    zs = z0 + hz * torch.arange(nz, device=dev, dtype=torch.float64)
    a = torch.arange(sub, device=dev, dtype=torch.float64) / sub
    Zg, B, A = torch.meshgrid(zs, a, a, indexing='ij')
    pos = torch.stack([A.reshape(-1), B.reshape(-1), Zg.reshape(-1)], 1).float()
    reg = st.get('reg')
    comps = []
    with torch.no_grad():
        for ph in (0.0, math.pi, 0.5 * math.pi):
            if reg is None:
                out = [big(p, phi_extra=torch.full((len(p),), ph, device=dev)) for p in pos.split(chunk)]
            else:          # channel c at (x, y) + reg[c]
                out = []
                for p in pos.split(chunk):
                    f = torch.full((len(p),), ph, device=dev)
                    out.append(torch.stack([big(p + torch.tensor([reg[ch][0], reg[ch][1], 0.0], device=dev), phi_extra=f)[:, ch]
                                            for ch in range(C)], 1))
            comps.append(torch.cat(out))                                    # [M,C,Pl,Pl]
    g0, gpi, g90 = comps
    inc = 0.5 * (g0 + gpi); cr = 0.5 * (g0 - gpi); ci = inc - g90
    if st.get('beta'):
        b = (float(st['beta']) * pos[:, 2])[:, None, None, None]
        cr, ci = cr * torch.cos(b) - ci * torch.sin(b), cr * torch.sin(b) + ci * torch.cos(b)
    G = [comp.reshape(nz, sub, sub, C, Pl, Pl).flip(1).flip(2).permute(3, 0, 4, 1, 5, 2).reshape(C, nz, Pl * sub, Pl * sub)
         for comp in (inc, cr, ci)]
    return torch.stack(G)
