"""Dense-scene variant of the 4Pi candidate-patch simulator (simcode/fourpi/sim4pi.py, imported read-only; only the neighbour
generator is replaced). For the high-density head (hd) experiment, 2026-11-15.

Main emitter, kinetics, SIM pattern, 4Pi phase, background, camera noise and the GT targets are FourPiPatchSim's, unchanged.
The neighbours ('other ON emitters' in the patch) follow a per-sample scene mode:
  iso      (p_iso)      no neighbour;
  v10      (p_v10)      the v10 rule: 1..3 neighbours at r log-uniform 0.6-7 px, z = z0 + N(0, 0.15 um);
  line     (p_line)     microtubule-like: a local ON density rho log-uniform [rho_lo, rho_hi] /um^2 sets a Poisson count
                        lam = cf * rho * A (A = the 15.6 x 15.6 px square around the patch centre, cf = 1.7 = the measured crowding
                        of the MT field over a uniform field); a fraction U(0.3, 0.8) lies on a line through the main emitter
                        (random direction, perpendicular jitter 0.2 px, z = z0 + slope * s, slope N(0, 0.02 um/px), + N(0, 0.02))
                        and on 0-2 other lines (crossings: random offset / direction, z offset U(-0.35, 0.35) um), the rest uniform
                        in the square with z = z0 + U(-0.35, 0.35);
  uniform  (rest)       the same density law, all uniform in the square, z = z0 + N(0, 0.15) or uniform over the z range (1:1).
In line / uniform scenes a CLOSE PAIR is added with p_close: one neighbour at r log-uniform [0.5, 2.5] px, z = z0 + U(-0.4, 0.4) um
(the fringe-ambiguous overlaps). Neighbours are kept inside |dx|, |dy| <= 7.8 px (the LUT's offset limit is 8.5 px incl. the
channel offset). Brightness / kinetics / 4Pi phase of the neighbours as in FourPiPatchSim. At most nb_cap neighbours per sample.
nnb = number of neighbours whose centre is inside the 15 x 15 patch (|dx|, |dy| < 7.5), as the sweep's crowding count.
"""
import math, os, sys
import torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.simulate.fourpi_patches import FourPiPatchSim

PX_UM = 0.128
HALF = 7.8


class Sim4PiHD(FourPiPatchSim):
    def __init__(self, lut, p_iso=0.2, p_v10=0.2, p_line=0.4, rho=(0.08, 2.0), cf=1.7, p_close=0.3, nb_cap=24, **kw):
        super().__init__(lut, **kw)
        self.p_iso, self.p_v10, self.p_line, self.rho, self.cf, self.p_close, self.nb_cap = p_iso, p_v10, p_line, rho, cf, p_close, nb_cap

    def neighbours(self, n, x, y, z, zr):
        """flat neighbour list: (idx [k] sample index, xn, yn, zn)"""
        d = self.dev; g = self.g
        mode = self.u(n)
        iso = mode < self.p_iso
        v10 = (mode >= self.p_iso) & (mode < self.p_iso + self.p_v10)
        line = (mode >= self.p_iso + self.p_v10) & (mode < self.p_iso + self.p_v10 + self.p_line)
        dens = ~(iso | v10)
        I, X, Y, Z = [], [], [], []
        # v10 rule
        nv = torch.where(v10, torch.randint(1, 4, (n,), device=d, generator=g), torch.zeros(n, dtype=torch.long, device=d))
        iv = torch.repeat_interleave(torch.arange(n, device=d), nv); k = len(iv)
        if k:
            r = self.lu(k, 0.6, 7.0); th = self.u(k, 0, 2 * math.pi)
            I.append(iv); X.append(x[iv] + r * torch.cos(th)); Y.append(y[iv] + r * torch.sin(th))
            Z.append(z[iv] + 0.15 * torch.randn(k, device=d, generator=g))
        # density scenes
        A = (2 * HALF * PX_UM) ** 2
        rho = self.lu(n, *self.rho)
        lam = torch.where(dens, self.cf * rho * A, torch.zeros(n, device=d))
        cnt = torch.poisson(lam, generator=g).long().clamp(max=self.nb_cap)
        idd = torch.repeat_interleave(torch.arange(n, device=d), cnt); k = len(idd)
        if k:
            on_line = line[idd]
            # line 0 through the main emitter, lines 1-2 crossings; per sample a line fraction and the line parameters
            fl = self.u(n, 0.3, 0.8); nlines = torch.randint(1, 4, (n,), device=d, generator=g)
            th0 = self.u(n * 3, 0, math.pi).reshape(n, 3)
            off = torch.cat([torch.zeros(n, 1, device=d), self.u(n * 2, -6.0, 6.0).reshape(n, 2)], 1)
            dzl = torch.cat([torch.zeros(n, 1, device=d), self.u(n * 2, -0.35, 0.35).reshape(n, 2)], 1)
            slope = 0.02 * torch.randn(n * 3, device=d, generator=g).reshape(n, 3)
            u_ = self.u(k)
            pick_line = on_line & (u_ < fl[idd])
            li = (self.u(k) * nlines[idd].float()).long().clamp(max=2)          # which line
            li = torch.where(self.u(k) < 0.6, torch.zeros_like(li), li)         # most on the main emitter's own tube
            li = torch.minimum(li, nlines[idd] - 1)
            t = th0[idd, li]; s = self.u(k, -11.0, 11.0); pj = 0.2 * torch.randn(k, device=d, generator=g)
            # line li: points p = c_l + s e_t + pj e_n, c_l = main + off_l e_n
            ex, ey = torch.cos(t), torch.sin(t); nx, ny = -ey, ex
            xl = x[idd] + (off[idd, li] + pj) * nx + s * ex; yl = y[idd] + (off[idd, li] + pj) * ny + s * ey
            zl = z[idd] + dzl[idd, li] + slope[idd, li] * s + 0.02 * torch.randn(k, device=d, generator=g)
            xu = self.u(k, -HALF, HALF); yu = self.u(k, -HALF, HALF)
            zu_line = z[idd] + self.u(k, -0.35, 0.35)
            zu_uni = torch.where(self.u(k) < 0.5, z[idd] + 0.15 * torch.randn(k, device=d, generator=g), self.u(k, *zr))
            zu = torch.where(on_line, zu_line, zu_uni)
            I.append(idd); X.append(torch.where(pick_line, xl, xu)); Y.append(torch.where(pick_line, yl, yu)); Z.append(torch.where(pick_line, zl, zu))
        # close pairs
        cp = dens & (self.u(n) < self.p_close)
        ic = torch.nonzero(cp)[:, 0]; k = len(ic)
        if k:
            r = self.lu(k, 0.5, 2.5); th = self.u(k, 0, 2 * math.pi)
            I.append(ic); X.append(x[ic] + r * torch.cos(th)); Y.append(y[ic] + r * torch.sin(th)); Z.append(z[ic] + self.u(k, -0.4, 0.4))
        if not I:
            e = torch.zeros(0, device=d); return e.long(), e, e, e
        I, X, Y, Z = (torch.cat(v) for v in (I, X, Y, Z))
        keep = (X.abs() <= HALF) & (Y.abs() <= HALF) & ((X - x[I]) ** 2 + (Y - y[I]) ** 2 > 0.25 ** 2)
        return I[keep], X[keep], Y[keep], Z[keep].clamp(*zr)

    def sample(self, n, F=(60.0, 2500.0), zr=(-0.45, 0.45), bg=(0.2, 2.0), jit=0.35,
               m=(0.6, 0.95), rn=1.14, offerr=0.2, p_on=0.6, eps_range=math.pi, **_):
        S = super().sample(n, F=F, zr=zr, bg=bg, p_nb=0.0, nb_max=1, jit=jit, m=m, rn=rn, offerr=offerr, p_on=p_on, eps_range=eps_range)
        d, P = self.dev, self.P
        x, y, z = S['x'], S['y'], S['z']
        I, xn, yn, zn = self.neighbours(n, x, y, z, zr)
        k = len(I)
        mu = S['mu0'].clone()
        if k:
            ang = torch.tensor([0, 0, 0, 1, 1, 1], device=d)
            Fn = self.lu(k, *F); qn = self.kinetics(k, p_on)
            Ntn = Fn[:, None] * qn * self.pattern(xn, yn, S['m'][I], S['eps'][I])
            Phin = S['Phi0'][I] + 0.05 * torch.randn(k, device=d, generator=self.g)
            doff = S['doff'][I]
            for c in range(4):
                pc = torch.stack([xn + doff[:, c, 0], yn + doff[:, c, 1], zn], 1)
                gimg = self.render_ch(pc, Phin, c)[:, ang]                       # [k,6,P,P]
                mu[:, :, c].index_add_(0, I, Ntn[:, :, None, None] * gimg)
        lam = mu + S['bg']
        img = torch.poisson(lam, generator=self.g) + rn * torch.randn(lam.shape, device=d, generator=self.g) \
            + offerr * torch.randn(n, 1, 4, P, P, device=d, generator=self.g)
        inpatch = (xn.abs() < 7.5) & (yn.abs() < 7.5)
        nnb = torch.bincount(I[inpatch], minlength=n) if k else torch.zeros(n, dtype=torch.long, device=d)
        S.update(img=img, mu=mu, nnb=nnb, nb=[], nb_flat=dict(idx=I, x=xn, y=yn, z=zn))
        return S
