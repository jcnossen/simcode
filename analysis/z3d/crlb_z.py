"""z3d design check: Cramér-Rao bound of (x, y, z) for one emitter on the 6-frame window sum (pattern-free: the pattern
informs x, y only), vectorial PSF of a calibration at its focus. Parameters x, y, z, N, bg; N window photons, bg per px
per frame (x 6 frames). usage: python analysis/z3d/crlb_z.py [calib] [N] [bg]"""
import sys
import numpy as np, torch
from simcode.sim.simulator import Simulator
from simcode.psf import render

calib = sys.argv[1] if len(sys.argv) > 1 else 'insitu/calib_1809_2_psf3d_avg.json'
N = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0; bg = 6 * (float(sys.argv[3]) if len(sys.argv) > 3 else 8.0)
sim = Simulator(calib, psf_roi=21)
px = sim.c['pixelsize_um'] * 1000
print(f'{calib}: focus {sim.focus:.3f} um, N {N:g}, bg {bg / 6:g}/px/frame; CRLB (nm) per emitter height')
print('z_um  crlb_x  crlb_y  crlb_z   psf_sum')
for z in np.arange(0.0, 0.31, 0.025):
    def mu(t):
        p = torch.tensor([[t[0], t[1], t[2]]], device='cuda', dtype=torch.float32)
        with torch.no_grad():
            return (t[3] * render(sim.psf, p)[0] + t[4]).double()
    t0 = [0.0, 0.0, float(z), N, bg]; h = [1e-3, 1e-3, 2e-3, 1.0, 0.1]
    m0 = mu(t0); J = []
    for i in range(5):
        a = list(t0); b = list(t0); a[i] += h[i]; b[i] -= h[i]
        J.append((mu(a) - mu(b)) / (2 * h[i]))
    J = torch.stack([j.reshape(-1) for j in J], 1)
    Fi = J.T @ (J / m0.reshape(-1, 1))
    C = torch.linalg.inv(Fi)
    s = C.diagonal().sqrt()
    print(f'{z:.3f}  {s[0] * px:6.2f}  {s[1] * px:6.2f}  {s[2] * 1000:6.1f}   {float((m0 - bg).sum() / N):.3f}')
