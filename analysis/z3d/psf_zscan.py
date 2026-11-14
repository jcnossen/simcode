"""z3d design check: PSF sharpness vs emitter height z_e and vs focus f for a model's PSF calibration; the focus-shift
equivalent gamma = d f / d z_e (which focus shift reproduces an emitter-height change).
usage: python analysis/z3d/psf_zscan.py [calib]"""
import sys, time
import numpy as np, torch
from simcode.sim.simulator import Simulator

calib = sys.argv[1] if len(sys.argv) > 1 else 'insitu/calib_1809_2_psf3d_avg.json'
t = time.time()
sim = Simulator(calib, psf_roi=41)
print('sim build', round(time.time() - t, 1), 's', tuple(sim.lut.shape), sim.lut.element_size() * sim.lut.numel() / 1e6, 'MB', 'focus', sim.focus)
psf = sim.psf


def P(ze, f):
    ze = torch.as_tensor(ze, device='cuda', dtype=torch.float32).reshape(-1); f = torch.as_tensor(f, device='cuda', dtype=torch.float32).reshape(-1)
    n = max(len(ze), len(f)); ze = ze.expand(n); f = f.expand(n)
    pos = torch.stack([torch.zeros(n, device='cuda'), torch.zeros(n, device='cuda'), ze], 1)
    with torch.no_grad():
        return psf(pos, f)


def width(i):
    o = torch.arange(i.shape[-1], device=i.device) - i.shape[-1] // 2
    w = i / i.sum((1, 2), keepdim=True)
    return ((w * (o[None, None, :] ** 2 + o[None, :, None] ** 2)).sum((1, 2)) / 2).sqrt()


zs = np.linspace(0, 0.6, 13)
I = P(zs, sim.focus); wd = width(I)
print('height scan at f_cal: z_e, peak, sum, rms width px')
for z, i, w in zip(zs, I, wd): print(f'{z:.2f} {float(i.max()):.4f} {float(i.sum()):.3f} {float(w):.3f}')
fs = sim.focus + np.linspace(-0.6, 0.6, 25)
I = P(0.05, fs); wd = width(I)
print('focus scan at z_e=0.05: f-fcal, peak, sum, width')
for f, i, w in zip(fs, I, wd): print(f'{f - sim.focus:+.2f} {float(i.max()):.4f} {float(i.sum()):.3f} {float(w):.3f}')
for dz in (0.05, 0.1, 0.15):
    ref = P(0.05 + dz, sim.focus); ref = ref / ref.sum()
    gs = np.linspace(0.5, 1.8, 131)
    q = P(0.05, sim.focus - gs * dz); q = q / q.sum((1, 2), keepdim=True)
    d = ((q - ref) ** 2).sum((1, 2))
    print(f'gamma (dz {dz}): {gs[int(d.argmin())]:.3f}  rel. residual {float(d.min() ** .5 / (ref ** 2).sum() ** .5):.4f}')
