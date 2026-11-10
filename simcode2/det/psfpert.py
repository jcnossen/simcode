"""
PSF-perturbed simulation (critic review 24 #1): a bank of Simulator PSF lookup tables, each rendered from the
calibrated vectorial PSF with a random perturbation:
  width   pixel-size scale (PSF width in pixels x w, w log-uniform in [w_lo, w_hi])
  zern    Gaussian jitter of every Zernike coefficient (sd zj waves rms) on top of the calibration
  focus   focus shift (uniform +- df um)
plus, per batch (sample()), a wider emitter z range (zmax_pert), and camera gain / offset errors (the frames are
converted with gain (1 + eg) and offset + eo ADU instead of the true ones).
"""
import math, copy
import numpy as np, torch
from simcode2.sim.simulator import Simulator
from simcode2.psf.vectorial import VectorialPSF

dev = 'cuda'


def build_lut(sim, width=1.0, zern_delta=None, focus_shift=0.0, zmax=None):
    """returns (lut [nz, ls+1, ls+1, R, R], zmax) for a perturbed PSF, rendered like Simulator.__init__"""
    c = sim.c
    zmax = zmax or sim.zmax
    psf = VectorialPSF(sim.R, pixelsize=c['pixelsize_um'] / width, NA=c['NA'], n_imm=c['n_imm'], n_med=c['n_med'],
                       wavelength=c['wavelength_um'], oversample=3, device=dev)
    with torch.no_grad():
        zc = torch.tensor(c['zernike_waves'], device=dev)
        if zern_delta is not None: zc = zc + zern_delta
        psf.zcoef.copy_(zc); psf.apod.copy_(torch.tensor(c['apod'], device=dev))
    zs = torch.linspace(0, zmax, sim.lz, device=dev)
    sub = (torch.arange(sim.ls + 1, device=dev) / sim.ls) - 0.5
    Z, SY, SX = torch.meshgrid(zs, sub, sub, indexing='ij')
    pos = torch.stack([SX.reshape(-1), SY.reshape(-1), Z.reshape(-1)], 1)
    with torch.no_grad():
        lut = torch.cat([psf(p, torch.full((len(p),), sim.focus + focus_shift, device=dev)) for p in pos.split(1024)])
    return lut.reshape(sim.lz, sim.ls + 1, sim.ls + 1, sim.R, sim.R), zmax


class PertBank:
    def __init__(self, sim, n=12, w=(0.92, 1.12), zj=0.02, df=0.1, zmax=0.35, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.sim = sim; self.items = []
        self.base = (sim.lut, sim.zmax, sim.zs)
        nz = len(sim.c['zernike_waves'])
        for i in range(n):
            u = torch.rand(3, generator=g)
            wi = math.exp(math.log(w[0]) + (math.log(w[1]) - math.log(w[0])) * float(u[0]))
            zd = (torch.randn(nz, generator=g) * zj).to(dev)
            fs = (float(u[1]) - 0.5) * 2 * df
            lut, zm = build_lut(sim, wi, zd, fs, zmax)
            self.items.append(dict(lut=lut, zmax=zm, width=wi, focus=fs, zern_sd=float(zd.std())))

    def use(self, i):
        it = self.items[i]
        self.sim.lut = it['lut']; self.sim.zmax = it['zmax']; self.sim.zs = torch.linspace(0, it['zmax'], self.sim.lz, device=dev)
        return it

    def reset(self):
        self.sim.lut, self.sim.zmax, self.sim.zs = self.base


def camera_error(o, sim, eg=0.1, eo=2.0):
    """re-convert frames with a wrong gain (x (1 + U(-eg, eg))) and offset (+ U(-eo, eo) ADU) per FOV"""
    nb = o['frames'].shape[0]
    ge = 1 + (torch.rand(nb, 1, 1, 1, device=dev) - 0.5) * 2 * eg
    oe = (torch.rand(nb, 1, 1, 1, device=dev) - 0.5) * 2 * eo
    adu = o['frames'] / sim.gain + sim.offset
    o['frames'] = (adu - (sim.offset + oe)) * sim.gain * ge
    return o
