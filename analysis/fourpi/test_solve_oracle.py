"""Consistency test of the explicit solve (simcode/fourpi/solve4pi.py) against the simulator: feed the GT read-out
targets (T, C) without noise and a loose prior around the truth; the solve must return the truth (all modes), and
the phase mode must return Phi0."""
import json, math, os, sys
import torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
from simcode.fourpi.lut4pi import LUT4Pi
from simcode.fourpi.sim4pi import Sim4Pi
from simcode.fourpi.solve4pi import Solve4Pi, psi_from_eps

st = json.load(open(f'{P4}/psf4pi/NPC_f_dxy15.json'))['psf']
lut = LUT4Pi(st, P=15)
S = Sim4Pi(lut, seed=5).sample(2000, p_nb=0.0, p_on=1.0)
n = 2000
g = torch.Generator(device='cuda').manual_seed(0)
o = dict(T=S['T'], C=S['C'], L=torch.zeros(n, 6, 4, 2, 2, device='cuda'),
         dxy=torch.stack([S['x'], S['y']], 1) + 0.1 * torch.randn(n, 2, device='cuda', generator=g),
         z=S['z'] + 0.02 * torch.randn(n, device='cuda', generator=g), ls=torch.log(torch.tensor([0.3, 0.3, 0.04], device='cuda')).expand(n, 3))
o['L'][..., 0, 0] = (S['T'] + 5).sqrt(); o['L'][..., 1, 1] = (S['C'] + 2).sqrt()
sv = Solve4Pi(lut)
psi = psi_from_eps(S['eps'])
for mode in ('on', 'free'):
    r = sv.solve(o, S['Phi0'], S['doff'], S['m'], psi, mode=mode)
    e = (r['th'][:, :3] - torch.stack([S['x'], S['y'], S['z']], 1)).abs()
    print(mode, 'max |err| x y (px), z (um):', e.max(0).values.tolist(), 'p99', e.quantile(0.99, 0).tolist())
r = sv.solve(o, torch.zeros_like(S['Phi0']), S['doff'], S['m'], psi, mode='free', fit_phi=True)
dphi = torch.remainder(r['th'][:, -1] - S['Phi0'] + math.pi, 2 * math.pi) - math.pi
print('phi mode: |dPhi| p50 / p99 (deg)', math.degrees(dphi.abs().median()), math.degrees(dphi.abs().quantile(0.99)))
