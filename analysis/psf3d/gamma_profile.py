"""(2026-11-11 in-situ 3D PSF) Held-out likelihood profile in the photoselection gamma at a FROZEN base PSF: writes copies of a
fit_insitu2 _psf.pt with extras gamma / dipole_M (s-polarised, e_a perpendicular to k_a of --pol-calib) for each gamma; run
psf_heldout on each (same held-out samples) and compare with psf_compare (paired NLL, moment contrast).
usage: python analysis/psf3d/gamma_profile.py <base _psf.pt> <out prefix> g1,g2,...  [pol calib json]"""
import sys, json
import torch
from simcode2.psf.vectorial import photosel_tensor
from simcode2 import config

base, outp, gs = sys.argv[1:4]; pc = sys.argv[4] if len(sys.argv) > 4 else 'insitu/calib_0.5nM.json'
S = torch.load(base, weights_only=False)
k = torch.tensor(json.load(open(config.resolve(pc)))['pattern']['k_rad_per_px'])
kd = k / k.norm(dim=1, keepdim=True); e = torch.stack([-kd[:, 1], kd[:, 0]], 1)
for g in [float(v) for v in gs.split(',')]:
    T = dict(S); ex = dict(S.get('extras', {}))
    ex['gamma'] = g; ex['e_dir'] = e.tolist(); ex['dipole_M'] = [photosel_tensor(torch.tensor(g), e[a]).tolist() for a in range(2)]
    T['extras'] = ex
    torch.save(T, f'{outp}_g{g:g}_psf.pt'); print('wrote', f'{outp}_g{g:g}_psf.pt')
