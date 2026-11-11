"""(2026-11-11 in-situ 3D PSF) Synthetic isolated-emitter track samples rendered through the SIMULATOR's PSF path
(sim.simulator.Simulator LUT, per-angle LUTs when the calib has a per-angle PSF), with the real samples' fitted parameters
(position, z, per-frame I, per-angle bg from a psf_heldout npz of a real track file), Poisson + the calib's per-pixel
read-noise map and per-pixel offset map at the sample's sensor position (the in-situ fit subtracts the global 99.3 ADU, as on
real data), gain 0.56, uint16. Same format as
insitu/extract_tracks (rois, mask, frames, corner, xy_init), so psf_heldout / psf_compare run on it unchanged.
usage: python analysis/psf3d/simtracks.py <calib.json> <real tracks.npz> <real heldout.npz> <out tracks.npz> [seed]"""
import sys
import numpy as np, torch
from simcode2.sim.simulator import Simulator

calib, tracks, ho, outp = sys.argv[1:5]; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
dev = 'cuda'; torch.manual_seed(seed)
D = np.load(tracks); H = np.load(ho)
sel = H['sel']; n = len(sel); R = D['rois'].shape[-1]; T = D['rois'].shape[1]; h = R // 2
sim = Simulator(calib, H=32, W=32, T=6, device=dev, psf_roi=21)
x = torch.tensor(H['x'] - D['corner'][sel, 0], device=dev, dtype=torch.float32)       # ROI coords (pixel centres at integers)
y = torch.tensor(H['y'] - D['corner'][sel, 1], device=dev, dtype=torch.float32)
z = torch.tensor(H['z'], device=dev, dtype=torch.float32)
with torch.no_grad():
    P = sim.render_psf_angles(x, y, z)                                                    # [A, n, 21, 21] centred on round(x, y)
cx, cy = torch.round(x).long(), torch.round(y).long()
o = torch.arange(R, device=dev)
# ROI pixel j <-> LUT pixel j - c + 10
iy = (o[None, :] - cy[:, None] + 10).clamp(0, 20); ix = (o[None, :] - cx[:, None] + 10).clamp(0, 20)
roi = P[:, torch.arange(n, device=dev)[:, None, None], iy[:, :, None], ix[:, None, :]]  # [A, n, R, R]
frames = torch.tensor(D['frames'][sel], device=dev); mask = torch.tensor(D['mask'][sel], device=dev)
ang = torch.where(mask, (frames % 6) // 3, torch.zeros_like(frames))
I = torch.tensor(H['I'], device=dev, dtype=torch.float32).clamp(min=0); bg = torch.tensor(H['bg'], device=dev, dtype=torch.float32).clamp(min=0.1)
A = roi.shape[0]
ps = roi[ang.clamp(max=A - 1), torch.arange(n, device=dev)[:, None]]                     # [n, T, R, R]
mu = I[..., None, None] * ps + torch.gather(bg, 1, ang)[..., None, None]
import json, os
cc = json.load(open(calib)); mp = cc['camera']['maps']; mp = mp if os.path.isabs(mp) else os.path.join(os.path.dirname(os.path.abspath(calib)), mp)
Mp = np.load(mp); cy0, cx0 = D['corner'][sel, 1], D['corner'][sel, 0]
ii = (cy0[:, None, None] + np.arange(R)[None, :, None]).clip(0, 119); jj = (cx0[:, None, None] + np.arange(R)[None, None, :]).clip(0, 119)
rnm = torch.tensor(Mp['readnoise_e'][ii, jj], device=dev, dtype=torch.float32)[:, None]; offm = torch.tensor(Mp['offset_adu'][ii, jj], device=dev, dtype=torch.float32)[:, None]
e = torch.poisson(mu) + rnm * torch.randn_like(mu)
adu = torch.round(e / 0.56 + offm).clamp(0, 65535) * mask[..., None, None]
np.savez_compressed(outp, rois=adu.cpu().numpy().astype(np.uint16), mask=D['mask'][sel], frames=D['frames'][sel], corner=D['corner'][sel],
                    xy_init=np.c_[x.cpu().numpy(), y.cpu().numpy()], photons_smlm=D['photons_smlm'][sel])
print('saved', outp, n, 'samples; per-angle PSF' if A == 2 else '; single PSF')
