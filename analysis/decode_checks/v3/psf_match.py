"""Does the DECODE training PSF (decode_paper.setup_psf -> dec_common.LutPSF) render the same images as the movie renderer
(papersim common.RendererV2 = simcode Simulator)? Centroid / image difference per z and sub-pixel offset."""
import os, sys, json
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(REPO, 'analysis', 'baselines', 'decode'), os.path.join(REPO, 'analysis', 'papersim'), REPO]
import decode_paper as DP, dec_common as D, common as C
R = C.RendererV2(40, 40)
DP.setup_psf('models/insitu/calib_1809_2_psf_v2.json', 0.0, 0.4)
luts, zmax, halo = D._LUTS
lp = D.LutPSF([-0.5, 39.5], [-0.5, 39.5], [40, 40], luts, zmax, halo=0.0)
out = {}
o = torch.arange(41, device='cuda', dtype=torch.float32) - 20
for z in [0.0, 0.1, 0.2, 0.3, 0.4]:
    rows = []
    for fx, fy in [(0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (-0.4, 0.3), (0.45, -0.45)]:
        x = torch.tensor([20.0 + fx], device='cuda'); y = torch.tensor([20.0 + fy], device='cuda'); zz = torch.tensor([z], device='cuda')
        a = R.sim.render_psf(x, y, zz)[0]; b = lp.rois(x, y, zz)[0]
        def cen(g): s = g.sum(); return float((g.sum(0) * o).sum() / s), float((g.sum(1) * o).sum() / s), float(s)
        ca, cb = cen(a), cen(b)
        rows.append(dict(fx=fx, fy=fy, mov_cx=ca[0] - fx, mov_cy=ca[1] - fy, dec_cx=cb[0] - fx, dec_cy=cb[1] - fy, mov_sum=ca[2], dec_sum=cb[2],
                         rel_l1=float((a - b).abs().sum() / a.sum()), peak_mov=[int(v) for v in np.unravel_index(int(a.argmax()), a.shape)],
                         peak_dec=[int(v) for v in np.unravel_index(int(b.argmax()), b.shape)]))
    out[z] = rows
    for r in rows: print(z, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
