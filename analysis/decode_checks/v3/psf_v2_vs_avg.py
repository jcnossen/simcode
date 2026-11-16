"""1809_2 PSF models at the pixel level: in-situ PSF v2 (calib_1809_2_psf_v2.json, the v3 DECODE / SIMCODE m1809 training PSF)
vs the averaged psf3d (calib_1809_2_psf3d_avg.json, the old real-data DECODE dec_1809_2's training PSF): DECODE LUT images
(decode_paper.psf_luts) at the same heights and sub-pixel offsets; centroid (11 x 11 px) and relative L1 difference.
usage: CUDA_VISIBLE_DEVICES=2 python analysis/decode_checks/v3/psf_v2_vs_avg.py"""
import os, sys, json
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(REPO, 'analysis', 'baselines', 'decode'), REPO]
import decode_paper as DP
A, _ = DP.psf_luts('models/insitu/calib_1809_2_psf_v2.json', lz=21, zmax=0.2)
Bv, _ = DP.psf_luts('models/insitu/calib_1809_2_psf3d_avg.json', lz=21, zmax=0.2)
R = A.shape[-1]; c = R // 2; o = torch.arange(-5, 6, device=A.device, dtype=torch.float32)
out = []
for iz in (5, 10, 15, 20):
    for (iy, ix) in ((16, 16), (16, 0), (16, 32), (0, 16), (32, 16), (8, 24)):
        a = A[iz, iy, ix, c - 5:c + 6, c - 5:c + 6]; b = Bv[iz, iy, ix, c - 5:c + 6, c - 5:c + 6]
        a = a / a.sum(); b = b / b.sum()
        cen = lambda g: (float((g.sum(0) * o).sum()), float((g.sum(1) * o).sum()))
        fx, fy = ix / 32 - 0.5, iy / 32 - 0.5
        ca, cb = cen(a), cen(b)
        out.append(dict(z_um=iz / 100, fx=fx, fy=fy, v2_centroid_minus_true=(ca[0] - fx, ca[1] - fy), avg_centroid_minus_true=(cb[0] - fx, cb[1] - fy),
                        rel_l1=float((a - b).abs().sum()), peak_ratio=float(a.max() / b.max())))
        print({k: (tuple(round(v, 3) for v in w) if isinstance(w, tuple) else round(w, 3)) for k, w in out[-1].items()})
json.dump(out, open(os.path.join(REPO, 'out', 'decode_checks', 'v3', 'psf_v2_vs_avg.json'), 'w'), indent=1)
