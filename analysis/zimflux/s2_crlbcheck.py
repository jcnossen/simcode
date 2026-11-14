"""Why are the paper's CRLBs (Fig. 8e: 11.7 / 6.0 nm) about 1.5x below ours (17.7 / 8.6 nm) on the same movie?
Recompute the CRLB of the classical localizations (eps_mod < 0.3, median cluster-like set) under alternative bookkeeping:
(a) as fitted; (b) background divided by 3 (b per pixel per frame used where the summed-image b belongs, i.e. the
eq. 1 b / LK applied to a per-frame b); (c) background 0; (d) photons x 2.
usage: python s2_crlbcheck.py <gpu> [S1 subfolder=f225/]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/zimflux')
os.environ.setdefault('ZF_S1_SUB', sys.argv[2] if len(sys.argv) > 2 else 'f225/')
from s1_common import *
from simcode.fourpi.lm import fisher_j
dev = torch.device(f'cuda:{sys.argv[1]}')
A = np.load(WORK + 'astig.npz'); Zf = np.load(WORK + 'zf.npz'); pj = json.load(open(WORK + 'psf_used.json'))
lut, f, _ = make_lut(dev, 1.0, variant=pj['variant'], focus_shift=pj['focus_shift'])
pat = pat_to_torch({k: v for k, v in json.load(open(WORK + 'pattern_final.json')).items() if k in ('q', 'kz', 'm', 'eta', 'psi', 'tb')}, dev)
gi = Zf['idx']; k = Zf['eps'] < 0.3
rng = np.random.default_rng(0); sel = rng.choice(np.nonzero(k)[0], 4000, replace=False); ii = gi[sel]
tha = torch.tensor(A['th'][ii], device=dev).float(); thz = torch.tensor(Zf['th'][sel], device=dev).float()
cn = torch.tensor(A['corner'][ii], device=dev).float(); t = torch.tensor(A['cycle'][ii], device=dev).float()
v1 = torch.full((len(ii), P * P), 2.44 * 3, device=dev); v3 = torch.full((len(ii), 3 * P * P), 2.44, device=dev)
res = {}
for lab, bs, ns in (('fitted', 1, 1), ('bg/3', 1 / 3, 1), ('bg0', 1e-3, 1), ('N x2', 1, 2)):
    a = tha.clone(); a[:, 4] *= bs; a[:, 3] *= ns; z = thz.clone(); z[:, 4] *= bs; z[:, 3] *= ns
    ca = torch.sqrt(torch.diagonal(torch.linalg.pinv(fisher_j(astig_model(lut), a, v1)), dim1=1, dim2=2))
    cz = torch.sqrt(torch.diagonal(torch.linalg.pinv(fisher_j(zf_model(lut, pat, cn[:, 1], cn[:, 0], t), z, v3)), dim1=1, dim2=2))
    res[lab] = [float(ca[:, 2].median() * 1000), float(cz[:, 2].median() * 1000)]
    print(lab, 'median CRLB z astig / zf (nm)', np.round(res[lab], 1), flush=True)
res['median_N_astig'] = float(A['th'][ii, 3].mean()); res['median_bg_sum_per_px'] = float(np.median(A['th'][ii, 4]))
json.dump(res, open(WORK + 'crlbcheck.json', 'w'), indent=1); print(res)
