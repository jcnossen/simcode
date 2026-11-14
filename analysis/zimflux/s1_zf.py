"""S1 step 4: k_z refinement by profile likelihood and the ZIMFLUX joint MLE (paper eq. 13, 3.6.2).
- k_z: for each candidate, the pattern (q, eta, m, psi(T)) is re-solved with k_z fixed (s1_pattern.fit_pattern), then the
  joint fit of 20k bright spots; the summed log-likelihood is maximised (parabola through the best 3). This replaces the
  paper's pattern-vs-PSF bias regression (eq. 33-36), which is attenuated by the astigmatic z error.
- Full joint fit of all good spots: Poisson LM on the 3 x 16 x 16 pixels, th = (x, y, z, N, b), start = astigmatic fit.
- Modulation error eps_mod = max_k |N w_k(r) - N_k| / (N w_k(r)) (eq. 37), N_k = per-frame intensities (position fixed).
usage: python s1_zf.py <gpu> [kz candidates comma-separated] -> out/zimflux/s1/zf.npz, kz_profile.json, pattern_final.json"""
import sys, json, numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/zimflux')
from s1_common import *
from s1_pattern import fit_pattern, good_mask
from simcode.fourpi.lm import lm_poisson_j, fisher_j
dev = torch.device(f'cuda:{sys.argv[1]}')
A = dict(np.load(WORK + 'astig.npz')); Z = np.load(WORK + 'rois.npz'); cam = np.load(WORK + 'camera.npz')
rois = Z['rois']; ncyc = int(Z['ncycles'])
pj = json.load(open(WORK + 'psf_used.json'))
lut, f, zern = make_lut(dev, 1.0, variant=None if pj['variant'] == 'none' else pj['variant'], focus_shift=pj['focus_shift'])
rn = torch.tensor(cam['rnvar'], device=dev, dtype=torch.float32); ar = torch.arange(P, device=dev)
g = good_mask(A); gi = np.nonzero(g)[0]
gp = g & (A['Nk'].sum(1) > 1000)
lo = torch.tensor([-7.0, -7.0, ZLO, 1.0, 0.003], device=dev); hi = torch.tensor([7.0, 7.0, ZHI, 1e6, 1e4], device=dev)


def joint(idx, pat, B=6144, iters=30, out_all=False):
    pt = pat_to_torch({k: pat[k] for k in ('q', 'kz', 'm', 'eta', 'psi', 'tb')}, dev)
    res = {k: [] for k in ('th', 'll', 'crlb', 'eps')}
    for i0 in range(0, len(idx), B):
        ii = idx[i0:i0 + B]; n = len(ii)
        d = torch.tensor(rois[ii], device=dev).float().reshape(n, -1)
        cn = torch.tensor(A['corner'][ii], device=dev).long()
        v = rn[(cn[:, 0, None] + ar)[:, :, None], (cn[:, 1, None] + ar)[:, None, :]].reshape(n, 1, -1).expand(n, 3, -1).reshape(n, -1)
        cx = cn[:, 1].float(); cy = cn[:, 0].float(); t = torch.tensor(A['cycle'][ii], device=dev).float()
        fj = zf_model(lut, pt, cx, cy, t)
        th0 = torch.tensor(A['th'][ii], device=dev).float(); th0[:, 4] = th0[:, 4] / 3
        th, ll, mu = lm_poisson_j(fj, d, th0, v, lo, hi, iters=iters)
        res['th'].append(th.cpu().numpy()); res['ll'].append(ll.cpu().numpy())
        if out_all:
            F = fisher_j(fj, th, v); cr = torch.sqrt(torch.diagonal(torch.linalg.pinv(F), dim1=1, dim2=2).clamp(min=0))
            w, _ = pattern_w(pt, cx + 7.5 + th[:, 0], cy + 7.5 + th[:, 1], th[:, 2], t)
            Nexp = th[:, 3, None] * w; Nest = torch.tensor(A['Nk'][ii], device=dev)
            res['crlb'].append(cr.cpu().numpy()); res['eps'].append(((Nexp - Nest).abs() / Nexp.clamp(min=1)).max(1).values.cpu().numpy())
    return {k: np.concatenate(v) for k, v in res.items() if v}


if len(sys.argv) > 2 and sys.argv[2] == 'final':   # pattern + k_z from s1_kz.py (track means, no EIV attenuation)
    pat = json.load(open(WORK + 'pattern_final.json'))
    r = joint(gi, pat, out_all=True)
    np.savez(WORK + 'zf.npz', idx=gi, **r)
    print('done', len(gi), 'eps<0.3', float((r['eps'] < 0.3).mean())); sys.exit()
q0 = json.load(open(WORK + 'pattern.json'))['q']
sub = np.nonzero(gp)[0]; sub = sub[np.argsort(-A['Nk'][sub].sum(1))[:20000]]
cands = [float(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else [-8.5, -10.5, -12.5, -14.5, -16.5, 12.5]
prof = {}
for kz in cands:
    pat = fit_pattern(dev, A, np.nonzero(gp)[0], ncyc, q0, kz_fixed=kz, verbose=False)
    r = joint(sub, pat, iters=20)
    prof[kz] = float(r['ll'].astype(np.float64).sum()); print('kz', kz, 'll', prof[kz], 'm', np.round(pat['m'], 3).tolist(), flush=True)
ks = np.array(sorted(prof)); ls = np.array([prof[k] for k in ks]); i = int(np.argmax(ls))
if 0 < i < len(ks) - 1:
    c = np.polyfit(ks[i - 1:i + 2], ls[i - 1:i + 2], 2); kbest = float(-c[1] / (2 * c[0]))
else:
    kbest = float(ks[i])
print('kz best', kbest, 'p_ax nm', 2 * np.pi / abs(kbest) * 1000, flush=True)
json.dump({'profile': {str(k): v for k, v in prof.items()}, 'kz_best': kbest, 'pax_nm': 2 * np.pi / abs(kbest) * 1000},
          open(WORK + 'kz_profile.json', 'w'), indent=1)
pat = fit_pattern(dev, A, np.nonzero(gp)[0], ncyc, q0, kz_fixed=kbest, verbose=False)
json.dump(pat, open(WORK + 'pattern_final.json', 'w'), indent=1)
r = joint(gi, pat, out_all=True)
np.savez(WORK + 'zf.npz', idx=gi, **r)
print('done', len(gi), 'eps<0.3', float((r['eps'] < 0.3).mean()))
