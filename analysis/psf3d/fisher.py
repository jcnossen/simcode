"""(2026-11-11 in-situ 3D PSF) Information matrix of global PSF parameters on held-out track samples, with the per-sample
nuisances (x, y, z, per-frame I, per-angle bg) profiled out (Schur complement), for a fit_insitu2 _psf.pt.
Globals: blur covariance (sxx, syy, sxy), photoselection gamma (and zeta), Noll 5 / 6 astigmatism, focus.
Prints the SEs (1 / sqrt of the profiled information), the correlation matrix and the eigenvalues of the correlation-scaled
information (near-zero = near-degenerate combination), and the eigenvector of the smallest one.
usage: python analysis/psf3d/fisher.py <fit_psf.pt> <tracks.npz> <heldout.npz> [n=1500] > out.txt"""
import sys, json, math
import numpy as np, torch
from simcode.psf.vectorial import VectorialPSF
from simcode.insitu.varpro import solve_linear

dev = 'cuda'
fit, tracks, ho = sys.argv[1:4]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
S = torch.load(fit, weights_only=False); ex = S.get('extras', {})
D = np.load(tracks); H = np.load(ho)
R = D['rois'].shape[-1]; c = (R - 1) / 2
model = VectorialPSF(R, oversample=2, NA=float(S['NA']), wavelength=0.70, n_med=1.33, device=dev)
with torch.no_grad(): model.zcoef.copy_(S['zcoef'].to(dev)); model.apod.copy_(S['apod'].to(dev))
for p_ in model.parameters(): p_.requires_grad_(False)
g, off, rn2, zmax = float(S['gain']), float(S['offset']), float(torch.exp(2 * S['logrn'])), float(S['zmax'])
glob = dict(focus=float(S['focus']), z5=float(S['zcoef'][0]), z6=float(S['zcoef'][1]))
blur = ex.get('blur_px2', [[0.0, 0.0, 0.0]])[0]
glob.update(sxx=blur[0], syy=blur[1], sxy=blur[2])
EDIR = torch.tensor(ex['e_dir'], device=dev) if 'e_dir' in ex else torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=dev)
glob.update(gamma=ex.get('gamma', 0.0))
if 'zeta' in ex: glob['zeta'] = ex['zeta']
names = list(glob)
idx = np.arange(min(n, len(H['sel'])))
sel = H['sel'][idx]
adu = torch.tensor(D['rois'][sel].astype(np.float32), device=dev); mask = torch.tensor(D['mask'][sel], device=dev)
frames = torch.tensor(D['frames'][sel], device=dev)
xyz0 = torch.tensor(np.c_[H['x'][idx] - D['corner'][sel, 0] - c, H['y'][idx] - D['corner'][sel, 1] - c, H['z'][idx]], device=dev, dtype=torch.float32)
group = torch.where(mask, (frames % 6) // 3, torch.zeros_like(frames))
data = (adu - off) * g
B, T = adu.shape[:2]


def psfT(xyz, gv):
    with torch.no_grad():
        zc = model.zcoef.clone(); zc[0] = gv['z5']; zc[1] = gv['z6']; model.zcoef.copy_(zc)
        cov = torch.tensor([gv['sxx'], gv['syy'], gv['sxy']], device=dev)
        out = []
        for k in range(2):
            e = torch.zeros(3, device=dev); e[:2] = EDIR[k]
            ze = gv.get('zeta', 1 / 3)
            base = torch.diag(torch.tensor([(1 - ze) / 2, (1 - ze) / 2, ze], device=dev))
            M = (1 - gv['gamma']) * base + gv['gamma'] * e[:, None] * e[None, :]
            out.append(model(xyz, gv['focus'], blur=cov, dipM=M))
        ps = torch.stack(out, 1)
        return torch.gather(ps, 1, group[..., None, None].expand(-1, -1, R, R)).reshape(B, T, -1)   # [B,T,P]


P0 = psfT(xyz0, glob)
I, bg = solve_linear(P0, data.reshape(B, T, -1), mask, group, rn2)
bgf = torch.gather(bg, 1, group)[..., None]
mu0 = I[..., None] * P0 + bgf
W = (mask[..., None].float() / (mu0.clamp(min=1e-3) + rn2)).reshape(B, -1).double()        # [B, T P]
h = 1e-3
def dmu_dglob(k):
    gp = dict(glob); gm = dict(glob); gp[k] += h; gm[k] -= h
    return (I[..., None] * (psfT(xyz0, gp) - psfT(xyz0, gm)) / (2 * h)).reshape(B, -1)
def dmu_dloc(j):
    dd = torch.zeros(3, device=dev); dd[j] = h if j < 2 else 1e-3
    return (I[..., None] * (psfT(xyz0 + dd, glob) - psfT(xyz0 - dd, glob)) / (2 * dd[j])).reshape(B, -1)
Jg = torch.stack([dmu_dglob(k) for k in names], -1).double()                                    # [B, TP, G]
Jl = [dmu_dloc(j) for j in range(3)]
Jl += [(P0 * (torch.arange(T, device=dev)[None, :, None] == t).float()).reshape(B, -1) for t in range(T)]
Jl += [((group == a_).float()[..., None].expand(-1, -1, R * R)).reshape(B, -1) for a_ in range(2)]
Jl = torch.stack(Jl, -1).double()                                                                # [B, TP, L]
Fgg = torch.einsum('bpi,bp,bpj->ij', Jg, W, Jg)
Fgl = torch.einsum('bpi,bp,bpj->bij', Jg, W, Jl); Fll = torch.einsum('bpi,bp,bpj->bij', Jl, W, Jl)
Fll = Fll + 1e-6 * torch.eye(Fll.shape[-1], device=dev, dtype=torch.float64) * Fll.diagonal(dim1=1, dim2=2).mean(1)[:, None, None]
Fp = Fgg - torch.einsum('bij,bjk,blk->il', Fgl, torch.linalg.inv(Fll), Fgl)
C = torch.linalg.inv(Fp).cpu().numpy(); se = np.sqrt(np.diag(C)) * math.sqrt(1.0)
cor = C / np.outer(se, se)
d = np.sqrt(np.diag(Fp.cpu().numpy())); Fn = Fp.cpu().numpy() / np.outer(d, d)
ev, evec = np.linalg.eigh(Fn)
print(f'{fit}: n {B} held-out samples (per {B} samples; SE scales as 1/sqrt(n))')
for k, s_ in zip(names, se): print(f'  {k:6s} value {glob[k]:+.4f}  SE {s_:.4f}  (conditional SE {1 / d[names.index(k)]:.4f})')
print('  correlation:\n' + '\n'.join('   ' + ' '.join(f'{v:+.2f}' for v in row) + f'  {k}' for row, k in zip(cor, names)))
print('  eigenvalues of the scaled information:', np.round(ev, 4).tolist())
print('  weakest combination:', {k: round(float(v), 3) for k, v in zip(names, evec[:, 0])})
