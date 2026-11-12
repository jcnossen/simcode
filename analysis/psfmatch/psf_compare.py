"""PSF-matched retrain (2026-11-11): how close are the movie PSFs? (decides whether 20 nM needs its own network set)
Uses the averaged (mean over pattern axes) z-stacks out/psf3d/final/zstack_<movie>_psf3d.npz (31 x 31 px, 108.33 nm).
Per z (weighted by the TIRF depth profile exp(-z / 0.082 um) over the stack range) and pair of movies:
  L1   = sum |A - B| of the unit-sum 31 x 31 images (0 = identical, 2 = disjoint)
  dE3  = difference of 3 x 3 / 11 x 11 energy (the spot-shape number of analysis/psf3d_nets/spotshape.py)
  m2   = second moment (px^2) of the 11 x 11 core, per movie
  ratio of the Fisher-information-based lateral CRLB (Poisson, 1000 ph, bg 10 / px) of A vs B
usage: python analysis/psfmatch/psf_compare.py > out/psfmatch/psf_compare.txt"""
import numpy as np

MOV = ['1809_2', '2p5nM_m2', '20nM_m2']
Z = {m: np.load(f'out/psf3d/final/zstack_{m}_psf3d.npz') for m in MOV}


def unit(a):
    a = np.clip(a, 0, None); return a / a.sum(axis=(-1, -2), keepdims=True)


def e3(a):
    c = a.shape[-1] // 2
    return a[..., c - 1:c + 2, c - 1:c + 2].sum((-1, -2)) / a[..., c - 5:c + 6, c - 5:c + 6].sum((-1, -2))


def m2(a):
    c = a.shape[-1] // 2; b = a[..., c - 5:c + 6, c - 5:c + 6]; b = b / b.sum((-1, -2), keepdims=True)
    y, x = np.mgrid[-5:6, -5:6]
    return (b * (x ** 2 + y ** 2)).sum((-1, -2))


def crlb(a, N=1000., bg=10.):
    mu = N * a + bg
    gy, gx = np.gradient(N * a, axis=(-2, -1))
    F = np.array([[(gx * gx / mu).sum((-1, -2)), (gx * gy / mu).sum((-1, -2))], [(gx * gy / mu).sum((-1, -2)), (gy * gy / mu).sum((-1, -2))]])
    F = np.moveaxis(F, (0, 1), (-2, -1))
    return np.sqrt(np.linalg.inv(F)[..., 0, 0] + np.linalg.inv(F)[..., 1, 1])


z = Z[MOV[0]]['z_um']
for m in MOV: assert np.allclose(Z[m]['z_um'], z)
w = np.exp(-np.clip(z, 0, None) / 0.082); w[z < 0] = 0; w /= w.sum()
U = {m: unit(Z[m]['mean']) for m in MOV}
print(f'# z range {z.min():.3f}..{z.max():.3f} um, TIRF weights exp(-z/0.082) on z >= 0; values = weighted mean over z (and at z = 0.055)')
iz = int(np.argmin(np.abs(z - 0.055)))
for m in MOV:
    print(f'{m:10s} E3 {np.sum(w * e3(U[m])):.4f} (z=.055: {e3(U[m])[iz]:.4f})  m2 {np.sum(w * m2(U[m])):.3f} px^2  CRLB(1000 ph, bg 10) {np.sum(w * crlb(U[m])):.4f} px')
for a, b in [('2p5nM_m2', '20nM_m2'), ('1809_2', '2p5nM_m2'), ('1809_2', '20nM_m2')]:
    L1 = np.abs(U[a] - U[b]).sum((-1, -2))
    print(f'{a} vs {b}: L1 {np.sum(w * L1):.4f} (z=.055 {L1[iz]:.4f})  dE3 {np.sum(w * (e3(U[a]) - e3(U[b]))):+.4f}  '
          f'm2 ratio {np.sum(w * m2(U[a])) / np.sum(w * m2(U[b])):.3f}  CRLB ratio {np.sum(w * crlb(U[a])) / np.sum(w * crlb(U[b])):.3f}')
# per-axis vs mean of one movie, for scale (the per-axis vs averaged ablation showed < 0.5 % network effect)
for m in MOV:
    L1 = np.abs(unit(Z[m]['angle0']) - unit(Z[m]['angle1'])).sum((-1, -2))
    print(f'{m}: angle0 vs angle1 L1 {np.sum(w * L1):.4f}')
