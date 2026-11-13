"""Pick good single nuclear-pore sites from the classical 4Pi-SIMFLUX NPC table (pixel-MLE rows) and render them:
top view (x-y, colour = z), the two rings separately, and side views. Selection is automatic and stated in the figure:
ring radius, radial scatter, angular coverage of both z rings, isolation.
usage: python analysis/fourpi_classical/npc_site.py [n_sites]"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter, convolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simcode.fourpi.zdrift import zdrift_struct
from simcode.fourpi.setup import PX_NM as PX
OUT = f'{P4}/classical'
nshow = int(sys.argv[1]) if len(sys.argv) > 1 else 6
C = np.load(f'{OUT}/real/cl_NPC_wv_all.npz')
k = C['keep_pix'] & np.isfinite(C['xpd']) & np.isfinite(C['zp'])
x = C['xpd'][k] * PX; y = C['ypd'][k] * PX; z = C['zp'][k] * 1000.0; w = (C['frame'][k] // 6)
sx = C['sxp'][k]; sz = C['szp'][k]
d = zdrift_struct(x, y, z, w); z = z - d[np.clip(w // 500, 0, len(d) - 1)]
print('rows', len(x), 'z drift ptp nm', float(np.ptp(d)))
b = 8.0; x0, y0 = x.min(), y.min()
H = np.histogram2d(y - y0, x - x0, bins=[np.arange(0, y.max() - y0 + b, b), np.arange(0, x.max() - x0 + b, b)])[0]
r = np.hypot(*np.meshgrid(np.arange(-12, 13) * b, np.arange(-12, 13) * b))
ker = np.exp(-0.5 * ((r - 55.0) / 8.0) ** 2); ker /= ker.sum()
cen = (r < 28).astype(float); cen /= cen.sum(); far = ((r > 85) & (r < 100)).astype(float); far /= far.sum()
Hs = np.minimum(H, 6)
resp = convolve(Hs, ker, mode='constant') - 1.0 * convolve(Hs, cen, mode='constant') - 0.5 * convolve(Hs, far, mode='constant')
pk = (resp == maximum_filter(resp, 21)) & (resp > np.percentile(resp, 99.5))
iy, ix = np.nonzero(pk); order = np.argsort(-resp[iy, ix]); iy, ix = iy[order][:4000], ix[order][:4000]
from scipy.spatial import cKDTree
T = cKDTree(np.c_[x, y])
def circ(px, py):
    A = np.c_[2 * px, 2 * py, np.ones(len(px))]; s = np.linalg.lstsq(A, px ** 2 + py ** 2, rcond=None)[0]
    return s[0], s[1], np.sqrt(max(s[2] + s[0] ** 2 + s[1] ** 2, 0))
S = []
for a, c in zip(iy, ix):
    cx, cy = x0 + (c + 0.5) * b, y0 + (a + 0.5) * b
    for _ in range(3):
        ii = np.array(T.query_ball_point([cx, cy], 85.0))
        if len(ii) < 150: break
        rr = np.hypot(x[ii] - cx, y[ii] - cy); m = (rr > 35) & (rr < 78)
        if m.sum() < 120: break
        cx, cy, R = circ(x[ii][m], y[ii][m])
    else:
        ii = np.array(T.query_ball_point([cx, cy], 85.0)); rr = np.hypot(x[ii] - cx, y[ii] - cy)
        ring = (rr > R - 16) & (rr < R + 16)
        if ring.sum() < 150 or not (48 < R < 64): continue
        zz = z[ii][ring]; zm = np.median(zz); lo = zz < zm; mu = [zz[lo].mean(), zz[~lo].mean()]
        for _ in range(20):
            lo = np.abs(zz - mu[0]) < np.abs(zz - mu[1]); mu = [zz[lo].mean(), zz[~lo].mean()] if lo.any() and (~lo).any() else mu
        sep = mu[1] - mu[0]; frac = min(lo.mean(), 1 - lo.mean())
        th = np.arctan2(y[ii][ring] - cy, x[ii][ring] - cx); sec = ((th + np.pi) / (2 * np.pi) * 16).astype(int) % 16
        cov_lo = len(np.unique(sec[lo])); cov_hi = len(np.unique(sec[~lo]))
        zs = np.sqrt((zz[lo].var() * lo.sum() + zz[~lo].var() * (~lo).sum()) / len(zz))
        inner = (rr < R - 25).mean(); outer = len(T.query_ball_point([cx, cy], 150.0)) - len(ii)
        rsd = 1.4826 * np.median(np.abs(rr[ring] - np.median(rr[ring])))
        S.append(dict(cx=cx, cy=cy, R=R, n=int(ring.sum()), rsd=rsd, sep=sep, frac=frac, cov_lo=cov_lo, cov_hi=cov_hi, zs=zs,
                      inner=inner, outer_frac=outer / max(len(ii), 1), zmid=(mu[0] + mu[1]) / 2))
print('candidate pores', len(S))
good = [s for s in S if s['cov_lo'] >= 13 and s['cov_hi'] >= 13 and 35 < s['sep'] < 62 and s['frac'] > 0.3 and s['inner'] < 0.06
        and s['outer_frac'] < 0.25 and s['rsd'] < 9 and s['zs'] < 14]
print('pass all criteria', len(good))
good.sort(key=lambda s: (s['zs'] / s['sep']) + s['rsd'] / 40 - 0.0005 * min(s['n'], 800))
json.dump(good[:50], open(f'{OUT}/sites/sites.json', 'w'), indent=1)
def img(px, py, cz, ext, bin_=1.5, sig=2.0, zr=None, cmap='turbo'):
    e = np.arange(ext[0], ext[1] + bin_, bin_)
    Hn = np.histogram2d(py, px, bins=[e, e])[0]; Hn = gaussian_filter(Hn, sig / bin_)
    if cz is None: return Hn / max(Hn.max(), 1e-9)
    Hz = np.histogram2d(py, px, bins=[e, e], weights=cz)[0]; Hz = gaussian_filter(Hz, sig / bin_) / np.maximum(Hn, 1e-9)
    col = plt.get_cmap(cmap)(np.clip((Hz - zr[0]) / (zr[1] - zr[0]), 0, 1))[..., :3]
    v = np.clip(Hn / np.percentile(Hn[Hn > 0], 99.5), 0, 1) ** 0.7
    return col * v[..., None]
n = min(nshow, len(good)); fig, ax = plt.subplots(n, 5, figsize=(17.5, 3.6 * n), squeeze=False)
for i, s in enumerate(good[:n]):
    ii = np.array(T.query_ball_point([s['cx'], s['cy']], 95.0)); px = x[ii] - s['cx']; py = y[ii] - s['cy']; pz = z[ii] - s['zmid']
    ext = (-95, 95); zr = (-45, 45)
    ax[i, 0].imshow(img(px, py, pz, ext, zr=zr), extent=[*ext, ext[1], ext[0]]); ax[i, 0].set_title(f'site {i + 1}: top view, colour = z (±45 nm)\n{len(ii)} localizations, ring radius {s["R"]:.1f} nm', fontsize=9)
    for j, (m, nm) in enumerate([(pz < 0, 'lower ring (z < 0)'), (pz >= 0, 'upper ring (z > 0)')]):
        ax[i, 1 + j].imshow(img(px[m], py[m], None, ext), cmap='inferno', extent=[*ext, ext[1], ext[0]]); ax[i, 1 + j].set_title(f'{nm}: {int(m.sum())} localizations', fontsize=9)
    for j, (pa, nm) in enumerate([(px, 'x'), (py, 'y')]):
        e = np.arange(-95, 96.5, 1.5); ez = np.arange(-80, 81.5, 1.5)
        Hn = gaussian_filter(np.histogram2d(pz, pa, bins=[ez, e])[0], 2.0 / 1.5)
        ax[i, 3 + j].imshow(Hn, cmap='inferno', extent=[-95, 95, 80, -80], aspect='equal'); ax[i, 3 + j].invert_yaxis()
        ax[i, 3 + j].set_title(f'side view {nm}-z; ring separation {s["sep"]:.0f} nm, z spread per ring {s["zs"]:.1f} nm', fontsize=9)
    for a_ in ax[i]:
        a_.plot([-85, -35], [a_.get_ylim()[0] * 0.88] * 2, 'w-', lw=3); a_.set_xticks([]); a_.set_yticks([])
fig.suptitle('NPC (Nup96), classical 4Pi-SIMFLUX pixel fit, individual pores picked automatically (complete rings in both z layers, isolated). Scale bar 50 nm; 1.5 nm bins, 2 nm blur.', fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(f'{OUT}/sites/npc_sites.png', dpi=140); print('saved', f'{OUT}/sites/npc_sites.png')
for s in good[:n]: print({k_: (round(v, 1) if isinstance(v, float) else v) for k_, v in s.items()})
