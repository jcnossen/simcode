"""
In-situ PSF refit on high-SNR data (critic review 11 #1, iteration 12 finding chi2/px ~2.3 on bright 12-frame sums).
Data: 12-frame sums of bright, isolated emitters that are on through the window (outer-cycle sums), from a localize_real
output with pattern-free references. Free: Zernike 5..21 (+ optional linear field dependence of Noll 5-11), focus;
per emitter x, y, z (TIRF range), B, bg + slope. Half of the emitters held out for validation (chi2 before/after).
"""
import argparse, json
import numpy as np
import tifffile, torch
from simcode.util.neighbours import nn_overlap
from scipy.spatial import cKDTree
from simcode.psf.vectorial import VectorialPSF
from simcode import config
from simcode.insitu import MOVIE_05, data, out, cam_maps

ap = argparse.ArgumentParser()
ap.add_argument('--locs', default=out('legacy', 'real_v6_p4_f0.npz'),
                help='locs npz with pattern-free references (a legacy localize_real output of the development tree: x_ref, y_ref, I_ref [N,12], I_pre, I_post, phot, z, wstart, crop, f0, f1; copies in <out>/insitu/legacy)')
ap.add_argument('--movie', default=MOVIE_05)
ap.add_argument('--calib', default='insitu/calib_0.5nM.json')
ap.add_argument('--field', type=int, default=1)
ap.add_argument('--iters', type=int, default=1500)
ap.add_argument('--out', default=out('psf_sums_fit.json'))
a = ap.parse_args()
a.movie, a.calib = data(a.movie), config.resolve(a.calib)
dev = 'cuda'; T = 12; P = 11; h = 5
L = np.load(a.locs)
c1, c2 = L['I_ref'][:, :6].sum(1), L['I_ref'][:, 6:].sum(1)
nn = nn_overlap(L['x_ref'], L['y_ref'], L['wstart'])
sel = np.isfinite(L['I_pre']) & (L['I_pre'] > 0.75 * c1) & np.isfinite(L['I_post']) & (L['I_post'] > 0.75 * c2) & (nn >= 6) & (L['phot'] > 1500)
xr, yr, ws, zz = L['x_ref'][sel], L['y_ref'][sel], L['wstart'][sel].astype(int), L['z'][sel]
calib = json.load(open(a.calib)); cam = calib['camera']; maps = np.load(cam_maps(a.calib, cam))
x0c, y0c, Sc = [int(v) for v in L['crop']]; f0, f1 = int(L['f0']), int(L['f1']); M = 8; X0, Y0, S = x0c - M, y0c - M, Sc + 2 * M
tif = tifffile.TiffFile(a.movie)
mov = np.stack([tif.pages[i].asarray()[Y0:Y0 + S, X0:X0 + S] for i in range(f0, f1)]).astype(np.float32)
mov = (mov - maps['offset_adu'][Y0:Y0 + S, X0:X0 + S]) * cam['gain_e_per_adu']
cx, cy = np.round(xr).astype(int) - X0, np.round(yr).astype(int) - Y0
ok = (cx >= h) & (cx < S - h) & (cy >= h) & (cy < S - h) & (ws - f0 + T <= f1 - f0)
cx, cy, ws, zz, xr, yr = cx[ok], cy[ok], ws[ok], zz[ok], xr[ok], yr[ok]
N = len(cx)
roi = torch.tensor(np.stack([mov[w - f0:w - f0 + T, y - h:y + h + 1, x - h:x + h + 1].sum(0) for w, x, y in zip(ws, cx, cy)]), device=dev)
rn2m = torch.tensor(maps['readnoise_e'][Y0:Y0 + S, X0:X0 + S] ** 2, device=dev)
rn2 = torch.stack([rn2m[y - h:y + h + 1, x - h:x + h + 1] for x, y in zip(cx, cy)]) * T
train = torch.tensor(np.random.default_rng(0).random(N) < 0.5, device=dev)
print(f'{N} emitters ({int(train.sum())} train / {int((~train).sum())} held out), median sum photons {roi.sum((1, 2)).median().item():.0f}')

psf = VectorialPSF(P, pixelsize=calib['pixelsize_um'], NA=calib['NA'], n_imm=calib['n_imm'], n_med=calib['n_med'],
                   wavelength=calib['wavelength_um'], oversample=3, device=dev)
with torch.no_grad(): psf.zcoef.copy_(torch.tensor(calib['zernike_waves'], device=dev))
nz = psf.zcoef.numel()
nfield = 7 if a.field else 0                          # Noll 5..11 get a linear field term
fieldc = torch.zeros(nfield, 2, device=dev, requires_grad=bool(a.field))
focus = torch.tensor(calib['focus_um'], device=dev, requires_grad=True)
fx = torch.tensor((xr - (x0c + Sc / 2)) / 64.0, device=dev, dtype=torch.float32)
fy = torch.tensor((yr - (y0c + Sc / 2)) / 64.0, device=dev, dtype=torch.float32)
E = dict(xy=torch.tensor(np.stack([xr - X0 - cx, yr - Y0 - cy], 1), device=dev, dtype=torch.float32),
         z=torch.logit(torch.tensor(np.clip(zz, 0.01, 0.19) / 0.2, device=dev, dtype=torch.float32)),
         logB=torch.log(roi.sum((1, 2)) * 0.8), bg=roi.amin((1, 2)).clone(), slope=torch.zeros(N, 2, device=dev))
for v in E.values(): v.requires_grad_(True)
o = torch.arange(P, device=dev) - h


def model(idx):
    off = None
    if a.field:
        off = torch.zeros(len(idx), nz, device=dev)
        off[:, :nfield] = fieldc[None, :, 0] * fx[idx, None] + fieldc[None, :, 1] * fy[idx, None]
    pos = torch.cat([E['xy'][idx], 0.2 * torch.sigmoid(E['z'][idx])[:, None]], 1)
    p = psf(pos, focus.expand(len(idx)), zcoef_offset=off)
    return torch.exp(E['logB'][idx])[:, None, None] * p + E['bg'][idx, None, None] + \
        E['slope'][idx, 0, None, None] * o[None, None, :] + E['slope'][idx, 1, None, None] * o[None, :, None]


def nll(idx):
    mu = model(idx).clamp(min=1e-3); var = mu + rn2[idx]
    return (0.5 * (roi[idx] - mu) ** 2 / var + 0.5 * torch.log(var)).sum()


def chi2(idx):
    with torch.no_grad():
        mu = model(idx).clamp(min=1e-3)
        return (((roi[idx] - mu) ** 2) / (mu + rn2[idx])).mean((1, 2))


allidx = torch.arange(N, device=dev)
opt_e = torch.optim.Adam(E.values(), lr=0.02)
# 1. emitter params only (current PSF) -> baseline chi2
for it in range(400):
    opt_e.zero_grad(); sum(nll(c) for c in allidx.split(1024)).backward(); opt_e.step()
base = torch.cat([chi2(c) for c in allidx.split(1024)])
print(f'baseline chi2/px median: train {base[train].median().item():.3f}  held-out {base[~train].median().item():.3f}')
# 2. global PSF refit on the train half (emitter params of all emitters keep adapting)
psf.zcoef.requires_grad_(True)
opt_g = torch.optim.Adam([{'params': [psf.zcoef, focus], 'lr': 2e-3}] + ([{'params': [fieldc], 'lr': 2e-3}] if a.field else []))
tr_idx = allidx[train]
for it in range(a.iters):
    opt_e.zero_grad(); opt_g.zero_grad()
    l_tr = sum(nll(c) for c in tr_idx.split(1024))
    l_tr.backward(); opt_g.step()
    sum(nll(c) for c in allidx[~train].split(1024)).backward(); opt_e.step()   # held-out: emitter params only
    if it % 300 == 0 or it == a.iters - 1:
        ch = torch.cat([chi2(c) for c in allidx.split(1024)])
        print(f'{it:5d} chi2/px median train {ch[train].median().item():.3f} held-out {ch[~train].median().item():.3f}  '
              f'focus {focus.item():.3f}  |dZ| {(psf.zcoef - torch.tensor(calib["zernike_waves"], device=dev)).norm().item():.3f}  '
              f'|field| {fieldc.norm().item() if a.field else 0:.3f}', flush=True)
ch = torch.cat([chi2(c) for c in allidx.split(1024)])
zf = (0.2 * torch.sigmoid(E['z'])).detach().cpu().numpy()
res = dict(n=N, chi2_base_train=float(base[train].median()), chi2_base_heldout=float(base[~train].median()),
           chi2_train=float(ch[train].median()), chi2_heldout=float(ch[~train].median()), focus=focus.item(),
           zernike=psf.zcoef.detach().cpu().tolist(), field=fieldc.detach().cpu().tolist() if a.field else None,
           z_median=float(np.median(zf)), z_at_bounds=float(np.mean((zf < 0.005) | (zf > 0.195))))
print(json.dumps({k: v for k, v in res.items() if k not in ('zernike', 'field')}))
print('zernike change (Noll 5..21):', np.round(np.array(res['zernike']) - np.array(calib['zernike_waves']), 3).tolist())
json.dump(res, open(a.out, 'w'), indent=1)
