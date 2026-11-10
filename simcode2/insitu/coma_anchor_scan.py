"""
Pattern-anchored profile scan of odd aberrations (critic review 13 #2/#3).
Anchor: per emitter and angle, the pattern-only position along k_a from an exact 3-step phase solve on I_ref (frames of
that angle, both cycles averaged) -> position phase k_a.r mod 2pi, unwrapped to the integer nearest the PSF reference.
The PSF-only position is biased by unmodelled odd aberrations (x-coma x z); the anchor is not (to first order).
For each candidate (Noll 7, Noll 8) the 12-frame sums (15x15, neighbour pixels masked) are fitted with x,y softly anchored
(prior sigma from the anchor's photon statistics), z with a Gaussian TIRF-like prior. Report total NLL and the bias
D = (PSF-fit position with weak prior) - anchor along k vs photon quartile for both angles, on two time halves.
"""
import argparse, json, math, os
import numpy as np, tifffile, torch
from simcode2.psf.vectorial import VectorialPSF
from simcode2.insitu.lm_refine import lm_refine
from simcode2.util.neighbours import nn_overlap
from simcode2 import config
from simcode2.insitu import MOVIE_05, data, out, cam_maps

ap = argparse.ArgumentParser()
ap.add_argument('--locs', default=out('legacy', 'real_v6_p5m_f0.npz'),
                help='locs npz (a legacy localize_real output of the development tree: x_ref, y_ref, I_ref [N,12], I_pre, I_post, phot, z, wstart, crop, f0, f1; copies in <out>/insitu/legacy)')
ap.add_argument('--pattern', default=os.path.join(config.root('out'), 'calib_inputs', 'pattern5_0.5nM.pt'),
                help='pattern .pt (k, phi, dphi, knot_frames; fit_pattern.py format)')
ap.add_argument('--movie', default=MOVIE_05)
ap.add_argument('--calib', default='insitu/calib_0.5nM.json')
ap.add_argument('--grid8', default='-0.08,-0.04,0,0.04,0.08')
ap.add_argument('--grid7', default='-0.04,0,0.04')
ap.add_argument('--max_emitters', type=int, default=5000)
ap.add_argument('--scan', default='coma')          # coma: grid8 x grid7 = Noll 8 / Noll 7 offsets; saf: grid8 = t_x, grid7 = a_x
ap.add_argument('--zproxy', type=float, default=0.0)   # >0: z fixed to delta*ln(B_ref/B_site) (TIRF height proxy, site brightness)
ap.add_argument('--out', default=out('coma_scan.json'))
a = ap.parse_args()
a.movie, a.calib = data(a.movie), config.resolve(a.calib)
dev = 'cuda'; T = 12; P = 15; h = P // 2; px = 108.33
L = np.load(a.locs)
c1, c2 = L['I_ref'][:, :6].sum(1), L['I_ref'][:, 6:].sum(1)
nn = nn_overlap(L['x_ref'], L['y_ref'], L['wstart'])
sel = np.isfinite(L['I_pre']) & (L['I_pre'] > 0.75 * c1) & np.isfinite(L['I_post']) & (L['I_post'] > 0.75 * c2) & \
      (L['phot'] > 1000) & (nn >= 3)
idx_sel = np.nonzero(sel)[0]
rng = np.random.default_rng(0)
if len(idx_sel) > a.max_emitters: idx_sel = np.sort(rng.choice(idx_sel, a.max_emitters, replace=False))
xr, yr, ws, I, ph = L['x_ref'][idx_sel], L['y_ref'][idx_sel], L['wstart'][idx_sel].astype(int), L['I_ref'][idx_sel], L['phot'][idx_sel]

# ---- site brightness (events of the same site across windows) -> TIRF height proxy (critic review 14)
from scipy.spatial import cKDTree as _KD
allsel = np.nonzero(sel)[0]
pts = np.stack([L['x_ref'][allsel], L['y_ref'][allsel]], 1)
pairs = _KD(pts).query_pairs(0.25, output_type='ndarray')
parent = np.arange(len(pts))
def _find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i_, j_ in pairs:
    ri, rj = _find(i_), _find(j_)
    if ri != rj: parent[ri] = rj
root = np.array([_find(i_) for i_ in range(len(pts))])
phot_all = L['phot'][allsel]
site_med = {r_: np.median(phot_all[root == r_]) for r_ in np.unique(root)}
Bsite_all = np.array([site_med[r_] for r_ in root])
pos_in_all = {v: i_ for i_, v in enumerate(allsel)}
Bsite = np.array([Bsite_all[pos_in_all[v]] for v in idx_sel])
B_ref = np.percentile(Bsite_all, 90)
zprox = np.clip(a.zproxy * np.log(B_ref / Bsite), 0, 0.3) if a.zproxy > 0 else None
print(f'sites: {len(np.unique(root))} for {len(allsel)} windows; B_site p10/p50/p90 {np.round(np.percentile(Bsite_all, [10, 50, 90]))}')

# ---- pattern anchor
Pt = torch.load(a.pattern); k = Pt['k'].numpy().astype(float); phi = Pt['phi'].numpy(); dphi = Pt['dphi'].numpy(); kf = Pt['knot_frames']
rel = np.exp(Pt['logrel'].numpy() - Pt['logrel'].numpy().mean()); m = Pt['m'].numpy()
fa = ws[:, None] + np.arange(T)[None]; s = fa % 6; an = s // 3
tt = fa / kf; i0 = np.minimum(np.floor(tt).astype(int), dphi.shape[1] - 2); w = np.clip(tt - i0, 0, 1)
off = phi[s] + (dphi[an, i0] * (1 - w) + dphi[an, i0 + 1] * w)          # I_t = A rel (1 + m cos(k.r - off_t))
phase = np.zeros((len(xr), 2)); sig_ph = np.zeros((len(xr), 2))
for q in (0, 1):
    msk = an == q
    It = np.where(msk, I / rel[s], 0)
    # linear LSQ per emitter: I = A + U cos(off) + V sin(off)  ->  k.r = atan2(V, U)  (cos(kr - off) = cos kr cos off + sin kr sin off)
    C, S_ = np.cos(off), np.sin(off)
    X = np.stack([msk * 1.0, msk * C, msk * S_], -1)                       # [N,T,3]
    A_ = np.einsum('ntk,ntl->nkl', X, X); b_ = np.einsum('ntk,nt->nk', X, It)
    coef = np.linalg.solve(A_ + 1e-9 * np.eye(3), b_[..., None])[..., 0]
    phase[:, q] = np.arctan2(coef[:, 2], coef[:, 1])
    Nq = np.maximum(coef[:, 0] * 6, 10)
    sig_ph[:, q] = np.sqrt(2.0 / (m[3 * q] ** 2 * Nq)) * 1.3                # rough phase sd (x1.3 for bg/read noise)
K = k                                                                       # rows: k_a
r_ref = np.stack([xr, yr], 1)
ph_ref = r_ref @ K.T                                                        # k_a . r_ref
nwrap = np.round((ph_ref - phase) / (2 * np.pi))
ph_un = phase + 2 * np.pi * nwrap
r_anchor = np.linalg.solve(K, ph_un.T).T                                    # [N,2] global px
kn = np.linalg.norm(K, axis=1)
sig_pos = sig_ph / kn[None]                                                 # px along each k (k ~ axis aligned)
print(f'{len(xr)} emitters; anchor - ref along k (median, nm): {np.round(np.median(((r_anchor - r_ref) @ (K / kn[:, None]).T), 0) * px, 2)}; '
      f'anchor sigma median {np.round(np.median(sig_pos, 0) * px, 2)} nm')

# ---- data: 15x15 12-frame sums, neighbour pixels masked (huge read noise)
calib = json.load(open(a.calib)); cam = calib['camera']; maps = np.load(cam_maps(a.calib, cam))
x0c, y0c, Sc = [int(v) for v in L['crop']]; f0, f1 = int(L['f0']), int(L['f1']); M = 10; X0, Y0, S = x0c - M, y0c - M, Sc + 2 * M
tif = tifffile.TiffFile(a.movie)
mov = np.stack([tif.pages[i].asarray()[Y0:Y0 + S, X0:X0 + S] for i in range(f0, f1)]).astype(np.float32)
mov = (mov - maps['offset_adu'][Y0:Y0 + S, X0:X0 + S]) * cam['gain_e_per_adu']
rnm = maps['readnoise_e'][Y0:Y0 + S, X0:X0 + S] ** 2
cx, cy = np.round(xr).astype(int) - X0, np.round(yr).astype(int) - Y0
okk = (cx >= h) & (cx < S - h) & (cy >= h) & (cy < S - h) & (ws - f0 + T <= f1 - f0)
roi = np.stack([mov[w_ - f0:w_ - f0 + T, y - h:y + h + 1, x - h:x + h + 1].sum(0) for w_, x, y in zip(ws, cx, cy)])
rn2 = np.stack([rnm[y - h:y + h + 1, x - h:x + h + 1] * T for x, y in zip(cx, cy)])
# neighbour mask: other localizations in temporally overlapping windows (not the same site)
allx, ally, allw = L['x_ref'], L['y_ref'], L['wstart']
from scipy.spatial import cKDTree
tree = cKDTree(np.stack([allx, ally], 1))
oo = np.arange(P) - h
for n_ in range(len(xr)):
    for j in tree.query_ball_point([xr[n_], yr[n_]], h + 4):
        if abs(allw[j] - ws[n_]) >= T or np.hypot(allx[j] - xr[n_], ally[j] - yr[n_]) < 1.0: continue
        dxp = allx[j] - (cx[n_] + X0); dyp = ally[j] - (cy[n_] + Y0)
        mm = (oo[None, :] - dxp) ** 2 + (oo[:, None] - dyp) ** 2 < 3.5 ** 2
        rn2[n_][mm] = 1e9
keep = okk
roi, rn2 = torch.tensor(roi[keep], device=dev)[:, None], torch.tensor(rn2[keep], device=dev, dtype=torch.float32)[:, None]
xr, yr, ws, ph, r_anchor, sig_pos, cx, cy = xr[keep], yr[keep], ws[keep], ph[keep], r_anchor[keep], sig_pos[keep], cx[keep], cy[keep]
Bsite = Bsite[keep]
if zprox is not None: zprox = zprox[keep]
ph = Bsite.copy()   # stratify by site brightness (critic review 14)
N = len(xr)
print(f'{N} emitters with data; masked pixel fraction {(rn2 > 1e8).float().mean().item():.3f}')
half = torch.tensor(ws < np.median(ws), device=dev)

psf = VectorialPSF(P, pixelsize=calib['pixelsize_um'], NA=calib['NA'], n_imm=calib['n_imm'], n_med=calib['n_med'],
                   wavelength=calib['wavelength_um'], oversample=3, device=dev)
z0 = torch.tensor(calib['zernike_waves'], device=dev); zidx = psf.zidx
i7, i8 = zidx.index(7), zidx.index(8)
one = torch.ones(N, device=dev)
ilf = lambda a_, b_: torch.ones(len(a_), 1, device=dev)
anc_rel = torch.tensor(np.stack([r_anchor[:, 0] - X0 - cx, r_anchor[:, 1] - Y0 - cy], 1), device=dev, dtype=torch.float32)
ref_rel = torch.tensor(np.stack([xr - X0 - cx, yr - Y0 - cy], 1), device=dev, dtype=torch.float32)
bg0 = torch.quantile(roi.reshape(N, -1), 0.2, dim=1)
B0 = (roi.reshape(N, -1) - bg0[:, None]).clamp(min=0).sum(1) * 1.25
Kn = torch.tensor(K / kn[:, None], device=dev, dtype=torch.float32)
phq = np.quantile(ph, [0.25, 0.5, 0.75])
pbin = np.digitize(ph, phq)
results = []
for c8 in [float(v) for v in a.grid8.split(',')]:
    for c7 in [float(v) for v in a.grid7.split(',')]:
        with torch.no_grad():
            psf.zcoef.copy_(z0); psf.saf.zero_()
            if a.scan == 'coma':
                psf.zcoef[i8] += c8; psf.zcoef[i7] += c7
            else:
                psf.saf[0] = c8; psf.saf[2] = c7
        row = dict(c8=c8, c7=c7)
        gshift = torch.zeros(2, device=dev)      # global PSF-frame vs pattern-frame offset (gauge), estimated from the free fit
        for mode in ['free', 'anchored']:
            if mode == 'free': xy0, sxy = ref_rel, 10 * one
            else: xy0, sxy = anc_rel + gshift[None], torch.tensor(sig_pos, device=dev, dtype=torch.float32).mean(1)
            zinit = torch.tensor(zprox, device=dev, dtype=torch.float32) if zprox is not None else 0.05 * one
            th0 = torch.stack([xy0[:, 0], xy0[:, 1], zinit, B0, bg0, bg0, 0 * one, 0 * one], 1)
            prior = torch.stack([sxy, sxy, (1e-4 if zprox is not None else 0.06) * one], 1)
            th, _, _ = lm_refine(roi, ilf, psf, calib['focus_um'], th0, torch.ones(N, 1, device=dev), rn2,
                                 torch.zeros(N, 1, dtype=torch.long, device=dev), prior_sigma=prior, prior_scale=1.0, gate=1e9,
                                 zmax=0.4, iters=40)
            with torch.no_grad():
                o = torch.arange(P, device=dev) - h
                mu = th[:, 3, None, None] * psf(th[:, :3], torch.full((N,), calib['focus_um'], device=dev)) + th[:, 4, None, None] + \
                     th[:, 6, None, None] * o[None, None, :] + th[:, 7, None, None] * o[None, :, None]
                var = mu.clamp(min=1e-3) + rn2[:, 0]; w_ = (rn2[:, 0] < 1e8).float()
                nll = ((0.5 * (roi[:, 0] - mu) ** 2 / var + 0.5 * torch.log(var)) * w_).sum((1, 2))
                chi = (((roi[:, 0] - mu) ** 2 / var) * w_).sum((1, 2)) / w_.sum((1, 2))
            if mode == 'anchored':
                row['nll'] = nll.sum().item(); row['nll_h0'] = nll[~half].sum().item(); row['nll_h1'] = nll[half].sum().item()
                row['chi2_med'] = chi.median().item(); row['z_at0'] = (th[:, 2] < 0.003).float().mean().item()
            else:
                Dk = (th[:, :2] - anc_rel) @ Kn.T
                gshift = torch.linalg.solve(Kn, Dk.median(0).values)            # gauge: global offset in px (x,y)
                row['gshift_nm'] = (gshift * px).tolist()
                D = (Dk.cpu().numpy() - Dk.median(0).values.cpu().numpy()[None]) * px   # minus the global offset [nm]
                row['D_by_phot'] = [[float(np.median(D[(pbin == b) & sel_h, q])) for b in range(4)] for q in (0, 1) for sel_h in [np.ones(N, bool)]]
                row['D_slope'] = [float(np.polyfit(np.log2(ph), D[:, q], 1)[0]) for q in (0, 1)]
                row['chi2_free'] = chi.median().item()
        results.append(row)
        print(f"c8 {c8:+.2f} c7 {c7:+.2f} | NLL(anchored) {row['nll']:.1f} | chi2 {row['chi2_med']:.3f} / free {row['chi2_free']:.3f} | "
              f"gshift {np.round(row['gshift_nm'], 1)} | D by photons a0 {np.round(row['D_by_phot'][0], 1)} a1 {np.round(row['D_by_phot'][1], 1)} | slope/doubling {np.round(row['D_slope'], 2)} | z@0 {row['z_at0']:.2f}", flush=True)
best = min(results, key=lambda r: r['nll'])
best_h0 = min(results, key=lambda r: r['nll_h0']); best_h1 = min(results, key=lambda r: r['nll_h1'])
print(f"best by NLL: c8 {best['c8']:+.2f} c7 {best['c7']:+.2f}; time halves: ({best_h0['c8']:+.2f},{best_h0['c7']:+.2f}) / ({best_h1['c8']:+.2f},{best_h1['c7']:+.2f})")
json.dump(dict(results=results, base_noll7=float(z0[i7]), base_noll8=float(z0[i8])), open(a.out, 'w'), indent=1)
