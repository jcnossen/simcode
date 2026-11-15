"""COPY of sweep4pi_eval.py (hd head experiment, 2026-11-15): --locs (network table), --bound (optional true pattern + field
table), --out (output directory); classical rows are not evaluated here (see the sweep eval.json). Metrics unchanged.
Original docstring:
4Pi-SIMCODE density sweep: ground-truth evaluation of every method / selection on one sweep movie (sweep4pi_movie.py).

Region: x, y in [8, 52) px (5.63 um square; candidates need a full 15 px patch). All lengths in nm (x, y = px * 128,
z = um * 1000 relative to the focus).

Primary (gtfrc.py of analysis/papersim, z3d worktree, imported read-only; dimension-agnostic):
  GT-FRC x-y   gtfrc.gt_frc of the x-y projection: reconstruction vs ALL labelling sites in the region, both rendered as
               histograms on the same field with 5 nm pixels, 1/7 crossing of the 5-shell smoothed curve.
  GT-FRC x-z   the same estimator pooled over x-z SECTIONS: the region is cut into y-slabs of 250 nm; per slab the (x, z)
               histograms (5 nm pixels, z field -450..+450 nm so that one-fringe ghosts at +-255 nm stay in the field and
               count as uncorrelated power) are Fourier transformed and the per-shell cross / auto powers are summed over the
               slabs before the correlation is formed (gtfrc.frc_images summed over sections; shell width 1 / 5632 nm).
  time to fidelity  both resolutions on the localizations with frame < N (frame = 6 w) on a sqrt(2) log grid of N, and the
               frames needed to reach each target (gtfrc.time_to_fidelity rule: log-interpolated, None = not reached).
Diagnostics:
  coverage     gtfrc.coverage_cv: 250 nm segments along each microtubule axis (sites assigned to the nearest axis point of
               their own microtubule), expected = the segment's share of ground-truth emitter-windows -> CV, Poisson floor, excess
  ghost        gtfrc.ghost_fraction (3D, 3 own sigma; no emitter ON in the localization's own window) and the wrong-fringe
               rate of matched localizations (|dz| > 100 nm)
  sigma-Jaccard  gtfrc.sigma_jaccard, 3D Mahalanobis gate 3 own sigma vs all labelling sites in the region
  errors       simeval4pi matching (per window, greedy within 64 nm laterally, against all GT units); recall over GT units >= 300 e,
               precision; robust (1.4826 MAD after the median offset) lateral / axial error over all matched and at 2-4k e
  hollow tube  localizations within 120 nm (3D) of an axis point that has no other microtubule within 200 nm (x-y):
               in-plane / axial cross-section dip (profile at the axis / mean of the two peaks, < 0.81 = resolved), ring radius
Methods: network = head v10 + explicit solve, in-situ pattern + phase field (locs_*_iterv10), under several selections
(SELECTIONS below); 'net_bound' = true pattern + field (if run); classical published-style ('cl_pub', its published
rejections) and pixel MLE ('cl_pix'); 'gt_units' = the true positions of every GT unit >= 300 e (the sampling limit).
Output: $P4/sweep4pi/<tag>/eval.json, locs_<method>.npz (selected, region), curves.npz
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi_sim/sweep4pi_eval.py --tag mtc_d03
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, f'{REPO}/analysis/fourpi')
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
from simeval4pi import gt_units, match

GTFRC = os.environ.get('GTFRC', os.path.join(REPO, 'analysis', 'papersim', 'gtfrc.py'))
_s = importlib.util.spec_from_file_location('gtfrc', GTFRC); G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)

NM = 128.0
LO, HI = 8.0, 52.0
PIX = 5.0
ZF = (-450.0, 450.0)
SLAB = 250.0
SEG = 250.0
K = 3.0
TARGETS_XY = (60, 40, 30, 25, 22, 20, 15)
TARGETS_XZ = (80, 60, 50, 40, 35, 30, 25)
# network selections: base = simeval4pi filters (sigma_z < 25 nm, sigma_xy < 0.12 px, N > 300)
SELECTIONS = {
    'net_all':    'every candidate',
    'net_base':   'base filters, chi2 selection off',
    'net_chi16':  'base + chi2/dof < 1.6 (showcase default)',
    'net_chi13':  'base + chi2/dof < 1.3 (stricter)',
    'net_sig0':   'base + sigma_xy < 7 nm and sigma_z < 16 nm',
    'net_sig1':   'base + sigma_xy < 5 nm and sigma_z < 12 nm',
    'net_sig2':   'base + sigma_xy < 3.5 nm and sigma_z < 8 nm',
    'net_chi16_sig1': 'base + chi2/dof < 1.6 + sigma_xy < 5 nm, sigma_z < 12 nm',
}


def frc_sections(pos, gt_img_sections, lo, hi, slab):
    """pooled x-z section FRC: pos [n, 3] nm; gt_img_sections list of (x, z) GT renders per y-slab -> dict(res_nm, freq, curve)"""
    if len(pos) == 0: return dict(res_nm=float('nan'), freq=[], curve=[], n=0)
    num = da = db = None
    ys = np.arange(lo[1], hi[1], slab)
    for k, y0 in enumerate(ys):
        s = (pos[:, 1] >= y0) & (pos[:, 1] < y0 + slab)
        a = G.render(pos[s][:, [0, 2]], (lo[0], ZF[0]), (hi[0], ZF[1]), PIX); b = gt_img_sections[k]
        A, B = np.fft.fftn(a), np.fft.fftn(b)
        df = 1.0 / (max(a.shape) * PIX)
        fr = np.sqrt(sum(g ** 2 for g in np.meshgrid(*[np.fft.fftfreq(n, PIX) for n in a.shape], indexing='ij')))
        kk = (fr / df).astype(int).ravel(); nmax = int(0.5 / PIX / df)
        n_ = np.bincount(kk, (A * B.conj()).real.ravel(), nmax + 1)[:nmax]
        a_ = np.bincount(kk, (np.abs(A) ** 2).ravel(), nmax + 1)[:nmax]; b_ = np.bincount(kk, (np.abs(B) ** 2).ravel(), nmax + 1)[:nmax]
        num, da, db = (n_, a_, b_) if num is None else (num + n_, da + a_, db + b_)
    c = num / np.sqrt(np.maximum(da * db, 1e-30)); c = np.convolve(c, np.ones(5) / 5, 'same')
    freq = np.arange(len(c)) * df
    return dict(res_nm=G.crossing(freq, c), freq=freq.tolist(), curve=c.tolist(), n=int(len(pos)))


def last_crossing(freq, curve, threshold=G.THRESHOLD, start=3):
    """1 / (highest frequency, from shell `start` on, at which the curve is still >= threshold): the curve stays below the
    threshold beyond it. Robust to the structural zeros of the hollow-tube spectrum (J0(2 pi f r), r = 24.6 nm: zeros at
    1/64 and 1/28 nm), where the first-crossing rule stops. nan if the curve is never >= threshold; the Nyquist period
    (2 pixels = 10 nm) if it never drops below."""
    c = np.asarray(curve); f = np.asarray(freq)
    if len(c) <= start: return float('nan')
    above = np.nonzero(c[start:] >= threshold)[0]
    if len(above) == 0: return float('nan')
    i = above[-1] + start
    return float(1.0 / f[i]) if i < len(c) - 1 else 2 * PIX


def ttf(res, grid, targets):
    reach = {}
    r = np.array(res, float)
    for t in targets:
        ok = np.isfinite(r) & (r <= t)
        if not ok.any(): reach[str(t)] = None; continue
        i = int(np.argmax(ok))
        if i == 0 or not np.isfinite(r[i - 1]): reach[str(t)] = float(grid[i]); continue
        u = (r[i - 1] - t) / (r[i - 1] - r[i])
        reach[str(t)] = float(np.exp(np.log(grid[i - 1]) + u * (np.log(grid[i]) - np.log(grid[i - 1]))))
    return reach


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='mtc_d03'); p.add_argument('--model', default='v3L_det')
    p.add_argument('--no_ttf', type=int, default=0)
    p.add_argument('--locs', required=True); p.add_argument('--bound', default=''); p.add_argument('--out', required=True)
    a = p.parse_args()
    d = f'{P4}/simmovie/{a.tag}'; out = a.out; os.makedirs(out, exist_ok=True)
    meta = json.load(open(f'{d}/meta.json')); T = meta['T']
    t0 = time.time()
    lo, hi = (LO * NM, LO * NM), (HI * NM, HI * NM)
    inr = lambda x, y, m=0.0: (x >= LO - m) & (x < HI + m) & (y >= LO - m) & (y < HI + m)
    # ---- ground truth
    S = np.load(f'{d}/gt_sites.npz')['sites']
    sreg = inr(S[:, 0], S[:, 1])
    GT = np.c_[S[sreg, 0] * NM, S[sreg, 1] * NM, S[sreg, 2] * 1000]; GTmt = S[sreg, 3].astype(int)
    U = gt_units(d, T)
    gall = inr(U['x'], U['y'], 1.0)
    GA = {k: v[gall] for k, v in U.items()}
    gsel = inr(GA['x'], GA['y']) & (GA['N'] >= 300)
    ev = np.load(f'{d}/gt_events.npz'); site_of_event = ev['site']
    # emitter-windows per labelling site (all GT units, any photons) -> coverage weight
    wsite = np.bincount(site_of_event[U['event']], minlength=len(S))[sreg].astype(float)
    # ON emitters per window (all GT units, 3D nm) for the ghost test
    EW = {}
    o = np.argsort(U['w'], kind='stable'); ws = U['w'][o]; cut = np.r_[0, np.nonzero(np.diff(ws))[0] + 1, len(ws)]
    for i0, i1 in zip(cut[:-1], cut[1:]):
        ii = o[i0:i1]; EW[int(ws[i0])] = np.c_[U['x'][ii] * NM, U['y'][ii] * NM, U['z'][ii] * 1000]
    # segments: 250 nm along each axis (axis points 2 nm apart, ordered per microtubule)
    AX = np.load(f'{d}/gt_axis.npz')['axis']; axmt = AX[:, 6].astype(int)
    arc = np.zeros(len(AX))
    for m_ in np.unique(axmt):
        ii = np.nonzero(axmt == m_)[0]; st = np.r_[0, np.linalg.norm(np.diff(AX[ii, :3] * 1000, axis=0), axis=1)]   # nm
        arc[ii] = np.cumsum(st)
    axseg_key = axmt * 100000 + np.floor(arc / SEG).astype(int)
    site_ax = np.zeros(len(GT), int)
    for m_ in np.unique(GTmt):
        ia = np.nonzero(axmt == m_)[0]; isite = np.nonzero(GTmt == m_)[0]
        _, j = cKDTree(AX[ia, :3] * 1000).query(GT[isite])
        site_ax[isite] = ia[j]
    key = axseg_key[site_ax]
    uk, seg = np.unique(key, return_inverse=True)
    axin = inr(AX[:, 0] * 1000 / NM, AX[:, 1] * 1000 / NM)
    seg_len = np.array([2.0 * np.sum((axseg_key == k_) & axin) for k_ in uk])
    tube_um = float(axin.sum() * 2.0 / 1000)
    g = np.load(f'{d}/gt_frames.npz')
    res = dict(tag=a.tag, dens_fov=meta['dens_actual'], T=T, region_um=(HI - LO) * NM / 1000, tube_um_region=tube_um,
               dens_region=float(inr(g['x'], g['y']).sum() / T / ((HI - LO) * NM / 1000) ** 2),
               n_sites_region=int(len(GT)), gt_units=int(gsel.sum()), gt_units_per_um=float(gsel.sum() / tube_um),
               n_segments=int(np.sum(seg_len >= SEG / 2)))
    # crowding: other ON emitters in the 15 x 15 px patch, same frame (as the showcase)
    fr = g['frame']; fx, fy = g['x'], g['y']
    o = np.argsort(fr, kind='stable'); cut = np.r_[0, np.nonzero(np.diff(fr[o]))[0] + 1, len(fr)]
    cnt = []
    for i0, i1 in zip(cut[:-1:7], cut[1::7]):     # every 7th frame
        ii = o[i0:i1]; q = inr(fx[ii], fy[ii]); jj = ii[q]
        if len(jj) == 0: continue
        cnt.append(((np.abs(fx[jj][:, None] - fx[ii][None]) < 7.5) & (np.abs(fy[jj][:, None] - fy[ii][None]) < 7.5)).sum(1) - 1)
    res['crowd_frame_mean'] = float(np.concatenate(cnt).mean())
    print(f"{a.tag}: FOV {res['dens_fov']:.2f} /um2/frame, region {res['dens_region']:.2f}, tube {tube_um:.0f} um in region, "
          f"GT units {res['gt_units']} ({res['gt_units_per_um']:.0f}/um), others in 15 px patch {res['crowd_frame_mean']:.2f}, "
          f"{res['n_segments']} segments  [{time.time() - t0:.0f}s]", flush=True)
    # ---- GT renders
    gimg_xy = G.render(GT[:, :2], lo, hi, PIX)
    ys = np.arange(lo[1], hi[1], SLAB)
    gimg_xz = [G.render(GT[(GT[:, 1] >= y0) & (GT[:, 1] < y0 + SLAB)][:, [0, 2]], (lo[0], ZF[0]), (hi[0], ZF[1]), PIX) for y0 in ys]
    # ---- methods
    M = {}
    L = dict(np.load(a.locs))
    sxy = np.maximum(L['sx'], L['sy']) * NM; szn = L['sz'] * 1000; c2 = L['chi2'] / L['dof']
    base = (L['sz'] < 0.025) & (L['sx'] < 0.12) & (L['sy'] < 0.12) & (L['N'] > 300)
    sig0 = (sxy < 7) & (szn < 16); sig1 = (sxy < 5) & (szn < 12); sig2 = (sxy < 3.5) & (szn < 8)
    selm = dict(net_all=np.ones(len(c2), bool), net_base=base, net_chi16=base & (c2 < 1.6), net_chi13=base & (c2 < 1.3),
                net_sig0=base & sig0, net_sig1=base & sig1, net_sig2=base & sig2, net_chi16_sig1=base & (c2 < 1.6) & sig1)
    for nm, s in selm.items():
        M[nm] = (L['x'], L['y'], L['z'], L['sx'], L['sy'], L['sz'], L['w'], s)
    fb = a.bound
    if fb and os.path.exists(fb):
        B = dict(np.load(fb)); bb = (B['sz'] < 0.025) & (B['sx'] < 0.12) & (B['sy'] < 0.12) & (B['N'] > 300)
        M['net_bound_base'] = (B['x'], B['y'], B['z'], B['sx'], B['sy'], B['sz'], B['w'], bb)
    fk = ''
    if os.path.exists(fk):
        Kc = dict(np.load(fk))
        M['cl_pub'] = (Kc['x'], Kc['y'], Kc['z'], Kc['sx'], Kc['sy'], Kc['sz'], Kc['w'], Kc['keep'])
        M['cl_pix'] = (Kc['xp'], Kc['yp'], Kc['zp'], Kc['sxp'], Kc['syp'], Kc['szp'], Kc['w'], Kc['keep_pix'])
    res['classical_present'] = os.path.exists(fk)
    gu = inr(U['x'], U['y']) & (U['N'] >= 300)
    M['gt_units'] = (U['x'][gu], U['y'][gu], U['z'][gu], np.full(gu.sum(), 1e-3), np.full(gu.sum(), 1e-3), np.full(gu.sum(), 1e-5), U['w'][gu], np.ones(gu.sum(), bool))
    grid = [int(round(500 * 2 ** (k / 2))) for k in range(30) if 500 * 2 ** (k / 2) < T] + [T]
    res['ttf_grid'] = grid; res['methods'] = {}; curves = {}
    Gx, Gy, Gz, GN = GA['x'][gsel], GA['y'][gsel], GA['z'][gsel], GA['N'][gsel]
    # axis points usable for the tube profile: no other microtubule within 200 nm in x-y
    usable = axin.copy()
    for m_ in np.unique(axmt):
        im = axmt == m_
        if (~im).any():
            dd, _ = cKDTree(AX[~im, :2]).query(AX[im, :2]); usable[np.nonzero(im)[0][dd < 0.2]] = False
    tz = AX[:, 3:6]; zh = np.array([0.0, 0.0, 1.0])
    n1 = np.cross(tz, zh); n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-9); n2 = np.cross(tz, n1); n2 *= np.sign(n2[:, 2:3] + 1e-12)
    tr3 = cKDTree(AX[:, :3])
    rr = 24.6

    def dip(e):
        h, b = np.histogram(e, np.arange(-80, 80.1, 4.0)); c = 0.5 * (b[1:] + b[:-1])
        h = np.convolve(h, [0.25, 0.5, 0.25], 'same')
        pk = max(h[(c < -rr * 0.5) & (c > -rr * 1.6)].max(), 1e-9), max(h[(c > rr * 0.5) & (c < rr * 1.6)].max(), 1e-9)
        return float(h[np.abs(c) < 4].mean() / np.mean(pk))
    for nm, (x, y, z, sx, sy, sz, w, ok) in M.items():
        ok = ok & inr(x, y) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z, w = x[ok], y[ok], z[ok], w[ok].astype(np.int64)
        P3 = np.c_[x * NM, y * NM, z * 1000]; SG = np.c_[sx[ok] * NM, sy[ok] * NM, sz[ok] * 1000]
        r = dict(n_locs=int(len(x)), desc=SELECTIONS.get(nm, nm), locs_per_um=float(len(x) / tube_um))
        # matching / errors
        ok1 = inr(x, y, 1.0)
        if len(x):
            mg, ml = match(GA['w'], GA['x'], GA['y'], w, x, y)
        else:
            mg, ml = np.full(len(GA['w']), -1), np.zeros(0, int)
        r['precision'] = float((ml >= 0).mean()) if len(x) else float('nan')
        m = mg[gsel]; mm = m >= 0; r['recall'] = float(mm.mean())
        ex = (x[m[mm]] - Gx[mm]) * NM; ey = (y[m[mm]] - Gy[mm]) * NM; ez = (z[m[mm]] - Gz[mm]) * 1000
        rb = lambda e: float(1.4826 * np.median(np.abs(e - np.median(e)))) if len(e) >= 30 else float('nan')
        r['lat_rob'] = 0.5 * (rb(ex) + rb(ey)); fringe = np.abs(ez) > 100
        r['ax_rob'] = rb(ez[~fringe]) if (~fringe).sum() >= 30 else float('nan')
        r['ax_rob_all'] = rb(ez); r['wrong_fringe'] = float(fringe.mean()) if mm.any() else float('nan')
        b24 = (GN[mm] >= 2000) & (GN[mm] < 4000)
        r['lat_rob_24'] = 0.5 * (rb(ex[b24]) + rb(ey[b24])); r['ax_rob_24'] = rb(ez[b24]); r['wrong_fringe_24'] = float(fringe[b24].mean()) if b24.any() else float('nan')
        # GT-FRC
        fxy = G.gt_frc(P3[:, :2], None, lo, hi, PIX, 1.0, gt_img=gimg_xy)
        fxz = frc_sections(P3, gimg_xz, lo, hi, SLAB)
        r['gtfrc_xy'] = fxy['res_nm']; r['gtfrc_xz'] = fxz['res_nm']
        r['gtfrc_last_xy'] = last_crossing(fxy['freq'], fxy['curve']) if len(fxy['freq']) else float('nan')
        r['gtfrc_last_xz'] = last_crossing(fxz['freq'], fxz['curve']) if len(fxz['freq']) else float('nan')
        curves[f'{nm}__xy'] = np.array(fxy['curve']); curves[f'{nm}__xz'] = np.array(fxz['curve'])
        curves['freq_xy'] = np.array(fxy['freq']) if len(fxy['freq']) else curves.get('freq_xy'); curves['freq_xz'] = np.array(fxz['freq']) if len(fxz['freq']) else curves.get('freq_xz')
        if not a.no_ttf:
            frm = 6 * w
            cxy = [G.gt_frc(P3[frm < N, :2], None, lo, hi, PIX, 1.0, gt_img=gimg_xy) for N in grid]
            cxz = [frc_sections(P3[frm < N], gimg_xz, lo, hi, SLAB) for N in grid]
            r['ttf_xy'] = [c['res_nm'] for c in cxy]; r['ttf_xz'] = [c['res_nm'] for c in cxz]
            r['ttf_last_xy'] = [last_crossing(c['freq'], c['curve']) if len(c['freq']) else float('nan') for c in cxy]
            r['ttf_last_xz'] = [last_crossing(c['freq'], c['curve']) if len(c['freq']) else float('nan') for c in cxz]
            r['frames_to_xy'] = ttf(r['ttf_xy'], grid, TARGETS_XY); r['frames_to_xz'] = ttf(r['ttf_xz'], grid, TARGETS_XZ)
            r['frames_to_last_xy'] = ttf(r['ttf_last_xy'], grid, TARGETS_XY); r['frames_to_last_xz'] = ttf(r['ttf_last_xz'], grid, TARGETS_XZ)
        if nm != 'gt_units':
            r['sigma_jaccard'] = G.sigma_jaccard(P3, SG, GT, K)
            r['ghost'] = G.ghost_fraction(P3, SG, w, EW, K)
            r['ghost_fixed'] = G.ghost_fraction(P3, np.tile([10.0, 10.0, 25.0], (len(P3), 1)), w, EW, K)   # fixed gate 30 / 30 / 75 nm
            r['coverage'] = G.coverage_cv(P3, SG, GT, seg, seg_len, wsite, SEG, K)
        # tube profile
        dd, ii = tr3.query(P3 / 1000)
        okp = (dd < 0.12) & usable[ii]
        dv = P3[okp] / 1000 - AX[ii[okp], :3]
        u = (dv * n1[ii[okp]]).sum(1) * 1000; v = (dv * n2[ii[okp]]).sum(1) * 1000
        rho = np.hypot(u, v)
        r['tube'] = dict(n=int(len(u)), dip_lat=dip(u) if len(u) > 100 else float('nan'), dip_ax=dip(v) if len(u) > 100 else float('nan'),
                         radius_med=float(np.median(rho)) if len(u) else float('nan'))
        np.savez(f'{out}/locs_{nm}.npz', x=P3[:, 0], y=P3[:, 1], z=P3[:, 2], sx=SG[:, 0], sy=SG[:, 1], sz=SG[:, 2], w=w, match=ml >= 0,
                 prof_u=u, prof_v=v)
        res['methods'][nm] = r
        cv = r.get('coverage', {}); gh = r.get('ghost', {}); sj = r.get('sigma_jaccard', {})
        print(f"{nm:15s} n {r['n_locs']:7d} rec {r['recall']:.2f} prec {r['precision']:.3f} GT-FRC xy {r['gtfrc_xy']:5.1f} xz {r['gtfrc_xz']:5.1f} "
              f"last xy {r['gtfrc_last_xy']:5.1f} xz {r['gtfrc_last_xz']:5.1f} ghost fixed {r.get('ghost_fixed', {}).get('ghost_fraction', np.nan):.3f} | "
              f"lat {r['lat_rob']:.1f} ax {r['ax_rob']:.1f} fringe {r['wrong_fringe']:.3f} ghost {gh.get('ghost_fraction', np.nan):.3f} "
              f"sigJ {sj.get('jaccard', np.nan):.3f} covCV {cv.get('cv', np.nan):.3f} (P {cv.get('poisson_cv', np.nan):.3f}) "
              f"dip {r['tube']['dip_lat']:.2f}/{r['tube']['dip_ax']:.2f} | frames-to (last) xy {r.get('frames_to_last_xy')} xz {r.get('frames_to_last_xz')} [{time.time() - t0:.0f}s]", flush=True)
    # GT itself (all labelling sites) tube profile, for the dip reference
    dd, ii = tr3.query(GT / 1000); okp = (dd < 0.12) & usable[ii]; dv = GT[okp] / 1000 - AX[ii[okp], :3]
    u = (dv * n1[ii[okp]]).sum(1) * 1000; v = (dv * n2[ii[okp]]).sum(1) * 1000
    res['gt_sites_tube'] = dict(n=int(len(u)), dip_lat=dip(u), dip_ax=dip(v), radius_med=float(np.median(np.hypot(u, v))))
    np.savez(f'{out}/prof_gt.npz', u=u, v=v)
    np.savez(f'{out}/curves.npz', **{k: v for k, v in curves.items() if v is not None})
    json.dump(res, open(f'{out}/eval.json', 'w'), indent=1, default=float)
    print(f'total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
