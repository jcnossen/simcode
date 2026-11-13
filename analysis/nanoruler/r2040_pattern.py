"""Pattern check on a cached sample: per-frame intensities of isolated emitters that stay ON through a 6-frame cycle, and
for every frame index j = t % 6 a regression of the intensity fraction on [1, cos(k r), sin(k r)] for the two pattern
wave vectors. Gives, per frame index: which orientation modulates it, phase, depth, relative power; no assumption on the
order. Also on-time from linking the moving-sum detections and the setup numbers (density, photons, bg, sigma).
Model: fraction_j = a_j (1 + d_j cos(k . r - phi_j)), r = (x, y) in px with pixel centres at integers.
Input: sample.tif (acquisition order), quick_smlm.py table of the 6-frame moving sum (frame = window start)."""
import argparse, json, sys, numpy as np, h5py, tifffile
from scipy.spatial import cKDTree
from scipy.optimize import minimize
ap = argparse.ArgumentParser()
ap.add_argument('--sample', required=True); ap.add_argument('--locs', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--gain', type=float, default=0.45); ap.add_argument('--offset', type=float, default=100.0)
ap.add_argument('--ky', default='-0.0024,1.8574'); ap.add_argument('--kx', default='1.8611,0.0017')
ap.add_argument('--cyc', type=int, default=6); ap.add_argument('--size', type=int, default=400)
a = ap.parse_args()
C = a.cyc
L = h5py.File(a.locs, 'r')['locs'][:]
good = (L['chi2'] < 3.6) & (L['snr'] > 8) & (L['sx'] > 1.4) & (L['sx'] < 2.4)
G = L[good]; nwin = int(L['frame'].max()) + 1
res = {'n_det': int(len(L)), 'n_good': int(len(G)), 'n_windows': nwin, 'good_per_window': float(len(G) / nwin), 'all_per_window': float(len(L) / nwin),
       'on_density_per_px': float(len(G) / nwin / a.size ** 2)}
for k, sc in (('photons', 1.0), ('bg', 1.0 / C), ('sx', 1.0), ('lpx', 1.0)):
    res[k if k != 'bg' else 'bg_per_px_per_frame'] = [float(v * sc) for v in np.percentile(G[k], [5, 50, 95])]
print('detections/window all %.2f good %.2f; ON density %.2e /px; window photons p5/50/95 %s; bg/px/frame %s; sigma %s' % (
    res['all_per_window'], res['good_per_window'], res['on_density_per_px'], np.round(res['photons']), np.round(res['bg_per_px_per_frame'], 3), np.round(res['sx'], 3)))
# ---- link good detections over consecutive window starts (0.6 px) -> track length in window starts
fr = G['frame'].astype(int); xy = np.stack([G['x'], G['y']], 1).astype(float)
order = np.argsort(fr, kind='stable'); fr = fr[order]; xy = xy[order]; G = G[order]
st = np.searchsorted(fr, np.arange(nwin + 1))
tid = -np.ones(len(G), int); nt = 0
prev = None
for f in range(nwin):
    s = slice(st[f], st[f + 1]); n = st[f + 1] - st[f]
    if n == 0: prev = None; continue
    ids = -np.ones(n, int)
    if prev is not None and len(prev[0]):
        dd, j = cKDTree(prev[0]).query(xy[s], distance_upper_bound=0.6)
        ok = np.isfinite(dd); ids[ok] = prev[1][j[ok]]
    new = ids < 0; ids[new] = nt + np.arange(new.sum()); nt += int(new.sum())
    tid[s] = ids; prev = (xy[s], ids)
tl = np.bincount(tid); t0 = np.zeros(nt, int); np.minimum.at(t0, tid, 10 ** 9) if False else None
first = np.full(nt, 10 ** 9); np.minimum.at(first, tid, fr); last = np.zeros(nt, int); np.maximum.at(last, tid, fr)
inner = (first > 0) & (last < nwin - 1)
res['track_len_windows'] = {'n': int(inner.sum()), 'mean': float(tl[inner].mean()), 'median': float(np.median(tl[inner])), 'p90': float(np.percentile(tl[inner], 90))}
big = inner & (tl >= C)
res['track_len_windows_ge_cyc'] = {'n': int(big.sum()), 'mean': float(tl[big].mean()), 'median': float(np.median(tl[big]))}
res['frac_inner_tracks_truncated_note'] = 'tracks touching the sample start / end excluded; a sample of %d windows truncates long events' % nwin
print('tracks (window starts, linked 0.6 px): n %d mean %.1f median %.0f p90 %.0f ; tracks >= %d starts: n %d mean %.1f median %.0f' % (
    inner.sum(), tl[inner].mean(), np.median(tl[inner]), np.percentile(tl[inner], 90), C, big.sum(), tl[big].mean(), np.median(tl[big])))
# ---- emitters ON through a full cycle: window start f (f % C == 0) with the same track at f - C and f + C, isolated (10 px)
key = {}
for i in range(len(G)): key[(int(fr[i]), int(tid[i]))] = i
sel = []
for f in range(C, nwin - C, C):
    s = np.arange(st[f], st[f + 1])
    if len(s) == 0: continue
    allf = L[L['frame'] == f]; tr = cKDTree(np.stack([allf['x'], allf['y']], 1))
    for i in s:
        if (f - C, tid[i]) in key and (f + C, tid[i]) in key and len(tr.query_ball_point(xy[i], 10.0)) == 1:
            sel.append(i)
sel = np.array(sel); print('full-cycle isolated emitter windows:', len(sel)); res['n_cycle_windows'] = int(len(sel))
t = tifffile.TiffFile(a.sample); R = 9; h = R // 2
uf = np.unique(fr[sel]); fmap = {}
from scipy.special import erf
yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(float)
I = np.zeros((len(sel), C)); B = np.zeros((len(sel), C))
bysf = {}
for n_, i in enumerate(sel): bysf.setdefault(int(fr[i]), []).append((n_, i))
for f, lst in bysf.items():
    mov = (np.stack([t.pages[f + j].asarray() for j in range(C)]).astype(np.float64) - a.offset) * a.gain
    for n_, i in lst:
        x, y = xy[i]; ix, iy = int(round(x)), int(round(y)); s = float(G['sx'][i])
        ex = 0.5 * (erf((xx - (x - ix) + 0.5) / (s * 2 ** 0.5)) - erf((xx - (x - ix) - 0.5) / (s * 2 ** 0.5)))
        ey = 0.5 * (erf((yy - (y - iy) + 0.5) / (s * 2 ** 0.5)) - erf((yy - (y - iy) - 0.5) / (s * 2 ** 0.5)))
        A = np.stack([(ex * ey).ravel(), np.ones(R * R)], 1)
        roi = mov[:, iy - h:iy + h + 1, ix - h:ix + h + 1].reshape(C, -1)
        sol = np.linalg.lstsq(A, roi.T, rcond=None)[0]; I[n_] = sol[0]; B[n_] = sol[1]
tot = I.sum(1); ok = tot > 300; I = I[ok]; B = B[ok]; tot = tot[ok]; P = xy[sel][ok]
q = I / tot[:, None]
res['cycle_emitters'] = {'n': int(ok.sum()), 'photons_window_median': float(np.median(tot)), 'bg_per_px_per_frame_median': float(np.median(B)),
                         'bg_per_frameindex': [float(v) for v in np.median(B, 0)], 'mean_fraction_per_frameindex': [float(v) for v in q.mean(0)],
                         'min_frame_photons_median': float(np.median(I.min(1)))}
print('n %d; window photons (linear PSF fit) median %.0f; bg/px/frame median %.3f; bg per frame index %s' % (ok.sum(), np.median(tot), np.median(B), np.round(np.median(B, 0), 3)))
print('mean intensity fraction per frame index:', np.round(q.mean(0), 4))
K = {'A(ky)': np.array([float(v) for v in a.ky.split(',')]), 'B(kx)': np.array([float(v) for v in a.kx.split(',')])}
def reg(qj, k):
    ph = P @ k; X = np.stack([np.ones(len(P)), np.cos(ph), np.sin(ph)], 1)
    c, *_ = np.linalg.lstsq(X, qj, rcond=None); fit = X @ c
    r2 = 1 - ((qj - fit) ** 2).sum() / ((qj - qj.mean()) ** 2).sum()
    return r2, c
def refine(js, k0):
    def cost(k): return sum(-reg(q[:, j], k)[0] for j in js)
    best = min((minimize(cost, k0 + d0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-9}) for d0 in (np.zeros(2),)), key=lambda r: r.fun)
    return best.x
res['per_frameindex'] = []
r2tab = np.array([[reg(q[:, j], K[n])[0] for n in K] for j in range(C)])
print('R^2 of fraction_j vs [1, cos, sin](k r), columns = k along y (axis A), k along x (axis B), k from the published report:'); print(np.round(r2tab, 3))
axis_of = r2tab.argmax(1); res['r2_table'] = r2tab.tolist(); res['axis_of_frameindex'] = [int(v) for v in axis_of]
names = list(K)
Kref = {}
for ai, n in enumerate(names):
    js = [j for j in range(C) if axis_of[j] == ai]
    Kref[n] = refine(js, K[n]) if js else K[n]
    kk = Kref[n]; print('axis %s: frame indices %s; k refined on the sample (%.4f, %.4f) rad/px, |k| %.4f, pitch %.4f px = %.2f nm at 65 nm/px, angle %.3f deg (report k %s)' % (
        n, js, kk[0], kk[1], np.hypot(*kk), 2 * np.pi / np.hypot(*kk), 2 * np.pi / np.hypot(*kk) * 65, np.degrees(np.arctan2(kk[1], kk[0])), K[n]))
res['k_refined'] = {n: [float(v) for v in Kref[n]] for n in names}; res['pitch_px'] = {n: float(2 * np.pi / np.hypot(*Kref[n])) for n in names}
for j in range(C):
    n = names[axis_of[j]]; r2, c = reg(q[:, j], Kref[n]); r2o, co = reg(q[:, j], K[n])
    dep = np.hypot(c[1], c[2]) / c[0]; phi = np.degrees(np.arctan2(c[2], c[1])); phio = np.degrees(np.arctan2(co[2], co[1]))
    res['per_frameindex'].append({'j': j, 'axis': n, 'r2': float(r2), 'a': float(c[0]), 'depth': float(dep), 'phase_deg': float(phi), 'phase_deg_report_k': float(phio), 'depth_report_k': float(np.hypot(co[1], co[2]) / co[0])})
    print('frame index %d: axis %s  R2 %.3f  a %.4f  depth %.3f  phase %8.2f deg (with report k: depth %.3f phase %8.2f)' % (j, n, r2, c[0], dep, phi, np.hypot(co[1], co[2]) / co[0], phio))
for ai, n in enumerate(names):
    js = [j for j in range(C) if axis_of[j] == ai]
    ph = [res['per_frameindex'][j]['phase_deg_report_k'] for j in js]
    steps = [float((ph[(i + 1) % len(ph)] - ph[i]) % 360) for i in range(len(ph))]
    asum = sum(res['per_frameindex'][j]['a'] for j in js)
    res.setdefault('phase_steps_deg', {})[n] = steps; res.setdefault('axis_power', {})[n] = float(asum)
    print('axis %s: frame indices %s, phase steps (deg, cyclic) %s, summed fraction %.4f' % (n, js, np.round(steps, 2), asum))
json.dump(res, open(a.out, 'w'), indent=1)
