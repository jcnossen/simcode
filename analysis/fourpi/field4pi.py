"""Field dependence of the 4Pi PSF, interference and pipeline quality, per channel, in tiles over the illuminated field
(x 60-300, y 0-200 px of each channel crop in ch0 coordinates; 5 x 4 tiles of 48 x 50 px).

A. Per-channel PSF shape vs the global model (items 1, 3): bright (> 2000 e), isolated (nn > 6 px), on-through-window
   network localizations of the 3 subset files per movie, matched (window, pixel) to the classical pass-1 rows that
   hold the unwarped per-channel 11 x 11 images. Per emitter and channel the frame-summed image d_c is compared with the
   global coherent model g_c at the emitter's own (x, y, z, 4Pi phase) (photons of angle 0 / 1 from the network T),
   amplitude + background fitted per channel (shape-only comparison). Per tile and channel: data second moments
   (r <= 4 px), data - model moment differences (dsx, dsy), ellipticity in 3 z bins, peak fraction (3 x 3 / r <= 4)
   data vs model, chi2/px, and the mean residual image (rms over pixels, as a fraction of the mean model peak).
   Registration residual = intensity centroid (r <= 3 px) of the data minus that of the fitted model (the model already
   contains the PSF-shape centroid shifts), i.e. what the polynomial map + model leave over.
B. Interference per tile (item 2), from the network channel totals T of all files: fractions f_c, pair differences
   u = f0 - f2, v = f1 - f3. For a 4Pi phase spread over the circle, u = a cos(Phi), v = b sin(Phi + eps):
   eps = asin(2 cov(u, v) / (a b)) = deviation of the inter-pair phase offset from quadrature; visibilities
   V_p = a / (f0 + f2), V_s = b / (f1 + f3); transmissions = mean f_c; within-pair complementarity = slope of f2 on f0.
   Same with the centre read-out C (higher visibility). Middle z tercile only (the visibility envelope depends on z).
C. Pipeline quality per tile (item 4), all files: kinetics-model fraction, chi2 of the free model per dof, DAFL x / y / z
   (adjacent windows), 4Pi fringe indicator |z - z_prior| > Lambda / 2, median sigma_Phi, SIM pattern coherence R per
   angle from the per-file pattern (network intensities at the solved positions, on-through rows).
Output: out/4pi/field/field_<cell>.json + tile-map figures.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/field4pi.py --cell NPC
"""
import argparse, glob, json, math, os, sys
import numpy as np, torch
from scipy.spatial import cKDTree
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
from register import PolyMap
from pattern import emitter_phase
from simcode.fourpi.lut4pi import LUT4Pi

OUT = f'{P4}/field'
NET = f'{P4}/net'
XT = np.linspace(60, 300, 6); YT = np.linspace(0, 200, 5)
NTX, NTY = 5, 4
from simcode.fourpi.setup import DELTA    # per-angle 4Pi phase offset per cell (SMLM_REPORT 5.2)
SUBSET = {'MT': (5, 6, 7), 'NPC': (10, 11, 12)}


def tile_of(x, y):
    tx = np.clip(np.searchsorted(XT, x) - 1, 0, NTX - 1); ty = np.clip(np.searchsorted(YT, y) - 1, 0, NTY - 1)
    inside = (x >= XT[0]) & (x < XT[-1]) & (y >= YT[0]) & (y < YT[-1])
    return np.where(inside, ty * NTX + tx, -1)


def part_a(cell, dev, model='v2'):
    st = json.load(open(f'{P4}/psf4pi/NPC_f_dxy15.json'))['psf']
    lut = LUT4Pi(st, P=11, dev=dev)
    pm = PolyMap.load(f'{P4}/smlm/reg/reg_insitu_{cell}.json')
    rows = []
    for fi in SUBSET[cell]:
        fl = f'{NET}/locs_{cell}_{fi:03d}_{model}.npz'; ff = f'{NET}/front_{cell}_{fi:03d}_{model}.npz'
        fp = f'{P4}/smlm/p1/p1_{cell}_{fi:03d}.npz'
        if not (os.path.exists(fl) and os.path.exists(fp)): continue
        L = np.load(fl); Fr = np.load(ff); P1 = np.load(fp, mmap_mode='r', allow_pickle=True)
        sel = L['on'] & (L['ton'] <= 0) & (L['toff'] >= 6) & (L['N'] > 2000) & (L['nn'] > 6) & (L['sz'] < 0.02)
        key = lambda w, x, y: (w.astype(np.int64) * 400 + y.astype(np.int64)) * 400 + x.astype(np.int64)
        kp = key(np.asarray(P1['w']), np.asarray(P1['cx']), np.asarray(P1['cy']))
        kf = key(Fr['w'], Fr['cx'], Fr['cy'])
        order = np.argsort(kp); pos = np.searchsorted(kp[order], kf); pos = np.clip(pos, 0, len(kp) - 1)
        hit = kp[order][pos] == kf
        idx = np.nonzero(sel & hit)[0]
        j1 = order[pos[idx]]
        srt = np.argsort(j1); idx, j1 = idx[srt], j1[srt]
        roi = np.asarray(P1['roi_ch'][j1]).astype(np.float32).sum(1)          # [n,4,11,11] frame sums
        corner = np.asarray(P1['corner'][j1])                                  # [n,4,2] (ix, iy)
        Tn = Fr['T'][idx]                                                      # [n,6,4]
        Na = np.stack([Tn[:, :3].sum((1, 2)), Tn[:, 3:].sum((1, 2))], 1)
        rows.append(dict(x=L['x'][idx], y=L['y'][idx], z=L['z'][idx], Phi0=L['Phi0'][idx], roi=roi, corner=corner, Na=Na))
        print(f'{cell} {fi}: {len(idx)} emitters for the PSF analysis', flush=True)
    R = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    n = len(R['x'])
    yy, xx = np.mgrid[0:11, 0:11].astype(np.float32)
    res = {k: np.zeros((n, 4)) for k in ('sx_d', 'sy_d', 'sx_m', 'sy_m', 'pk_d', 'pk_m', 'chi2', 'cx_res', 'cy_res', 'peak_m')}
    resid_img = np.zeros((NTY * NTX, 4, 11, 11)); peak_acc = np.zeros((NTY * NTX, 4)); cnt_t = np.zeros(NTY * NTX)
    tile = tile_of(R['x'], R['y'])
    for c in range(4):
        mx, my = pm.inv(c, R['x'], R['y'])
        px = mx - (R['corner'][:, c, 0] + 5); py = my - (R['corner'][:, c, 1] + 5)
        for i in range(0, n, 8192):
            s = slice(i, i + 8192)
            pos = torch.tensor(np.c_[px[s], py[s], R['z'][s]], device=dev, dtype=torch.float32)
            Ph = torch.tensor(R['Phi0'][s], device=dev, dtype=torch.float32)
            g = (lut.evaluate_ch(pos, Ph, c) * torch.tensor(R['Na'][s, 0], device=dev)[:, None, None]
                 + lut.evaluate_ch(pos, Ph + DELTA[cell], c) * torch.tensor(R['Na'][s, 1], device=dev)[:, None, None]).cpu().numpy()
            d = R['roi'][s, c]
            # amplitude + bg LSQ (shape-only)
            G = g.reshape(len(g), -1); D = d.reshape(len(d), -1)
            a11 = (G * G).sum(1); a12 = G.sum(1); a22 = G.shape[1]; r1 = (G * D).sum(1); r2 = D.sum(1)
            det = a11 * a22 - a12 ** 2
            A = (a22 * r1 - a12 * r2) / det; b = (a11 * r2 - a12 * r1) / det
            m = A[:, None, None] * g + b[:, None, None]
            var = np.maximum(m, 1) + 6 * 1.3
            res['chi2'][s, c] = ((d - m) ** 2 / var).mean((1, 2))
            for nm, img in (('d', d - b[:, None, None]), ('m', A[:, None, None] * g)):
                cxm = px[s] + 5; cym = py[s] + 5
                r2_ = (xx[None] - cxm[:, None, None]) ** 2 + (yy[None] - cym[:, None, None]) ** 2
                w4 = (r2_ <= 16).astype(np.float32); w3 = (r2_ <= 9).astype(np.float32)
                tot = (img * w4).sum((1, 2)).clip(1)
                mx_ = (img * w4 * xx).sum((1, 2)) / tot; my_ = (img * w4 * yy).sum((1, 2)) / tot
                res[f'sx_{nm}'][s, c] = np.sqrt(np.clip((img * w4 * (xx - mx_[:, None, None]) ** 2).sum((1, 2)) / tot, 0, None))
                res[f'sy_{nm}'][s, c] = np.sqrt(np.clip((img * w4 * (yy - my_[:, None, None]) ** 2).sum((1, 2)) / tot, 0, None))
                ic = np.clip(np.round(cym).astype(int), 1, 9); jc = np.clip(np.round(cxm).astype(int), 1, 9)
                pk = np.array([img[k, ic[k] - 1:ic[k] + 2, jc[k] - 1:jc[k] + 2].sum() for k in range(len(img))])
                res[f'pk_{nm}'][s, c] = pk / tot
                t3 = (img * w3).sum((1, 2)).clip(1)
                cen = ((img * w3 * xx).sum((1, 2)) / t3, (img * w3 * yy).sum((1, 2)) / t3)
                if nm == 'd': cd = cen
                else: cm = cen
            res['cx_res'][s, c] = cd[0] - cm[0]; res['cy_res'][s, c] = cd[1] - cm[1]
            res['peak_m'][s, c] = (A[:, None, None] * g).reshape(len(g), -1).max(1)
            tt = tile[s]
            for k in np.nonzero(tt >= 0)[0]:
                resid_img[tt[k], c] += (d[k] - m[k]); peak_acc[tt[k], c] += res['peak_m'][s, c][k]
                if c == 0: cnt_t[tt[k]] += 1
    zq = np.quantile(R['z'], [0, 1 / 3, 2 / 3, 1])
    zb = np.clip(np.searchsorted(zq, R['z']) - 1, 0, 2)
    out = dict(n=int(n), z_terciles_um=zq.tolist(), tiles=[])
    for t in range(NTY * NTX):
        s = tile == t
        e = dict(tile=t, tx=t % NTX, ty=t // NTX, n=int(s.sum()))
        if s.sum() >= 50:
            med = lambda v: np.nanmedian(v[s], 0).tolist()
            e.update(sx_d=med(res['sx_d']), sy_d=med(res['sy_d']), dsx=med(res['sx_d'] - res['sx_m']), dsy=med(res['sy_d'] - res['sy_m']),
                     pk_d=med(res['pk_d']), pk_m=med(res['pk_m']), chi2=med(res['chi2']),
                     reg_x=med(res['cx_res']), reg_y=med(res['cy_res']),
                     reg_rel_x=med(res['cx_res'] - res['cx_res'][:, :1]), reg_rel_y=med(res['cy_res'] - res['cy_res'][:, :1]),
                     resid_frac=(np.sqrt(((resid_img[t] / max(cnt_t[t], 1)) ** 2).reshape(4, -1).mean(1)) / (peak_acc[t] / max(cnt_t[t], 1))).tolist())
            ell = []
            for b_ in range(3):
                sb = s & (zb == b_)
                ell.append(np.nanmedian(((res['sx_d'] - res['sy_d']) / (res['sx_d'] + res['sy_d']))[sb], 0).tolist() if sb.sum() >= 20 else None)
            e['ellip_d_zbins'] = ell
            ellm = []
            for b_ in range(3):
                sb = s & (zb == b_)
                ellm.append(np.nanmedian(((res['sx_m'] - res['sy_m']) / (res['sx_m'] + res['sy_m']))[sb], 0).tolist() if sb.sum() >= 20 else None)
            e['ellip_m_zbins'] = ellm
            e['n_zbins'] = [int((s & (zb == b_)).sum()) for b_ in range(3)]
        out['tiles'].append(e)
    return out


def fit_interf(f, nit=12):
    """joint fit of f_c = tau_c (1 + V_pair(c) cos(Phi_e + phi_c)) over emitters e (fractions f [n,4]):
    tile parameters tau_0..3 (sum 1), V_p (ch0, ch2), V_s (ch1, ch3), phi_1..3 (phi_0 = 0); Phi_e free per emitter.
    Alternating: Phi_e in closed form (linear in cos / sin), then the tile parameters by least squares.
    Does not need Phi uniformly spread (needs some spread). Returns dict."""
    from scipy.optimize import least_squares
    tau = f.mean(0); V = np.array([0.3, 0.3]); ph = np.radians([0.0, 270.0, 180.0, 90.0])
    pair = np.array([0, 1, 0, 1])
    for it in range(nit):
        # Phi_e: f_c - tau_c = tau_c V_c (cos Phi cos ph_c - sin Phi sin ph_c) -> LSQ in (cos Phi, sin Phi)
        A = np.stack([tau * V[pair] * np.cos(ph), -tau * V[pair] * np.sin(ph)], 1)        # [4,2]
        sol = np.linalg.lstsq(A, (f - tau).T, rcond=None)[0].T                            # [n,2]
        Phi = np.arctan2(sol[:, 1], sol[:, 0])
        def r(p):
            t = np.r_[p[0:3], 1 - p[0:3].sum()]; Vv = p[3:5]; pp = np.r_[0.0, p[5:8]]
            return (f - t * (1 + Vv[pair] * np.cos(Phi[:, None] + pp))).ravel()
        p0 = np.r_[tau[:3], V, ph[1:]]
        p = least_squares(r, p0).x
        tau = np.r_[p[0:3], 1 - p[0:3].sum()]; V = p[3:5]; ph = np.r_[0.0, p[5:8]]
    R = float(np.abs(np.mean(np.exp(1j * Phi))))
    d = np.degrees(ph)
    return dict(tau=tau.tolist(), V_p=float(V[0]), V_s=float(V[1]), phi_deg=(d % 360).tolist(),
                eps_deg=float(((d[1] - d[0] - 270 + 180) % 360) - 180), pair02_deg=float(((d[2] - d[0] + 180) % 360) - 180 + 0),
                pair13_deg=float(((d[3] - d[1] + 180) % 360) - 180), arm_ratio_proxy=float(V[0] / V[1]), R_Phi=R,
                rms_resid=float(np.sqrt(np.mean(r(p) ** 2))))


def part_b(cell, model='v2'):
    out = dict(tiles=[])
    acc = {k: [] for k in ('T', 'C', 'x', 'y', 'z', 'phi')}
    for ff in sorted(glob.glob(f'{NET}/front_{cell}_[0-9][0-9][0-9]_{model}.npz')):
        fl = ff.replace('front_', 'locs_')
        if not os.path.exists(fl): continue
        F = np.load(ff); L = np.load(fl)
        s = (L['Tsum'] > 2000) & (L['nn'] > 6) & (L['sz'] < 0.025)
        acc['T'].append(F['T'][s].sum(1)); acc['C'].append(F['C'][s].sum(1))
        acc['x'].append(L['x'][s]); acc['y'].append(L['y'][s]); acc['z'].append(L['z'][s]); acc['phi'].append(L['phi'][s])
    R = {k: np.concatenate(v) for k, v in acc.items()}
    zq = np.quantile(R['z'], [1 / 3, 2 / 3])
    mid = (R['z'] >= zq[0]) & (R['z'] < zq[1])
    tile = tile_of(R['x'], R['y'])
    out['n'] = int(len(R['x'])); out['z_mid_um'] = zq.tolist()
    for t in range(NTY * NTX):
        e = dict(tile=t, tx=t % NTX, ty=t // NTX)
        s = (tile == t) & mid
        e['n'] = int(s.sum())
        if s.sum() >= 200:
            # the pair-covariance estimator needs the 4Pi phase spread over the circle: valid for R_Phi <~ 0.25
            # (synthetic check: bias -1 deg at R 0, -4 at 0.27, -17 at 0.6, -30 at 0.84)
            e['R_Phi'] = float(np.abs(np.mean(np.exp(1j * R['phi'][s])))); e['valid'] = e['R_Phi'] <= 0.25
            for nm in ('T', 'C'):
                f = R[nm][s] / R[nm][s].sum(1, keepdims=True)
                u = f[:, 0] - f[:, 2]; v = f[:, 1] - f[:, 3]
                a = np.sqrt(2 * u.var()); b = np.sqrt(2 * v.var())
                eps = math.degrees(math.asin(np.clip(2 * np.cov(u, v)[0, 1] / (a * b), -1, 1)))
                sl = np.polyfit(f[:, 0], f[:, 2], 1)[0]; sl2 = np.polyfit(f[:, 1], f[:, 3], 1)[0]
                e[nm] = dict(tau=f.mean(0).tolist(), V_p=float(a / (f[:, 0] + f[:, 2]).mean()), V_s=float(b / (f[:, 1] + f[:, 3]).mean()),
                             eps_deg=eps, slope_02=float(sl), slope_13=float(sl2), arm_ratio_proxy=float(a / b))
        out['tiles'].append(e)
    return out


def dafl_xyz(w, x, y, z, r=0.35):
    order = np.argsort(w, kind='stable'); ws = w[order]
    st = np.searchsorted(ws, np.arange(ws.max() + 2))
    I, J = [], []
    for wv in range(ws.max()):
        a = order[st[wv]:st[wv + 1]]; b = order[st[wv + 1]:st[wv + 2]]
        if len(a) == 0 or len(b) == 0: continue
        dd, k = cKDTree(np.c_[x[b], y[b]]).query(np.c_[x[a], y[a]], k=2, distance_upper_bound=2 * r)
        ok = (dd[:, 0] < r) & ~(dd[:, 1] < 2 * r)
        I.append(a[ok]); J.append(b[k[ok, 0]])
    return np.concatenate(I), np.concatenate(J)


def part_c(cell, model='v2'):
    summ = json.load(open(f'{NET}/locs_{cell}_{model}_summary.json'))
    acc = {k: [] for k in ('x', 'y', 'on', 'c2f', 'c2o', 'sphi', 'N', 'tile', 'pair_dx', 'pair_dy', 'pair_dz', 'pair_t', 'dphi0', 'dphi1', 'tile_e', 'fringe')}
    for fl in sorted(glob.glob(f'{NET}/locs_{cell}_[0-9][0-9][0-9]_{model}.npz')):
        fi = int(fl.split('_')[-2])
        if str(fi) not in summ or 'pattern' not in summ[str(fi)]: continue
        L = np.load(fl); F = np.load(fl.replace('locs_', 'front_'))
        good = (L['sz'] < 0.025) & (L['sx'] < 0.12) & (L['sy'] < 0.12) & (L['N'] > 300)
        t = tile_of(L['x'], L['y'])
        for k, v in (('x', L['x']), ('y', L['y']), ('on', L['on']), ('c2f', L['chi2_free']), ('c2o', L['chi2_on']),
                     ('sphi', L['sphi']), ('N', L['N']), ('tile', t)):
            acc[k].append(v[good])
        acc['fringe'].append((np.abs(L['z'] - L['zprior']) > 0.123)[good])
        # DAFL pairs (good rows)
        g = np.nonzero(good)[0]
        i, j = dafl_xyz(L['w'][g], L['x'][g], L['y'][g], L['z'][g])
        i, j = g[i], g[j]
        pm_ = (0.5 * (L['N'][i] + L['N'][j]) >= 2000) & (0.5 * (L['N'][i] + L['N'][j]) < 4000)   # photon-matched bin
        i, j = i[pm_], j[pm_]
        acc['pair_dx'].append((L['x'][i] - L['x'][j]) * 128); acc['pair_dy'].append((L['y'][i] - L['y'][j]) * 128)
        acc['pair_dz'].append((L['z'][i] - L['z'][j]) * 1000); acc['pair_t'].append(t[i])
        # SIM pattern phase residual per emitter (on-through, bright) with the file's pattern
        pat = summ[str(fi)]['pattern']
        s = L['on'] & (L['ton'] <= 0) & (L['toff'] >= 6) & (L['N'] > 1500) & (L['nn'] > 6)
        I = F['T'][s].sum(2)
        for a in range(2):
            k = np.array(pat[f'angle{a}']['k']); psi = np.radians(pat[f'angle{a}']['psi_deg'])
            ph, mm, _ = emitter_phase(I[:, 3 * a:3 * a + 3], psi)
            Xp = np.c_[F['cx'][s] + F['dxy'][s, 0], F['cy'][s] + F['dxy'][s, 1]]            # pattern-free network prior
            acc[f'dphi{a}'].append(np.angle(np.exp(1j * (ph - Xp @ k))))
        acc['tile_e'].append(t[s])
    R = {k: np.concatenate(v) for k, v in acc.items()}
    rsd = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) / math.sqrt(2) if len(v) > 50 else None
    out = dict(tiles=[])
    for t in range(NTY * NTX):
        s = R['tile'] == t; sp = R['pair_t'] == t; se = R['tile_e'] == t
        e = dict(tile=t, tx=t % NTX, ty=t // NTX, n=int(s.sum()), n_pairs=int(sp.sum()), n_pattern=int(se.sum()))
        if s.sum() >= 200:
            e.update(kin_frac=float(R['on'][s].mean()), chi2_free_per_dof=float(np.median(R['c2f'][s]) / (48 - 9)),
                     chi2_kin_per_dof=float(np.median(R['c2o'][s & R['on']]) / (48 - 6)) if (s & R['on']).sum() > 50 else None,
                     fringe_frac=float(R['fringe'][s].mean()), sphi_med_deg=float(np.degrees(np.median(R['sphi'][s]))),
                     dafl_x=rsd(R['pair_dx'][sp]), dafl_y=rsd(R['pair_dy'][sp]), dafl_z=rsd(R['pair_dz'][sp]),
                     R0=float(np.abs(np.mean(np.exp(1j * R['dphi0'][se])))) if se.sum() > 50 else None,
                     R1=float(np.abs(np.mean(np.exp(1j * R['dphi1'][se])))) if se.sum() > 50 else None,
                     mean_dphi0_deg=float(np.degrees(np.angle(np.mean(np.exp(1j * R['dphi0'][se]))))) if se.sum() > 50 else None,
                     mean_dphi1_deg=float(np.degrees(np.angle(np.mean(np.exp(1j * R['dphi1'][se]))))) if se.sum() > 50 else None)
        out['tiles'].append(e)
    return out


def tilemap(ax, vals, title, fmt='{:.2f}', cmap='coolwarm', center=None):
    import matplotlib.pyplot as plt
    A = np.full((NTY, NTX), np.nan)
    for t, v in enumerate(vals):
        if v is not None: A[t // NTX, t % NTX] = v
    vmax = np.nanmax(np.abs(A - (center or 0))) if center is not None else None
    im = ax.imshow(A, cmap=cmap, vmin=(center - vmax) if center is not None else None, vmax=(center + vmax) if center is not None else None)
    for t in range(NTY * NTX):
        if not np.isnan(A[t // NTX, t % NTX]): ax.text(t % NTX, t // NTX, fmt.format(A[t // NTX, t % NTX]), ha='center', va='center', fontsize=6)
    ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def figures(cell, A, B, C):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    g = lambda T, k, c=None: [(t[k][c] if c is not None else t[k]) if k in t and t[k] is not None else None for t in T['tiles']]
    fig, axs = plt.subplots(6, 4, figsize=(13, 13))
    for c in range(4):
        tilemap(axs[0, c], [v * 128 if v is not None else None for v in g(A, 'dsx', c)], f'ch{c} data-model sx (nm)', '{:.0f}', center=0)
        tilemap(axs[1, c], [v * 128 if v is not None else None for v in g(A, 'dsy', c)], f'ch{c} data-model sy (nm)', '{:.0f}', center=0)
        tilemap(axs[2, c], [t['ellip_d_zbins'][1][c] if 'ellip_d_zbins' in t and t['ellip_d_zbins'][1] is not None else None for t in A['tiles']], f'ch{c} ellipticity, mid z (data)', '{:.3f}', center=None, cmap='viridis')
        tilemap(axs[3, c], [v * 100 if v is not None else None for v in g(A, 'resid_frac', c)], f'ch{c} mean residual rms (% of peak)', '{:.1f}', cmap='viridis')
        kx, ky, lab = ('reg_x', 'reg_y', 'centroid data-model') if c == 0 else ('reg_rel_x', 'reg_rel_y', 'registration (rel. ch0)')
        tilemap(axs[4, c], [v * 128 if v is not None else None for v in g(A, kx, c)], f'ch{c} {lab} x (nm)', '{:.0f}', center=0)
        tilemap(axs[5, c], [v * 128 if v is not None else None for v in g(A, ky, c)], f'ch{c} {lab} y (nm)', '{:.0f}', center=0)
    fig.suptitle(f'{cell}: per-channel PSF vs the global model, tiles of 48 x 50 px (x 60-300, y 0-200)', fontsize=10)
    plt.tight_layout(); plt.savefig(f'{OUT}/field_psf_{cell}.png', dpi=120); plt.close()
    fig, axs = plt.subplots(3, 4, figsize=(13, 7))
    TT = [t.get('T') if t.get('valid') else None for t in B['tiles']]
    tilemap(axs[0, 0], [t['eps_deg'] if t else None for t in TT], 'inter-pair phase - 90 deg (T), valid tiles (deg)', '{:.1f}', center=0)
    tilemap(axs[0, 1], [t.get('R_Phi') for t in B['tiles']], 'R of the 4Pi phase (valid if <= 0.25)', '{:.2f}', cmap='viridis')
    tilemap(axs[0, 2], [t['V_p'] if t else None for t in TT], 'visibility p pair (T)', '{:.2f}', cmap='viridis')
    tilemap(axs[0, 3], [t['V_s'] if t else None for t in TT], 'visibility s pair (T)', '{:.2f}', cmap='viridis')
    for c in range(4):
        tilemap(axs[1, c], [t['tau'][c] if t else None for t in TT], f'tau ch{c} (T)', '{:.3f}', cmap='viridis')
    tilemap(axs[2, 0], [t['arm_ratio_proxy'] if t else None for t in TT], 'V_p / V_s amplitude ratio', '{:.2f}', cmap='viridis')
    tilemap(axs[2, 1], [t.get('R0') for t in C['tiles']], 'SIM coherence R angle 0 (prior pos.)', '{:.2f}', cmap='viridis')
    tilemap(axs[2, 2], [t.get('R1') for t in C['tiles']], 'SIM coherence R angle 1 (prior pos.)', '{:.2f}', cmap='viridis')
    tilemap(axs[2, 3], [t.get('kin_frac') for t in C['tiles']], 'kinetics-model fraction', '{:.2f}', cmap='viridis')
    fig.suptitle(f'{cell}: interference and pattern per tile', fontsize=10)
    plt.tight_layout(); plt.savefig(f'{OUT}/field_interf_{cell}.png', dpi=120); plt.close()
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.5))
    for ax, k, tt, f in ((axs[0, 0], 'dafl_x', 'DAFL x, 2-4k e (nm)', '{:.1f}'), (axs[0, 1], 'dafl_y', 'DAFL y, 2-4k e (nm)', '{:.1f}'), (axs[0, 2], 'dafl_z', 'DAFL z, 2-4k e (nm)', '{:.1f}'),
                         (axs[1, 0], 'chi2_free_per_dof', 'chi2/dof free model', '{:.2f}'), (axs[1, 1], 'fringe_frac', '|z - z_prior| > L/2 fraction', '{:.3f}'),
                         (axs[1, 2], 'sphi_med_deg', 'median sigma_Phi (deg)', '{:.0f}')):
        tilemap(ax, [t.get(k) for t in C['tiles']], tt, f, cmap='viridis')
    fig.suptitle(f'{cell}: pipeline quality per tile', fontsize=10)
    plt.tight_layout(); plt.savefig(f'{OUT}/field_pipe_{cell}.png', dpi=120); plt.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cell', default='NPC'); p.add_argument('--parts', default='abc')
    a = p.parse_args()
    from simcode.util.proctitle import auto; auto('4pi-field')
    os.makedirs(OUT, exist_ok=True)
    fn = f'{OUT}/field_{a.cell}.json'
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    if 'a' in a.parts: res['A'] = part_a(a.cell, 'cuda'); json.dump(res, open(fn, 'w'), indent=1)
    if 'b' in a.parts: res['B'] = part_b(a.cell); json.dump(res, open(fn, 'w'), indent=1)
    if 'c' in a.parts: res['C'] = part_c(a.cell); json.dump(res, open(fn, 'w'), indent=1)
    if all(k in res for k in 'ABC'): figures(a.cell, res['A'], res['B'], res['C'])
    print('done', fn)


if __name__ == '__main__':
    main()
