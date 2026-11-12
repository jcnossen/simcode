"""Metrics of the paper simulations (S1 / S2 density movies, S3 moving movie) for the 4 methods.

Rows per method ('full' = what each method outputs, as the preprint scored it):
  smlm    <run>/base/smlm/smlm.npy (chi^2-filtered, a window at every frame)
  sf      <run>/base/sf/sf.hdf5 (pattern estimated from SMLM)      sf_gtp  <run>/base/sf_gtp/sf.hdf5 (ground-truth pattern)
  decode  <run>/decode/locs.npz (all rows; 6-frame sums at window starts 0, 3 mod 6)
  v2      <run>/v2_<netset>/locs_p2_sim.npz rows `strict` (the user's choice: SIMCODE v2 is scored on its STRICT subset);
          v2sel = 'SIMCODE v2 all' = rows `sel`; v2gt = strict rows of the diagnostic pass 2 on
          the ground-truth pattern (run_simcode.py --gt_diag; not the method); v2q45 = 'SIMCODE v2, u <= 0.45' = the main v2 row
          (user 2026-11-12): rows with a rank (p >= 0.63, inside the field) and strict_u <= 0.45, not intersected with sel;
          'v2sel' ('SIMCODE v2 all') is no longer reported (loader kept)
  v1      <run>/v1/locs.npz: the preprint's SIMCODE, merged 'sfhd-ndi+smlm' rows (run_simcode_v1.py), its MVU sigma
Metrics:
  counts   n_all, n_disjoint (window start % 6 == 0, inside the FOV), n_frc
  FRC      the fair protocol of analysis/paper/frc_fair.py (frc_halves = simcode2.evalproto.lib.frc_std): disjoint windows,
           max(sigma_x, sigma_y) < 0.15 px with each method's own sigma, alternating 300-frame halves, 5-nm renders, 1/7.
           Simulation-specific: region = the whole simulated FOV [0, W) (not the sensor crop 5-115 px) and NO drift correction
           (the simulations have no drift, so the DME step of frc_fair is skipped).
  Jaccard  vs number of frames, ported verbatim from the preprint: smlmtorch 5a4fda9 scripts/jaccard.py compute_jaccard_points
           / find_nearby and scripts/fig_simulation_2D_results.py jaccard_timeline (GT = all GT tubule points ds_gt.pos,
           est = the method's positions with frame < f for f = (i + 1) nframes / timebins, distance <= 10 nm, 120 bins
           (Fig 2: jaccard_timeline_plot(numframes=60000, timebins=120, threshold_dist_nm=10)); computed on the 'full' rows
           (as the preprint: no crlb filter) and, secondary, on the FRC rows.
  GT match (not in the preprint; sanity / precision): each FRC-set localization is matched to the nearest emitter that is ON
           in its window (within 100 nm) -> recall of GT emitters with >= 150 window photons, mean offset (convention check),
           RMS error per axis, median |err| / sigma.
  FRC references (coordinator request 2026-11-12, see frc_reference): the method's own FRC-set rows moved to the true position
           of their ON emitter + Gaussian noise at its robust sigma (frc_ideal_nm) / plain RMS (frc_ideal_rms_nm) / its own error
           vectors shuffled between rows (frc_ideal_shuffled_nm); frc_floor = all GT events at 1 / 5 / 10 nm (sigma 0 never crosses 1/7).
  frc_nm_sustained: the first frequency after which the curve stays below 1/7 for 0.01 / nm (hollow tubes have a spectral
           null at 2.405 / (2 pi R) that the first crossing can stop at; 65 nm for R = 25 nm).
  Moving (S3): per 1000-frame block Jaccard at 10 / 20 / 40 nm, GT = the trajectory points of the block (as
           smlmtorch fig_moving_tubules.py compute_jaccards; frames strictly inside (i bs, (i + 1) bs) as there).
usage: python analysis/papersim/metrics.py <run dir> [--movie <movie dir>] [--nframes N] [--netset m1809] [--kind density|moving]
outputs: <run dir>/metrics.json, metrics.png"""
import os, sys, json, argparse, glob
import numpy as np
from scipy.spatial import KDTree
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis', 'paper'))
import common as C

SIGMA_MAX = 0.15
BLK = 300


# --------------------------------------------------------------- verbatim port: smlmtorch 5a4fda9 scripts/jaccard.py
def find_nearby(A, B, max_distance):
    tree_A = KDTree(A)
    tree_B = KDTree(B)
    dist, ix = tree_A.query(B)
    A_in_B = dist <= max_distance
    dist, ix = tree_B.query(A)
    B_in_A = dist <= max_distance
    return A_in_B, B_in_A


def compute_jaccard_points(A, B, max_distance):
    A_in_B, B_in_A = find_nearby(A, B, max_distance)
    total = len(A) + len(B)
    return ((len(A) / total) * A_in_B.sum() / len(A) +
            (len(B) / total) * B_in_A.sum() / len(B))


# --------------------------------------------------------------- 5a4fda9 scripts/fig_simulation_2D_results.py jaccard_timeline
def jaccard_timeline(pos, frame, gt, nframes, timebins=120, threshold_dist=10 / C.PX_NM):
    """= jaccard_timeline(ds_est, ds_gt, timebins, threshold_dist) with nframes = the movie length (the preprint used
    ds_est.numFrames after ds[ds.frame < numframes], i.e. the same up to the last localized frame)"""
    jac = np.full(timebins, np.nan)
    for i in range(timebins):
        f = int((i + 1) * nframes / timebins)
        p = pos[frame < f, :2]
        if len(p): jac[i] = compute_jaccard_points(gt[:, :2], p, threshold_dist)
    return (np.arange(timebins) + 1) * nframes / timebins, jac


# --------------------------------------------------------------- loading
def load_method(kind, path):
    if kind in ('v1', 'v1gt'):          # preprint SIMCODE: the merged 'sfhd-ndi+smlm' rows (run_simcode_v1.py)
        z = np.load(path); i = z['ndi_idx']
        return dict(x=z['mrg_x'].astype(np.float64), y=z['mrg_y'].astype(np.float64), f=z['frame'][i].astype(np.int64),
                    sx=z['mrg_sx'].astype(np.float64), sy=z['mrg_sy'].astype(np.float64), photons=z['ndi_photons'].astype(np.float64))
    if kind in ('v2', 'v2sel', 'v2gt', 'v2q45'):
        z = np.load(path)
        if kind == 'v2sel': m = z['sel'].astype(bool)
        elif kind == 'v2q45':     # user decision 2026-11-12 01:20: ranked rows (p >= 0.63, inside the crop) with strict_u <= 0.45; no `sel`
            m = np.nan_to_num(z['strict_u'].astype(np.float64), nan=2.0) <= 0.45
        else: m = z['strict'].astype(bool)
        return dict(x=z['x'][m].astype(np.float64), y=z['y'][m].astype(np.float64), f=z['frame'][m].astype(np.int64),
                    sx=z['sx'][m].astype(np.float64), sy=z['sy'][m].astype(np.float64), photons=z['photons'][m].astype(np.float64))
    import frc_fair
    return frc_fair.load({'sf_gtp': 'sf', 'decode_sim': 'decode', 'decode_pre': 'decode'}.get(kind, kind), path)


def u_all(x, y, sig, base, tile=10.0):
    """rank of the 'v2, u <= q' selections (2026-11-12): within every 10 x 10 px tile (tiles from (0, 0)), the quantile
    (rank + 0.5) / n of the mean calibrated sigma among ALL base rows (`sel`) of the tile, so u = 1 is exactly the base set.
    = strict.py's strict_u extended from the p >= 0.63 rows to every `sel` row (nan outside the base)"""
    u = np.full(len(x), np.nan)
    i = np.nonzero(base)[0]
    t = (np.floor(x[i] / tile).astype(np.int64) * 100003 + np.floor(y[i] / tile).astype(np.int64))
    o = np.lexsort((sig[i], t)); ts = t[o]
    start = np.r_[0, np.nonzero(np.diff(ts))[0] + 1]; cnt = np.diff(np.r_[start, len(ts)])
    rank = np.arange(len(ts)) - np.repeat(start, cnt)
    u[i[o]] = (rank + 0.5) / np.repeat(cnt, cnt)
    return u


def method_paths(run, netset='m1809'):
    P = dict(smlm=f'{run}/base/smlm/smlm.npy', sf=f'{run}/base/sf/sf.hdf5', sf_gtp=f'{run}/base/sf_gtp/sf.hdf5',
             decode=f'{run}/decode/locs.npz', v2=f'{run}/v2_{netset}/locs_p2_sim.npz')
    P['v2q45'] = P['v2']                                   # 'SIMCODE v2, u <= 0.45' (main v2 row; 'v2 all' removed 2026-11-12)
    P['v1'] = f'{run}/v1/locs.npz'
    P['decode_sim'] = f'{run}/decode_sim/locs.npz'
    P['decode_pre'] = f'{run}/decode_pre/locs.npz'      # the preprint's own DECODE model (decode-3f-g1.3), Gaussian-PSF reproduction      # DECODE retrained for the simulation ranges (dec_sim1809)
    P['v1gt'] = f'{run}/v1gt/locs.npz'           # diagnostic: preprint SIMCODE given the GT pattern
    P['v2gt'] = f'{run}/v2_{netset}/locs_p2gt_sim.npz'        # diagnostic: pass 2 on the ground-truth pattern
    return {k: v for k, v in P.items() if os.path.exists(v)}


def frc_set(T, W, nframes):
    ok = np.isfinite(T['x']) & np.isfinite(T['y']) & np.isfinite(T['sx']) & np.isfinite(T['sy']) & (T['f'] + 6 <= nframes)
    inside = (T['x'] >= 0) & (T['x'] < W) & (T['y'] >= 0) & (T['y'] < W)
    disj = ok & (T['f'] % 6 == 0) & inside
    filt = disj & (np.maximum(T['sx'], T['sy']) < SIGMA_MAX)
    return ok & inside, disj, filt


def gt_windows(gt, nframes, starts):
    """per disjoint window start: GT emitters ON in [f, f + 6) with their window photons -> dict f -> (x, y, ph)"""
    from common import illum
    P = dict(k=gt['k'], phi=gt['phi'], m=gt['m'], rel=gt['rel'])
    t0, t1 = gt['ev_t0'], gt['ev_t1']; ex, ey = gt['ev_x'], gt['ev_y']
    I6 = gt['ev_I6'] if 'ev_I6' in gt else np.full(len(t0), C.I6)     # per-event brightness (hollow sims) or the constant 500
    o = np.argsort(t0); t0, t1, ex, ey, I6 = t0[o], t1[o], ex[o], ey[o], I6[o]
    maxdur = float((t1 - t0).max())
    out = {}
    for f in starts:
        lo = np.searchsorted(t0, f - maxdur); hi = np.searchsorted(t0, f + 6)
        i = np.arange(lo, hi); i = i[(t1[i] > f) & (t0[i] < f + 6)]
        ph = np.zeros(len(i))
        for t in range(6):
            onf = np.clip(np.minimum(t1[i], f + t + 1) - np.maximum(t0[i], f + t), 0, 1)
            ph += I6[i] / 6 * illum(P, ex[i], ey[i], np.full(len(i), f + t)) * onf
        out[f] = (ex[i], ey[i], ph)
    return out


def frc_sustained(freq, curve, width=0.01):
    """resolution (nm) at the first frequency after which the FRC stays below 1/7 for `width` (1/nm). = the usual first crossing
    for a smooth curve; differs when the structure's spectrum has a null (hollow tubes: J0 zero at f = 2.405 / (2 pi R), 65 nm
    for R = 25 nm) where the curve dips below 1/7 and recovers"""
    freq = np.asarray(freq); c = np.asarray(curve); w = max(int(width / max(freq[1] - freq[0], 1e-12)), 1)
    for i in range(3, len(c) - w):
        if (c[i:i + w] < 1 / 7).all():
            return float(1 / freq[i])
    return float('nan')


def row_truth(T, m, GW, rmax=150 / C.PX_NM, ph_min=30.0):
    """true position of the nearest ON emitter (>= ph_min window photons) of each row's window, within rmax; nan otherwise"""
    tx = np.full(len(T['x']), np.nan); ty = np.full(len(T['x']), np.nan)
    idx = np.nonzero(m)[0]; fs = T['f'][idx]
    for f in np.unique(fs):
        if f not in GW: continue
        gx, gy, gph = GW[f]; k = gph >= ph_min; gx, gy = gx[k], gy[k]
        if not len(gx): continue
        ii = idx[fs == f]
        d = np.hypot(T['x'][ii, None] - gx[None], T['y'][ii, None] - gy[None]); j = d.argmin(1); ok = d[np.arange(len(ii)), j] < rmax
        tx[ii[ok]] = gx[j[ok]]; ty[ii[ok]] = gy[j[ok]]
    return tx, ty


def frc_reference(T, m, GW, rms_nm, seed=0):
    """FRC references built on the method's own FRC-set rows moved to the true position of their ON emitter (same rows, windows,
    count, repeats and halves; rows without an ON emitter within 150 nm are dropped, their number is returned):
      frc_ideal_nm          + Gaussian noise at the method's ROBUST sigma (1.4826 MAD of its GT errors, mean of x, y): the
                            count- and core-precision limit
      frc_ideal_rms_nm      + Gaussian noise at its plain GT RMS (outliers spread as if Gaussian)
      frc_ideal_shuffled_nm + its own GT error vectors, shuffled between rows (same error distribution incl. the tails, no
                            correlation with position / event): actual FRC worse than this = structured or correlated errors
      frc_ideal_sigma0_nm   the true positions alone (sampling limit)"""
    import frc_fair
    tx, ty = row_truth(T, m, GW)
    k = m & np.isfinite(tx)
    rng = np.random.default_rng(seed); f = T['f'][k]
    ex = (T['x'][k] - tx[k]); ey = (T['y'][k] - ty[k])
    mad = lambda e: 1.4826 * np.median(np.abs(e - np.median(e)))
    s_rob = 0.5 * (mad(ex) + mad(ey)); s_rms = float(np.mean(rms_nm)) / C.PX_NM
    g = lambda s: frc_fair.frc_halves(tx[k] + rng.normal(0, s, k.sum()), ty[k] + rng.normal(0, s, k.sum()), f)[0]
    p = rng.permutation(k.sum())
    return dict(frc_ideal_nm=float(g(s_rob)), frc_ideal_rms_nm=float(g(s_rms)),
                frc_ideal_shuffled_nm=float(frc_fair.frc_halves(tx[k] + ex[p], ty[k] + ey[p], f)[0]),
                frc_ideal_sigma0_nm=float(frc_fair.frc_halves(tx[k], ty[k], f)[0]), n_ideal=int(k.sum()),
                n_unmatched=int((m & ~np.isfinite(tx)).sum()), sigma_robust_nm=float(s_rob * C.PX_NM), sigma_rms_nm=float(np.mean(rms_nm)))


def frc_floor(gt, nframes, sigmas=(1, 5, 10), seed=0):
    """FRC of all GT events (one point per event at its true position, half by the event's start frame) + Gaussian noise sigma"""
    import frc_fair
    f = np.floor(gt['ev_t0']).astype(np.int64); k = (f >= 0) & (f < nframes)
    rng = np.random.default_rng(seed); out = {}
    for s in sigmas:
        x = gt['ev_x'][k] + rng.normal(0, s / C.PX_NM, k.sum()); y = gt['ev_y'][k] + rng.normal(0, s / C.PX_NM, k.sum())
        out[str(s)] = float(frc_fair.frc_halves(x, y, f[k])[0])
    out['n_events'] = int(k.sum())
    return out


def gt_match(T, m, GW, W, rmax=100 / C.PX_NM, ph_min=150.0, edge=2.0):
    """nearest ON emitter of the same window (unique per GT emitter: nearest localization kept)"""
    idx = np.nonzero(m)[0]
    errs, sig, n_gt, n_hit = [], [], 0, 0
    fs = T['f'][idx]
    for f in np.unique(fs):
        if f not in GW: continue
        gx, gy, gph = GW[f]
        tg = (gph >= ph_min) & (gx >= edge) & (gx < W - edge) & (gy >= edge) & (gy < W - edge)
        n_gt += int(tg.sum())
        ii = idx[fs == f]
        if not len(ii) or not len(gx): continue
        d = np.hypot(T['x'][ii, None] - gx[None], T['y'][ii, None] - gy[None])
        j = d.argmin(1); dm = d[np.arange(len(ii)), j]
        ok = dm < rmax
        # one localization per GT emitter (closest)
        best = {}
        for a, (jj, dd) in enumerate(zip(j, dm)):
            if ok[a] and (jj not in best or dd < best[jj][1]): best[jj] = (a, dd)
        for jj, (a, dd) in best.items():
            errs.append((T['x'][ii[a]] - gx[jj], T['y'][ii[a]] - gy[jj])); sig.append((T['sx'][ii[a]], T['sy'][ii[a]]))
            if tg[jj]: n_hit += 1
    if not errs: return dict(n_matched=0)
    e = np.array(errs) * C.PX_NM; s = np.array(sig) * C.PX_NM
    return dict(n_matched=int(len(e)), recall_gt_ph150=n_hit / max(n_gt, 1), n_gt_ph150=n_gt, mean_offset_nm=e.mean(0).round(2).tolist(),
                rms_nm=np.sqrt((e ** 2).mean(0)).round(2).tolist(),
                rms_trim2_nm=np.sqrt((e[np.hypot(*e.T) <= np.quantile(np.hypot(*e.T), 0.98)] ** 2).mean(0)).round(2).tolist(),
                median_sigma_nm=np.median(s, 0).round(2).tolist(), median_abs_err_over_sigma=np.median(np.abs(e) / s, 0).round(3).tolist(),
                frac_matched_of_set=float(len(e) / max(m.sum(), 1)), frac_err_gt30=float(np.mean(np.hypot(*e.T) > 30)))


def density_metrics(run, md, nframes, netset, timebins=120):
    import frc_fair
    g = json.load(open(os.path.join(md, 'gen.json'))); W = g['W']
    gt = dict(np.load(os.path.join(md, 'gt.npz')))
    pts = gt['pts']
    starts = np.arange(0, nframes - 5, 6)
    GW = gt_windows(gt, nframes, starts)
    R = dict(run=run, movie=md, nframes=nframes, netset=netset, W=W, frc_floor=frc_floor(gt, nframes), protocol=dict(
        frc='disjoint windows (start % 6 == 0), max(sx, sy) < 0.15 px, alternating 300-frame halves, whole FOV, no drift',
        jaccard='5a4fda9 compute_jaccard_points, 10 nm, %d bins, all GT points' % timebins), methods={})
    for name, path in method_paths(run, netset).items():
        T = load_method(name, path)
        full, disj, filt = frc_set(T, W, nframes)
        r = dict(path=path, n_all=int(len(T['x'])), n_full_in_fov=int(full.sum()), n_disjoint=int(disj.sum()), n_frc=int(filt.sum()))
        if filt.sum() > 50:
            res, freq, curve = frc_fair.frc_halves(T['x'][filt], T['y'][filt], T['f'][filt])
            r['frc_nm'] = res; r['frc_curve'] = curve.tolist(); r['frc_freq'] = freq.tolist()
            r['frc_nm_nofilter'] = frc_fair.frc_halves(T['x'][disj], T['y'][disj], T['f'][disj])[0]
            r['frc_nm_sustained'] = frc_sustained(freq, curve)
        pos = np.c_[T['x'], T['y']]
        fb, jac = jaccard_timeline(pos[full], T['f'][full], pts, nframes, timebins)
        r['jaccard_frames'] = fb.tolist(); r['jaccard'] = jac.tolist(); r['jaccard_end'] = float(jac[-1]); r['jaccard_peak'] = float(np.nanmax(jac))
        _, jf = jaccard_timeline(pos[filt], T['f'][filt], pts, nframes, timebins)
        r['jaccard_frcset'] = jf.tolist(); r['jaccard_frcset_end'] = float(jf[-1])
        r['gt_match'] = gt_match(T, filt, GW, W)
        if r['gt_match'].get('rms_nm') and filt.sum() > 50:
            r.update(frc_reference(T, filt, GW, r['gt_match']['rms_nm']))
        R['methods'][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}), flush=True)
    return R


def moving_metrics(run, md, nframes, netset, bs=1000, thr_nm=(10, 20, 40)):
    g = json.load(open(os.path.join(md, 'gen.json'))); W = g['W']
    traj = np.load(os.path.join(md, 'traj.npy'), mmap_mode='r'); tstep = int(g['traj_step'])
    nb = nframes // bs
    R = dict(run=run, movie=md, nframes=nframes, blocks=nb, block=bs, traj_step=tstep, thresholds_nm=list(thr_nm), methods={})
    for name, path in method_paths(run, netset).items():
        T = load_method(name, path)
        J = np.full((len(thr_nm), nb), np.nan)
        for i in range(nb):
            sel = (T['f'] > i * bs) & (T['f'] < (i + 1) * bs)
            if sel.sum() == 0: continue
            pts = np.asarray(traj[i * bs // tstep:(i + 1) * bs // tstep]).reshape(-1, 2)
            for k, t in enumerate(thr_nm):
                J[k, i] = compute_jaccard_points(np.c_[T['x'][sel], T['y'][sel]], pts, t / C.PX_NM)
        # per-block FRC (2026-11-12): within each block the halves alternate every 60 frames (10 disjoint windows); disjoint
        # windows only, max sigma < 0.15 px; a moving structure only stays put within a block
        import frc_fair
        fb = np.full(nb, np.nan)
        for i in range(nb):
            sel = (T['f'] >= i * bs) & (T['f'] < (i + 1) * bs) & (T['f'] % 6 == 0) & (np.maximum(T['sx'], T['sy']) < 0.15)
            if sel.sum() > 200: fb[i] = frc_fair.frc_halves(T['x'][sel], T['y'][sel], T['f'][sel], blk=60)[0]
        R['methods'][name] = dict(path=path, n_all=int(len(T['x'])), jaccard=J.tolist(), jaccard_mean=np.nanmean(J, 1).tolist(),
                                  frc_block_nm=fb.tolist(), frc_block_median_nm=float(np.nanmedian(fb)) if np.isfinite(fb).any() else None,
                                  n_per_block_mean=float(len(T['x']) / max(nb, 1)))
        print(name, R['methods'][name]['jaccard_mean'], R['methods'][name]['frc_block_median_nm'], flush=True)
    return R


def plot(R, fn):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = dict(smlm='tab:blue', sf='tab:green', sf_gtp='olive', decode='tab:orange', v2='tab:red', v2sel='salmon', v2gt='purple', v2q45='darkred', v1='tab:brown', v1gt='peru', decode_sim='darkorange', decode_pre='orangered')
    lab = dict(smlm='SMLM', sf='SIMFLUX', sf_gtp='SIMFLUX (GT pattern)', decode='DECODE', v2='SIMCODE v2 (strict)', v2sel='SIMCODE v2 all', v2gt='v2 strict, GT pattern (diag.)', v2q45='SIMCODE v2, u <= 0.45', v1='SIMCODE v1 (preprint)', v1gt='SIMCODE v1, GT pattern (diag.)', decode_sim='DECODE (retrained dec_sim1809)', decode_pre='DECODE (preprint model decode-3f-g1.3)')
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for k, r in R['methods'].items():
        if 'frc_curve' in r:
            ax[0].plot(r['frc_freq'], r['frc_curve'], color=col[k], label=f"{lab[k]} {r['frc_nm']:.1f} nm")
        ax[1].plot(r['jaccard_frames'], r['jaccard'], color=col[k], lw=2, label=lab[k])
    ax[0].axhline(1 / 7, color='grey', ls='--'); ax[0].set_xlabel('spatial frequency (1/nm)'); ax[0].set_ylabel('FRC'); ax[0].legend(fontsize=8)
    ax[0].set_xlim(0, 0.12)
    ax[1].set_ylim(0, 1); ax[1].set_xlabel('frames'); ax[1].set_ylabel('Jaccard (10 nm)'); ax[1].legend(fontsize=8)
    ks = list(R['methods'])
    ax[2].bar(range(len(ks)), [R['methods'][k]['n_frc'] for k in ks], color=[col[k] for k in ks])
    ax[2].set_xticks(range(len(ks))); ax[2].set_xticklabels([lab[k] for k in ks], rotation=25, fontsize=8); ax[2].set_ylabel('localizations in the FRC set')
    fig.suptitle(f"{os.path.relpath(R['run'], C.OUT)}: {R['nframes']} frames", fontsize=10)
    plt.tight_layout(); plt.savefig(fn, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--movie', default=None)
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--netset', default='m1809')
    ap.add_argument('--kind', default=None)
    ap.add_argument('--timebins', type=int, default=120)
    a = ap.parse_args()
    run = os.path.abspath(a.run); md = os.path.abspath(a.movie or a.run)
    g = json.load(open(os.path.join(md, 'gen.json')))
    nf = a.nframes or g['nframes']
    kind = a.kind or g['kind']
    if kind == 'moving':
        R = moving_metrics(run, md, nf, a.netset)
        json.dump(R, open(os.path.join(run, f'metrics_{a.netset}.json'), 'w'), indent=1)
        return
    R = density_metrics(run, md, nf, a.netset, a.timebins)
    json.dump(R, open(os.path.join(run, f'metrics_{a.netset}.json'), 'w'), indent=1)
    plot(R, os.path.join(run, f'metrics_{a.netset}.png'))
    print('written', os.path.join(run, f'metrics_{a.netset}.json'))


if __name__ == '__main__':
    main()
