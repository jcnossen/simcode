"""
Localization table and the recommended strict product of a pipeline output directory.

Table (one row per stored localization = a 6-frame window of an emitter; windows start every 3 frames):
  frame       window start (absolute movie frame; the window covers frame .. frame + 5)
  t           time centroid of the window's network intensities (absolute frame, fractional)
  x, y        final position = explicit pattern solve (px, sensor coordinates of the movie, NOT drift corrected)
  sx, sy      calibrated per-axis sigma of x, y (px)
  photons     network window photons (sum of the 6 per-frame network intensities, label units)
  I           [6] network per-frame intensities
  p           detector probability;  q_dof  intensity-domain chi2 / dof of the solve;  kin  on/off-kinetics model chosen
  dnll        NLL gap to the best competing fringe (pass 1 only; nan for pass 2)
  x_prior, y_prior   pattern-free network position (px)
  good        p >= 0.63 (the evaluated operating point)
  sel         phase-calibrated detection selection (per photon band the same pass fraction as p >= 0.63, taken as the top p
              within each of 6 x 6 sub-pixel (frac x, frac y) cells) -> sub-pixel-uniform selection; the recommended full
              output for count-based uses (renders, drift)
  disjoint    frame % 6 == 0 (non-overlapping windows)
  strict_u    per-tile sigma rank quantile in [0, 1] (rows with good = True; nan otherwise), sigma = mean of sx, sy
  strict      the recommended strict product: strict_u <= q
Strict product (critic 37 "per-tile sigma quantile, matched coverage"): within every 10-px tile of the 5-115 px crop the
fraction q of the good rows with the smallest calibrated sigma is kept. Per-tile ranking makes the selection neutral to the
illumination envelope and the depth field (a global sigma threshold selects the bright FOV centre). q:
  default (density)        q from the GT-free row density rho = good rows in the crop per window start per um^2, log-linear
                           between anchors and clamped outside them (Q_DENSITY; critic 40 #1). The anchors are the
                           SIMCODE-coverage-matched q of the three real movies (0.5 / 2.5 / 20 nM; mean of the fixed-c and the
                           automatic-c runs, which agree to <= 0.016): pass 1 rho 0.49 -> q 0.275, rho >= 0.87 -> q 0.12; pass 2
                           rho 0.64 -> 0.25, 1.69 -> 0.12, >= 2.22 -> 0.10. The per-tile q is insensitive to the solve
                           calibration (the sigma ranking within a tile hardly changes), unlike a sigma threshold. Check
                           (analysis/evalproto/strictq.py, covpool protocol, own-DME drift): occupied 20-nm bins / SIMCODE's =
                           0.98-1.02 (pass 1, automatic c and fixed c, all three movies; in-sample: the anchors come from these movies); see docs/RESULTS.md "Reproduce".
  --strict_q <q>           fixed q
  --strict_sigma_ref <nm>  the earlier rule: q such that the product covers as many occupied 20-nm bins as the global set
                           sigma <= s_ref. Valid only for the solve calibration it was derived on: 7.0 nm (pass 1) matches
                           SIMCODE's coverage with the fixed-c calibration (--autoc 0) but NOT with the automatic c (the
                           default; sigma is ~10-17 % larger there and the matched threshold is 7.6 / 8.2 / 8.2 nm, so 7.0 nm
                           gives q 0.17 / 0.05 / 0.06 and 0.82 / 0.71 / 0.68 of SIMCODE's bins at 0.5 / 2.5 / 20 nM).
  Coverage = occupied 20-nm bins of the (optionally drift-corrected, --drift) positions in the crop. The strict product's
  sigma is overconfident in its best bins (winner's curse, 1.3-1.9x in sim): quote NN / GT RMS for it, never its sigma.
usage: python -m simcode2.strict <out dir> [--tag est|est2] [--drift drift.npy] [--strict_q q | --strict_sigma_ref nm] [--format npz,csv]
"""
import argparse, glob, json, os
import numpy as np

PX_NM_DEFAULT = 108.33
P_GOOD = 0.63
CROP = (5.0, 5.0, 110.0)


def load_rows(out, tag):
    fs = sorted(glob.glob(f'{out}/pass2_{tag}_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
    if not fs: raise FileNotFoundError(f'no pass2_{tag}_*.npz in {out}')
    Q = {}
    for f in fs:
        d = np.load(f)
        for k in d.files:
            if d[k].ndim: Q.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in Q.items()}


def phase_thr(p, x, y, ph, thr=P_GOOD, bands=(0, 700, 1200, 2000, 1e9), nc=6):
    """phase-calibrated detection threshold (GT-free): per network-photon band the global pass fraction q = mean(p >= thr);
    per cell of a nc x nc grid of (frac x, frac y) keep the top q of p -> bool keep (equal N per band, uniform pass rate per
    sub-pixel phase)"""
    keep = np.zeros(len(p), bool)
    cx = np.minimum((np.mod(x, 1) * nc).astype(int), nc - 1); cy = np.minimum((np.mod(y, 1) * nc).astype(int), nc - 1)
    for lo, hi in zip(bands[:-1], bands[1:]):
        b = (ph >= lo) & (ph < hi)
        if not b.any(): continue
        q = np.mean(p[b] >= thr)
        for i in range(nc):
            for j in range(nc):
                m = b & (cx == i) & (cy == j)
                if m.sum() == 0: continue
                t = np.quantile(p[m], 1 - q) if q < 1 else -1
                keep |= m & (p >= t)
    return keep


def tile_of(x, y, crop=CROP, size=10.0):
    x0, y0, sz = crop; n = int(round(sz / size))
    ix = np.clip(((x - x0) / size).astype(int), 0, n); iy = np.clip(((y - y0) / size).astype(int), 0, n)
    return iy * (n + 1) + ix


def bin20(x, y, px_nm, crop=CROP):
    x0, y0, sz = crop; nb = int(sz * px_nm / 20)
    ix = np.clip(((x - x0) / sz * nb).astype(int), 0, nb - 1); iy = np.clip(((y - y0) / sz * nb).astype(int), 0, nb - 1)
    return iy * nb + ix


def n_for_cov(keys_sorted, target):
    """smallest N such that the first N rows (in this order) cover >= target distinct keys"""
    u, first = np.unique(keys_sorted, return_index=True)
    fi = np.sort(first)
    return int(fi[min(target, len(fi)) - 1] + 1) if len(fi) else 0


def tile_rank(x, y, sig):
    """per-10-px-tile rank quantile of sigma: (rank + 0.5) / n(tile), stable order"""
    o = np.argsort(sig, kind='stable'); t = tile_of(x, y)
    u = np.empty(len(sig))
    for tt in np.unique(t):
        it = o[t[o] == tt]
        u[it] = (np.arange(len(it)) + 0.5) / len(it)
    return u


# density -> q anchors (rho = good rows in the crop per window start per um^2; log-linear, clamped), see the module docstring
Q_DENSITY = {'est': ((0.49, 0.275), (0.87, 0.12)),
             'est2': ((0.64, 0.25), (1.69, 0.12), (2.22, 0.10))}


def q_from_density(rho, tag='est'):
    a = np.array(Q_DENSITY.get(tag, Q_DENSITY['est']), float)
    return float(np.interp(np.log(max(rho, 1e-6)), np.log(a[:, 0]), a[:, 1]))


def strict_product(x, y, sig, good, px_nm, q=None, sigma_ref_nm=None, xc=None, yc=None, frame=None, tag='est'):
    """-> (u [n] (nan where not good), strict [n] bool, info). x, y raw px; xc, yc drift-corrected px (coverage / tiles;
    default = raw); sig = mean calibrated sigma (px); frame = window starts (for the density rule).
    q given -> fixed; else sigma_ref_nm given -> coverage of the global sigma <= sigma_ref set; else the density rule."""
    xc = x if xc is None else xc; yc = y if yc is None else yc
    x0, y0, sz = CROP
    inc = good & (xc >= x0) & (xc < x0 + sz) & (yc >= y0) & (yc < y0 + sz)
    ii = np.nonzero(inc)[0]
    u = np.full(len(x), np.nan)
    strict = np.zeros(len(x), bool)
    info = dict(n_good_crop=int(len(ii)), px_nm=px_nm)
    if not len(ii): return u, strict, info
    ug = tile_rank(xc[ii], yc[ii], sig[ii]); u[ii] = ug
    keys = bin20(xc[ii], yc[ii], px_nm)
    if q is not None:
        info.update(mode='fixed')
    elif sigma_ref_nm is not None:
        ref = sig[ii] * px_nm <= sigma_ref_nm
        target = len(np.unique(keys[ref]))
        ou = np.argsort(ug, kind='stable')
        Nq = n_for_cov(keys[ou], target)
        q = float(ug[ou[Nq - 1]]) if Nq else 0.0
        info.update(mode='sigma_ref', sigma_ref_nm=sigma_ref_nm, coverage_target_bins=int(target))
    else:
        if frame is None: raise ValueError('the density rule needs the window starts (frame)')
        n_starts = len(np.unique(frame))
        area_um2 = (sz * px_nm * 1e-3) ** 2
        rho = len(ii) / max(n_starts, 1) / area_um2
        q = q_from_density(rho, tag)
        info.update(mode='density', density_per_um2_per_start=float(rho), n_window_starts=int(n_starts))
    sel = ug <= q
    strict[ii[sel]] = True
    info.update(q=float(q), n_strict=int(sel.sum()), coverage_bins=int(len(np.unique(keys[sel]))),
                coverage_bins_good=int(len(np.unique(keys))), median_sigma_nm=float(np.median(sig[ii[sel]]) * px_nm) if sel.any() else float('nan'),
                median_photons=None)
    return u, strict, info


SIGMA_REF = {'est': 7.0, 'est2': 7.4}   # the earlier sigma_ref rule's values (fixed-c calibration); opt-in only


def build_table(out, tag, px_nm=PX_NM_DEFAULT, drift=None, strict_q=None, sigma_ref_nm=None):
    """pipeline output dir -> (table dict of numpy columns, info dict)"""
    R = load_rows(out, tag)
    n = len(R['x_hy0'])
    T = dict(frame=R['wstart'].astype(np.int64), t=R['frame'].astype(np.float64), x=R['x_hy0'].astype(np.float64),
             y=R['y_hy0'].astype(np.float64), sx=R['sx_cal'].astype(np.float32), sy=R['sy_cal'].astype(np.float32),
             photons=R['phot'].astype(np.float32), I=R['I_net'].astype(np.float32), p=R['p'].astype(np.float32),
             q_dof=R['q_dof'].astype(np.float32), kin=R['kin_hy'].astype(bool),
             dnll=R['dnll'].astype(np.float32) if 'dnll' in R else np.full(n, np.nan, np.float32),
             x_prior=R['x_prior'].astype(np.float64), y_prior=R['y_prior'].astype(np.float64))
    T['good'] = T['p'] >= P_GOOD
    T['sel'] = phase_thr(T['p'], T['x'], T['y'], T['photons'])
    T['disjoint'] = T['frame'] % 6 == 0
    xc = yc = None
    if drift is not None:
        d = np.load(drift, allow_pickle=True)
        if isinstance(d, np.lib.npyio.NpzFile):
            d = d['R'].item()['drift_full'] if 'R' in d.files else d['drift']
        fi = np.clip(T['frame'] + 3, 0, len(d) - 1)
        xc = T['x'] - d[fi, 0]; yc = T['y'] - d[fi, 1]
        T['x_dc'] = xc; T['y_dc'] = yc
    sig = 0.5 * (T['sx'] + T['sy']).astype(np.float64)
    u, strict, info = strict_product(T['x'], T['y'], sig, T['good'], px_nm, q=strict_q, sigma_ref_nm=sigma_ref_nm, xc=xc, yc=yc,
                                     frame=T['frame'], tag='est2' if tag.startswith('est2') else 'est')
    T['strict_u'] = u.astype(np.float32); T['strict'] = strict
    info['median_photons'] = float(np.median(T['photons'][strict])) if strict.any() else float('nan')
    info.update(tag=tag, n_rows=int(n), n_good=int(T['good'].sum()), n_sel=int(T['sel'].sum()), drift=drift or '',
                columns={k: str(v.dtype) + ('' if v.ndim == 1 else f'[{v.shape[1]}]') for k, v in T.items()})
    return T, info


def write_table(T, info, path_base, formats=('npz',)):
    out = []
    if 'npz' in formats:
        np.savez(path_base + '.npz', **T, px_nm=np.array(info['px_nm'])); out.append(path_base + '.npz')
    if 'csv' in formats:
        cols = [k for k in T if T[k].ndim == 1] + [f'I{t}' for t in range(T['I'].shape[1])]
        M = np.column_stack([T[k].astype(np.float64) for k in T if T[k].ndim == 1] + [T['I'][:, t] for t in range(T['I'].shape[1])])
        np.savetxt(path_base + '.csv', M, delimiter=',', header=','.join(cols), comments='', fmt='%.6g'); out.append(path_base + '.csv')
    if 'hdf5' in formats:
        import h5py
        with h5py.File(path_base + '.hdf5', 'w') as f:
            for k, v in T.items(): f.create_dataset(k, data=v, compression='gzip')
            f.attrs['info'] = json.dumps(info, default=str)
        out.append(path_base + '.hdf5')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='localization table + strict product of a simcode2 output directory')
    ap.add_argument('out')
    ap.add_argument('--tag', default='est', help="'est' = pass 1 (pattern solve of the pass-1 rows), 'est2' = pass 2")
    ap.add_argument('--name', default='')
    ap.add_argument('--px_nm', type=float, default=0.0, help='pixel size (nm); default: from the run (runtime.json calib) or 108.33')
    ap.add_argument('--drift', default='', help='drift [F,2] px (.npy, .npz with key drift, or a DME v2 result npz), subtracted at frame + 3')
    ap.add_argument('--strict_q', type=float, default=None, help='fixed per-tile quantile (default: from the row density)')
    ap.add_argument('--strict_sigma_ref', type=float, default=None,
                    help='nm; opt-in earlier rule (7.0 pass 1 / 7.4 pass 2, valid with the fixed-c calibration --autoc 0 only)')
    ap.add_argument('--format', default='npz')
    a = ap.parse_args(argv)
    px = a.px_nm or PX_NM_DEFAULT
    T, info = build_table(a.out, a.tag, px, a.drift or None, a.strict_q, a.strict_sigma_ref)
    base = os.path.join(a.out, a.name or ('locs_p1' if a.tag == 'est' else 'locs_p2' if a.tag == 'est2' else f'locs_{a.tag}'))
    files = write_table(T, info, base, a.format.split(','))
    json.dump(info, open(base + '_info.json', 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in info.items() if k != 'columns'}, default=str))
    print('written', *files)


if __name__ == '__main__':
    main()
