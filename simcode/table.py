"""
Localization table of a pipeline output directory, and the paper's selection.

Table (one row per stored localization = a 6-frame window of an emitter; windows start every 3 frames):
  frame       window start (absolute movie frame; the window covers frame .. frame + 5)
  t           time centroid of the window's network intensities (absolute frame, fractional)
  x, y        final position = explicit pattern solve (px, sensor coordinates of the movie, NOT drift corrected)
  sx, sy      calibrated per-axis sigma of x, y (px; after the model's selection-aware sigma rescale, --sig_u)
  photons     network window photons (sum of the 6 per-frame network intensities, label units)
  I           [6] network per-frame intensities
  p           detector probability (the score);  q_dof  intensity-domain chi2 / dof of the solve;  kin  on/off-kinetics model chosen
  dnll        NLL gap to the best competing fringe (pass 1 only; nan for pass 2)
  x_prior, y_prior   pattern-free network position (px)
  good        p >= 0.63 (the evaluated operating point)        } kept in the tables until the dataset-format pass
  sel         phase-calibrated detection selection (per photon band the same pass fraction as p >= 0.63, taken as the top p
              within each of 6 x 6 sub-pixel (frac x, frac y) cells) -> sub-pixel-uniform selection; the recommended full
              output for count-based uses (renders, drift)
  disjoint    frame % 6 == 0 (non-overlapping windows)
  x_dc, y_dc  only with --drift: drift-corrected x, y (drift subtracted at frame + 3)
The paper's selection (user decision 2026-11-13) is a GLOBAL threshold on the calibrated localization precision:
  select(T, max_precision_nm=12, min_score=0.63, in_region=True)
  = max(sx, sy) * px_nm <= max_precision_nm on the rows with score p >= min_score inside the output region (crop; x_dc, y_dc
  when the table has them), i.e. "calibrated precision better than 12 nm in both axes", the same kind of rule as the
  max(sigma) < 0.15 px precision filter used for SMLM / SIMFLUX. The table has no column for it; the threshold of the run
  (--sigma_max, default SIGMA_MAX_NM) and px_nm are stored in the table file and the info json (n_main = its row count).
  simcode.dataset.Dataset.select is the same rule on a loaded table (crop from the table's info json).
--subset <nm,...> writes the rows of the selection with those thresholds as small files locs_<pass>_sig<X>.npz (sig12; sig12p5
for 12.5) with x, y, frame, sx, sy, photons and sel = True, the format the figure and FRC scripts read.
The selection-aware sigma rescale (--sig_u, the model's) multiplies sx, sy of the ranked rows (p >= 0.63, inside the crop) by
g(u), u = the per-10-px-tile rank quantile of the calibrated sigma (tile_rank; computed for the rescale only, not stored).
(Removed 2026-11-14: the per-tile rule 'u <= 0.45', the strict product (per-tile sigma quantile, column strict) and the
strict_u column; docs/CLEANUP_LOG.md.)
The crop (x0, y0, size px) is the output region of the run: read from the run's pass-1 files (their 'crop' entry) unless
--crop is given; (5, 5, 110) if neither is available.
usage: python -m simcode.strict <out dir> [--tag est|est2] [--drift drift.npy] [--format npz,csv] [--sigma_max nm]
                                [--subset 12,14] [--sig_u <json>] [--crop x0,y0,size]
"""
import argparse, glob, json, os
import numpy as np

PX_NM_DEFAULT = 108.33
P_GOOD = 0.63
CROP = (5.0, 5.0, 110.0)
SIGMA_MAX_NM = 12.0     # the paper's selection: max(sx, sy) * px_nm <= SIGMA_MAX_NM (nm) on the rows with p >= P_GOOD in the crop


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


def run_crop(out):
    """output region (x0, y0, size) of a pipeline run: the 'crop' entry of its pass-1 files; CROP if there is none"""
    fs = sorted(glob.glob(f'{out}/pass1_*.npz'))
    if fs:
        d = np.load(fs[0])
        if 'crop' in d.files: return tuple(float(v) for v in d['crop'])
    return CROP


def tile_rank(x, y, sig, crop=CROP):
    """per-10-px-tile rank quantile of sigma: (rank + 0.5) / n(tile), stable order"""
    o = np.argsort(sig, kind='stable'); t = tile_of(x, y, crop)
    u = np.empty(len(sig))
    for tt in np.unique(t):
        it = o[t[o] == tt]
        u[it] = (np.arange(len(it)) + 0.5) / len(it)
    return u


def in_crop(x, y, crop=CROP):
    """x0 <= x < x0 + size and y0 <= y < y0 + size (crop = (x0, y0, size) px)"""
    x0, y0, sz = crop
    return (x >= x0) & (x < x0 + sz) & (y >= y0) & (y < y0 + sz)


def precision_nm(T, px_nm=None):
    """calibrated localization precision of each row (nm): max(sx, sy) * px_nm (px_nm None: the table's own, else PX_NM_DEFAULT)"""
    if px_nm is None: px_nm = float(T['px_nm']) if 'px_nm' in T else PX_NM_DEFAULT
    return np.maximum(np.asarray(T['sx'], np.float64), np.asarray(T['sy'], np.float64)) * px_nm


def select(T, max_precision_nm=SIGMA_MAX_NM, min_score=P_GOOD, in_region=True, crop=None, px_nm=None):
    """row mask of the paper's selection -> bool [n]. T = a table (dict or loaded npz) with x, y, sx, sy, p.
    max_precision_nm: max(sx, sy) * px_nm <= this (nm; None = no precision condition); min_score: p >= this (None = none);
    in_region: inside the crop (x0, y0, size) px (crop None: T['crop'] if the table has it, else CROP), with the drift-corrected
    x_dc, y_dc when the table has them (as the ranking of the table build); px_nm None: the table's own, else PX_NM_DEFAULT."""
    n = len(T['x']); m = np.ones(n, bool)
    if min_score is not None: m &= np.asarray(T['p']) >= min_score
    if in_region:
        if crop is None: crop = tuple(float(v) for v in T['crop']) if 'crop' in T else CROP
        x, y = (T['x_dc'], T['y_dc']) if 'x_dc' in T and 'y_dc' in T else (T['x'], T['y'])
        m &= in_crop(np.asarray(x), np.asarray(y), crop)
    if max_precision_nm is not None: m &= precision_nm(T, px_nm) <= max_precision_nm
    return m


def subset_name(max_precision_nm=SIGMA_MAX_NM):
    """file-name part of a subset file: 'sig12' for 12 nm, 'sig12p5' for 12.5"""
    return 'sig' + f'{max_precision_nm:g}'.replace('.', 'p')


def write_subset(T, path_base, max_precision_nm=SIGMA_MAX_NM, crop=None, px_nm=None):
    """rows of select(T, max_precision_nm) as a small table <path_base>_sig<X>.npz (sel = True, x, y, frame, sx, sy, photons,
    scalar sigma_max) -> file name"""
    from simcode.fileio.work import atomic_savez
    k = select(T, max_precision_nm, crop=crop, px_nm=px_nm)
    fn = f'{path_base}_{subset_name(max_precision_nm)}.npz'
    atomic_savez(fn, sel=np.ones(int(k.sum()), bool), x=T['x'][k], y=T['y'][k], frame=T['frame'][k], sx=T['sx'][k], sy=T['sy'][k],
                 photons=T['photons'][k], sigma_max=np.array(float(max_precision_nm)))
    return fn


def subset_thresholds(subset, sigma_max=SIGMA_MAX_NM):
    """--subset value -> list of thresholds (nm): None = [sigma_max], '' = [], '12,14' = [12.0, 14.0]"""
    if subset is None: return [float(sigma_max)]
    return [float(v) for v in str(subset).split(',') if v.strip()]


def build_table(out, tag, px_nm=PX_NM_DEFAULT, drift=None, *, sig_u=None, crop=None, sigma_max=SIGMA_MAX_NM):
    """pipeline output dir -> (table dict of numpy columns, info dict). sig_u (None = no rescale): json of the model's
    selection-aware sigma rescale sx, sy x g(u) (analysis/psf3d_nets/sigcal.py; keys u_mid, g), u = per-10-px-tile rank
    quantile of the calibrated sigma among the rows with p >= 0.63 inside the crop (tile_rank; drift-corrected positions with
    --drift); only those rows are rescaled. crop = (x0, y0, size) of the output region (None: the run's own, run_crop(out)).
    sigma_max (nm): threshold of the paper's selection (select(T, sigma_max)) on the final (rescaled) sigma; stored with the
    table, its row count is info['n_main']."""
    crop = run_crop(out) if crop is None else tuple(float(v) for v in crop)
    R = load_rows(out, tag)
    n = len(R['x_hy0'])
    T = dict(frame=R['wstart'].astype(np.int64), t=R['frame'].astype(np.float64), x=R['x_hy0'].astype(np.float64),
             y=R['y_hy0'].astype(np.float64), sx=R['sx_cal'].astype(np.float32), sy=R['sy_cal'].astype(np.float32),
             photons=R['phot'].astype(np.float32), I=R['I_net'].astype(np.float32), p=R['p'].astype(np.float32),
             q_dof=R['q_dof'].astype(np.float32), kin=R['kin_hy'].astype(bool),
             dnll=R['dnll'].astype(np.float32) if 'dnll' in R else np.full(n, np.nan, np.float32),
             x_prior=R['x_prior'].astype(np.float64), y_prior=R['y_prior'].astype(np.float64))
    zinfo = ''
    if 'z_hy0' in R and tag.endswith('2') and 'p1_row' not in R:
        # pass-2 files written before the pass-1 z (2026-11-14) carry the pass-2 head's z, which is not used: no z columns
        zinfo = 'none (pass-2 files without p1_row: the pass-2 head z of an earlier run is not used; re-run pass 2)'
        print(f'build_table {out} [{tag}]: {zinfo}', flush=True)
    elif 'z_hy0' in R:
        # 3D heads (2026-11-14): z (px of the lateral pixel, relative to the PSF calibration's focal plane) after the solve, its
        # sd and covariances with x, y (px^2); NaN for 2D heads. Pass 1 (est): the pass-1 head's z conditioned on the pass-1
        # solved x, y. Pass 2 (est2): the LINKED PASS-1 candidate's z prior conditioned on the pass-2 solved x, y
        # (sc.pass1_z_columns; NaN for rows without a pass-1 partner, p1_row = -1)
        zinfo = 'pass 1: the pass-1 head z | pass-1 x, y' if not tag.endswith('2') else 'pass 1: the linked pass-1 head z | pass-2 x, y'
        T.update(z=R['z_hy0'].astype(np.float64), sz=R['sz_cal'].astype(np.float32), cov_xz=R['cov_xz'].astype(np.float32),
                 cov_yz=R['cov_yz'].astype(np.float32), z_prior=R['z_prior'].astype(np.float64))
    T['good'] = T['p'] >= P_GOOD
    T['sel'] = phase_thr(T['p'], T['x'], T['y'], T['photons'])
    T['disjoint'] = T['frame'] % 6 == 0
    xc, yc = T['x'], T['y']
    if drift is not None:
        d = np.load(drift, allow_pickle=True)
        if isinstance(d, np.lib.npyio.NpzFile):
            d = d['R'].item()['drift_full'] if 'R' in d.files else d['drift']
        fi = np.clip(T['frame'] + 3, 0, len(d) - 1)
        xc = T['x'] - d[fi, 0]; yc = T['y'] - d[fi, 1]
        T['x_dc'] = xc; T['y_dc'] = yc
    ii = np.nonzero(T['good'] & in_crop(xc, yc, crop))[0]
    info = dict(n_good_crop=int(len(ii)), px_nm=px_nm, crop=list(crop))
    if sig_u and len(ii):
        sig = 0.5 * (T['sx'] + T['sy']).astype(np.float64)
        u = tile_rank(xc[ii], yc[ii], sig[ii], crop)
        G_ = json.load(open(sig_u))
        g_ = np.interp(u, G_['u_mid'], G_['g']).astype(np.float32)
        T['sx'][ii] *= g_; T['sy'][ii] *= g_
    if sig_u: info['sig_u'] = sig_u
    k = select(T, sigma_max, crop=crop, px_nm=px_nm)
    info.update(main_rule='max(sx, sy) * px_nm <= sigma_max on the rows with p >= 0.63 inside the crop', sigma_max=float(sigma_max),
                n_main=int(k.sum()), median_photons=float(np.median(T['photons'][k])) if k.any() else float('nan'))
    if zinfo: info['z'] = zinfo
    if 'z' in T and 'p1_row' in R:      # pass-2 rows without a pass-1 partner (z NaN): all rows / the paper's selection
        info.update(n_no_p1=int((R['p1_row'] < 0).sum()), n_main_no_p1=int((R['p1_row'][k] < 0).sum()))
    info.update(tag=tag, n_rows=int(n), n_good=int(T['good'].sum()), n_sel=int(T['sel'].sum()), drift=drift or '',
                columns={k: str(v.dtype) + ('' if v.ndim == 1 else f'[{v.shape[1]}]') for k, v in T.items()})
    return T, info


def write_table(T, info, path_base, formats=('npz',)):
    out = []
    if 'npz' in formats:
        from simcode.fileio.work import atomic_savez
        atomic_savez(path_base + '.npz', **T, px_nm=np.array(info['px_nm']), sigma_max=np.array(info.get('sigma_max', SIGMA_MAX_NM)))
        out.append(path_base + '.npz')
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


def build_parser():
    ap = argparse.ArgumentParser(description='localization table (+ selection subset files) of a simcode output directory')
    ap.add_argument('out')
    ap.add_argument('--tag', default='est', help="'est' = pass 1 (pattern solve of the pass-1 rows), 'est2' = pass 2")
    ap.add_argument('--name', default='')
    ap.add_argument('--px_nm', type=float, default=0.0, help='pixel size (nm); default: from the run (runtime.json calib) or 108.33')
    ap.add_argument('--drift', default='', help='drift [F,2] px (.npy, .npz with key drift, or a DME v2 result npz), subtracted at frame + 3')
    ap.add_argument('--format', default='npz')
    ap.add_argument('--sig_u', default='', help='opt-in selection-aware sigma rescale json (analysis/psf3d_nets/sigcal.py)')
    ap.add_argument('--sigma_max', type=float, default=SIGMA_MAX_NM, help='nm; threshold of the paper\'s selection (max(sx, sy) <= sigma_max on the rows with p >= 0.63 inside the crop; stored in the table)')
    ap.add_argument('--subset', default='', help="also write the selection with these thresholds (nm) as small files, e.g. '12,14' -> <table>_sig12.npz, <table>_sig14.npz")
    ap.add_argument('--crop', default='', help='x0,y0,size (px) of the output region (selection, sigma-rescale tiles); default: the output region of the run')
    ap.add_argument('--dest', default='', help='directory for the table files (default: the run directory)')
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    px = a.px_nm or PX_NM_DEFAULT
    if a.sig_u:
        from simcode import config
        a.sig_u = config.resolve(a.sig_u)                # a path, or a name relative to the models directory
    T, info = build_table(a.out, a.tag, px, a.drift or None, sig_u=a.sig_u or None,
                          crop=[float(v) for v in a.crop.split(',')] if a.crop else None, sigma_max=a.sigma_max)
    if a.dest: os.makedirs(a.dest, exist_ok=True)
    base = os.path.join(a.dest or a.out, a.name or ('locs_p1' if a.tag == 'est' else 'locs_p2' if a.tag == 'est2' else f'locs_{a.tag}'))
    files = write_table(T, info, base, a.format.split(','))
    files += [write_subset(T, base, s, crop=info['crop'], px_nm=px) for s in subset_thresholds(a.subset)]
    from simcode.fileio.work import atomic_json
    atomic_json(base + '_info.json', info, indent=1, default=str)
    print(json.dumps({k: v for k, v in info.items() if k != 'columns'}, default=str))
    print('written', *files)
    # the settings of this rebuild go into the run record of the directory (run_config.json 'steps'); command.txt = a plain log
    try:
        import sys
        from simcode.util import runconfig as RC
        name = os.path.basename(base)
        RC.log_command(a.out, 'simcode.strict', argv)
        RC.append_step(a.out, dict(
            step='strict', time=RC.now(), command='python -m simcode.strict ' + ' '.join(sys.argv[1:] if argv is None else argv),
            settings=dict(out=os.path.abspath(a.out), tag=a.tag, name=name, dest=os.path.abspath(a.dest) if a.dest else '', px_nm=px,
                          drift=os.path.abspath(a.drift) if a.drift else '', sig_u=a.sig_u, sigma_max=a.sigma_max,
                          subset=a.subset, format=a.format, crop=info['crop'], p_good=P_GOOD),
            files={k: RC.file_record(v) for k, v in (('sig_u', a.sig_u), ('drift', a.drift)) if v},
            code={k: v for k, v in RC.code_record(dme=False).items() if k.startswith(('version', 'git_', 'modified'))},
            result={k: v for k, v in info.items() if k != 'columns'}, outputs=[os.path.abspath(f) for f in files]),
            outputs=files, selection={name: RC.selection_summary(info)} if not a.dest else None)
    except OSError as e:
        print(f'strict: run_config.json of {a.out} not updated ({e})')


if __name__ == '__main__':
    main()
