"""
Localization tables of a pipeline output directory, and the paper's selection.

Every run writes FOUR tables (2026-11-17), one row per stored localization = a 6-frame window of an emitter (windows start every
3 frames):
  locs_p1_net     pass 1, the emitter network's own output (pattern-free position prior)          tag est,  kind net
  locs_p1_solve   pass 1, the explicit pattern solve of those rows                                 tag est,  kind solve
  locs_p2_net     pass 2, the pattern-aware emitter network's own output  = THE RESULT TABLE        tag est2, kind net
  locs_p2_solve   pass 2, the explicit pattern solve of those rows                                 tag est2, kind solve
The two tables of a pass have the same rows in the same order (row i of locs_p2_net is row i of locs_p2_solve); pass-2 rows
carry p1_index = the row of the linked pass-1 candidate in the pass-1 tables (-1: no partner among the stored pass-1 rows).
The result table (DEFAULT_TABLE, what Dataset.from_run returns by default, the paper's SIMCODE result) is locs_p2_net: on
held-out simulations the pass-2 network is within 1-6 % of the solve's precision and its sigma is calibrated, the solve's is
overconfident (analysis/sigma_joint, paper branch).
Columns:
  frame       window start (absolute movie frame; the window covers frame .. frame + 5)
  t           time centroid of the window's network intensities (absolute frame, fractional)
  x, y        position (px, sensor coordinates of the movie, NOT drift corrected): net = the emitter network's position
              (the calibration's constant prior_off removed); solve = the posterior mean of the pattern solve
  sx, sy      per-axis sigma of x, y (px): net = the network's sd (position block of its joint (x, y, I_1..6) covariance);
              solve = the solve's posterior sd (mixture over the fringe modes and the two kinetic models). No factor on either.
  z, sz, cov_xz, cov_yz, z_prior   (3D networks) z (px of the lateral pixel, relative to the PSF calibration's focal plane), its
              sd and covariances with x, y (px^2): the network's 3D prior (pass 2: of the linked PASS-1 candidate) conditioned on
              this table's x, y and sx, sy (simcode/solve/z_conditioning.py); NaN for pass-2 rows without a pass-1 partner
  photons     network window photons (sum of the 6 per-frame network intensities, label units);  I  [6] network intensities
  p           detector probability (the score)
  x_prior, y_prior   the raw network position (px; before the prior_off correction)
  solve only: q_dof  intensity-domain chi2 / dof of the solve at its MAP mode; kin  posterior weight of the kinetic (on/off
              within the window) model, 0-1 (simcode/solve/solve.model_average); dnll  NLL gap to the best competing fringe
  good        p >= 0.63 (the evaluated operating point)
  sel         phase-calibrated detection selection (per photon band the same pass fraction as p >= 0.63, taken as the top p
              within each of 6 x 6 sub-pixel (frac x, frac y) cells) -> sub-pixel-uniform selection
  disjoint    frame % 6 == 0 (non-overlapping windows)
  p1_index    pass 2 only, see above
  x_dc, y_dc  only with --drift: drift-corrected x, y (drift subtracted at frame + 3)
The paper's selection is a GLOBAL threshold on the reported localization precision, the same for every table:
  select(T, max_precision_nm=12, min_score=0.63, in_region=True)
  = max(sx, sy) * px_nm <= max_precision_nm on the rows with score p >= min_score inside the output region (crop; x_dc, y_dc
  when the table has them). It is applied when a table is loaded (simcode.dataset.Dataset.select, the crop and pixel size from
  the table's metadata); no selection is stored (the info json counts its rows at SIGMA_MAX_NM: n_main).
Files (write_table): <name>.hdf5 = the torch_psf Dataset format (docs/DATASET_FORMAT.md; metadata pixel_nm, crop, tag, kind,
info) and <name>_info.json (counts, crop, columns). CSV / Picasso: exports of the loaded Dataset (save('x.csv'), save_picasso).
(Removed 2026-11-17: the selection-aware sigma rescale sx, sy x g(u) (u = per-10-px-tile sigma rank) and the pass-2
network-prior variant files pass2_est2b_*; 2026-11-14: the per-tile rule 'u <= 0.45', the strict product; docs/CLEANUP_LOG.md.)
The crop (x0, y0, size px) is the output region of the run: read from the run's pass-1 files (their 'crop' entry) unless
--crop is given; (5, 5, 110) if neither is available.
usage: python -m simcode.table <out dir> [--table locs_p2_net|...|all] [--drift drift.npy] [--crop x0,y0,size]
"""
import argparse, glob, json, os
import numpy as np

PX_NM_DEFAULT = 108.33
P_GOOD = 0.63
CROP = (5.0, 5.0, 110.0)
SIGMA_MAX_NM = 12.0     # the paper's selection: max(sx, sy) * px_nm <= SIGMA_MAX_NM (nm) on the rows with p >= P_GOOD in the crop
# the four tables of a run: name -> (tag of the chunk files, kind)
TABLES = dict(locs_p1_net=('est', 'net'), locs_p1_solve=('est', 'solve'), locs_p2_net=('est2', 'net'), locs_p2_solve=('est2', 'solve'))
DEFAULT_TABLE = 'locs_p2_net'           # the result table of a run (and the paper's SIMCODE result)
TABLE_DOC = dict(locs_p1_net='pass 1, emitter network output (position prior, its sd)', locs_p1_solve='pass 1, pattern solve',
                 locs_p2_net='pass 2, emitter network output (position prior, its sd) = the result table',
                 locs_p2_solve='pass 2, pattern solve')


def table_name(pass_, kind):
    """(1 | 2, 'net' | 'solve') -> the table's base name"""
    if kind not in ('net', 'solve'): raise ValueError(f'kind {kind!r}: net or solve')
    return f'locs_p{int(pass_)}_{kind}'


def chunk_files(out, tag):
    """[(f0, path)] of the chunk files pass2_<tag>_<f0>.npz of a run, by f0"""
    fs = sorted(glob.glob(f'{out}/pass2_{tag}_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
    if not fs: raise FileNotFoundError(f'no pass2_{tag}_*.npz in {out}')
    return [(int(f.rsplit('_', 1)[1][:-4]), f) for f in fs]


def load_rows(out, tag):
    """the rows of the chunk files of a tag, concatenated in chunk order; 'chunk' = the chunk start f0 of each row"""
    Q = {}
    for f0, f in chunk_files(out, tag):
        d = np.load(f); n = None
        for k in d.files:
            if d[k].ndim:
                Q.setdefault(k, []).append(d[k]); n = len(d[k]) if n is None else n
        Q.setdefault('chunk', []).append(np.full(n or 0, f0, np.int64))
    return {k: np.concatenate(v) for k, v in Q.items()}


def pass1_index(out, R):
    """pass-2 rows -> the row index of their linked pass-1 candidate (p1_row, an index into pass1_<f0>.npz) in the pass-1
    tables (the 'out' rows of the pass-1 chunk files, concatenated in chunk order); -1 if none or not a stored pass-1 row"""
    j = np.full(len(R['p1_row']), -1, np.int64); off = 0
    for f in sorted(glob.glob(f'{out}/pass1_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4])):
        f0 = int(f.rsplit('_', 1)[1][:-4])
        o = np.load(f)['out'].astype(bool)
        rank = np.where(o, np.cumsum(o) - 1 + off, -1)
        m = (R['chunk'] == f0) & (R['p1_row'] >= 0)
        j[m] = rank[R['p1_row'][m]]
        off += int(o.sum())
    return j


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


def build_table(out, tag, px_nm=PX_NM_DEFAULT, drift=None, *, kind='net', crop=None):
    """pipeline output dir -> (table dict of numpy columns, info dict) of one table: tag 'est' (pass 1) | 'est2' (pass 2), kind
    'net' (the emitter network's own output) | 'solve' (the pattern solve). crop = (x0, y0, size) of the output region (None:
    the run's own, run_crop(out)). info['n_main'] = the row count of the paper's selection (select at SIGMA_MAX_NM)."""
    if kind not in ('net', 'solve'): raise ValueError(f'kind {kind!r}: net or solve')
    crop = run_crop(out) if crop is None else tuple(float(v) for v in crop)
    R = load_rows(out, tag)
    n = len(R['x_hy0'])
    p2 = tag.endswith('2')
    if 'x_net' not in R:
        raise ValueError(f'build_table {out} [{tag}]: chunk files of a run before 2026-11-17 (no network columns x_net, ...): re-run '
                         'the pass-1 solve / pass-2 stages')
    if p2 and 'p1_row' not in R:      # their z is the pass-2 head's, not the linked pass-1 z
        raise ValueError(f'build_table {out} [{tag}]: pass-2 chunk files of a run before 2026-11-14 (no p1_row): re-run pass 2')
    f = lambda k: R[k].astype(np.float64)
    if kind == 'net':
        xy = dict(x=f('x_net'), y=f('y_net'), sx=R['sx_net'].astype(np.float32), sy=R['sy_net'].astype(np.float32))
        zc = ('z_net', 'sz_net', 'cov_xz_net', 'cov_yz_net')
    else:
        xy = dict(x=f('x_hy0'), y=f('y_hy0'), sx=R['sx_cal'].astype(np.float32), sy=R['sy_cal'].astype(np.float32))
        zc = ('z_hy0', 'sz_cal', 'cov_xz', 'cov_yz')
    T = dict(frame=R['wstart'].astype(np.int64), t=R['frame'].astype(np.float64), **xy,
             photons=R['phot'].astype(np.float32), I=R['I_net'].astype(np.float32), p=R['p'].astype(np.float32))
    if kind == 'solve':
        T.update(q_dof=R['q_dof'].astype(np.float32), kin=R['kin_hy'].astype(np.float32),
                 dnll=R['dnll'].astype(np.float32) if 'dnll' in R else np.full(n, np.nan, np.float32))
    T.update(x_prior=f('x_prior'), y_prior=f('y_prior'))
    # z (3D networks, 2026-11-14): the network's 3D prior conditioned on this table's x, y. Pass 1: the pass-1 network's;
    # pass 2: the LINKED PASS-1 candidate's (z_conditioning.pass1_z_columns / network_columns; NaN without a pass-1 partner)
    zinfo = ('pass 1: the pass-1 network z' if not p2 else 'the linked pass-1 network z') + \
            (' | the network x, y (its own sd)' if kind == 'net' else ' | the solved x, y (posterior sd)')
    if zc[0] in R:      # (2D networks before 2026-11-14: no z columns)
        T.update(z=f(zc[0]), sz=R[zc[1]].astype(np.float32), cov_xz=R[zc[2]].astype(np.float32), cov_yz=R[zc[3]].astype(np.float32),
                 z_prior=f('z_prior'))
    else: zinfo = ''
    T['good'] = T['p'] >= P_GOOD
    T['sel'] = phase_thr(T['p'], T['x'], T['y'], T['photons'])
    T['disjoint'] = T['frame'] % 6 == 0
    if p2: T['p1_index'] = pass1_index(out, R)
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
    k = select(T, SIGMA_MAX_NM, crop=crop, px_nm=px_nm)
    info.update(main_rule=f'max(sx, sy) * px_nm <= {SIGMA_MAX_NM:g} nm on the rows with p >= 0.63 inside the crop (applied at load)',
                n_main=int(k.sum()), median_photons=float(np.median(T['photons'][k])) if k.any() else float('nan'),
                median_sigma_nm=float(np.median(precision_nm(T, px_nm)[k])) if k.any() else float('nan'))
    if zinfo: info['z'] = zinfo
    if p2:      # pass-2 rows without a pass-1 partner (z NaN): all rows / the paper's selection
        info.update(n_no_p1=int((R['p1_row'] < 0).sum()), n_main_no_p1=int((R['p1_row'][k] < 0).sum()))
    info.update(tag=tag, kind=kind, n_rows=int(n), n_good=int(T['good'].sum()), n_sel=int(T['sel'].sum()), drift=drift or '',
                columns={c: str(v.dtype) + ('' if v.ndim == 1 else f'[{v.shape[1]}]') for c, v in T.items()})
    return T, info


def write_tables(out, px_nm=PX_NM_DEFAULT, drift=None, crop=None, dest=None, names=None):
    """the four tables of a run (or `names`) -> {name: (info, files)}; files in dest (default: the run directory)"""
    from simcode.io.work import atomic_json
    res = {}
    for name in (names or TABLES):
        tag, kind = TABLES[name]
        T, info = build_table(out, tag, px_nm, drift, kind=kind, crop=crop)
        info['table'] = name; info['table_doc'] = TABLE_DOC[name]; info['default'] = name == DEFAULT_TABLE
        base = os.path.join(dest or out, name)
        files = [write_table(T, info, base)]
        atomic_json(base + '_info.json', info, indent=1, default=str)
        res[name] = (info, files)
    return res


def write_table(T, info, path_base):
    """table columns + info -> <path_base>.hdf5 (torch_psf Dataset format: simcode.dataset.Dataset with the metadata pixel_nm,
    crop, tag, info) -> the file name"""
    from simcode.dataset import Dataset
    meta = dict(px_nm=info['px_nm'], crop=[float(v) for v in info['crop']], tag=info.get('tag', ''), kind=info.get('kind', ''),
                table=info.get('table', ''), info={k: v for k, v in info.items() if k != 'columns'})
    return Dataset.from_columns(T, meta).save(path_base + '.hdf5')


def build_parser():
    ap = argparse.ArgumentParser(description='localization tables of a simcode output directory (torch_psf Dataset HDF5)')
    ap.add_argument('out')
    ap.add_argument('--table', default='all', help="one of " + ', '.join(TABLES) + " or 'all' (default: all four)")
    ap.add_argument('--px_nm', type=float, default=0.0, help='pixel size (nm); default 108.33')
    ap.add_argument('--drift', default='', help='drift [F,2] px (.npy, .npz with key drift, or a DME v2 result npz), subtracted at frame + 3')
    ap.add_argument('--crop', default='', help='x0,y0,size (px) of the output region (selection); default: the output region of the run')
    ap.add_argument('--dest', default='', help='directory for the table files (default: the run directory)')
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    px = a.px_nm or PX_NM_DEFAULT
    if a.table != 'all' and a.table not in TABLES: raise SystemExit(f'--table {a.table!r}: one of {", ".join(TABLES)} or all')
    if a.dest: os.makedirs(a.dest, exist_ok=True)
    res = write_tables(a.out, px, a.drift or None, [float(v) for v in a.crop.split(',')] if a.crop else None, a.dest or None,
                       None if a.table == 'all' else [a.table])
    for name, (info, files) in res.items():
        print(name, json.dumps({k: v for k, v in info.items() if k != 'columns'}, default=str))
        print('written', *files)
    # the settings of this rebuild go into the run record of the directory (run_config.json 'steps'); command.txt = a plain log
    try:
        import sys
        from simcode.util import runconfig as RC
        RC.log_command(a.out, 'simcode.table', argv)
        outs = [f for _, fs in res.values() for f in fs]
        RC.append_step(a.out, dict(
            step='table', time=RC.now(), command='python -m simcode.table ' + ' '.join(sys.argv[1:] if argv is None else argv),
            settings=dict(out=os.path.abspath(a.out), tables=list(res), dest=os.path.abspath(a.dest) if a.dest else '', px_nm=px,
                          drift=os.path.abspath(a.drift) if a.drift else '', crop=next(iter(res.values()))[0]['crop'], p_good=P_GOOD),
            files={k: RC.file_record(v) for k, v in (('drift', a.drift),) if v},
            code={k: v for k, v in RC.code_record(dme=False).items() if k.startswith(('version', 'git_', 'modified'))},
            result={n: {k: v for k, v in i.items() if k != 'columns'} for n, (i, _) in res.items()}, outputs=[os.path.abspath(f) for f in outs]),
            outputs=outs, selection={n: RC.selection_summary(i) for n, (i, _) in res.items()} if not a.dest else None)
    except OSError as e:
        print(f'table: run_config.json of {a.out} not updated ({e})')


if __name__ == '__main__':
    main()
