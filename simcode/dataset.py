"""
SIMCODE's localization Dataset: torch_psf.Dataset plus SIMCODE's loaders and the paper's selection.

torch_psf.Dataset (torch-psf docs/DATASET_FORMAT.md, docs/API.md) holds the localizations as a dict of CPU tensors and does
everything generic: fields by key or attribute, indexing (masks, slices, indices), filters (filter, filter_finite,
filter_photons, filter_crlb, filter_frames; in-place variants with a trailing _), crop / crop_from_edge, concatenate and +,
render, frc, info, and the files: save / load (.hdf5 = the torch-psf format, .csv = ThunderSTORM), save_picasso /
load_picasso. This module adds:

  drift     the drift as state of the object (as torch-psf before 2026-11-20; torch-psf's own Dataset now returns the drift
            as a torch_psf.Drift result and has no drift state): ds.drift ([F, D] px per movie frame or None),
            ds.drift_applied, set_drift(drift, info), drift_at(), apply_drift(drift=None), undo_drift(),
            estimate_drift(n_frames, rows, dims, apply, device, return_info) (DME v2: torch_psf Dataset.estimate_drift);
            the drift goes with every object indexed / copied from ds and with concatenate; save('.hdf5') writes it in the
            file's /drift group (attributes applied, info) and load reads it back (torch-psf's own reader reads the record
            of an applied drift)

  loaders   Dataset.load(path)  a run directory (from_run), .hdf5 (load_table: a table of simcode.run / simcode.table, the
                                torch-psf format), .npz (load_npz: a table of a run before 2026-11-15, a subset file of
                                that time, or the paper's DECODE table: from_decode), .npy (load_npy: the preprint code's Dataset / SFDataset file), else
                                torch_psf's load (.hdf5: this format or Picasso, e.g. the paper's sf.hdf5; .csv: ThunderSTORM)
            from_run(run_dir, pass_=2, kind='net'), load_table, from_columns, load_npz, from_decode, from_smlmtorch(ds), load_npy
            window=<frames>: frames per localization (the n_frames field) for files that do not record it (the paper's
            SMLM / SIMFLUX tables of summed 6-frame windows: window=6)
  select    select(max_precision_nm=12, min_score=0.63, in_region=True) = simcode.table.select on this table: the paper's
            selection (calibrated precision max(crlb_x, crlb_y) * pixel_nm <= 12 nm, score >= 0.63, inside the run's crop
            meta['crop']; the drift-corrected x_dc, y_dc when the table has them); None switches a condition off.
            mask(...) is the boolean row mask.

    from simcode.dataset import Dataset
    ds = Dataset.load('out/run1')                     # a run directory (its result table locs_p2_solve.hdf5) or a table file
    sel = ds.select()                                 # the paper's selection
    ds.estimate_drift(rows=ds.good & ds.disjoint)     # DME v2 (the dme package: simcode.util.drift.dme_package())
    sel = ds.select().apply_drift(ds.drift)
    img = sel.render(pixel_nm=10); res = sel.frc()
    sel.save('run1_sel.hdf5')                         # torch-psf format; sel.save_picasso(...), sel.save('x.csv')

Fields of a v2 table (simcode/table.py) in this layout (docs/DATASET_FORMAT.md of SIMCODE lists them):
  pos [n, 2] = (x, y) px, crlb [n, 2] = (sx, sy) px (calibrated sigma); tables of 3D heads (columns z, sz, cov_xz, cov_yz):
  pos [n, 3] = (x, y, z) with z in px of the lateral pixel relative to the PSF calibration's focal plane, crlb [n, 3], cov
  [n, 3] = (xy, xz, yz) px^2; frame = window start, n_frames = 6 (window length), photons, score = p (detector probability), intensities [n, 6] = I (network per-frame intensities), and the other
  columns under their names (t, q_dof, kin, dnll (solve tables), x_prior, y_prior, good, sel, disjoint, p1_index, x_dc, y_dc).
  Metadata: pixel_nm (px_nm of the table), crop, tag, kind ('net' | 'solve'), table, info (the table's info json), run_config,
  model, n_frames, image_size, pass, source, format.
Other sources: DECODE phot -> photons, sphot -> crlb_photons, prob -> score, z (nm) kept as the field z; the preprint code's
estim.pos / crlb.pos -> pos / crlb, estim.ibg -> intensities, intensities_bg, chisq, group, roi_id, roipos as fields (a z
column becomes the field z, crlb_z: its own unit, not part of pos).
"""
import json, os
import numpy as np
import torch
from simcode import table as SP
from simcode.psf import Dataset as _TorchPSFDataset, Drift

_RENAME = dict(p='score', I='intensities')                     # v2 table column -> field
_META = dict(px_nm='pixel_nm')                                 # v2 table metadata -> torch_psf names


def _f64(v):
    """drift array (numpy / list / tensor) -> float64 CPU tensor"""
    return (v.detach() if torch.is_tensor(v) else torch.from_numpy(np.array(v))).to(torch.float64)


def _read_json(path):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError):
        return None


DECODE_FRAME_LABEL = 'window start (pad same, 2026-11-15 fix)'      # = analysis/baselines/decode/dec_common.FRAME_LABEL


def _decode_fixed(z):
    if 'info' not in z.files or z['info'].ndim != 0: return False
    try: return json.loads(str(z['info'])).get('frame_label') == DECODE_FRAME_LABEL
    except ValueError: return False


def decode_table(path):
    """The DECODE table to read for a DECODE output path (<dir>/locs.npz). DECODE tables written before 2026-11-15 label each
    row one 6-frame window early (the decode fork's InferenceDataset has pad=None: output i is the centre sum i + 1; fixed in
    analysis/baselines/decode/dec_common.process_padded). Their corrected copy (frames + 6, the original unchanged) is
    <dir>/locs_fixed.npz (analysis/baselines/decode/fix_frames.py): returned when path has the old labels and the copy exists,
    else path (a table of the fixed inference has info['frame_label'])."""
    fx = os.path.join(os.path.dirname(path), 'locs_fixed.npz')
    if os.path.abspath(path) == os.path.abspath(fx) or not os.path.exists(fx): return path
    if os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(fx):
        with np.load(path, allow_pickle=False) as z:
            if _decode_fixed(z): return path              # rerun with the fixed inference after the copy was made
    return fx


class Dataset(_TorchPSFDataset):
    """torch_psf.Dataset with SIMCODE's loaders, select() and the drift state (module docstring)."""

    def __init__(self, fields, meta=None, drift=None, drift_applied=False):
        super().__init__(fields, meta)
        self.drift = None if drift is None else _f64(drift)
        self.drift_applied = bool(drift_applied)

    def __getitem__(self, idx):
        r = super().__getitem__(idx)
        if isinstance(r, Dataset):
            r.drift, r.drift_applied = self.drift, self.drift_applied
        return r

    def copy(self):
        r = super().copy()
        r.drift, r.drift_applied = (None if self.drift is None else self.drift.clone()), self.drift_applied
        return r

    def to(self, device):
        r = super().to(device)
        r.drift, r.drift_applied = (None if self.drift is None else self.drift.to(device)), self.drift_applied
        return r

    @classmethod
    def concatenate(cls, items, set_group=False):
        """Rows of several datasets: the fields they all have, metadata of the first (n_frames = the largest); set_group:
        field `group` = index of the source. The drift of the first when all have the same (else none); refused when some
        are drift corrected and some are not."""
        items = list(items)
        if not items:
            raise ValueError('nothing to concatenate')
        if len({d.drift_applied for d in items}) > 1:
            raise ValueError('some datasets are drift corrected and some are not')
        keys = [k for k in items[0].fields if all(k in d.fields for d in items)]
        F = {k: torch.cat([d.fields[k] for d in items]) for k in keys}
        if set_group:
            F['group'] = torch.cat([torch.full((len(d),), i, dtype=torch.int64) for i, d in enumerate(items)])
        meta = dict(items[0].meta)
        if any(d.meta.get('n_frames') for d in items):
            meta['n_frames'] = max(d.num_frames for d in items)
        d0 = items[0].drift
        same = all((d.drift is None and d0 is None) or (d.drift is not None and d0 is not None and torch.equal(d.drift, d0))
                   for d in items)
        return cls(F, meta, d0 if same else None, items[0].drift_applied)

    # ------------------------------------------------------------------ drift (module docstring)
    def drift_at(self):
        """Drift [n, D'] (px) at each localization's centre frame (clipped to the drift's frame range)."""
        if self.drift is None:
            raise ValueError('no drift: estimate_drift() or set_drift() first')
        return Drift(self.drift).at(self.center_frame)

    def set_drift(self, drift, info=None):
        """Store a drift [F, D'] (px per movie frame, D' <= D) without applying it. Returns self."""
        if self.drift_applied:
            raise ValueError('positions are drift corrected: undo_drift() first')
        self.drift = _f64(drift)
        if info is not None:
            self.meta['drift'] = info
        return self

    def apply_drift(self, drift=None):
        """pos -= drift[centre frame] (the stored drift, or `drift`). Returns self."""
        if self.drift_applied:
            raise ValueError('positions are already drift corrected')
        if drift is not None:
            self.drift = _f64(drift)
        d = self.drift_at()
        p = self.fields['pos'].clone()
        p[:, :d.shape[1]] -= d.to(p)
        self.fields['pos'] = p
        self.drift_applied = True
        return self

    def undo_drift(self):
        """Add the applied drift back. Returns self."""
        if not self.drift_applied:
            raise ValueError('no drift correction has been applied')
        d = self.drift_at()
        p = self.fields['pos'].clone()
        p[:, :d.shape[1]] += d.to(p)
        self.fields['pos'] = p
        self.drift_applied = False
        return self

    def estimate_drift(self, n_frames=None, rows=None, dims=2, apply=False, device=None, return_info=False):
        """Drift with DME v2 (torch_psf Dataset.estimate_drift: rows `rows` (mask) or all, only those with finite positions and
        0 < crlb < 1 px; time = centre frame; dims 2 or 3). Stores the drift [F, dims] px in self.drift and the run record in
        meta['drift']; returns the drift (and the record with return_info). apply=True also subtracts it."""
        if self.drift_applied:
            raise ValueError('positions are already drift corrected: undo_drift() first')
        D = super().estimate_drift(n_frames=n_frames, rows=rows, dims=dims, device=device)
        self.drift = D.curve
        self.meta['drift'] = D.info
        if apply:
            self.apply_drift()
        return (self.drift, D.info) if return_info else self.drift

    def info(self):
        R = super().info()
        R['drift'] = 'none' if self.drift is None else ('applied' if self.drift_applied else 'stored, not applied')
        return R

    def save_hdf5(self, path):
        """torch_psf Dataset.save_hdf5, plus the drift (when set) as the dataset /drift with the attributes applied and info
        (the record meta['drift'])."""
        out = super().save_hdf5(path)
        if self.drift is not None:
            import h5py
            with h5py.File(out, 'a') as f:
                d = f.create_dataset('drift', data=self.drift.cpu().numpy())
                d.attrs['applied'] = self.drift_applied
                d.attrs['info'] = json.dumps(self.meta.get('drift', {}), default=str)
        return out

    @classmethod
    def load_hdf5(cls, path):
        """torch_psf Dataset.load_hdf5, plus the drift of the file's /drift (when it has one)."""
        import h5py, warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='File holds an unapplied drift curve')
            L = super().load_hdf5(path)
        with h5py.File(path, 'r') as f:
            if 'drift' in f:
                L.drift = torch.from_numpy(f['drift'][...]).to(torch.float64)
                L.drift_applied = bool(f['drift'].attrs.get('applied', False))
                info = f['drift'].attrs.get('info')
                if info is not None and 'drift' not in L.meta:
                    L.meta['drift'] = json.loads(info)
        return L

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_columns(cls, C, meta, window=6, drift=None, drift_applied=False):
        """table columns x, y, frame (+ sx, sy, z, sz and the others of simcode.table.build_table) -> this layout (pos / crlb /
        cov, score, intensities; metadata px_nm -> pixel_nm)"""
        C = dict(C)
        F = dict(pos=np.stack([np.asarray(C.pop('x'), np.float64), np.asarray(C.pop('y'), np.float64)], 1),
                 frame=np.asarray(C.pop('frame')))
        if 'sx' in C and 'sy' in C:
            F['crlb'] = np.stack([C.pop('sx'), C.pop('sy')], 1).astype(np.float32)
        if 'cov_xz' in C and 'z' in C and 'sz' in C and np.isfinite(np.asarray(C['z'])).any():
            # v2 table of a 3D head: z (px, relative to the PSF's focal plane) in pos, its sd in crlb, cov = (xy, xz, yz)
            F['pos'] = np.concatenate([F['pos'], np.asarray(C.pop('z'), np.float64)[:, None]], 1)
            F['crlb'] = np.concatenate([F['crlb'], np.asarray(C.pop('sz'), np.float32)[:, None]], 1)
            xz, yz = np.asarray(C.pop('cov_xz'), np.float32), np.asarray(C.pop('cov_yz'), np.float32)
            F['cov'] = np.stack([np.zeros_like(xz), xz, yz], 1)
        if 'sz' in C: F['crlb_z'] = C.pop('sz')
        F['n_frames'] = np.full(len(F['frame']), int(window), np.int32)
        for k, v in C.items():
            F[_RENAME.get(k, k)] = np.ascontiguousarray(v)
        meta = {_META.get(k, k): v for k, v in meta.items() if k != 'window'}
        meta.setdefault('pixel_nm', SP.PX_NM_DEFAULT)
        return cls(F, meta, drift, drift_applied)

    @classmethod
    def load_npz(cls, path, columns=None, info=True, window=6):
        """A table file of a run before 2026-11-15 (locs_p1.npz / locs_p2.npz / a subset file of that time), or the
        paper's DECODE table (from_decode). columns: load only these (x, y, frame always; a 660 MB production table takes a
        few seconds in full). info: also read <table>_info.json, and summary.json / run_config.json of the directory when
        they exist (crop, model, frame count, image size, the run record)."""
        z = np.load(path, allow_pickle=False)
        if 'phot' in z.files and 'photons' not in z.files: return cls.from_decode(path, window)
        n = len(z['x']); C = {}; meta = dict(source=os.path.abspath(path), format='simcode npz')
        want = None if columns is None else set(columns) | {'x', 'y', 'frame'}
        for k in z.files:
            if want is not None and k not in want and k not in ('px_nm', 'sigma_max'): continue
            v = z[k]
            if v.ndim == 0:
                meta[k] = v.item() if v.dtype.kind in 'fiub' else str(v)
            elif len(v) == n:
                C[k] = v
        if info: cls._run_meta(path, meta)
        return cls.from_columns(C, meta, meta.get('window', window))

    @staticmethod
    def _run_meta(path, meta):
        """metadata of a table file of a run directory: <table>_info.json, summary.json / run_config.json of the directory (crop,
        model, frame count, image size, the run record)"""
        d = os.path.dirname(os.path.abspath(path)); base = os.path.splitext(os.path.abspath(path))[0]
        I = _read_json(base + '_info.json')
        if I is not None:
            meta.setdefault('info', I)
            if 'crop' in I: meta.setdefault('crop', [float(v) for v in I['crop']])
        if 'crop' not in meta and os.path.exists(os.path.join(d, 'pass1_0.npz')):          # tables made before the info json had it
            meta['crop'] = [float(v) for v in SP.run_crop(d)]
        S = _read_json(os.path.join(d, 'summary.json')) or {}
        R = _read_json(os.path.join(d, 'run_config.json'))
        if R is not None:
            meta.setdefault('run_config', R); mv = R.get('movie') or {}
            nf = mv.get('frames_processed') or mv.get('n_frames')
            if nf: meta.setdefault('n_frames', int(nf))
            if mv.get('frame_shape'): meta.setdefault('image_size', [int(v) for v in mv['frame_shape']])
            m = R.get('model'); m = m.get('name') if isinstance(m, dict) else m
            if m: meta.setdefault('model', m)
        m = S.get('model')
        if isinstance(m, str): meta.setdefault('model', m)

    @classmethod
    def load_table(cls, path, info=True):
        """a table file of simcode.run / simcode.table (<name>.hdf5, the torch_psf format with SIMCODE's fields) with the run's
        metadata (info: _run_meta)"""
        L = cls.load_hdf5(path); L.meta['format'] = 'simcode hdf5'
        if info: cls._run_meta(path, L.meta)
        return L

    @classmethod
    def from_run(cls, run_dir, pass_=2, kind='solve', table=None, **kw):
        """A localization table of a run directory of simcode.run (four tables per run, simcode/table.py TABLES): pass_ = 2 (the
        pattern-aware pass, the default) or 1, kind = 'solve' (the pattern solve, the default: locs_p2_solve = the result table,
        simcode.table.DEFAULT_TABLE; 2026-11-19) or 'net' (the emitter network's own output: locs_p<pass>_net); table: another
        base name instead (e.g. 'locs_p2' of a run before 2026-11-17, whose locs_p<pass> tables were the solve with the removed
        sigma rescale). meta['pass'] and
        meta['kind'] record which table it is. This is what the pipeline's dataset_pass1() / dataset_pass2() return; the
        paper's selection: ds.select()."""
        n = table or SP.table_name(pass_, kind)
        n = os.path.splitext(n)[0] if n.endswith(('.npz', '.hdf5')) else n
        for ext in ('.hdf5', '.npz'):
            f = os.path.join(run_dir, n + ext)
            if not os.path.exists(f): continue
            L = cls.load_table(f, info=kw.get('info', True)) if ext == '.hdf5' else cls.load_npz(f, **kw)
            b = os.path.basename(f)
            if b.startswith('locs_p') and b[6:7].isdigit(): L.meta.setdefault('pass', int(b[6]))
            if n in SP.TABLES: L.meta.setdefault('kind', SP.TABLES[n][1])
            return L
        old = [t for t in ('locs_p2', 'locs_p1') if any(os.path.exists(os.path.join(run_dir, t + e)) for e in ('.hdf5', '.npz'))]
        raise FileNotFoundError(f'no {n} (.hdf5 / .npz) in {run_dir}' + (f' (a run before 2026-11-17 with {old}: give table=..., or '
                                'rebuild the tables of a re-run)' if old else ''))

    @classmethod
    def from_decode(cls, path, window=6):
        """A DECODE table as written for the paper (npz with frame, x, y, z, sx, sy, phot, sphot, bg, prob, info): phot ->
        photons, sphot -> crlb_photons, prob -> score; z (nm) and the other columns keep their names. A table without
        info['frame_label'] (written before 2026-11-15) labels its rows one window early (decode_table) -> a warning."""
        z = np.load(path, allow_pickle=False); n = len(z['x'])
        if not _decode_fixed(z):
            import warnings
            warnings.warn(f'{path}: DECODE table with the pre-2026-11-15 frame labels (one window early); read '
                          f'decode_table(path) = the locs_fixed.npz next to it (analysis/baselines/decode/fix_frames.py)')
        ren = dict(phot='photons', sphot='crlb_photons', prob='score')
        C = {ren.get(k, k): z[k] for k in z.files if z[k].ndim and len(z[k]) == n}
        meta = dict(source=os.path.abspath(path), format='decode npz')
        if 'info' in z.files and z['info'].ndim == 0: meta['info'] = str(z['info'])
        return cls.from_columns(C, meta, window)

    @classmethod
    def from_smlmtorch(cls, ds, window=1, px_nm=SP.PX_NM_DEFAULT):
        """From a Dataset / SFDataset object of the preprint code (smlmtorch), or its structured array (ds.data): estim.pos ->
        pos (a z column -> the field z), crlb.pos -> crlb (crlb_z), estim.photons / bg -> photons / bg, frame, chisq, group,
        roi_id; SFDataset estim.ibg [n, patterns, 2] -> intensities and intensities_bg. window: frames per localization (6 for
        the paper's summed-window SMLM / SIMFLUX tables; it sets the time of a row for the drift)."""
        data = ds if isinstance(ds, np.ndarray) else ds.data; e, c = data['estim'], data['crlb']
        C = dict(x=e['pos'][:, 0], y=e['pos'][:, 1], frame=np.asarray(data['frame']), sx=c['pos'][:, 0], sy=c['pos'][:, 1],
                 photons=e['photons'], bg=e['bg'])
        if e['pos'].shape[1] > 2: C.update(z=e['pos'][:, 2], sz=c['pos'][:, 2])
        for k in data.dtype.names:                                 # chisq, group, roi_id, roipos and any extra field
            if k not in ('frame', 'estim', 'crlb'): C[k] = np.asarray(data[k])
        if 'ibg' in e.dtype.names: C.update(intensities=e['ibg'][..., 0], intensities_bg=e['ibg'][..., 1])
        if 'sigma' in e.dtype.names: C.update(psf_sx=e['sigma'][:, 0], psf_sy=e['sigma'][:, 1])
        meta = dict(format='smlmtorch', pixel_nm=float(px_nm))
        if hasattr(ds, 'imgshape'): meta['image_size'] = [int(v) for v in ds.imgshape]
        return cls.from_columns({k: np.ascontiguousarray(v) for k, v in C.items()}, meta, window)

    @classmethod
    def load_npy(cls, path, window=1, px_nm=SP.PX_NM_DEFAULT):
        """The numpy file of the preprint code's Dataset.save (.npy: image shape, then the structured array), read with numpy
        only. Fields as from_smlmtorch."""
        with open(path, 'rb') as s:
            imgshape = np.load(s); data = np.load(s)
        L = cls.from_smlmtorch(data, window, px_nm)
        L.meta.update(image_size=[int(v) for v in imgshape], source=os.path.abspath(path), format='smlmtorch npy')
        return L

    @classmethod
    def load(cls, path, window=None, **kw):
        """By kind of path: a run directory (from_run), .npz (load_npz: v2 table or the paper's DECODE table), .npy (load_npy: the preprint code's file), else torch_psf.Dataset.load (.hdf5 / .h5: torch-psf format or
        Picasso; .csv: ThunderSTORM, needs pixel_nm=). window: frames per localization (the n_frames field) when the file
        does not record it (the paper's SMLM / SIMFLUX tables: 6); keywords go to that reader."""
        path = str(path)
        if os.path.isdir(path): return cls.from_run(path, **kw)
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npz': return cls.load_npz(path, window=window or 6, **kw)
        if ext == '.npy': return cls.load_npy(path, window=window or 1, **kw)
        L = super().load(path, **kw)
        if ext in ('.hdf5', '.h5') and 'tag' in L.meta and 'info' in L.meta: cls._run_meta(path, L.meta)   # a table of simcode.run
        L.meta.setdefault('pixel_nm', SP.PX_NM_DEFAULT)
        if window is not None: L['n_frames'] = torch.full((len(L),), int(window), dtype=torch.int32)
        return L

    def columns(self):
        """this dataset as the table columns of simcode.table.build_table (numpy): x, y (z), sx, sy (sz), cov_xz, cov_yz, p,
        I and the other fields under their names, + px_nm and crop of the metadata (the dict simcode.table.select reads)"""
        F = self.numpy(); T = {}
        T['x'], T['y'] = F['pos'][:, 0], F['pos'][:, 1]
        if F['pos'].shape[1] > 2: T['z'] = F['pos'][:, 2]
        if 'crlb' in F:
            T['sx'], T['sy'] = F['crlb'][:, 0], F['crlb'][:, 1]
            if F['crlb'].shape[1] > 2: T['sz'] = F['crlb'][:, 2]
        if 'cov' in F: T['cov_xz'], T['cov_yz'] = F['cov'][:, 1], F['cov'][:, 2]
        back = {v: k for k, v in _RENAME.items()}
        for k, v in F.items():
            if k not in ('pos', 'crlb', 'cov', 'n_frames'): T[back.get(k, k)] = v
        T['px_nm'] = self.pixel_nm
        if self.meta.get('crop') is not None: T['crop'] = np.array(self.meta['crop'], np.float64)
        return T

    # ------------------------------------------------------------------ the paper's selection
    def _table(self):
        """this dataset as the v2 table dict simcode.table.select reads"""
        F = self.numpy(); p, c = F['pos'], F['crlb']
        T = dict(x=p[:, 0], y=p[:, 1], sx=c[:, 0], sy=c[:, 1], px_nm=self.pixel_nm)
        if 'score' in F: T['p'] = F['score']
        for k in ('x_dc', 'y_dc'):
            if k in F: T[k] = F[k]
        if self.meta.get('crop') is not None: T['crop'] = np.array(self.meta['crop'], np.float64)
        return T

    def mask(self, max_precision_nm=SP.SIGMA_MAX_NM, min_score=SP.P_GOOD, in_region=True):
        """Row mask (bool tensor) of the paper's selection = simcode.table.select on this table: max(crlb_x, crlb_y) *
        pixel_nm <= max_precision_nm (nm), score >= min_score, inside the run's crop (meta['crop'], else
        simcode.table.CROP; x_dc, y_dc when the table has them); None switches a condition off."""
        return torch.from_numpy(SP.select(self._table(), max_precision_nm, min_score, in_region))

    def select(self, max_precision_nm=SP.SIGMA_MAX_NM, min_score=SP.P_GOOD, in_region=True):
        """The rows of the paper's selection as a new object (self[self.mask(...)]); the parameters in meta['selection']."""
        L = self[self.mask(max_precision_nm, min_score, in_region)]
        L.meta['selection'] = dict(max_precision_nm=max_precision_nm, min_score=min_score, in_region=bool(in_region))
        return L


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2: raise SystemExit('usage: python -m simcode.dataset <run dir | table file> [output file (.hdf5 / .csv)] [max precision nm]')
    L = Dataset.load(sys.argv[1])
    if len(sys.argv) > 3: L = L.select(float(sys.argv[3]))
    print(L)
    if len(sys.argv) > 2: print('written', L.save(sys.argv[2]))
