"""
Dataset: the localizations of one sample as an object (columns as numpy arrays + metadata + drift), with selection,
filters, crop, merge, drift correction, rendering, FRC and file formats. It is the v2 counterpart of the preprint code's
dataset classes (smlmtorch Dataset / SFDataset): the same method names where they make sense (load / save, saveHDF5 /
loadHDF5, saveNumpy / loadNumpy, merge and +, filter, crlb_filter, crop, cropFromEdge, autocrop, pos / crlb / photons /
background / frame, numFrames, indicesPerFrame, addFields, applyDrift / undoDrift / isDriftCorrected, info, fromEstimates)
and it reads their files. `Localizations` is an alias of the class (simcode.locs re-exports both).

    from simcode.dataset import Dataset
    ds = Dataset.load('out/run1')                     # a run directory (locs_p2.npz, else locs_p1.npz) or a table file
    ds = Dataset.from_run('out/run1', pass_=2)        # the same, the pass given
    m = ds.select('main')                             # the paper's selection: max(sx, sy) <= 12 nm on the ranked rows
    ds.estimateDrift(); m = ds.select('main').applyDrift()
    img = m.render(pixel_nm=10); res = m.frc()        # 2D histogram [ny, nx]; FRC resolution (nm)
    m.save('run1_main.hdf5')                          # Picasso; also .npz (v2 table, lossless), .csv (ThunderSTORM), .npy

Differences from the preprint class: positions are the columns x, y (pos is a derived [n, 2] array, not a view), filters and
crops return a new object instead of changing this one, every field is a plain column (addFields(name=array)), and
applyDrift subtracts drift[frame + window // 2] (window centre).

Units and conventions (the v2 table's, simcode/strict.py):
  x, y        camera pixels, sensor coordinates of the movie; an integer value is a pixel CENTRE; x = column, y = row
  sx, sy      localization precision (one sigma) of x, y in camera pixels (v2: calibrated sigma; preprint code / Picasso: CRLB;
              DECODE: its predicted sigma). precision_nm = max(sx, sy) * px_nm
  z, sz       optional; the unit of the source file (v2 tables have no z)
  frame       first movie frame of the localization (0-based). v2, and the paper's SMLM / SIMFLUX / DECODE tables, localize
              windows of 6 frames: frame = window start, the window covers frame .. frame + window - 1
  photons     v2: sum of the network per-frame intensities of the window (label units); other sources: their photon count
  bg          background per pixel, if the source has it (v2 tables have none)
  I           [n, window] per-frame intensities, if present
  p, strict_u, good, sel, strict, disjoint, ...   the v2 columns, see simcode/strict.py
  px_nm       camera pixel size in the sample (nm); positions in nm = x * px_nm
  drift       [F, 2] camera pixels per movie frame; corrected position = x - drift[frame + window // 2] (window centre)
Every other column of a source file is kept under its own name, and further per-localization fields can be added
(ds.addFields(name=array) or ds['name'] = array): all columns are carried through indexing, merge, save and load. Columns
are read as attributes (ds.x, ds.photons) or ds['x']; ds[mask], ds[slice], ds[indices] give a new object with those rows
(metadata and drift are carried along).

Formats (load / save pick by the file extension):
  .npz    the v2 table (simcode.strict.write_table layout: one array per column, scalars px_nm, u_max, sigma_max). save()
          adds the metadata (json) and the drift under the keys locs_meta, locs_drift, locs_drift_applied; the readers of the
          existing tables ignore those. Lossless. DECODE tables of the paper (analysis/baselines; key 'phot') are recognised.
  .hdf5   Picasso: dataset 'locs' (structured: frame u4, x, y, photons, sx, sy, bg, lpx, lpy, ... f4) + a .yaml next to it.
          In Picasso sx, sy are the PSF widths and lpx, lpy the localization precision, so this class's sx, sy are stored as
          lpx, lpy, and Picasso's sx, sy are the columns psf_sx, psf_sy here. float32 positions.
  .csv    ThunderSTORM: "frame","x [nm]","y [nm]","sigma [nm]","intensity [photon]","offset [photon]","uncertainty [nm]", ...
          ThunderSTORM counts frames from 1 and puts the centre of the first pixel at 0.5 px; both are converted.
  .npy    the preprint code's Dataset / SFDataset numpy file (image shape, then one structured array: roi_id, frame,
          estim (pos, photons, bg, ibg), crlb (same), chisq, group, roipos, + extra fields), read and written with numpy only
          (smlmtorch is not imported). float32 positions, no metadata: use .npz to keep everything.
Dependencies: numpy; h5py only for .hdf5; scipy only for render(sigma_nm > 0); the `dme` package only for estimate_drift.
"""
import json, os
import numpy as np
from simcode import strict as SP

BOOL_COLS = ('good', 'sel', 'strict', 'disjoint', 'kin')
META_KEY, DRIFT_KEY, APPLIED_KEY = 'locs_meta', 'locs_drift', 'locs_drift_applied'
PSF_SIGMA_PX = 1.3          # nominal PSF width written to Picasso / ThunderSTORM files when the table has none (not measured)
_PICASSO_YAML = ('Byte Order', 'Camera', 'Data Type', 'File', 'Frames', 'Height', 'Width')


def _jsonable(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.generic): return o.item()
    return str(o)


def _read_json(path):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError):
        return None


def _group_indexed(cols):
    """columns I0, I1, ... (how 2D columns are written to csv / hdf5) -> one column I [n, k]"""
    import re
    G = {}
    for k in cols:
        m = re.fullmatch(r'(.*\D)(\d+)', k)
        if m: G.setdefault(m.group(1), {})[int(m.group(2))] = k
    for base, d in G.items():
        if base in cols or len(d) < 2 or sorted(d) != list(range(len(d))): continue
        cols[base] = np.stack([cols.pop(d[i]) for i in range(len(d))], 1)
    return cols


class Dataset:
    """Localizations of one sample: columns (numpy arrays of one length), metadata (dict `meta`) and an optional drift.
    See the module docstring for units, conventions and file formats.

    Attributes: cols (dict name -> array), meta (dict: px_nm, window, nframes, imgshape [h, w], crop [x0, y0, size], model,
    source, u_max, sigma_max, info = the table's info json, run_config = the run record if the run has one, drift_info),
    drift ([F, 2] px or None), drift_applied (bool)."""

    def __init__(self, cols, meta=None, drift=None, drift_applied=False):
        """cols: dict with at least x, y, frame (arrays of one length; not copied). meta: dict (px_nm default 108.33,
        window default 6). drift: [F, 2] px; drift_applied: whether x, y already have it subtracted."""
        cols = dict(cols)
        for k in ('x', 'y', 'frame'):
            if k not in cols: raise KeyError(f'Dataset needs the column {k!r}')
        n = len(cols['x'])
        for k, v in cols.items():
            cols[k] = v = np.asarray(v)
            if v.ndim == 0 or len(v) != n: raise ValueError(f'column {k!r}: shape {v.shape}, expected {n} rows')
        self.cols = cols
        self.meta = dict(meta or {})
        self.meta.setdefault('px_nm', SP.PX_NM_DEFAULT); self.meta.setdefault('window', 6)
        self.drift = None if drift is None else np.asarray(drift)
        self.drift_applied = bool(drift_applied)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_arrays(cls, x, y, frame, sx=None, sy=None, photons=None, bg=None, z=None, px_nm=SP.PX_NM_DEFAULT, window=6,
                    imgshape=None, nframes=None, **columns):
        """From arrays (the counterpart of the preprint code's Dataset.fromEstimates). x, y, sx, sy in camera pixels, frame =
        first frame of each localization, window = frames per localization; further columns by keyword (p=..., I=...)."""
        C = dict(x=np.asarray(x), y=np.asarray(y), frame=np.asarray(frame).astype(np.int64))
        for k, v in dict(sx=sx, sy=sy, photons=photons, bg=bg, z=z, **columns).items():
            if v is not None: C[k] = np.asarray(v)
        meta = dict(px_nm=float(px_nm), window=int(window), source='arrays')
        if imgshape is not None: meta['imgshape'] = [int(v) for v in imgshape]
        if nframes is not None: meta['nframes'] = int(nframes)
        return cls(C, meta)

    fromEstimates = from_arrays          # (the preprint class's fromEstimates took a parameter matrix; here: arrays by name)

    @classmethod
    def from_table(cls, path, columns=None, info=True):
        """A v2 table file (locs_p1.npz / locs_p2.npz / a subset file / a file written by save()). columns: load only these
        (x, y, frame are always loaded; a 660 MB production table takes a few seconds in full). info: also read
        <table>_info.json, and summary.json / run_config.json of the directory when they exist (crop, model, frame count,
        image size, the run record)."""
        z = np.load(path, allow_pickle=False)
        if 'phot' in z.files and 'photons' not in z.files: return cls.from_decode(path)
        n = len(z['x']); C = {}; meta = dict(source=os.path.abspath(path), format='simcode npz')
        want = None if columns is None else set(columns) | {'x', 'y', 'frame'}
        drift = None; applied = False
        for k in z.files:
            if k == META_KEY: meta = {**json.loads(str(z[k])), **meta}; continue
            if k == DRIFT_KEY: drift = z[k]; continue
            if k == APPLIED_KEY: applied = bool(z[k]); continue
            if want is not None and k not in want and k not in ('px_nm', 'u_max', 'sigma_max'): continue
            v = z[k]
            if v.ndim == 0:
                meta[k] = v.item() if v.dtype.kind in 'fiub' else str(v)
            elif len(v) == n:
                C[k] = v
        if info:
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
                if nf: meta.setdefault('nframes', int(nf))
                if mv.get('frame_shape'): meta.setdefault('imgshape', [int(v) for v in mv['frame_shape']])
                m = R.get('model'); m = m.get('name') if isinstance(m, dict) else m
                if m: meta.setdefault('model', m)
            m = S.get('model', S.get('netset'))
            if isinstance(m, str): meta.setdefault('model', m)
        return cls(C, meta, drift, applied)

    @classmethod
    def from_run(cls, run_dir, pass_=2, table=None, **kw):
        """The localization table of a run directory of simcode.run: pass_ = 2 (locs_p2.npz, the pattern-aware pass, the
        default) or 1 (locs_p1.npz); pass_ None: pass 2 if it exists, else pass 1; table: another base name instead
        (e.g. 'locs_p2_sig12'). meta['pass'] records the pass. Keywords as from_table (columns=..., info=...). This is what
        the pipeline's dataset_pass1() / dataset_pass2() return; select with ds.select('main')."""
        names = [table] if table else [f'locs_p{int(pass_)}'] if pass_ is not None else ['locs_p2', 'locs_p1']
        for n in names:
            f = os.path.join(run_dir, n if n.endswith('.npz') else n + '.npz')
            if os.path.exists(f):
                L = cls.from_table(f, **kw); b = os.path.basename(f)
                if b.startswith('locs_p') and b[6:7].isdigit(): L.meta.setdefault('pass', int(b[6]))
                return L
        raise FileNotFoundError(f'no {" / ".join(names)} (.npz) in {run_dir}')

    @classmethod
    def from_decode(cls, path, window=6):
        """A DECODE table as written for the paper (npz with frame, x, y, z, sx, sy, phot, sphot, bg, prob, info): phot ->
        photons, sphot -> photons_sigma, prob -> p; the other columns keep their names."""
        z = np.load(path, allow_pickle=False); n = len(z['x'])
        ren = dict(phot='photons', sphot='photons_sigma', prob='p')
        C = {ren.get(k, k): z[k] for k in z.files if z[k].ndim and len(z[k]) == n}
        meta = dict(source=os.path.abspath(path), format='decode npz', window=int(window))
        if 'info' in z.files and z['info'].ndim == 0: meta['info'] = str(z['info'])
        return cls(C, meta)

    @classmethod
    def from_smlmtorch(cls, ds, window=1, px_nm=SP.PX_NM_DEFAULT):
        """From a Dataset / SFDataset object of the preprint code (smlmtorch), or its structured array (ds.data): estim.pos ->
        x, y (z), crlb.pos -> sx, sy (sz), estim.photons / bg -> photons / bg, frame, chisq, group, roi_id; SFDataset
        estim.ibg [n, patterns, 2] -> I (intensities) and I_bg (backgrounds). window: frames per localization (6 for the
        paper's summed-window SMLM / SIMFLUX tables; it sets the time of a row for the drift)."""
        data = ds if isinstance(ds, np.ndarray) else ds.data; e, c = data['estim'], data['crlb']
        C = dict(x=e['pos'][:, 0], y=e['pos'][:, 1], frame=np.asarray(data['frame']), sx=c['pos'][:, 0], sy=c['pos'][:, 1],
                 photons=e['photons'], bg=e['bg'])
        if e['pos'].shape[1] > 2: C.update(z=e['pos'][:, 2], sz=c['pos'][:, 2])
        for k in data.dtype.names:                                 # chisq, group, roi_id, roipos and any extra field
            if k not in ('frame', 'estim', 'crlb'): C[k] = np.asarray(data[k])
        if 'ibg' in e.dtype.names: C.update(I=e['ibg'][..., 0], I_bg=e['ibg'][..., 1])
        if 'sigma' in e.dtype.names: C.update(psf_sx=e['sigma'][:, 0], psf_sy=e['sigma'][:, 1])
        meta = dict(format='smlmtorch', window=int(window), px_nm=float(px_nm))
        if hasattr(ds, 'imgshape'): meta['imgshape'] = [int(v) for v in ds.imgshape]
        return cls({k: np.ascontiguousarray(v) for k, v in C.items()}, meta)

    @classmethod
    def from_smlmtorch_npy(cls, path, window=1, px_nm=SP.PX_NM_DEFAULT):
        """The numpy file of the preprint code's Dataset.save, or of saveNumpy here (.npy: image shape, then the structured
        array), read with numpy only. Columns as from_smlmtorch; extra fields by their names."""
        with open(path, 'rb') as s:
            imgshape = np.load(s); data = np.load(s)
        L = cls.from_smlmtorch(data, window, px_nm)
        L.meta.update(imgshape=[int(v) for v in imgshape], source=os.path.abspath(path), format='smlmtorch npy')
        return L

    @classmethod
    def from_picasso(cls, path, window=1, px_nm=None):
        """A Picasso hdf5 file (also the preprint code's saveHDF5 output, e.g. the paper's sf.hdf5), read with h5py: x, y,
        frame, photons, bg by name; lpx, lpy (lpz) -> sx, sy (sz) = localization precision; Picasso's sx, sy (PSF widths) ->
        psf_sx, psf_sy; every other field by its name. Image size, frame count and pixel size (px_nm None) from the .yaml
        next to the file when it exists."""
        import h5py
        with h5py.File(path, 'r') as f:
            locs = f['locs'][:]
            bools = set(json.loads(f['locs'].attrs['bool_columns'])) if 'bool_columns' in f['locs'].attrs else set(BOOL_COLS)
        ren = dict(lpx='sx', lpy='sy', lpz='sz', sx='psf_sx', sy='psf_sy')
        C = {}
        for k in locs.dtype.names:
            v = np.ascontiguousarray(locs[k]); k2 = ren.get(k, k)
            if k in ('ellipticity', 'net_gradient') and not v.any(): continue      # placeholders of the writers
            C[k2] = v.astype(bool) if k2 in bools else v.astype(np.int64) if k == 'frame' else v
        C = _group_indexed(C)
        for k in bools & set(C): C[k] = C[k].astype(bool)
        meta = dict(source=os.path.abspath(path), format='picasso hdf5', window=int(window))
        Y = _read_picasso_yaml(os.path.splitext(path)[0] + '.yaml')
        if 'Height' in Y and 'Width' in Y: meta['imgshape'] = [int(Y['Height']), int(Y['Width'])]
        if 'Frames' in Y: meta['nframes'] = int(Y['Frames'])
        if 'Window' in Y and window == 1: meta['window'] = int(Y['Window'])
        px = px_nm or Y.get('Pixelsize')
        if px: meta['px_nm'] = float(px)
        return cls(C, meta)

    @classmethod
    def from_thunderstorm(cls, path, px_nm=SP.PX_NM_DEFAULT, window=1, frame_base=1, offset_px=0.5):
        """A ThunderSTORM csv. x = "x [nm]" / px_nm - offset_px (ThunderSTORM: centre of the first pixel = 0.5 px; here 0),
        frame = "frame" - frame_base (ThunderSTORM counts from 1), sx, sy = "uncertainty_x / _y [nm]" if present, else
        "uncertainty [nm]" (or "uncertainty_xy [nm]") for both; "sigma [nm]" -> psf_sx, psf_sy (px); intensity -> photons,
        offset -> bg, chi2 -> chisq; other columns by name."""
        with open(path) as f:
            hdr = [h.strip().strip('"') for h in f.readline().strip().split(',')]
        M = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
        R = {h: M[:, i] for i, h in enumerate(hdr)}
        g = lambda *names: next((R.pop(n) for n in names if n in R), None)
        C = dict(x=g('x [nm]') / px_nm - offset_px, y=g('y [nm]') / px_nm - offset_px, frame=np.round(g('frame')).astype(np.int64) - frame_base)
        ux, uy, u = g('uncertainty_x [nm]'), g('uncertainty_y [nm]'), g('uncertainty [nm]', 'uncertainty_xy [nm]')
        if ux is None and u is not None: ux = uy = u
        if ux is not None: C.update(sx=(ux / px_nm).astype(np.float32), sy=(uy / px_nm).astype(np.float32))
        for src, dst, sc in (('z [nm]', 'z', 1), ('uncertainty_z [nm]', 'sz', 1), ('intensity [photon]', 'photons', 1), ('offset [photon]', 'bg', 1),
                             ('chi2', 'chisq', 1), ('bkgstd [photon]', 'bg_std', 1)):
            v = g(src)
            if v is not None: C[dst] = v.astype(np.float32) * sc
        s = g('sigma [nm]')
        if s is not None: C['psf_sx'] = C['psf_sy'] = (s / px_nm).astype(np.float32)
        R.pop('id', None)
        for k, v in R.items():
            k = k.split(' [')[0]
            C[k] = v > 0.5 if k in BOOL_COLS else v
        return cls(_group_indexed(C), dict(source=os.path.abspath(path), format='thunderstorm csv', px_nm=float(px_nm), window=int(window)))

    @classmethod
    def load(cls, path, **kw):
        """By kind of path: a run directory (from_run), .npz (v2 table, or the paper's DECODE table), .hdf5 / .h5 (Picasso),
        .npy (preprint code's Dataset file), .csv (ThunderSTORM). Keywords go to that reader (window=6 for the paper's
        SMLM / SIMFLUX tables; DECODE and v2 tables are 6-frame windows by default)."""
        if os.path.isdir(path): return cls.from_run(path, **{'pass_': None, **kw})
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npz':
            w = kw.pop('window', None); L = cls.from_table(path, **kw)
            if w: L.meta['window'] = int(w)
            return L
        if ext in ('.hdf5', '.h5'): return cls.from_picasso(path, **kw)
        if ext == '.npy': return cls.loadNumpy(path, **kw)
        if ext == '.csv': return cls.from_thunderstorm(path, **kw)
        raise ValueError(f'unknown localization file type {ext!r} ({path})')

    # ------------------------------------------------------------------ columns, rows
    def __len__(self):
        return len(self.cols['x'])

    def __getattr__(self, name):
        cols = self.__dict__.get('cols')
        if cols is not None and name in cols: return cols[name]
        raise AttributeError(f'{type(self).__name__} has no attribute or column {name!r}')

    def __contains__(self, name):
        return name in self.cols

    def __getitem__(self, idx):
        """L['x'] -> the column; L[mask] / L[slice] / L[indices] -> a new Localizations with those rows (the arrays are
        copies for masks and index arrays, views for slices; metadata and drift are carried along)."""
        if isinstance(idx, str): return self.cols[idx]
        if isinstance(idx, (int, np.integer)): idx = [idx]
        if not isinstance(idx, slice):
            idx = np.asarray(idx)
            if idx.dtype == bool and len(idx) != len(self): raise IndexError(f'mask of {len(idx)} rows for {len(self)} localizations')
        return type(self)({k: v[idx] for k, v in self.cols.items()}, self._meta_copy(), self.drift, self.drift_applied)

    def __setitem__(self, name, value):
        """L['name'] = array: add or replace a column (one value per row)."""
        if not isinstance(name, str): raise TypeError('only columns can be assigned: L[name] = array')
        v = np.asarray(value)
        if v.ndim == 0 or len(v) != len(self): raise ValueError(f'column {name!r}: shape {v.shape}, expected {len(self)} rows')
        self.cols[name] = v

    def _meta_copy(self):
        return dict(self.meta)

    def copy(self):
        """A deep copy (columns, metadata, drift)."""
        return type(self)({k: v.copy() for k, v in self.cols.items()}, json.loads(json.dumps(self.meta, default=_jsonable)),
                          None if self.drift is None else self.drift.copy(), self.drift_applied)

    @property
    def columns(self):
        """Column names."""
        return list(self.cols)

    @property
    def px_nm(self):
        """Camera pixel size in the sample (nm)."""
        return float(self.meta['px_nm'])

    @property
    def window(self):
        """Frames per localization (6 for v2: one pattern cycle)."""
        return int(self.meta.get('window', 1))

    @property
    def t_offset(self):
        """Frames from `frame` (window start) to the time of the localization (window centre) = window // 2."""
        return self.window // 2

    @property
    def nframes(self):
        """Number of movie frames: the recorded count (metadata) if known, else last window start + window."""
        if self.meta.get('nframes'): return int(self.meta['nframes'])
        return int(self.cols['frame'].max()) + self.window if len(self) else 0

    numFrames = nframes

    @property
    def background(self):
        """The column bg (the preprint class's name for it)."""
        return self.cols['bg']

    @property
    def dims(self):
        """2, or 3 when there is a z column."""
        return 3 if 'z' in self.cols else 2

    @property
    def imgshape(self):
        """[height, width] of the camera image (px) if known, else None."""
        return self.meta.get('imgshape')

    def addFields(self, **fields):
        """Add (or replace) per-localization fields: ds.addFields(name=array, ...), one value (or one row of a 2D array) per
        localization. They are columns like the others: carried through indexing, merge, save and load. Returns self."""
        for k, v in fields.items(): self[k] = v
        return self

    @property
    def fields(self):
        """Column names (alias of columns)."""
        return list(self.cols)

    @property
    def pos(self):
        """[n, 2] (or [n, 3] with z) positions; x, y in camera pixels."""
        return np.stack([self.cols[k] for k in ('x', 'y', 'z') if k in self.cols], 1)

    @property
    def pos_nm(self):
        """[n, 2] x, y in nm (x * px_nm)."""
        return np.stack([self.cols['x'], self.cols['y']], 1).astype(np.float64) * self.px_nm

    @property
    def crlb(self):
        """[n, 2] sx, sy (camera pixels): the per-axis localization precision (the preprint class's crlb.pos)."""
        return np.stack([self.cols['sx'], self.cols['sy']], 1)

    @property
    def precision_nm(self):
        """max(sx, sy) * px_nm per row (nm): the quantity the main selection thresholds."""
        return SP.precision_nm(self.cols, self.px_nm)

    def table(self):
        """The v2 table as a dict: the columns plus the scalars px_nm, u_max, sigma_max (when known), i.e. what
        simcode.strict.select, simcode.util.drift.drift_of_table and the other table readers take."""
        T = dict(self.cols); T['px_nm'] = np.array(self.px_nm)
        for k in ('u_max', 'sigma_max'):
            if self.meta.get(k) is not None: T[k] = np.array(float(self.meta[k]))
        return T

    def indices_per_frame(self):
        """List of length nframes: the row indices whose `frame` is that frame (empty arrays where there are none)."""
        f = self.cols['frame'].astype(np.int64); n = max(self.nframes, int(f.max()) + 1 if len(f) else 0)
        o = np.argsort(f, kind='stable'); cnt = np.bincount(f, minlength=n)
        return np.split(o, np.cumsum(cnt)[:-1]) if n else []

    indicesPerFrame = indices_per_frame

    # ------------------------------------------------------------------ selection, filters, crop, merge
    def mask(self, rule='main', sigma_max=None, u_max=None):
        """Row mask of a named selection = simcode.strict.select on this table. 'main' (the paper's): rows with a rank
        (strict_u finite: p >= 0.63, inside the crop) and max(sx, sy) * px_nm <= sigma_max nm (None: the value stored with the
        table, else 12); 'u': strict_u <= u_max (None: stored, else 0.45); 'strict', 'sel', 'good': that column; 'all'."""
        return SP.select(self.table(), rule, u_max=u_max, sigma_max=sigma_max)

    def select(self, rule='main', sigma_max=None, u_max=None):
        """The rows of a named selection as a new object: self[self.mask(rule, ...)]. The rule is recorded in meta['selection']."""
        L = self[self.mask(rule, sigma_max=sigma_max, u_max=u_max)]
        L.meta['selection'] = dict(rule=rule, **({'sigma_max': sigma_max} if sigma_max is not None else {}), **({'u_max': u_max} if u_max is not None else {}))
        return L

    def filter(self, mask):
        """self[mask] (the preprint class's filter; not in place)."""
        return self[mask]

    def filter_finite(self):
        """Rows with finite x, y (and sx, sy when present)."""
        m = np.ones(len(self), bool)
        for k in ('x', 'y', 'sx', 'sy'):
            if k in self.cols: m &= np.isfinite(self.cols[k])
        return self[m]

    def filter_photons(self, min_photons=None, max_photons=None):
        """Rows with min_photons <= photons <= max_photons (None: no bound)."""
        p = self.cols['photons']; m = np.ones(len(self), bool)
        if min_photons is not None: m &= p >= min_photons
        if max_photons is not None: m &= p <= max_photons
        return self[m]

    def filter_sigma(self, max_nm=None, max_px=None):
        """Rows with max(sx, sy) <= max_nm (nm) and / or < max_px (camera pixels; strict inequality, as the preprint class's
        crlb_filter). No rank / crop condition: for the paper's selection use select('main')."""
        s = np.maximum(self.cols['sx'], self.cols['sy']).astype(np.float64); m = np.ones(len(self), bool)
        if max_nm is not None: m &= s * self.px_nm <= max_nm
        if max_px is not None: m &= s < max_px
        return self[m]

    def crlb_filter(self, max_px, min_photons=None):
        """The preprint class's crlb_filter: max(sx, sy) < max_px (camera pixels), optionally photons > min_photons. Not in place."""
        L = self.filter_sigma(max_px=max_px)
        return L if min_photons is None else L[L.cols['photons'] > min_photons]

    def filter_frames(self, start=0, stop=None, whole_windows=True):
        """Rows of the frames [start, stop): window start >= start and (whole_windows) window end <= stop, else start < stop."""
        f = self.cols['frame']; m = f >= start
        if stop is not None: m &= (f + (self.window if whole_windows else 1)) <= stop
        return self[m]

    def filter_region(self, x0, y0, x1, y1):
        """Rows with x0 <= x < x1 and y0 <= y < y1 (camera pixels)."""
        x, y = self.cols['x'], self.cols['y']
        return self[(x >= x0) & (x < x1) & (y >= y0) & (y < y1)]

    def crop(self, x0, y0, x1=None, y1=None, size=None, shift=False):
        """Rows inside [x0, x1) x [y0, y1) camera pixels (or size: a square x0 .. x0 + size, the (x0, y0, size) convention
        of the pipeline crop). The region is recorded in meta['region']. shift=True also moves the origin to (x0, y0)
        (the preprint class's crop); then meta['origin'] holds the total shift and the per-tile columns (strict_u) keep
        the values they had in sensor coordinates."""
        if size is not None: x1, y1 = x0 + size, y0 + size
        L = self.filter_region(x0, y0, x1, y1); L.meta['region'] = [float(x0), float(y0), float(x1), float(y1)]
        if shift:
            L.cols['x'] = L.cols['x'] - x0; L.cols['y'] = L.cols['y'] - y0
            o = L.meta.get('origin', [0.0, 0.0]); L.meta['origin'] = [o[0] + float(x0), o[1] + float(y0)]
            L.meta['region'] = [0.0, 0.0, float(x1 - x0), float(y1 - y0)]
        return L

    def crop_from_edge(self, dist):
        """Rows at least dist camera pixels inside the image (needs meta['imgshape']); the preprint class's cropFromEdge
        without the origin shift."""
        h, w = self.meta['imgshape']
        return self.filter_region(dist, dist, w - 1 - dist, h - 1 - dist)

    cropFromEdge = crop_from_edge

    def autocrop(self):
        """Crop to the bounding box of the localizations (floor / ceil of the extremes) and move the origin there."""
        if not len(self): return self[:]
        x, y = self.cols['x'], self.cols['y']
        return self.crop(np.floor(x.min()), np.floor(y.min()), np.floor(x.max()) + 1, np.floor(y.max()) + 1, shift=True)

    @classmethod
    def concatenate(cls, items, set_group=False):
        """Rows of several objects in one (the preprint class's merge): the columns they all have, metadata of the first
        (nframes = the largest). set_group: a column `group` = index of the source object. All must have the same drift
        state; the drift array is kept only when all share the same one."""
        items = list(items)
        if not items: raise ValueError('nothing to concatenate')
        if len({L.drift_applied for L in items}) > 1: raise ValueError('some objects are drift corrected and some are not')
        keys = [k for k in items[0].cols if all(k in L.cols for L in items)]
        C = {k: np.concatenate([L.cols[k] for L in items]) for k in keys}
        if set_group: C['group'] = np.concatenate([np.full(len(L), i, np.int32) for i, L in enumerate(items)])
        meta = items[0]._meta_copy(); d0 = items[0].drift
        same = all(L.drift is d0 or (L.drift is not None and d0 is not None and np.array_equal(L.drift, d0)) for L in items)
        if any(L.meta.get('nframes') for L in items): meta['nframes'] = max(L.nframes for L in items)
        if not same and items[0].drift_applied: meta['drift_note'] = 'concatenated from objects with different drifts (already subtracted)'
        return cls(C, meta, d0 if same else None, items[0].drift_applied)

    merge = concatenate

    def __add__(self, other):
        """a + b = Localizations.concatenate([a, b])."""
        return type(self).concatenate([self, other])

    # ------------------------------------------------------------------ drift
    def drift_at(self, frame=None):
        """Drift (px) [n, 2] at the time of each row: drift[frame + window // 2], clipped to the drift's frame range."""
        if self.drift is None: raise ValueError('no drift: estimate_drift() or set_drift() first')
        f = self.cols['frame'] if frame is None else np.asarray(frame)
        return self.drift[np.clip(f.astype(np.int64) + self.t_offset, 0, len(self.drift) - 1), :2]

    def estimate_drift(self, nframes=None, rows=None, apply=False, device=None):
        """Lateral drift with DME v2 (simcode.util.drift; needs the `dme` package and a GPU) -> [F, 2] camera pixels, stored
        in self.drift (details in meta['drift_info']). Rows used: `rows` (mask) if given, else the good and disjoint windows
        when the table has those columns, else all (drift_of_table's rule); always only rows with a finite position and
        0 < sigma < 1 px. Time of a row = frame + window // 2. nframes: default last window start + 6 (drift_of_table).
        Estimate on the full table, then select: the drift is carried to every object indexed from this one.
        apply=True also subtracts it (apply_drift)."""
        if self.drift_applied: raise ValueError('positions are already drift corrected: undo_drift() first')
        from simcode.util import drift as DR
        if rows is None:
            d, info = DR.drift_of_table(self.table(), nframes, return_info=True, t_offset=self.t_offset, device=device)
        else:
            F = int(nframes) if nframes else int(self.cols['frame'].max()) + 6
            c = self.cols; m = np.asarray(rows)
            pos, fr, crlb = DR.drift_rows(c['x'][m], c['y'][m], c['frame'][m], c['sx'][m], c['sy'][m], F, self.t_offset)
            d, i = DR.dme_v2(pos, fr, crlb, F, return_info=True, device=device)
            d = np.asarray(d)[:F, :2]; info = dict(n=int(len(pos)), chosen=i['chosen'], version=i['version'])
        self.drift = np.asarray(d); self.meta['drift_info'] = dict(method='DME v2', **{k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()})
        if apply: self.apply_drift()
        return self.drift

    def set_drift(self, drift, info=None):
        """Store a drift [F, 2] (camera pixels per movie frame; e.g. a saved estimate) without applying it. Returns self."""
        if self.drift_applied: raise ValueError('positions are already drift corrected: undo_drift() first')
        self.drift = np.asarray(drift)
        if info is not None: self.meta['drift_info'] = info
        return self

    def apply_drift(self, drift=None):
        """Subtract the drift from x, y in this object: x -= drift[frame + window // 2] (drift None: the stored one). The
        column arrays are replaced, not modified, so objects that share them are unaffected. Returns self."""
        if self.drift_applied: raise ValueError('positions are already drift corrected')
        if drift is not None: self.drift = np.asarray(drift)
        d = self.drift_at()
        self.cols['x'] = self.cols['x'] - d[:, 0].astype(self.cols['x'].dtype); self.cols['y'] = self.cols['y'] - d[:, 1].astype(self.cols['y'].dtype)
        self.drift_applied = True
        return self

    def undo_drift(self):
        """Add the applied drift back to x, y (the drift stays stored). Returns self."""
        if not self.drift_applied: raise ValueError('no drift correction has been applied')
        d = self.drift_at()
        self.cols['x'] = self.cols['x'] + d[:, 0].astype(self.cols['x'].dtype); self.cols['y'] = self.cols['y'] + d[:, 1].astype(self.cols['y'].dtype)
        self.drift_applied = False
        return self

    applyDrift, undoDrift, estimateDrift = apply_drift, undo_drift, estimate_drift

    @property
    def is_drift_corrected(self):
        """Whether x, y have the stored drift subtracted."""
        return self.drift_applied

    isDriftCorrected = is_drift_corrected

    # ------------------------------------------------------------------ render, FRC
    def region(self):
        """Default render region (x0, y0, x1, y1) camera pixels: meta['region'] (set by crop) if present, else the whole
        image (-0.5 .. size - 0.5: an integer coordinate is a pixel centre) if the image size is known, else the bounding box."""
        if self.meta.get('region'): return tuple(self.meta['region'])
        if self.meta.get('imgshape') and not self.meta.get('origin'):
            h, w = self.meta['imgshape']; return (-0.5, -0.5, w - 0.5, h - 0.5)
        x, y = self.cols['x'], self.cols['y']
        return (float(np.floor(x.min())), float(np.floor(y.min())), float(np.ceil(x.max())), float(np.ceil(y.max())))

    def render(self, pixel_nm=10.0, region=None, sigma_nm=0.0, weights=None, return_extent=False):
        """2D histogram image [ny, nx] float32 (row = y, column = x) with pixels of pixel_nm nm. region = (x0, y0, x1, y1) in
        camera pixels (default: region()); sigma_nm > 0: Gaussian blur of that width (scipy); weights: a column name or an
        array (default: counts). return_extent: also (x0, x1, y0, y1) in camera pixels of the image edges (for imshow with
        origin='lower', or pass extent as [x0, x1, y1, y0] with the default origin)."""
        x0, y0, x1, y1 = self.region() if region is None else region
        s = self.px_nm / pixel_nm; nx = max(int(np.ceil((x1 - x0) * s - 1e-9)), 1); ny = max(int(np.ceil((y1 - y0) * s - 1e-9)), 1)
        w = self.cols[weights] if isinstance(weights, str) else weights
        img = np.histogram2d(self.cols['y'], self.cols['x'], bins=(ny, nx), range=[[y0, y0 + ny / s], [x0, x0 + nx / s]], weights=w)[0].astype(np.float32)
        if sigma_nm > 0:
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, sigma_nm / pixel_nm)
        return (img, (x0, x0 + nx / s, y0, y0 + ny / s)) if return_extent else img

    def frc(self, blk=300, px=5.0, px_nm=None, return_curve=False):
        """Fourier ring correlation resolution (nm) between the rows of the even and the odd blocks of blk frames (half =
        frame // blk % 2), with the one implementation simcode.evalproto.frc.frc: px-nm histogram renders, 5-ring smoothed
        curve, 1/7 threshold (nan if the curve never drops below it). Uses the current x, y: drift-correct first.
        px_nm None: this table's pixel size. return_curve: (resolution, frequencies 1/nm, curve)."""
        from simcode.evalproto.frc import frc
        x, y, f = self.cols['x'], self.cols['y'], self.cols['frame']
        h = (f // blk) % 2 == 0
        return frc(x[h], y[h], x[~h], y[~h], px=px, px_nm=self.px_nm if px_nm is None else px_nm, return_curve=return_curve)

    # ------------------------------------------------------------------ summary
    def info(self):
        """Summary as a dict: counts, frames, pixel size, columns, model, drift state, and when the columns exist the median
        precision (nm), median photons and the row counts of the selections."""
        c = self.cols; n = len(self)
        R = dict(n=n, nframes=self.nframes, window=self.window, px_nm=self.px_nm, columns=self.columns)
        for k in ('imgshape', 'crop', 'region', 'model', 'source', 'format', 'selection', 'sigma_max', 'u_max'):
            if self.meta.get(k) is not None: R[k] = self.meta[k]
        R['drift'] = 'none' if self.drift is None else ('applied' if self.drift_applied else 'stored, not applied')
        if self.drift is not None: R['drift_ptp_nm'] = (np.ptp(self.drift[:, :2], 0) * self.px_nm).round(1).tolist()
        if n and 'sx' in c and 'sy' in c: R['median_precision_nm'] = float(np.nanmedian(self.precision_nm))
        if n and 'photons' in c: R['median_photons'] = float(np.nanmedian(c['photons']))
        if 'strict_u' in c: R['n_main'] = int(self.mask('main').sum()); R['n_u'] = int(self.mask('u').sum())
        for k in ('good', 'sel', 'strict'):
            if k in c: R['n_' + k] = int(c[k].sum())
        return R

    def __repr__(self):
        I = self.info(); s = f'Dataset: {I["n"]} rows, {I["nframes"]} frames (window {I["window"]}), px {I["px_nm"]:g} nm'
        if 'model' in I: s += f', model {I["model"]}'
        if 'median_precision_nm' in I: s += f', median precision {I["median_precision_nm"]:.1f} nm'
        if 'median_photons' in I: s += f', median photons {I["median_photons"]:.0f}'
        if 'n_main' in I: s += f', main selection {I["n_main"]}'
        if 'selection' in I: s += f', selection {I["selection"].get("rule")}'
        return s + f', drift {I["drift"]}, columns {" ".join(I["columns"])}'

    # ------------------------------------------------------------------ save
    def save(self, path, **kw):
        """By extension: .npz (save_npz: v2 table, lossless), .npy (saveNumpy: the preprint code's layout), .hdf5 / .h5
        (saveHDF5: Picasso), .csv (saveCSVThunderStorm). Returns the path."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npz': return self.save_npz(path, **kw)
        if ext == '.npy': return self.saveNumpy(path, **kw)
        if ext in ('.hdf5', '.h5'): return self.save_picasso(path, **kw)
        if ext == '.csv': return self.save_thunderstorm(path, **kw)
        raise ValueError(f'unknown localization file type {ext!r} ({path})')

    def save_npz(self, path):
        """The v2 table format, lossless: every column with its dtype, the scalars px_nm / u_max / sigma_max as
        simcode.strict.write_table, plus the metadata (json, key locs_meta) and the drift (locs_drift, locs_drift_applied).
        Existing table readers (np.load(...)['x'], simcode.strict.select) read it unchanged. If the drift is applied the
        stored x, y are the corrected ones."""
        from simcode.fileio.work import atomic_savez
        A = self.table()
        for k in (META_KEY, DRIFT_KEY, APPLIED_KEY):
            if k in A: raise ValueError(f'column name {k!r} is reserved')
        meta = {k: v for k, v in self.meta.items() if k != 'px_nm'}
        A[META_KEY] = np.array(json.dumps(meta, default=_jsonable)); A[APPLIED_KEY] = np.array(self.drift_applied)
        if self.drift is not None: A[DRIFT_KEY] = self.drift
        return atomic_savez(path, **A)

    def saveNumpy(self, path):
        """.npz: save_npz. .npy: the preprint code's Dataset numpy layout (np.save of the image shape, then of one structured
        array with roi_id, frame i4, estim (pos [2 or 3], photons, bg f4, ibg [window, 2] when there are per-frame
        intensities I: I and I_bg), crlb (pos = sx, sy; the rest 0), chisq, group, roipos, and every other column as an
        extra field with its own dtype), so that code's loader reads it. float32 positions; metadata and drift are not
        stored (image shape [0, 0] if unknown). Returns the path."""
        if path.lower().endswith('.npz'): return self.save_npz(path)
        c = self.cols; n = len(self); D = self.dims
        est = [('pos', '<f4', (D,)), ('photons', '<f4'), ('bg', '<f4')]
        if 'psf_sx' in c: est.append(('sigma', '<f4', (2,)))
        if 'I' in c: est.append(('ibg', '<f4', (c['I'].shape[1], 2)))
        used = {'x', 'y', 'z', 'sx', 'sy', 'sz', 'frame', 'photons', 'bg', 'I', 'I_bg', 'psf_sx', 'psf_sy', 'roi_id', 'chisq', 'group', 'roipos'}
        extra = [(k, v.dtype.str) + ((v.shape[1:],) if v.ndim > 1 else ()) for k, v in c.items() if k not in used]
        A = np.zeros(n, np.dtype([('roi_id', '<i4'), ('frame', '<i4'), ('estim', est), ('crlb', est), ('chisq', '<f4'), ('group', '<i4'),
                                   ('roipos', '<i4', (2,))] + extra))
        A['frame'] = c['frame']; e, cr = A['estim'], A['crlb']
        for i, (k, sk) in enumerate((('x', 'sx'), ('y', 'sy'), ('z', 'sz'))[:D]):
            e['pos'][:, i] = c[k]
            if sk in c: cr['pos'][:, i] = c[sk]
        for k in ('photons', 'bg'):
            if k in c: e[k] = c[k]
        if 'psf_sx' in c: e['sigma'][:, 0] = c['psf_sx']; e['sigma'][:, 1] = c.get('psf_sy', c['psf_sx'])
        if 'I' in c:
            e['ibg'][..., 0] = c['I']
            if 'I_bg' in c: e['ibg'][..., 1] = c['I_bg']
        for k in ('roi_id', 'chisq', 'group', 'roipos'):
            if k in c: A[k] = c[k]
        for k, *_ in extra: A[k] = c[k]
        tmp = path + '.tmp'
        with open(tmp, 'wb') as s:
            np.save(s, np.array(self.meta.get('imgshape') or [0, 0])); np.save(s, A)
        os.replace(tmp, path)
        return path

    @classmethod
    def loadNumpy(cls, path, **kw):
        """.npz: from_table (v2 table / a file of save_npz). .npy: from_smlmtorch_npy (the preprint code's file or one
        written by saveNumpy; keywords window, px_nm)."""
        if path.lower().endswith('.npz'): return cls.from_table(path, **kw)
        return cls.from_smlmtorch_npy(path, **kw)

    def _flat_columns(self, skip):
        """1D columns (2D ones split into name0, name1, ...) except `skip`, bools as uint8 -> dict"""
        out = {}
        for k, v in self.cols.items():
            if k in skip: continue
            if v.dtype == bool: v = v.astype(np.uint8)
            if v.ndim == 1: out[k] = v
            elif v.ndim == 2:
                for i in range(v.shape[1]): out[f'{k}{i}'] = v[:, i]
        return out

    def _psf_sigma(self, psf_sigma_px):
        c = self.cols; n = len(self)
        s = psf_sigma_px or self.meta.get('psf_sigma_px') or PSF_SIGMA_PX
        return (c['psf_sx'] if 'psf_sx' in c and psf_sigma_px is None else np.full(n, s, np.float32),
                c['psf_sy'] if 'psf_sy' in c and psf_sigma_px is None else np.full(n, s, np.float32))

    def save_picasso(self, path, extra=True, psf_sigma_px=None):
        """Picasso-compatible hdf5 (dataset 'locs') + the .yaml Picasso needs next to it. Fields: frame u4; x, y, photons,
        sx, sy, bg, lpx, lpy, ellipticity, net_gradient f4 (the fields of Picasso Localize; the same names and dtypes as the
        paper's sf.hdf5), z / lpz if there is a z. lpx, lpy = this table's sx, sy (localization precision, camera pixels);
        Picasso's sx, sy (PSF width) = the columns psf_sx, psf_sy if present, else the nominal psf_sigma_px (default 1.3, not
        measured); bg = 0 if the table has none; ellipticity, net_gradient = 0. x, y are the current ones (drift-corrected
        if applied), in this class's convention (integer = pixel centre), as in the paper's tables. extra: also every other
        column under its own name (bool as uint8, their names in the attribute bool_columns of 'locs'; [n, k] columns as
        name0 .. name{k-1}); from_picasso reads them back with their dtypes.
        The yaml also records Pixelsize (nm) and Window."""
        import h5py
        c = self.cols; n = len(self); f4 = np.float32
        psx, psy = self._psf_sigma(psf_sigma_px)
        z0 = np.zeros(n, f4)
        F = [('frame', '<u4', c['frame']), ('x', '<f4', c['x']), ('y', '<f4', c['y']), ('photons', '<f4', c.get('photons', z0)),
             ('sx', '<f4', psx), ('sy', '<f4', psy), ('bg', '<f4', c.get('bg', z0)), ('lpx', '<f4', c.get('sx', z0)),
             ('lpy', '<f4', c.get('sy', z0)), ('ellipticity', '<f4', z0), ('net_gradient', '<f4', z0)]
        if 'z' in c: F += [('z', '<f4', c['z'])] + ([('lpz', '<f4', c['sz'])] if 'sz' in c else [])
        if extra:
            used = {'frame', 'x', 'y', 'photons', 'bg', 'sx', 'sy', 'z', 'sz', 'psf_sx', 'psf_sy'}; names = {f[0] for f in F}
            for k, v in self._flat_columns(used).items():
                if k in names: raise ValueError(f'column {k!r} clashes with a Picasso field')
                F.append((k, v.dtype.str, v))
        A = np.zeros(n, dtype=[(k, t) for k, t, _ in F])
        for k, _, v in F: A[k] = v
        tmp = path + '.tmp'
        with h5py.File(tmp, 'w') as f:
            f.create_dataset('locs', data=A).attrs['bool_columns'] = json.dumps([k for k, v in c.items() if v.dtype == bool])
        os.replace(tmp, path)
        h, w = self.meta.get('imgshape') or (int(np.ceil(c['y'].max())) + 1 if n else 0, int(np.ceil(c['x'].max())) + 1 if n else 0)
        Y = {'Byte Order': '<', 'Camera': 'unknown', 'Data Type': 'uint16', 'File': os.path.abspath(path), 'Frames': self.nframes,
             'Height': int(h), 'Width': int(w), 'Pixelsize': self.px_nm, 'Window': self.window, 'Generated by': 'simcode.dataset',
             'Drift corrected': bool(self.drift_applied)}
        with open(os.path.splitext(path)[0] + '.yaml', 'w') as f:
            f.write(''.join(f'{k}: {json.dumps(v) if isinstance(v, (str, bool)) else v}\n' for k, v in Y.items()))
        return path

    def save_thunderstorm(self, path, extra=True, psf_sigma_px=None, frame_base=1, offset_px=0.5):
        """ThunderSTORM csv: "id","frame","x [nm]","y [nm]",("z [nm]",)"sigma [nm]","intensity [photon]","offset [photon]",
        "uncertainty [nm]","uncertainty_x [nm]","uncertainty_y [nm]". x [nm] = (x + offset_px) * px_nm (ThunderSTORM: centre of
        the first pixel = 0.5 px), frame = frame + frame_base (ThunderSTORM counts from 1), uncertainty = sqrt((sx^2 + sy^2) / 2)
        in nm, sigma = PSF width (psf_sx / psf_sy mean, else the nominal psf_sigma_px, default 1.3 px; not measured), offset =
        bg (0 if none). extra: the other columns under their names (bool as 0 / 1, [n, k] columns as name0 ..). Positions are
        written with 1e-4 nm resolution."""
        c = self.cols; n = len(self); px = self.px_nm; z0 = np.zeros(n)
        psx, psy = self._psf_sigma(psf_sigma_px)
        sx = c['sx'].astype(np.float64) if 'sx' in c else z0; sy = c['sy'].astype(np.float64) if 'sy' in c else z0
        F = [('id', np.arange(1, n + 1), '%d'), ('frame', c['frame'].astype(np.int64) + frame_base, '%d'),
             ('x [nm]', (c['x'].astype(np.float64) + offset_px) * px, '%.4f'), ('y [nm]', (c['y'].astype(np.float64) + offset_px) * px, '%.4f')]
        if 'z' in c: F.append(('z [nm]', c['z'], '%.4f'))
        F += [('sigma [nm]', 0.5 * (psx + psy).astype(np.float64) * px, '%.4f'), ('intensity [photon]', c.get('photons', z0), '%.8g'),
              ('offset [photon]', c.get('bg', z0), '%.8g'), ('uncertainty [nm]', np.sqrt(0.5 * (sx ** 2 + sy ** 2)) * px, '%.5f'),
              ('uncertainty_x [nm]', sx * px, '%.5f'), ('uncertainty_y [nm]', sy * px, '%.5f')]
        if 'sz' in c: F.append(('uncertainty_z [nm]', c['sz'], '%.5f'))
        if extra:
            used = {'frame', 'x', 'y', 'photons', 'bg', 'sx', 'sy', 'z', 'sz', 'psf_sx', 'psf_sy'}
            for k, v in self._flat_columns(used).items(): F.append((k, v, '%d' if v.dtype.kind in 'iub' else '%.9g'))
        tmp = path + '.tmp'
        np.savetxt(tmp, np.column_stack([np.asarray(v, np.float64) for _, v, _ in F]) if n else np.zeros((0, len(F))), delimiter=',',
                   header=','.join(f'"{k}"' for k, _, _ in F), comments='', fmt=[f for _, _, f in F])
        os.replace(tmp, path)
        return path


def _read_picasso_yaml(path):
    """the top-level `key: value` lines of a Picasso .yaml (first document wins) -> dict; {} if there is no file"""
    Y = {}
    try:
        with open(path) as f:
            for line in f:
                if line[:1] in (' ', '\t', '-', '#', '\n') or ':' not in line: continue
                k, v = line.split(':', 1); v = v.strip()
                if k in Y or not v: continue
                try: Y[k] = json.loads(v)
                except ValueError: Y[k] = v.strip('\'"')
    except OSError:
        pass
    return Y


Dataset.saveHDF5 = Dataset.save_picasso
Dataset.loadHDF5 = Dataset.from_picasso
Dataset.saveCSVThunderStorm = Dataset.save_thunderstorm
Dataset.loadCSVThunderStorm = Dataset.from_thunderstorm
Localizations = Dataset


def load(path, **kw):
    """Dataset.load(path, **kw)"""
    return Dataset.load(path, **kw)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2: raise SystemExit('usage: python -m simcode.dataset <run dir | table file> [output file (.npz / .hdf5 / .csv)] [selection rule]')
    L = load(sys.argv[1])
    if len(sys.argv) > 3: L = L.select(sys.argv[3])
    print(L)
    if len(sys.argv) > 2: print('written', L.save(sys.argv[2]))
