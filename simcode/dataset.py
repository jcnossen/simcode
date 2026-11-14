"""
SIMCODE's localization Dataset: torch_psf.Dataset plus SIMCODE's loaders and the paper's selection.

torch_psf.Dataset (torch-psf docs/DATASET_FORMAT.md, docs/API.md) holds the localizations as a dict of CPU tensors and does
everything generic: fields by key or attribute, indexing (masks, slices, indices), filters (filter, filter_finite,
filter_photons, filter_crlb, filter_frames; in-place variants with a trailing _), crop / crop_from_edge, concatenate and +,
drift (set_drift, drift_at, apply_drift, undo_drift, estimate_drift = DME v2), render, frc, info, and the files: save / load
(.hdf5 = the torch-psf format, .csv = ThunderSTORM), save_picasso / load_picasso. This module adds:

  loaders   Dataset.load(path)  a run directory (from_run), .npz (load_npz: a v2 table of simcode.run / simcode.strict, a
                                subset file, a file of the earlier SIMCODE Dataset.save_npz, or the paper's DECODE table:
                                from_decode), .npy (load_npy: the preprint code's Dataset / SFDataset file), else
                                torch_psf's load (.hdf5: this format or Picasso, e.g. the paper's sf.hdf5; .csv: ThunderSTORM)
            from_run(run_dir, pass_=2), load_npz, from_decode, from_smlmtorch(ds), load_npy
            window=<frames>: frames per localization (the n_frames field) for files that do not record it (the paper's
            SMLM / SIMFLUX tables of summed 6-frame windows: window=6)
  select    select(max_precision_nm=12, min_score=0.63, in_region=True) = simcode.strict.select on this table: the paper's
            selection (calibrated precision max(crlb_x, crlb_y) * pixel_nm <= 12 nm, score >= 0.63, inside the run's crop
            meta['crop']; the drift-corrected x_dc, y_dc when the table has them); None switches a condition off.
            mask(...) is the boolean row mask.

    from simcode.dataset import Dataset
    ds = Dataset.load('out/run1')                     # a run directory (locs_p2.npz, else locs_p1.npz) or a table file
    sel = ds.select()                                 # the paper's selection
    ds.estimate_drift(rows=ds.good & ds.disjoint)     # DME v2 (the dme package: simcode.util.drift.dme_package())
    sel = ds.select().apply_drift(ds.drift)
    img = sel.render(pixel_nm=10); res = sel.frc()
    sel.save('run1_sel.hdf5')                         # torch-psf format; sel.save_picasso(...), sel.save('x.csv')

Fields of a v2 table (simcode/strict.py) in this layout (docs/DATASET_FORMAT.md of SIMCODE lists them):
  pos [n, 2] = (x, y) px, crlb [n, 2] = (sx, sy) px (calibrated sigma), frame = window start, n_frames = 6 (window
  length), photons, score = p (detector probability), intensities [n, 6] = I (network per-frame intensities), and the other
  columns under their names (t, q_dof, kin, dnll, x_prior, y_prior, good, sel, disjoint, x_dc, y_dc, strict_u, strict).
  Metadata: pixel_nm (px_nm of the table), crop, sigma_max, info (the table's info json), run_config, model, n_frames,
  image_size, pass, source, format.
Other sources: DECODE phot -> photons, sphot -> crlb_photons, prob -> score, z (nm) kept as the field z; the preprint code's
estim.pos / crlb.pos -> pos / crlb, estim.ibg -> intensities, intensities_bg, chisq, group, roi_id, roipos as fields (a z
column becomes the field z, crlb_z: its own unit, not part of pos).
"""
import json, os
import numpy as np
import torch
from simcode import strict as SP
from simcode.psf import Dataset as _TorchPSFDataset

META_KEY, DRIFT_KEY, APPLIED_KEY = 'locs_meta', 'locs_drift', 'locs_drift_applied'   # keys of the earlier SIMCODE npz save
_RENAME = dict(p='score', I='intensities')                     # v2 table column -> field
_META = dict(px_nm='pixel_nm', imgshape='image_size', nframes='n_frames')   # earlier SIMCODE metadata -> torch_psf names


def _read_json(path):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError):
        return None


class Dataset(_TorchPSFDataset):
    """torch_psf.Dataset with SIMCODE's loaders and select() (module docstring)."""

    # ------------------------------------------------------------------ loaders
    @classmethod
    def _from_columns(cls, C, meta, window, drift=None, drift_applied=False):
        """columns x, y, frame (+ sx, sy, z, sz and others) -> pos / crlb layout"""
        C = dict(C)
        F = dict(pos=np.stack([np.asarray(C.pop('x'), np.float64), np.asarray(C.pop('y'), np.float64)], 1),
                 frame=np.asarray(C.pop('frame')))
        if 'sx' in C and 'sy' in C:
            F['crlb'] = np.stack([C.pop('sx'), C.pop('sy')], 1).astype(np.float32)
        if 'sz' in C: F['crlb_z'] = C.pop('sz')
        F['n_frames'] = np.full(len(F['frame']), int(window), np.int32)
        for k, v in C.items():
            F[_RENAME.get(k, k)] = np.ascontiguousarray(v)
        meta = {_META.get(k, k): v for k, v in meta.items() if k != 'window'}
        meta.setdefault('pixel_nm', SP.PX_NM_DEFAULT)
        return cls(F, meta, drift, drift_applied)

    @classmethod
    def load_npz(cls, path, columns=None, info=True, window=6):
        """A v2 table file (locs_p1.npz / locs_p2.npz / a subset file / a file of the earlier SIMCODE Dataset.save_npz), or the
        paper's DECODE table (from_decode). columns: load only these (x, y, frame always; a 660 MB production table takes a
        few seconds in full). info: also read <table>_info.json, and summary.json / run_config.json of the directory when
        they exist (crop, model, frame count, image size, the run record)."""
        z = np.load(path, allow_pickle=False)
        if 'phot' in z.files and 'photons' not in z.files: return cls.from_decode(path, window)
        n = len(z['x']); C = {}; meta = dict(source=os.path.abspath(path), format='simcode npz')
        want = None if columns is None else set(columns) | {'x', 'y', 'frame'}
        drift = None; applied = False
        for k in z.files:
            if k == META_KEY: meta = {**json.loads(str(z[k])), **meta}; continue
            if k == DRIFT_KEY: drift = z[k]; continue
            if k == APPLIED_KEY: applied = bool(z[k]); continue
            if want is not None and k not in want and k not in ('px_nm', 'sigma_max'): continue
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
                if nf: meta.setdefault('n_frames', int(nf))
                if mv.get('frame_shape'): meta.setdefault('image_size', [int(v) for v in mv['frame_shape']])
                m = R.get('model'); m = m.get('name') if isinstance(m, dict) else m
                if m: meta.setdefault('model', m)
            m = S.get('model')
            if isinstance(m, str): meta.setdefault('model', m)
        return cls._from_columns(C, meta, meta.get('window', window), drift, applied)

    @classmethod
    def from_run(cls, run_dir, pass_=2, table=None, **kw):
        """The localization table of a run directory of simcode.run: pass_ = 2 (locs_p2.npz, the pattern-aware pass, the
        default) or 1 (locs_p1.npz); pass_ None: pass 2 if it exists, else pass 1; table: another base name instead
        (e.g. 'locs_p2_sig12'). meta['pass'] records the pass. Keywords as load_npz (columns=..., info=...). This is what
        the pipeline's dataset_pass1() / dataset_pass2() return; the paper's selection: ds.select()."""
        names = [table] if table else [f'locs_p{int(pass_)}'] if pass_ is not None else ['locs_p2', 'locs_p1']
        for n in names:
            f = os.path.join(run_dir, n if n.endswith('.npz') else n + '.npz')
            if os.path.exists(f):
                L = cls.load_npz(f, **kw); b = os.path.basename(f)
                if b.startswith('locs_p') and b[6:7].isdigit(): L.meta.setdefault('pass', int(b[6]))
                return L
        raise FileNotFoundError(f'no {" / ".join(names)} (.npz) in {run_dir}')

    @classmethod
    def from_decode(cls, path, window=6):
        """A DECODE table as written for the paper (npz with frame, x, y, z, sx, sy, phot, sphot, bg, prob, info): phot ->
        photons, sphot -> crlb_photons, prob -> score; z (nm) and the other columns keep their names."""
        z = np.load(path, allow_pickle=False); n = len(z['x'])
        ren = dict(phot='photons', sphot='crlb_photons', prob='score')
        C = {ren.get(k, k): z[k] for k in z.files if z[k].ndim and len(z[k]) == n}
        meta = dict(source=os.path.abspath(path), format='decode npz')
        if 'info' in z.files and z['info'].ndim == 0: meta['info'] = str(z['info'])
        return cls._from_columns(C, meta, window)

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
        return cls._from_columns({k: np.ascontiguousarray(v) for k, v in C.items()}, meta, window)

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
        """By kind of path: a run directory (from_run), .npz (load_npz: v2 table, earlier SIMCODE save, or the paper's DECODE
        table), .npy (load_npy: the preprint code's file), else torch_psf.Dataset.load (.hdf5 / .h5: torch-psf format or
        Picasso; .csv: ThunderSTORM, needs pixel_nm=). window: frames per localization (the n_frames field) when the file
        does not record it (the paper's SMLM / SIMFLUX tables: 6); keywords go to that reader."""
        path = str(path)
        if os.path.isdir(path): return cls.from_run(path, **{'pass_': None, **kw})
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npz': return cls.load_npz(path, window=window or 6, **kw)
        if ext == '.npy': return cls.load_npy(path, window=window or 1, **kw)
        L = super().load(path, **kw)
        L.meta.setdefault('pixel_nm', SP.PX_NM_DEFAULT)
        if window is not None: L['n_frames'] = torch.full((len(L),), int(window), dtype=torch.int32)
        return L

    # ------------------------------------------------------------------ the paper's selection
    def _table(self):
        """this dataset as the v2 table dict simcode.strict.select reads"""
        F = self.numpy(); p, c = F['pos'], F['crlb']
        T = dict(x=p[:, 0], y=p[:, 1], sx=c[:, 0], sy=c[:, 1], px_nm=self.pixel_nm)
        if 'score' in F: T['p'] = F['score']
        for k in ('x_dc', 'y_dc'):
            if k in F: T[k] = F[k]
        if self.meta.get('crop') is not None: T['crop'] = np.array(self.meta['crop'], np.float64)
        return T

    def mask(self, max_precision_nm=SP.SIGMA_MAX_NM, min_score=SP.P_GOOD, in_region=True):
        """Row mask (bool tensor) of the paper's selection = simcode.strict.select on this table: max(crlb_x, crlb_y) *
        pixel_nm <= max_precision_nm (nm), score >= min_score, inside the run's crop (meta['crop'], else
        simcode.strict.CROP; x_dc, y_dc when the table has them); None switches a condition off."""
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
