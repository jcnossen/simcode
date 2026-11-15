"""The one place where SIMCODE uses torch_psf (github.com/jcnossen/torch-psf; pinned in pyproject.toml). Every other module
imports the PSF models, the PSF / calibration file reader, the fits, the movie readers and the localization Dataset from
here, so a change of the torch_psf API is a change of this file only.

From torch_psf (names as there): PSF (base class; PSF.load(path | dict, roisize, device, angle, oversample): calibration
json, torch-psf PSF file or psf.zip -> model, angle = per-pattern-angle PSF; psf.to_spline), VectorialPSF, CubicSplinePSF
(knots= / knot_u0_px / knot_z0_um wrap a stored knot array), GaussianPSF, AstigmaticGaussianPSF, FourPiPSF,
FourPiSplinePSF, Optics, Camera, ZStackEstimator (bead z-stacks), Dataset (localizations: torch_psf.dataset; SIMCODE's
layer is simcode/dataset.py), TiffMovie, ArrayMovie, open_movie, solve_linear (closed-form I / bg of the variable
projection; torch_psf.fit), Tracks, extract_rois, fit_tracks (in-situ PSF fit on tracks; torch_psf.estimation),
spot_deviance, filter_frames, photosel_tensor, gauss_blur_os, dipole_tensor, noll_to_nm, emitter_height_pos_fn,
PSFZip / save_zip / is_psf_zip / convert_calibration (the psf.zip format, torch-psf docs/PSF_FORMAT.md).

SIMCODE-specific:

  has_angle_psf(c), angle_kwargs(c, dev)   per-pattern-angle PSF terms of a calibration dict (SIM illumination angle =
                                           step // 3 = frame % 6 // 3): 'psf_dipole_M_angle' [2][3][3], 'psf_blur_angle_px2'
                                           [2][3]. PSF.load(path, angle=0 / 1 / None) gives that angle's model (None = the
                                           angle average: mean dipole tensor, mean blur covariance).
  render(psf, pos, **kw)                   normalized PSF at the model's focus for a VectorialPSF (kw = per-call dipM / blur)
                                           or a GaussianPSF (calib key 'psf_gauss_sigma_um'; z / focus / kw do not apply).
  spline(psf, zmax, roisize)               the PSF spline of the solve and the head template (psf.to_spline): knots 6 x
                                           oversampled, 10 nm in z over emitter heights 0 .. zmax plus 3 planes each side,
                                           offset limit P / 2 + 1 px, at the model's focus (or focus_um).

Calibration files (the one reader of the non-PSF entries of a calibration): a SIMCODE calibration is either the calib json
(PSF keys at the top level, 'camera' {gain_e_per_adu, offset_adu, readnoise_e, maps: npz next to the json}, 'pattern',
tirf_depth_um, emitter_z_*, ...) or a psf.zip (the PSF in channel 0, the camera with its per-pixel maps as arrays in the
zip, the pattern in 'patterns', tirf_depth_um / emitter_z_* in 'acquisition', source / psf_v2 in 'provenance').

  read_calib(path)            -> the calibration dict in the calib-json layout, for both kinds. For a psf.zip it is built
                                 from the zip's blocks (calib_from_psfjson); a zip converted from a calib json
                                 (provenance.converted_from) gives that json's exact values (float64 scalars, the
                                 calibrated camera scalars, keys the zip has no place for) when its blocks agree with
                                 them. camera.maps of a zip with maps = ZIP_MAPS (the maps are inside the zip).
  camera_maps(path, c=None)   -> {'offset_adu': [H, W], 'readnoise_e': [H, W], ...} (float32) or None: the npz of a calib
                                 json or the camera arrays of a psf.zip.
  camera_maps_path(path, c)   -> the file that holds the maps (the npz; the zip itself for a psf.zip), or None.
  calib_gaps(path)            -> what the zip's blocks alone do not reproduce of its converted_from json (tests / report).
  write_calib_zip(json, out)  -> psf.zip of a calib json (torch_psf convert_calibration; report= passed to save_zip).
  fourpi_state(path)          -> the psf4pi state dict of a 4Pi PSF file: json {'psf': state} or a psf.zip (FourPiPSF /
                                 FourPiSplinePSF with its source).
  fourpi_psf(state)           -> torch_psf FourPiPSF of a state whose state() returns that state verbatim (save_zip then
                                 stores the fitted values exactly; see fourpi_psf).
"""
import json, os
import numpy as np
import torch
from torch_psf import (PSF, VectorialPSF, CubicSplinePSF, GaussianPSF, AstigmaticGaussianPSF, FourPiPSF, FourPiSplinePSF,
                       Optics, Camera, ZStackEstimator, Dataset)
from torch_psf.movie import TiffMovie, ArrayMovie, open_movie
from torch_psf.fit import solve_linear
from torch_psf.spots import Tracks, extract_rois
from torch_psf.detect import filter_frames
from torch_psf.estimation import fit_tracks, spot_deviance
from torch_psf.vectorial import photosel_tensor, gauss_blur_os, dipole_tensor
from torch_psf.zernike import noll_to_nm
from torch_psf.spline import emitter_height_pos_fn
from torch_psf import psfzip as _pz
from torch_psf.psfzip import PSFZip, save_zip, is_psf_zip, convert_calibration

ANGLE_KEYS = ('psf_dipole_M_angle', 'psf_blur_angle_px2')
ZIP_MAPS = '<psf.zip camera arrays>'        # camera.maps of a psf.zip calibration: the maps are arrays inside the zip
PROVENANCE_KEYS = ('source', 'psf_v2', 'provenance_note')   # calib-json keys that torch_psf's converter puts in provenance


def has_angle_psf(c):
    """True if the calib dict c has a per-pattern-angle PSF (angle = step // 3 = frame % 6 // 3)"""
    return any(c.get(k) is not None for k in ANGLE_KEYS)


def angle_kwargs(c, dev):
    """per-pattern-angle VectorialPSF.forward kwargs [kw0, kw1] (dipM / blur of that angle) of a calib dict, or None"""
    if not has_angle_psf(c):
        return None
    out = []
    for k in range(2):
        kw = {}
        if c.get('psf_dipole_M_angle') is not None: kw['dipM'] = torch.tensor(c['psf_dipole_M_angle'][k], device=dev, dtype=torch.float32)
        if c.get('psf_blur_angle_px2') is not None: kw['blur'] = torch.tensor(c['psf_blur_angle_px2'][k], device=dev, dtype=torch.float32)
        out.append(kw)
    return out


def render(psf, pos, **kw):
    """normalized PSF [N, P, P] at pos [N, 3] (x, y px rel. to the ROI centre, z um) at psf.focus_um"""
    if isinstance(psf, VectorialPSF):
        return psf(pos, psf.focus_um, **kw)
    return psf.render(pos)


def spline(psf, zmax=0.2, roisize=None, focus_um=None, oversample=6, dz_um=0.01, zpad=3, xy_limit=None, chunk=4):
    """CubicSplinePSF of psf (VectorialPSF or GaussianPSF) for roisize x roisize patches (default psf.roisize) and
    emitter heights 0 .. zmax um at the focus psf.focus_um (or focus_um; vectorial only)"""
    P = psf.roisize if roisize is None else roisize
    kw = dict(oversample=oversample, dz_um=dz_um, zpad=zpad, xy_limit=P / 2 + 1 if xy_limit is None else xy_limit, chunk=chunk)
    if focus_um is not None:
        kw['pos_fn'] = emitter_height_pos_fn(focus_um)
    return psf.to_spline(P, (0.0, zmax), **kw)


# ============================================================================================ calibration files
def _same(a, b):
    """a (from the zip's blocks) agrees with b (the original json): equal up to float32 rounding of the numbers (the zip
    writer stores some values through float32 tensors); dicts: every key of a in b and agreeing"""
    if isinstance(a, dict):
        return isinstance(b, dict) and all(k in b and _same(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or np.float32(a) == np.float32(b)
    return a == b


def calib_from_psfjson(meta):
    """the calibration dict (calib-json layout) from the blocks of a psf.json: the channel-0 model keys (vectorial: the
    calib names pixelsize_um, NA, ..., zernike_*, focus_um, apod, psf_* terms, plus the stored sampling npupil /
    oversample / roisize / z_range_um; gaussian: pixelsize_um and psf_gauss_sigma_um), 'camera' (scalars; maps =
    ZIP_MAPS when the zip holds maps), 'pattern' (patterns without its 'type'), every acquisition key at the top level
    and source / psf_v2 / provenance_note from the provenance."""
    chans = meta['channels']
    if len(chans) != 1 or chans[0]['type'] not in ('vectorial', 'gaussian'):
        raise ValueError(f"SIMCODE calibration: a single-channel vectorial / gaussian psf.zip expected "
                         f"({len(chans)} channels, types {sorted({c['type'] for c in chans})})")
    ch = chans[0]; m = dict(ch['model']); c = {}
    prov = meta.get('provenance') or {}
    for k in PROVENANCE_KEYS:
        if k in prov: c[k] = prov[k]
    opt = dict(meta.get('optics') or {}, **(m.get('optics') or {}))
    for k, ko in (('pixelsize_um', 'pixel_um'), ('wavelength_um', 'wavelength_um'), ('NA', 'NA'), ('n_imm', 'n_imm'), ('n_med', 'n_med')):
        if ko in opt: c[k] = opt[ko]
    if ch['type'] == 'vectorial':
        m.pop('optics', None); c.update(m)
    else:
        g = m['gaussian']; c['pixelsize_um'] = g['pixel_um']
        sx, sy = g['sigma_px']
        if sx != sy: raise ValueError(f'SIMCODE Gaussian calibration: isotropic sigma expected, got {g["sigma_px"]}')
        c['psf_gauss_sigma_um'] = float(sx) * float(g['pixel_um'])
    cam = meta.get('camera')
    if cam:
        c['camera'] = {k: v for k, v in cam.items() if k != 'arrays'}
        if cam.get('arrays'): c['camera']['maps'] = ZIP_MAPS
    pat = meta.get('patterns')
    if pat:
        c['pattern'] = {k: v for k, v in pat.items() if k != 'type'}
    c.update(meta.get('acquisition') or {})
    return c


def _exact(c, orig):
    """the original calib json of a converted zip, with camera.maps -> ZIP_MAPS, when the zip's blocks c agree with it
    (else None). Camera scalars of a camera with maps are not compared: the zip stores the map medians there."""
    cam, ocam = c.get('camera'), orig.get('camera')
    for k, v in c.items():
        if k not in orig: continue
        if k == 'camera':
            if not isinstance(ocam, dict): return None
            for kk, vv in v.items():
                if kk in ('maps',) or kk not in ocam: continue
                if kk in ('offset_adu', 'readnoise_e') and v.get('maps') == ZIP_MAPS: continue
                if not _same(vv, ocam[kk]): return None
        elif k == 'pattern':
            o = dict(orig['pattern']); o.setdefault('note', orig.get('pattern_note'))
            if not _same(v, o): return None
        elif not _same(v, orig[k]):
            return None
    out = json.loads(json.dumps(orig))
    if isinstance(cam, dict) and isinstance(out.get('camera'), dict) and cam.get('maps') == ZIP_MAPS:
        out['camera']['maps'] = ZIP_MAPS
    return out


def _converted_from(meta):
    cf = (meta.get('provenance') or {}).get('converted_from') or {}
    return cf.get('json') if isinstance(cf.get('json'), dict) else None


def read_calib(path, exact=True):
    """SIMCODE calibration dict (calib-json layout) of a calib json or a psf.zip (module doc). exact=False: a zip's blocks
    only (ignore provenance.converted_from)"""
    path = str(path)
    if not is_psf_zip(path):
        with open(path) as f:
            return json.load(f)
    meta, _ = _pz.read_zip(path)
    c = calib_from_psfjson(meta)
    orig = _converted_from(meta) if exact else None
    if orig is not None:
        e = _exact(c, orig)
        if e is not None:
            return e
    return c


def calib_gaps(path):
    """{key: (value from the zip's blocks, value of the converted_from json)} for every top-level / camera / pattern key
    where the blocks alone differ from the original calibration json (keys missing in the blocks: value None); {} for a
    json or a zip without converted_from"""
    if not is_psf_zip(str(path)): return {}
    meta, _ = _pz.read_zip(str(path))
    orig = _converted_from(meta)
    if orig is None: return {}
    c = calib_from_psfjson(meta); out = {}
    for k in sorted(set(orig) | set(c)):
        a, b = c.get(k), orig.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            for kk in sorted(set(a) | set(b)):
                if a.get(kk) != b.get(kk): out[f'{k}.{kk}'] = (a.get(kk), b.get(kk))
        elif a != b:
            out[k] = (a, b)
    return out


def camera_maps_path(path, c=None):
    """file holding the camera maps of a calibration: the npz of a calib json ('camera.maps', relative to the json's
    directory unless absolute), the psf.zip itself, or None when the calibration has no maps"""
    path = str(path)
    if is_psf_zip(path):
        c = c if c is not None else read_calib(path)
        return os.path.abspath(path) if (c.get('camera') or {}).get('maps') else None
    c = c if c is not None else read_calib(path)
    m = (c.get('camera') or {}).get('maps')
    if not m: return None
    return m if os.path.isabs(m) else os.path.join(os.path.dirname(os.path.abspath(path)), m)


def camera_maps(path, c=None):
    """per-pixel camera maps of a calibration: {'offset_adu': [H, W], 'readnoise_e': [H, W], (other npz entries)} or None"""
    path = str(path)
    if is_psf_zip(path):
        meta, arrays = _pz.read_zip(path)
        A = (meta.get('camera') or {}).get('arrays') or {}
        return {k: arrays[m] for k, m in A.items()} or None
    p = camera_maps_path(path, c)
    if p is None: return None
    with np.load(p) as M:
        return {k: M[k] for k in M.files}


def write_calib_zip(json_path, out=None, **kw):
    """psf.zip of a SIMCODE calib json (torch_psf.psfzip.convert_calibration: lossless; the json is kept in
    provenance.converted_from); out default <json without .json>.psf.zip; kw (report=, provenance=, ...) -> save_zip.
    Returns the zip path."""
    out = out or (os.path.splitext(str(json_path))[0] + '.psf.zip')
    convert_calibration(json_path, out, **kw)
    return out


# ============================================================================================ 4Pi PSF files
def fourpi_psf(state, device=None, z_range=(-0.6, 0.6), **kw):
    """torch_psf FourPiPSF of a psf4pi state (bit-identical to simcode.fourpi.psf4pi.psf_from_state). Its state() returns
    the given state verbatim (torch_psf's state() recomputes tau = softmax(log tau) and float32-rounds NA / n, which does
    not reproduce the fitted values exactly), so save_zip / to_spline store the fitted state as it is."""
    p = FourPiPSF.from_state(state, device=device, z_range=z_range, **kw)
    s0 = json.loads(json.dumps(state))
    p.state = lambda: json.loads(json.dumps(s0))
    return p


def fourpi_state(path, root=None):
    """psf4pi state dict of a 4Pi PSF file: a json with key 'psf' (out/4pi/psf4pi/*.json), or a psf.zip of a FourPiPSF
    (top-level source) / FourPiSplinePSF (its source block). path: a file, or a name looked up as
    <root>/psf4pi/<name>.json, then <models>/fourpi/<name>.psf.zip (root = the 4Pi work directory)"""
    path = str(path)
    if not os.path.isfile(path):
        from simcode import config
        cands = ([os.path.join(root, 'psf4pi', path + '.json')] if root else []) + [config.model(f'fourpi/{path}.psf.zip')]
        path = next((c for c in cands if os.path.isfile(c)), cands[0])
    if is_psf_zip(path):
        meta, _ = _pz.read_zip(path)
        src = meta.get('source') or {}
        if src.get('type') != 'fourpi':
            raise ValueError(f'{path}: no fourpi source model')
        return src['model']['fourpi']
    with open(path) as f:
        return json.load(f)['psf']
