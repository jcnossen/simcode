"""The one place where SIMCODE uses torch_psf (github.com/jcnossen/torch-psf; pinned in pyproject.toml). Every other module
imports the PSF models, the PSF / calibration file reader, the fits, the movie readers and the localization Dataset from
here, so a change of the torch_psf API is a change of this file only.

From torch_psf (names as there): PSF (base class; PSF.load(path | dict, roisize, device, angle, oversample): calibration
json, torch-psf PSF file or psf.zip -> model, angle = per-pattern-angle PSF; psf.to_spline), VectorialPSF, CubicSplinePSF
(knots= / knot_u0_px / knot_z0_um wrap a stored knot array), GaussianPSF, AstigmaticGaussianPSF, FourPiPSF,
FourPiSplinePSF, Optics, Camera, ZStackEstimator (bead z-stacks), Dataset (localizations: torch_psf.dataset; SIMCODE's
layer is simcode/dataset.py), TiffMovie, ArrayMovie, open_movie, solve_linear (closed-form I / bg of the variable
projection; torch_psf.fit), Tracks, extract_rois, fit_tracks (in-situ PSF fit on tracks; torch_psf.estimation),
heldout_tracks / paired_nll / split_checkerboard (held-out check of a frozen PSF on track samples, the paired per-sample
NLL difference, the checkerboard split; torch_psf.estimation), spot_deviance, maps_from_movie (per-pixel sCMOS offset / read-noise maps from a sparse movie; torch_psf.camera),
white_photon_transfer (camera gain from the white background noise; torch_psf.camera), fit_gaussian_free_sigma
(spot fit with a free Gaussian width; torch_psf.gaussian), filter_frames, gauss_blur_os, dipole_tensor, emitter_height_pos_fn, save_zip / is_psf_zip /
convert_calibration (the psf.zip format, torch-psf docs/PSF_FORMAT.md).

SIMCODE-specific:

  has_angle_psf(c)                         per-pattern-angle PSF terms of a calibration dict (SIM illumination angle =
                                           step // 3 = frame % 6 // 3): 'psf_dipole_M_angle' [2][3][3], 'psf_blur_angle_px2'
                                           [2][3]. vectorial_psf(path, angle=0 / 1 / None) gives that angle's VectorialPSF (None
                                           = the angle average: mean dipole tensor, mean blur covariance); calib_psf(path,
                                           angle=...) its spline (None = the mean of the angle images). Note: torch_psf
                                           PSF.load(<psf.zip>, angle=k) ignores the angle (the per-angle terms of a calibration
                                           zip are in provenance.unused_model_keys; read_calib restores them).
  calib_psf(path, roisize, angle, device)  THE PSF renderer of SIMCODE and of the baselines (2026-11-19; analysis/baselines
                                           zpsf: SMLM, SIMFLUX, DECODE): the integrated cubic B-spline of the calibration
                                           (torch_psf IntegratedSplinePSF: point samples of the exact vectorial model on
                                           SPLINE's knot grid, the exact pixel integral of the spline; 1.7e-5 of the peak
                                           from the vectorial model at 'exact' pixel integration). A psf.zip written with
                                           it (write_calib_zip) carries the coefficients (per pattern angle + their mean);
                                           any other calibration (calib json, a psf.zip without them) builds them on load
                                           from its vectorial model (integrated_spline; cached per file). angle 0 / 1 =
                                           that pattern angle, None = the mean of the angle images. A Gaussian calibration
                                           ('psf_gauss_sigma_um') gives its GaussianPSF.
  integrated_spline(calib, device)         the IntegratedSplinePSF of a calibration (path or calib dict, e.g. a perturbed
                                           one) built from its vectorial model on the SPLINE grid (all angles).
  vectorial_psf(path, roisize, angle)      the calibration's VectorialPSF (the source model; the PSF fits, the CRLB).

Calibration files (the one reader of the non-PSF entries of a calibration): a SIMCODE calibration is either the calib json
(PSF keys at the top level, 'camera' {gain_e_per_adu, offset_adu, readnoise_e, maps: npz next to the json}, 'pattern',
tirf_depth_um, emitter_z_*, ...) or a psf.zip (the PSF in channel 0, the camera with its per-pixel maps as arrays in the
zip, the pattern in 'patterns', tirf_depth_um / emitter_z_* in 'acquisition', source / psf_v2 in 'provenance').

  read_calib(path)            -> the calibration dict in the calib-json layout, for both kinds. For a psf.zip it is built
                                 from the zip's blocks (calib_from_psfjson: model, optics, camera with its calibrated
                                 scalars and free-form keys, patterns incl. pattern_note, acquisition, provenance incl.
                                 unused_model_keys); a zip converted from a calib json gives that json's values
                                 (torch-psf >= 52c75c5). camera.maps of a zip with maps = ZIP_MAPS (the maps are in the zip).
  camera_maps(path, c=None)   -> {'offset_adu': [H, W], 'readnoise_e': [H, W], ...} (float32) or None: the npz of a calib
                                 json or the camera arrays (+ camera.extra_arrays) of a psf.zip.
  camera_maps_path(path, c)   -> the file that holds the maps (the npz; the zip itself for a psf.zip), or None.
  write_calib_zip(json, out)  -> psf.zip of a calib json (torch_psf convert_calibration; report= passed to save_zip).
  fourpi_state(path)          -> the coherent_psf state dict of a 4Pi PSF file: json {'psf': state} or a psf.zip (FourPiPSF /
                                 FourPiSplinePSF with its source).
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
from torch_psf.estimation import fit_tracks, spot_deviance, heldout_tracks, paired_nll, split_checkerboard
from torch_psf.vectorial import gauss_blur_os, dipole_tensor
from torch_psf.pixint import IntegratedSplinePSF
from torch_psf import psfzip as _pz
from torch_psf.psfzip import save_zip, is_psf_zip, convert_calibration
from torch_psf.camera import maps_from_movie, white_photon_transfer
from torch_psf.gaussian import fit_gaussian_free_sigma

ANGLE_KEYS = ('psf_dipole_M_angle', 'psf_blur_angle_px2')
ZIP_MAPS = '<psf.zip camera arrays>'        # camera.maps of a psf.zip calibration: the maps are arrays inside the zip
PROVENANCE_KEYS = ('source', 'psf_v2', 'provenance_note')   # calib-json keys that torch_psf's converter puts in provenance


def has_angle_psf(c):
    """True if the calib dict c has a per-pattern-angle PSF (angle = step // 3 = frame % 6 // 3)"""
    return any(c.get(k) is not None for k in ANGLE_KEYS)


# ============================================================================================ the PSF renderer
# knot grid of the integrated cubic B-spline of a calibration (IntegratedSplinePSF.from_vectorial): emitter heights 0 - 0.4 um
# (user decision 2026-11-19; 6 extra planes each side absorb the boundary of the B-spline prefilter), 10 nm planes, 4 knots / px
# (3 knots / px: 5.3e-5 of the peak from the exact vectorial model on the v4 PSFs, 4: 1.7e-5; out/onepsf), offsets up to 2 px
# on a 41 x 41 ROI (a smaller ROI supports larger offsets: 15 px at 11 x 11)
SPLINE = dict(z_range=(0.0, 0.4), roisize=41, xy_limit=2.0, oversample=4, dz_um=0.01, zpad=6)
_SPLINES = {}                       # (file, mtime, size, device) -> IntegratedSplinePSF of calib_psf (built on load)


def vectorial_psf(calib, roisize=15, angle=None, device='cuda'):
    """the VectorialPSF of a calibration (path: calib json or psf.zip, with or without the integrated spline; or a calib dict)
    at the calibration's sampling; angle 0 / 1 = that pattern angle's, None = the angle average (torch_psf semantics)"""
    c = calib if isinstance(calib, dict) else read_calib(calib)
    if c.get('psf_gauss_sigma_um') is not None:
        raise ValueError('a Gaussian calibration has no vectorial model')
    if isinstance(calib, dict) or not is_psf_zip(str(calib)):
        return PSF.load(calib, roisize=roisize, device=device, angle=angle)
    return PSF.load(c, roisize=roisize, device=device, angle=angle)          # the zip's vectorial keys (calib layout)


def integrated_spline(calib, device='cuda', **grid):
    """the IntegratedSplinePSF (all pattern angles: coefs_angle when the calibration has per-angle terms) of a calibration (path
    or calib dict) built from its vectorial model on the SPLINE grid (grid: overrides of SPLINE)"""
    g = dict(SPLINE, **grid)
    c = calib if isinstance(calib, dict) else read_calib(calib)
    vecs = ([vectorial_psf(calib, g['roisize'], k, device) for k in range(2)] if has_angle_psf(c)
            else vectorial_psf(calib, g['roisize'], None, device))
    v0 = vecs[0] if isinstance(vecs, list) else vecs
    if v0.pixel_integration != 'spectral':
        import warnings
        warnings.warn(f"{'calibration dict' if isinstance(calib, dict) else os.path.basename(str(calib))}: fitted with pixel_integration "
                      f"{v0.pixel_integration!r}; its PSF spline is the exact pixel integral of the same parameters (up to ~1 % of "
                      f"the peak from the {v0.pixel_integration!r} rendering)", stacklevel=2)
    return IntegratedSplinePSF.from_vectorial(vecs, **g)


def _zip_channel_type(path):
    import zipfile
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read('psf.json').decode('utf-8'))['channels'][0]['type']


def calib_psf(calib, roisize=None, angle=None, device='cuda', z_range=None):
    """THE PSF of a calibration (module doc): IntegratedSplinePSF for roisize x roisize ROIs (default SPLINE roisize) of pattern
    angle `angle` (None = the mean of the angle images); from the psf.zip's integrated_spline channel, else built from the
    vectorial model (integrated_spline, cached per file / device); calib may also be a calib dict (built, not cached).
    z_range: heights (um) the spline must cover (default SPLINE's); one beyond the stored spline's range is built from the
    vectorial model on the union of both ranges (e.g. heights below the coverslip). A Gaussian calibration: its GaussianPSF."""
    c = calib if isinstance(calib, dict) else read_calib(calib)
    P = SPLINE['roisize'] if roisize is None else int(roisize)
    if c.get('psf_gauss_sigma_um') is not None:
        return PSF.load(calib, roisize=P, device=device)
    zr = tuple(SPLINE['z_range']) if z_range is None else (float(z_range[0]), float(z_range[1]))
    covers = lambda r: r[0] <= zr[0] + 1e-9 and r[1] >= zr[1] - 1e-9
    grid = {} if covers(SPLINE['z_range']) else dict(z_range=(min(zr[0], SPLINE['z_range'][0]), max(zr[1], SPLINE['z_range'][1])))
    if isinstance(calib, dict):
        spl = integrated_spline(c, device, **grid)
    else:
        path = os.path.abspath(str(calib)); st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size, str(torch.device(device)), tuple(grid.get('z_range', ())))
        spl = _SPLINES.get(key)
        if spl is None:
            if is_psf_zip(path) and _zip_channel_type(path) == 'integrated_spline':
                spl = PSF.load(path, device=device)
                if not covers(spl.z_range):
                    spl = integrated_spline(path, device, z_range=(min(zr[0], spl.z_range[0]), max(zr[1], spl.z_range[1])))
            else:
                spl = integrated_spline(path, device, **grid)
            _SPLINES[key] = spl
    if spl.n_angles == 0 and has_angle_psf(c):
        raise ValueError('a calibration with per-angle PSF terms but a spline without per-angle coefficients')
    out = spl.for_angle(angle) if angle is not None else spl
    return out if out.roisize == P else out.resized(P)


# ============================================================================================ calibration files
def calib_from_psfjson(meta):
    """the calibration dict (calib-json layout) from the blocks of a psf.json: the channel-0 model keys (vectorial: the
    calib names pixelsize_um, NA, ..., zernike_*, focus_um, apod, psf_* terms, plus the stored sampling npupil /
    oversample / roisize / z_range_um; gaussian: pixelsize_um and psf_gauss_sigma_um), the optics keys (also those a
    Gaussian model does not use), the PSF-model keys the model does not use (provenance.unused_model_keys), 'camera' (the
    camera dict with its calibrated scalars and free-form keys; maps = ZIP_MAPS when the zip holds maps), 'pattern'
    (patterns without 'type' and 'pattern_note') and 'pattern_note', every acquisition key at the top level and source /
    psf_v2 / provenance_note from the provenance. Zips written before torch-psf 52c75c5 (pattern note in
    patterns['note'], camera scalars = map medians) read with these limits."""
    chans = meta['channels']
    if len(chans) != 1 or chans[0]['type'] not in ('vectorial', 'gaussian', 'integrated_spline'):
        raise ValueError(f"SIMCODE calibration: a single-channel vectorial / gaussian / integrated_spline psf.zip expected "
                         f"({len(chans)} channels, types {sorted({c['type'] for c in chans})})")
    ch = chans[0]; c = {}
    if ch['type'] == 'integrated_spline':          # the integrated spline (calib_psf) with the calibration's vectorial model as source
        if (ch.get('source') or {}).get('type') != 'vectorial':
            raise ValueError('SIMCODE calibration: an integrated_spline channel needs its vectorial source')
        ch = dict(ch['source'], name=ch['name'])
    m = dict(ch['model'])
    prov = meta.get('provenance') or {}
    for k in PROVENANCE_KEYS:
        if k in prov: c[k] = prov[k]
    c.update(prov.get('unused_model_keys') or {})
    opt = dict(meta.get('optics') or {}, **(m.get('optics') or {}))
    for k, ko in (('pixelsize_um', 'pixel_um'), ('wavelength_um', 'wavelength_um'), ('NA', 'NA'), ('n_imm', 'n_imm'), ('n_med', 'n_med')):
        if ko in opt: c[k] = opt[ko]
    if ch['type'] == 'vectorial':
        m.pop('optics', None); c.update({k: v for k, v in m.items() if v is not None})    # null = term not used (absent in a json)
    else:
        g = m['gaussian']; c['pixelsize_um'] = g['pixel_um']
        sx, sy = g['sigma_px']
        if sx != sy: raise ValueError(f'SIMCODE Gaussian calibration: isotropic sigma expected, got {g["sigma_px"]}')
        c['psf_gauss_sigma_um'] = float(sx) * float(g['pixel_um'])
    cam = meta.get('camera')
    if cam:
        c['camera'] = {k: v for k, v in cam.items() if k not in ('arrays', 'extra_arrays')}
        if cam.get('arrays'): c['camera']['maps'] = ZIP_MAPS
    pat = meta.get('patterns')
    if pat:
        pat = dict(pat); pat.pop('type', None)
        note = pat.pop('pattern_note', None)
        if note is not None: c['pattern_note'] = note
        c['pattern'] = pat
    c.update(meta.get('acquisition') or {})
    return c


def read_calib(path):
    """SIMCODE calibration dict (calib-json layout) of a calib json or a psf.zip (module doc; a zip: its blocks,
    calib_from_psfjson)"""
    path = str(path)
    if not is_psf_zip(path):
        with open(path) as f:
            return json.load(f)
    meta, _ = _pz.read_zip(path)
    return calib_from_psfjson(meta)


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
    """per-pixel camera maps of a calibration: {'offset_adu': [H, W], 'readnoise_e': [H, W], (other npz entries; a psf.zip:
    camera.extra_arrays)} or None"""
    path = str(path)
    if is_psf_zip(path):
        meta, arrays = _pz.read_zip(path)
        cam = meta.get('camera') or {}
        A = dict(cam.get('extra_arrays') or {}, **(cam.get('arrays') or {}))
        return {k: arrays[m] for k, m in A.items()} or None
    p = camera_maps_path(path, c)
    if p is None: return None
    with np.load(p) as M:
        return {k: M[k] for k in M.files}


def write_calib_zip(json_path, out=None, spline=True, **kw):
    """psf.zip of a SIMCODE calib json (torch_psf.psfzip.convert_calibration: lossless; the json is kept in
    provenance.converted_from); out default <json without .json>.psf.zip; spline (default): channel 0 is the integrated
    spline of the calibration on the SPLINE grid (per pattern angle; the vectorial model as its source: read_calib reads the
    calibration from there), so that calib_psf reads the coefficients instead of building them (a Gaussian calibration
    has none); kw (report=, provenance=, device=, ...) -> convert_calibration / save_zip. Returns the zip path."""
    out = out or (os.path.splitext(str(json_path))[0] + '.psf.zip')
    with open(json_path) as f:
        gauss = json.load(f).get('psf_gauss_sigma_um') is not None
    if spline and not gauss:
        kw['integrated_spline'] = dict(SPLINE)
    convert_calibration(json_path, out, **kw)
    return out


# ============================================================================================ 4Pi PSF files
def fourpi_state(path, root=None):
    """coherent_psf state dict of a 4Pi PSF file: a json with key 'psf' (out/4pi/psf4pi/*.json), or a psf.zip of a FourPiPSF
    (top-level source) / FourPiSplinePSF (its source block; without torch_psf's 'raw' parameter block). path: a file, or a name looked up as
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
        st = dict(src['model']['fourpi']); st.pop('raw', None)        # torch_psf's exact parameters (FourPiPSF.state)
        return st
    with open(path) as f:
        return json.load(f)['psf']
