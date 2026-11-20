"""The one place where SIMCODE imports torch_psf (github.com/jcnossen/torch-psf; pinned in pyproject.toml). Every other module
imports the PSF models, the PSF / calibration file reader, the fits, the movie readers and the localization Dataset from
here, so a renamed or moved torch_psf name is a change of this file only; the methods of these objects that other modules
call are listed in docs/TORCH_PSF.md, with what SIMCODE requires of a new torch-psf pin (the rendered pixels of the final
calibrations must not change) and the migration to torch-psf's unified PSF classes (psf-unify, 2026-11-20).

From torch_psf (names as there): PSF (base class; PSF.load(path | dict, roisize, device, angle, oversample): calibration
json, torch-psf PSF file or psf.zip -> model, angle = per-pattern-angle PSF; psf.to_spline; one render signature
render(pos, grad, *, dipM, phi, xy, focus, angle)), VectorialPSF, SplinePSF (the spline tables: pixel or density values,
Catmull-Rom or cubic B-spline kernel; knots= / knot_u0_px / knot_z0_um wrap a stored knot array), to_spline (a model or a
per-angle list -> SplinePSF, kernel / integrated / grid explicit), ZMode (z as emitter height or stage defocus), GaussianPSF,
AstigmaticGaussianPSF, FourPiPSF, FourPiSplinePSF, Optics, Camera, ZStackEstimator (bead z-stacks), Dataset / Drift
(localizations and a drift curve: torch_psf.dataset; SIMCODE's layer is simcode/dataset.py), TiffMovie, ArrayMovie,
open_movie, solve_linear (closed-form I / bg of the variable projection; torch_psf.fit), Tracks, extract_rois, fit_tracks
(in-situ PSF fit on tracks; torch_psf.estimation), heldout_tracks / paired_nll / split_checkerboard (held-out check of a
frozen PSF on track samples, the paired per-sample NLL difference, the checkerboard split; torch_psf.estimation),
spot_deviance, maps_from_movie (per-pixel sCMOS offset / read-noise maps from a sparse movie; torch_psf.camera),
white_photon_transfer (camera gain from the white background noise; torch_psf.camera), fit_gaussian_free_sigma
(spot fit with a free Gaussian width; torch_psf.gaussian), filter_frames, save_zip / is_psf_zip / convert_calibration
(the psf.zip format, torch-psf docs/PSF_FORMAT.md). Other modules call methods of these objects directly; the methods
SIMCODE relies on are listed in docs/TORCH_PSF.md.

SIMCODE-specific:

  has_angle_psf(c)                         per-pattern-angle PSF terms of a calibration dict (SIM illumination angle =
                                           step // 3 = frame % 6 // 3): 'psf_dipole_M_angle' [2][3][3], 'psf_blur_angle_px2'
                                           [2][3]. vectorial_psf(path, angle=0 / 1 / None) gives that angle's VectorialPSF (None
                                           = the angle average: mean dipole tensor, mean blur covariance); calib_psf(path,
                                           angle=...) its spline (None = the mean of the angle images). torch_psf
                                           PSF.load(<psf.zip>, angle=k) reads the per-angle terms (it refuses angle=k for a
                                           model without them); read_calib restores them from older zips
                                           (provenance.unused_model_keys).
  calib_psf(path, roisize, angle, device)  THE PSF renderer of SIMCODE and of the baselines (2026-11-19; analysis/baselines
                                           zpsf: SMLM, SIMFLUX, DECODE): the integrated cubic B-spline of the calibration
                                           (torch_psf SplinePSF, kernel 'bspline', integrated: point samples of the
                                           exact vectorial model on SPLINE's knot grid interpolated by the cubic B-spline,
                                           the exact pixel integral of the spline; psf.zip type 'integrated_spline'; 1.7e-5 of the peak
                                           from the vectorial model at 'exact' pixel integration). A psf.zip written with
                                           it (write_calib_zip) carries the coefficients (per pattern angle + their mean);
                                           any other calibration (calib json, a psf.zip without them) builds them on load
                                           from its vectorial model (integrated_spline; cached per file). angle 0 / 1 =
                                           that pattern angle, None = the mean of the angle images. A Gaussian calibration
                                           ('psf_gauss_sigma_um') gives its GaussianPSF.
  integrated_spline(calib, device)         the integrated B-spline (SplinePSF) of a calibration (path or calib dict, e.g. a
                                           perturbed one) built from its vectorial model on the SPLINE grid with SPLINE_KIND
                                           (all angles; torch_psf to_spline, kernel and grid passed explicitly).
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
  fourpi_state(path)          -> the 4Pi PSF state (FourPiPSF parameter dict) of a 4Pi PSF file: json {'psf': state} or a
                                 psf.zip (FourPiPSF / FourPiSplinePSF with its source).

4Pi PSF (torch_psf FourPiPSF / FourPiSplinePSF; SIMCODE has no 4Pi PSF code of its own since 2026-11-20):

  fourpi_psf(state, device, roisize, zmode)  the FourPiPSF of a 4Pi PSF state (the bead fit's parameter dict,
                                 analysis/fourpi/fit_psf4pi.py; FourPiPSF.state()). Also reads the per-arm terms of SIMCODE's
                                 in-situ states ('arms' {'top' | 'bot': {NA, n_med, apod, saf, pupil_phase, pupil_amp}},
                                 analysis/fourpi_plain/insitu4pi.py): FourPiPSF.from_state reads 'arms' {'top' | 'bottom'}
                                 without a per-arm n_med, so 'bot' is renamed and n_med is set on the arm. zmode (None = the
                                 state's) = torch_psf ZMode(model, zmode).
  fourpi_spline(state, roisize, z_range, oversample, dz_um, zpad, device)  THE 4Pi PSF of the networks, the simulators and
                                 the solve: the FourPiSplinePSF (FourPiPSF.to_spline; 4Pi phase analytic) of a state on
                                 SIMCODE's grid (default 15 px, 4 knots / px, z -0.6 .. 0.6 um in 10 nm planes, zpad 3; the
                                 networks pass their checkpoint's zlut). State keys of the in-situ states that torch_psf
                                 does not read (fourpi_knots): 'reg' (per-channel image offsets [C][2] px = FourPiPSF
                                 channel_offsets, baked into the table by to_spline), 'beta' (a 4Pi phase linear in z, rad /
                                 um: cr + i ci rotated by beta z per plane; not in FourPiPSF) and 'spline' (a refined knot
                                 table, analysis/fourpi_plain/p4spline_refine.py: K [3, C, nz, n, n] of 'spline_base' (default
                                 the file's vector_state); the table is K + table(state) - table(spline_base) on the file's
                                 grid, i.e. exactly K for the base state itself; roisize is kept, the file's grid replaces
                                 z_range / oversample / dz_um / zpad).
  fourpi_knots(state, ...)     the knots [3 (inc, cr, ci), C, nz, n, n] of that table (with reg / beta; no 'spline').
  fourpi_component(spline, k, c, pos, grad)  component k (0 inc, 1 cr, 2 ci) of channel c of a FourPiSplinePSF at pos [N, 3]:
                                 [N, P, P] (grad: value, d/dx, d/dy, d/dz); one table of the FourPiSplinePSF's knot bank
                                 (spline.spline, a SplinePSF with the axes (phase, channel)).
  fourpi_channel(spline, pos, phi, c, grad)  channel c of a FourPiSplinePSF at pos [N, 3] and the 4Pi phase phi [N]: [N, P, P]
                                 (grad: value, d/dx, d/dy, d/dz, d/dphi); FourPiSplinePSF.evaluate gives all channels.
"""
import json, os
import numpy as np
import torch
from torch_psf import (PSF, VectorialPSF, SplinePSF, to_spline, ZMode, GaussianPSF, AstigmaticGaussianPSF, FourPiPSF,
                       FourPiSplinePSF, Optics, Camera, ZStackEstimator, Dataset, Drift)
from torch_psf.movie import TiffMovie, ArrayMovie, open_movie
from torch_psf.fit import solve_linear
from torch_psf.spots import Tracks, extract_rois
from torch_psf.detect import filter_frames
from torch_psf.estimation import fit_tracks, spot_deviance, heldout_tracks, paired_nll, split_checkerboard
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
# knot grid of the integrated cubic B-spline of a calibration (torch_psf to_spline, SPLINE_KIND): emitter heights 0 - 0.4 um
# (user decision 2026-11-19; 6 extra planes each side absorb the boundary of the B-spline prefilter), 10 nm planes, 4 knots / px
# (3 knots / px: 5.3e-5 of the peak from the exact vectorial model on the v4 PSFs, 4: 1.7e-5; out/onepsf), offsets up to 2 px
# on a 41 x 41 ROI (a smaller ROI supports larger offsets: 15 px at 11 x 11)
SPLINE = dict(z_range=(0.0, 0.4), roisize=41, xy_limit=2.0, oversample=4, dz_um=0.01, zpad=6)
# its interpolant, passed explicitly so that torch-psf's default for new tables does not change SIMCODE's renders: the cubic
# B-spline of point samples of the PSF density, integrated exactly over each pixel (psf.zip type 'integrated_spline'). The
# trained models (m1809, m25) use it: keep it until a retrain.
SPLINE_KIND = dict(kernel='bspline', integrated=True, backend='sep')
_SPLINES = {}                       # (file, mtime, size, device) -> the SplinePSF of calib_psf (built on load)


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
    """the integrated cubic B-spline (torch_psf SplinePSF; all pattern angles: an 'angle' spline axis when the calibration has
    per-angle terms) of a calibration (path or calib dict) built from its vectorial model on the SPLINE grid with SPLINE_KIND
    (grid: overrides of SPLINE)"""
    g = dict(SPLINE, **grid)
    c = calib if isinstance(calib, dict) else read_calib(calib)
    vecs = ([vectorial_psf(calib, g['roisize'], k, device) for k in range(2)] if has_angle_psf(c)
            else vectorial_psf(calib, g['roisize'], None, device))
    v0 = vecs[0] if isinstance(vecs, list) else vecs
    if v0.pixel_integration != 'exact':
        import warnings
        warnings.warn(f"{'calibration dict' if isinstance(calib, dict) else os.path.basename(str(calib))}: fitted with pixel_integration "
                      f"{v0.pixel_integration!r}; its PSF spline is the exact pixel integral of the same parameters (up to ~1 % of "
                      f"the peak from the {v0.pixel_integration!r} rendering)", stacklevel=2)
    return to_spline(vecs, **SPLINE_KIND, **g)


def _zip_channel_type(path):
    import zipfile
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read('psf.json').decode('utf-8'))['channels'][0]['type']


def calib_psf(calib, roisize=None, angle=None, device='cuda', z_range=None):
    """THE PSF of a calibration (module doc): the integrated spline (SplinePSF) for roisize x roisize ROIs (default SPLINE roisize) of pattern
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
    out = spl.for_angle(angle) if angle is not None and spl.n_angles else spl       # no angle terms: the same for every angle
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
        kw['integrated_spline'] = dict(SPLINE, **SPLINE_KIND)
    convert_calibration(json_path, out, **kw)
    return out


# ============================================================================================ 4Pi PSF files
def fourpi_state(path, root=None):
    """4Pi PSF state (FourPiPSF parameter dict) of a 4Pi PSF file: a json with key 'psf' (out/4pi/psf4pi/*.json), or a psf.zip of a FourPiPSF
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


# ============================================================================================ 4Pi PSF models
FOURPI_TABLE_KEYS = ('reg', 'beta', 'spline', 'spline_base')     # state keys read by fourpi_spline / fourpi_knots, not torch_psf
# interpolant of the 4Pi tables (the 4Pi networks are trained on it; passed explicitly, as SPLINE_KIND): Catmull-Rom of the
# pixel values, grouped-convolution evaluation
FOURPI_KIND = dict(kernel='catmull', integrated=False, backend='conv')
_FOURPI_SPLINE = dict(kernel=FOURPI_KIND['kernel'], backend=FOURPI_KIND['backend'])     # FourPiSplinePSF(knots, ...)


def fourpi_psf(state, device=None, roisize=None, zmode=None):
    """torch_psf FourPiPSF of a 4Pi PSF state (module doc): FourPiPSF.from_state, plus SIMCODE's in-situ per-arm terms
    ('arms' with 'bot' and a per-arm n_med); roisize default meta.R, zmode default the state's ('emitter'; else torch_psf
    ZMode(model, zmode))"""
    arms = state.get('arms') or {}
    s = state
    if 'bot' in arms:          # SIMCODE in-situ state: {'top' | 'bot': {NA, n_med, apod, saf, pupil_phase, pupil_amp}}
        s = dict(state)
        s['arms'] = {nm: {k: v for k, v in (arms.get(src) or {}).items() if k != 'n_med'} for nm, src in (('top', 'top'), ('bottom', 'bot'))}
    p = FourPiPSF.from_state(s, device=device, roisize=roisize)
    if zmode is not None:
        p = ZMode(p, zmode)
    if 'bot' in arms:
        with torch.no_grad():
            for arm, src in ((p.top, 'top'), (p.bot, 'bot')):
                v = (arms.get(src) or {}).get('n_med')
                if v is not None:
                    arm.n_med.copy_(torch.as_tensor(v, dtype=arm.n_med.dtype).reshape(arm.n_med.shape))
    return p


def _rotate_beta(K, beta, z_range, dz_um, zpad):
    """knots [3, C, nz, n, n] with the coherent term cr + i ci of every plane rotated by beta z (state key 'beta', rad / um)"""
    nz = K.shape[2]
    z = (z_range[0] - zpad * dz_um + dz_um * torch.arange(nz, device=K.device, dtype=torch.float64)).float()
    b = (float(beta) * z)[None, :, None, None]
    cr, ci = K[1], K[2]
    return torch.stack([K[0], cr * torch.cos(b) - ci * torch.sin(b), cr * torch.sin(b) + ci * torch.cos(b)])


def fourpi_knots(state, roisize=15, z_range=(-0.6, 0.6), oversample=4, dz_um=0.01, zpad=3, device=None, chunk=128,
                 xy_limit=None, spline=False):
    """knots [3 (inc, cr, ci), C, nz, n, n] of the 4Pi spline table of a state (FourPiPSF.to_spline of fourpi_psf(state)
    in the movie geometry zmode 'emitter'), with the state keys 'reg' (channel_offsets of the model) and 'beta' (module
    doc); spline=True: (knots, the FourPiSplinePSF of to_spline: the table itself when there is no 'beta')"""
    p = fourpi_psf({k: v for k, v in state.items() if k not in FOURPI_TABLE_KEYS}, device=device, zmode='emitter').eval()
    if state.get('reg') is not None:
        with torch.no_grad():
            p.channel_offsets.copy_(torch.as_tensor(state['reg'], dtype=torch.float32).reshape(p.channel_offsets.shape))
    with torch.no_grad():
        T = p.to_spline(roisize, tuple(z_range), **FOURPI_KIND, oversample=oversample, dz_um=dz_um, zpad=zpad,
                        xy_limit=xy_limit, chunk=chunk)
    K = T.knots.reshape(*T.spline.spline_index.shape, *T.knots.shape[-3:])          # the knot bank [3 C, ...] -> [3, C, ...]
    if state.get('beta'):
        K = _rotate_beta(K, state['beta'], z_range, dz_um, zpad)
        T = None
    return (K, T) if spline else K


def fourpi_spline(state, roisize=15, z_range=(-0.6, 0.6), oversample=4, dz_um=0.01, zpad=3, device=None, chunk=128):
    """THE 4Pi PSF: the FourPiSplinePSF of a 4Pi PSF state on SIMCODE's grid (module doc; state keys reg / beta / spline)"""
    if state.get('spline'):
        return _fourpi_refined(state, roisize, device, chunk)
    K, T = fourpi_knots(state, roisize, z_range, oversample, dz_um, zpad, device, chunk, spline=True)
    if T is not None:
        return T
    return FourPiSplinePSF(roisize, tuple(z_range), K, oversample=oversample, dz_um=dz_um, zpad=zpad, tau=state['tau'],
                           device=device, pixel_um=state['meta']['px'], **_FOURPI_SPLINE)


def _fourpi_refined(state, roisize, device, chunk):
    """FourPiSplinePSF of a state with a refined knot table (state key 'spline'; module doc)"""
    f = torch.load(state['spline'], weights_only=False, map_location='cpu')
    S, hz, zpad, zr = int(f['S']), float(f['dz']), int(f['zpad']), (float(f['z_range'][0]), float(f['z_range'][1]))
    K = f['K'].to(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
    base = dict(f['vector_state'] if state.get('spline_base') is None else state['spline_base'])
    own = {k: v for k, v in state.items() if k not in ('spline', 'spline_base')}
    if json.dumps(base, sort_keys=True) != json.dumps(own, sort_keys=True):     # another state: K + its change of the table
        g = dict(roisize=f['roisize'], z_range=zr, oversample=S, dz_um=hz, zpad=zpad, device=K.device, chunk=chunk,
                 xy_limit=f['xy_limit'])
        d = fourpi_knots(own, **g) - fourpi_knots(base, **g)
        if d.shape != K.shape:
            raise ValueError(f"{state['spline']}: knots {tuple(K.shape)}, the table of its grid {tuple(d.shape)}")
        K = K + d
    return FourPiSplinePSF(roisize, zr, K, oversample=S, dz_um=hz, zpad=zpad, tau=state['tau'], device=K.device,
                           pixel_um=state['meta']['px'], **_FOURPI_SPLINE)


def fourpi_component(spline, k, c, pos, grad=False):
    """component k (0 inc, 1 cr, 2 ci) of channel c of a FourPiSplinePSF at pos [N, 3] (x, y px, z um): [N, P, P]; grad: (value,
    d/dx, d/dy, d/dz). One spline of the knot bank (spline.spline: SplinePSF with the axes (phase, channel)), without
    the spline's channel_offsets / intensity_ratio (SIMCODE's tables have the offsets baked in)"""
    T = spline.spline
    if T.axes != ('phase', 'channel'):
        raise ValueError(f'fourpi_component: a 4Pi table with the axes (phase, channel) expected, not {T.axes}')
    return T.evaluate_splines(pos, T.spline_index[k, c], grad)


def fourpi_channel(spline, pos, phi, c, grad=False):
    """channel c of a FourPiSplinePSF at pos [N, 3] (x, y px, z um) and 4Pi phase phi [N]: inc + cos(phi) cr - sin(phi) ci
    [N, P, P]; grad: (value, d/dx, d/dy, d/dz, d/dphi)"""
    cP, sP = torch.cos(phi)[:, None, None], torch.sin(phi)[:, None, None]
    if not grad:
        inc, cr, ci = (fourpi_component(spline, k, c, pos) for k in range(3))
        return inc + cP * cr - sP * ci
    a, b, e = (fourpi_component(spline, k, c, pos, True) for k in range(3))
    return [a[k] + cP * b[k] - sP * e[k] for k in range(4)] + [-sP * b[0] - cP * e[0]]
