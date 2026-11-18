"""Acquisition config of the automatic pipeline (docs/AUTO_PIPELINE.md section 1): defaults = the DMD-SIMFLUX setup.
load(cfg) -> nested dict with every default filled in; unknown keys raise. cfg: a ConfigDict (simcode/util/config_dict.py), a
dict, a yaml / json path, or a Python file that defines `config`. load_all(cfg) -> list (a Python file with `configs`).

  setup         widefield (default) | fourpi: the set-up plug-in (simcode/setups/), which adds its own config section
                (fourpi: simcode/setups/fourpi.py) and chooses the stages
  channels      null (one channel = the whole frame) or the list of detection channels on the camera
                [{name, crop: [x0, y0, w, h], mirror, page, psf, weight}] (simcode/channels/layout.py)
  registration  of the channels to channel 0 (simcode/channels/registration.py): mode insitu (per file, from the
                movie itself; no beads) | file (one map json for every file: map) | none; order of the polynomial; the
                settings of the in-situ chain (frames per file, frames per sum, detection / fit / matching)
  pattern.pattern_frames  which frame of a pattern cycle has which pattern: one row per pattern orientation, its frames in
                phase-step order (the preprint code's pattern_frames). [[0, 1, 2], [3, 4, 5]]: the 3 steps of orientation 1,
                then the 3 of orientation 2; [[0, 2, 4], [1, 3, 5]]: the orientations alternate. The networks of this package
                are trained for 2 orientations x 3 steps, so only that shape is accepted (in any frame order).
  camera.offset_map  npz with a per-pixel offset map 'off' [C, H, W] of the channel crops (ADU; instead of offset_adu)
  psf.lut, psf.lut_sub, psf.lut_inner  deprecated (2026-11-18), accepted and ignored: the spline-table variant of the
                in-situ PSF fit was removed; the psf stage always runs the direct vectorial fit (torch_psf fit_tracks), which
                was the default (lut: false). lut: true logs a warning.
  spots.maxframe  frames the spot finder and the tracks use (null = the whole movie, the default since 2026-11-18; it was 30000)
  psf.nsamples  track samples of each PSF fit (default 32000 since 2026-11-18, was 8000; at most the samples there are). The
                sampling sd of the fit falls as 1 / sqrt(nsamples); measured on 1809_2 (whole movie, 87468 samples): focus
                3.8 / 2.8 / 1.9 nm, Zernike rms 2.6 / 2.4 / 1.1 mwaves, line blur sxx 5 / 2 / 4 x 1e-3 px^2, held-out paired NLL
                bl - base 0.009 / 0.004 / 0.002 and ps3 - bl 0.003 / 0.002 / 0.002 nats per sample at 8000 / 16000 / 32000. The
                fit time grows linearly (RTX 3090, 32000: base 134 s, bl 178 s, ps3 449 s)
  psf.init      the starts of the multi-start base fit (simcode/pipeline/psf_stage.py _inits): 'zero', 'f<focus um>' (zero
                Zernikes), 'a<focus um>' (four astigmatism starts Z5 / Z6 = +- psf.astig_init_waves; added 2026-11-18: starts
                without astigmatism cannot leave the zero-astigmatism set), 'beads' (the bead fit and its mirrored twin), a
                calibration json / _psf.pt. Default [zero, f0.1, f0.2, f0.3, a0.1, beads]
  psf.astig_init_waves  astigmatism amplitude of the 'a<focus>' starts (default 0.05 waves)
  psf.init_nsamples  samples of the multi-start base fits (psf.init) that choose the start (default 8000; null = psf.nsamples);
                the base fit from the chosen start then uses psf.nsamples. With init_nsamples >= nsamples the start fits are the
                base fits (the behaviour before 2026-11-18)
  psf.heldout_nsamples  cb1 samples of the held-out test (6000)
  psf.spread_k  sampling-spread diagnostic (2026-11-18): the cb0 fits and the held-out test repeated on spread_k independent
                sample draws (default 3, i.e. 2 extra draws; 0 = off); the sd of focus, Zernikes, blur, gamma / zeta and of the
                held-out NLL (and its paired differences) for a fit of psf.nsamples samples: psf/spread.json, summary 'spread',
                report 'Sampling spread'. Each extra draw costs about one cb0 fit chain + one held-out test
  psf.spread_nsamples  samples per draw (default 8000; null = psf.nsamples; at most 3/4 of the cb0 samples). The sd is scaled
                to psf.nsamples by sqrt(spread_nsamples / psf.nsamples) (the 1 / sqrt(n) law held on 1809_2 for 8000 -> 32000
                within the scatter of the estimates); draws of psf.nsamples cost about 3 x more (1809_2: ~15 instead of ~5 min each)
  train.heldout  held-out set of the emitter trainer (python -m simcode.emitter.heldout --set 5 --out <file>; the model
                recipe's set); null: the trainer runs without it
A Python config file can define `configs` / `config` as a function of keyword arguments (e.g. the 4Pi configs:
configs(work, data) = the 4Pi work folder and the raw-movie folder); load / load_all(cfg, params) call it with params
(python -m simcode.pipeline <file> --work <dir> --data <dir>)."""
import copy, json, math, os
from collections.abc import Mapping

DEFAULTS = {
    'name': None, 'movie': None, 'out': None,
    'beads': None,
    'setup': 'widefield',
    'camera': dict(type='scmos', gain_e_per_adu=None, offset_adu=None, offset_map=None, readnoise_e=None, dark=None, readnoise_from=None,
                   map_frames=20000, gain_frames=12000),
    'channels': None,
    'registration': dict(mode='insitu', map=None, order=2, frames=0, frame0=0, sum=6, chunk=600, calib_frames=1200, calib_n=3000,
                         calib_p0=[1.2, 1.2, 0.18], astig='auto', roi=13, threshold=8.0, sigma=[1.5, 5.0], isolation=6, nms=7,
                         d_um=0.27, z_range=[-0.6, 0.6], z_range_sym=[0.03, 0.8], k_dev=3.0, max_crlb=0.15, zlim=[-0.5, 0.5],
                         zlim_sym=[0.0, 0.75], flips=['none', 'lr', 'ud', 'both'], r1=2.0, k_sigma=4.0, kfold=5),
    'optics': dict(pixel_um=0.10833, NA=1.49, wavelength_um=0.70, n_imm=1.518, n_med=1.33, tirf_depth_um=0.082),
    'pattern': dict(pattern_frames=[[0, 1, 2], [3, 4, 5]], cache=None, pitch_px=[1.6, 2.8]),
    'hints': dict(photons_window=None, density_per_um2=None, bg_photons_px=None),
    'spots': dict(maxframe=None, sum=6, thr=3.5, roi=11, track_roi=11, isolation_px=7.0, link_px=0.8,
                  chi2_max=3.6, snr_min=8.0, sigma_range=[0.75, 1.35], tracks_from='all'),
    'psf': dict(init=['zero', 'f0.1', 'f0.2', 'f0.3', 'a0.1', 'beads'], astig_init_waves=0.05, nsamples=32000, iters=600, zmax_um=0.2, blur='line', photosel=3, lut=False, heldout=True,
                readnoise_e=1.9, offset_adu=None, lut_sub=4, lut_inner=1, heldout_nsamples=6000, spread_k=3, spread_nsamples=8000, init_nsamples=8000),
    'train': dict(model=None, quick=False, quick_steps=30, heldout=None,
                  steps=dict(detector=40000, emitter=18000, detector_pass2=40000, detector_pass2_mix=40000, detector_pass2_cont=40000, emitter_pass2=24000,
                             emitter_pass2_long=50000, emitter_pass2_final=50000),
                  init=dict(detector=None, emitter='pretrain_emitter_pass1.pt'), extra=dict(detector='', emitter='', pass2=''),
                  gates=dict(sigma_zcal=[0.9, 1.15], rms_ratio_max=1.05), template_model='m1809'),
    'process': dict(model=None, calib=None, crop=None, chunk=6000, nframes=0, chunk_list='', det_batch=64,
                    cycle_aligned_only=False, extra=[]),
    'gpus': [0],
}
BEAD_DEFAULTS = dict(stacks=[], dz_um=0.010, sign=1.0, offset_adu=None, gain_e_per_adu=None, holdout=0, iters=300, roi=26)


def _merge(d, u, path=''):
    for k, v in u.items():
        if k not in d:
            raise KeyError(f'unknown config key {path + k!r}')
        if isinstance(d[k], dict) and isinstance(v, dict) and k not in ('steps',):
            _merge(d[k], v, path + k + '.')
        elif isinstance(d[k], dict) and isinstance(v, dict):
            d[k].update(v)
        else:
            d[k] = v
    return d


def _read(cfg, params=None):
    """any accepted form -> [(plain dict, source file or None), ...]; params: the arguments of a Python config file's
    configs / config function"""
    from simcode.util.config_dict import ConfigDict, load_python
    if isinstance(cfg, Mapping):
        u = ConfigDict(cfg).to_dict()
        return [(u, u.pop('_file', None))]          # '_file': a config that was already resolved (resolving again is a no-op)
    if isinstance(cfg, (list, tuple)):
        return [r for c in cfg for r in _read(c, params)]
    src = os.path.abspath(os.fspath(cfg))
    if src.endswith('.py'):
        return [(c.to_dict(), src) for c in load_python(src, params=params)]
    if params: raise ValueError(f'{src}: arguments {sorted(params)} apply to a Python config file with a configs(...) / config(...) function')
    txt = open(src).read()
    if src.endswith(('.yaml', '.yml')):
        import yaml
        u = yaml.safe_load(txt)
    else:
        u = json.loads(txt)
    if not isinstance(u, dict): raise ValueError(f'{src}: the top level of a config must be a mapping')
    u.pop('_file', None)                             # a resolved config written by the pipeline (config_resolved.json)
    return [(u, src)]


def load_all(cfg, params=None):
    """every acquisition config of cfg (a Python file with a list `configs`, a list of configs, or any single form),
    each validated and with the defaults filled in; params: the arguments of a Python file's configs(...) function"""
    L = [_resolve(u, src) for u, src in _read(cfg, params)]
    names = [c['name'] for c in L]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup: raise ValueError(f'config: acquisition names must be unique, repeated: {dup}')
    return L


def load(cfg, params=None):
    """one acquisition config (ConfigDict, dict, yaml / json path, or a Python file that defines one `config`); params: the
    arguments of a Python file's config(...) function"""
    L = _read(cfg, params)
    if len(L) != 1:
        raise ValueError(f'{cfg}: {len(L)} acquisition configs; LocalizationPipeline takes one (use '
                         f'LocalizationPipeline.from_file(file) or python -m simcode.pipeline <file> [--only name])')
    return _resolve(*L[0])


def _resolve(u, src=None):
    from simcode import setups
    c = copy.deepcopy(DEFAULTS)
    name = u.get('setup', DEFAULTS['setup'])
    if name not in setups.NAMES: raise ValueError(f"config: setup must be one of {setups.NAMES}, not {name!r}")
    S = setups.get(name)
    c.update(copy.deepcopy(S.defaults()))
    if u.get('beads'):
        b = dict(BEAD_DEFAULTS); bu = u.pop('beads')
        for k in bu:
            if k not in b: raise KeyError(f'unknown config key beads.{k!r}')
        b.update(bu); c['beads'] = b; _merge(c, u)
    else:
        u.pop('beads', None); _merge(c, u)
    for k in ('name', 'movie', 'out'):
        if not c[k]: raise ValueError(f'config: {k!r} is required')
        if not isinstance(c[k], str): raise ValueError(f'config: {k!r} must be a string (quote it in yaml: 1809_2 reads as the number 18092)')
    c['_file'] = src
    validate(c)
    S.validate(c)
    return c


def slot_of_frame(c):
    """acquisition frame j of a cycle -> slot (orientation * steps + step), from pattern.pattern_frames"""
    PF = c['pattern']['pattern_frames']
    slot = [None] * sum(len(r) for r in PF)
    for a, row in enumerate(PF):
        for s, j in enumerate(row): slot[int(j)] = a * len(row) + s
    return slot


def validate(c):
    P = c['pattern']; PF = P['pattern_frames']
    if not (isinstance(PF, (list, tuple)) and len(PF) == 2 and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in PF)):
        raise ValueError(f'pattern.pattern_frames {PF!r}: the networks are trained for 2 orientations x 3 phase steps, '
                         'so 2 rows of 3 frame indices are needed, e.g. [[0, 1, 2], [3, 4, 5]]')
    if sorted(int(j) for r in PF for j in r) != list(range(6)):
        raise ValueError(f'pattern.pattern_frames {PF!r} must use each of the frames 0-5 of a cycle once')
    if not (0 < P['pitch_px'][0] < P['pitch_px'][1]):
        raise ValueError('pattern.pitch_px must be [lo, hi] with 0 < lo < hi')
    if c['camera']['type'] not in ('scmos', 'emccd'):
        raise ValueError('camera.type: scmos | emccd')
    if c['psf']['blur'] not in ('none', 'line') or c['psf']['photosel'] not in (0, 3):
        raise ValueError('psf.blur: none | line; psf.photosel: 0 | 3')
    mf = c['spots']['maxframe']
    if mf is not None and (not isinstance(mf, int) or isinstance(mf, bool) or mf <= 0):
        raise ValueError(f'spots.maxframe: null (the whole movie) or a number of frames > 0, not {mf!r}')
    for k_ in ('nsamples', 'heldout_nsamples', 'init_nsamples', 'spread_nsamples'):
        v = c['psf'][k_]
        if not (v is None and k_ in ('init_nsamples', 'spread_nsamples')) and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
            raise ValueError(f'psf.{k_}: a number of samples > 0' + (' or null (= psf.nsamples)' if k_ in ('init_nsamples', 'spread_nsamples') else '') + f', not {v!r}')
    k = c['psf']['spread_k']
    if not isinstance(k, int) or isinstance(k, bool) or k < 0 or k == 1:
        raise ValueError(f'psf.spread_k: 0 (no sampling-spread diagnostic) or the number of sample draws >= 2, not {k!r}')
    if c['psf']['lut']:
        import warnings
        warnings.warn('psf.lut is deprecated and ignored: the spline-table PSF fit was removed, the psf stage runs the direct '
                      'vectorial fit (the default, lut: false)', stacklevel=2)
    if c['channels'] is not None:
        from simcode.channels.layout import ChannelLayout
        ChannelLayout.from_config(c)                     # field checks (crop, mirror, equal crop sizes, names)
    r = c['registration']
    if r['mode'] not in ('insitu', 'file', 'none'): raise ValueError('registration.mode: insitu | file | none')
    if r['mode'] == 'file' and not r['map']: raise ValueError('registration.mode file needs registration.map')
    if r['astig'] not in ('auto', 'on', 'off', True, False): raise ValueError('registration.astig: auto | on | off')


def needs_reorder(c):
    return slot_of_frame(c) != list(range(6))


def sigma_px(c):
    """diffraction-limited Gaussian sigma estimate (px): 0.21 lambda / NA, broadened ~10 % by pixel integration"""
    o = c['optics']
    return 0.21 * o['wavelength_um'] / o['NA'] / o['pixel_um'] * 1.1


def dump(c, path):
    from simcode.io.work import atomic_json
    atomic_json(path, c, indent=1, default=str)
