"""Acquisition config of the automatic pipeline (docs/AUTO_PIPELINE.md section 1): defaults = the DMD-SIMFLUX setup.
load(cfg) -> nested dict with every default filled in; unknown keys raise. cfg: a ConfigDict (simcode/util/cfgdict.py), a
dict, a yaml / json path, or a Python file that defines `config`. load_all(cfg) -> list (a Python file with `configs`).

  setup         widefield (default) | fourpi: the set-up plug-in (simcode/setups/), which adds its own config section
                (fourpi: simcode/setups/fourpi.py) and chooses the stages
  channels      null (one channel = the whole frame) or the list of detection channels on the camera
                [{name, crop: [x0, y0, w, h], mirror, page, psf, weight}] (simcode/channels/layout.py)
  registration  of the channels to channel 0 (simcode/channels/registration.py): mode insitu (per file, from the
                movie itself; no beads) | file (one map json for every file: map) | none; order of the polynomial; the
                settings of the in-situ chain (frames per file, frames per sum, detection / fit / matching)
  camera.offset_map  npz with a per-pixel offset map 'off' [C, H, W] of the channel crops (ADU; instead of offset_adu)"""
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
    'pattern': dict(orientations=2, steps=3, frame_order='blocked', cache=None, pitch_px=[1.6, 2.8]),
    'hints': dict(photons_window=None, density_per_um2=None, bg_photons_px=None),
    'spots': dict(maxframe=30000, sum=6, thr=3.5, roi=11, track_roi=11, isolation_px=7.0, link_px=0.8,
                  chi2_max=3.6, snr_min=8.0, sigma_range=[0.75, 1.35], tracks_from='all'),
    'psf': dict(init=['zero', 'f0.1', 'f0.2', 'f0.3', 'beads'], nsamples=8000, iters=600, zmax_um=0.2, blur='line', photosel=3, lut=False, heldout=True,
                readnoise_e=1.9, offset_adu=None, lut_sub=4, lut_inner=1, heldout_nsamples=6000),
    'train': dict(model=None, quick=False, quick_steps=30,
                  steps=dict(detector=40000, emitter=18000, detector_pass2=40000, detector_pass2_mix=40000, detector_pass2_cont=40000, emitter_pass2=24000,
                             emitter_pass2_long=50000, emitter_pass2_final=50000),
                  init=dict(detector=None, emitter='hd_sc5nb.pt'), extra=dict(detector='', emitter='', pass2=''),
                  gates=dict(sigma_zcal=[0.9, 1.15], rms_ratio_max=1.05), real_mult_from='m1809'),
    'process': dict(model=None, calib=None, crop=None, chunk=6000, nframes=0, chunk_list='', det_batch=64,
                    cycle_aligned_only=False, extra=[]),
    'gpus': [1, 2],
}
BEAD_DEFAULTS = dict(stacks=[], dz_um=0.010, sign=1.0, offset_adu=None, gain_e_per_adu=None, holdout=0, iters=300, roi=26)
ORDERS = {'blocked': [0, 1, 2, 3, 4, 5], 'interleaved': [0, 3, 1, 4, 2, 5]}   # acquisition frame j of a cycle -> slot


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


def _read(cfg):
    """any accepted form -> [(plain dict, source file or None), ...]"""
    from simcode.util.cfgdict import ConfigDict, load_python
    if isinstance(cfg, Mapping):
        u = ConfigDict(cfg).to_dict()
        return [(u, u.pop('_file', None))]          # '_file': a config that was already resolved (resolving again is a no-op)
    if isinstance(cfg, (list, tuple)):
        return [r for c in cfg for r in _read(c)]
    src = os.path.abspath(os.fspath(cfg))
    if src.endswith('.py'):
        return [(c.to_dict(), src) for c in load_python(src)]
    txt = open(src).read()
    if src.endswith(('.yaml', '.yml')):
        import yaml
        u = yaml.safe_load(txt)
    else:
        u = json.loads(txt)
    if not isinstance(u, dict): raise ValueError(f'{src}: the top level of a config must be a mapping')
    u.pop('_file', None)                             # a resolved config written by the pipeline (config_resolved.json)
    return [(u, src)]


def load_all(cfg):
    """every acquisition config of cfg (a Python file with a list `configs`, a list of configs, or any single form),
    each validated and with the defaults filled in"""
    L = [_resolve(u, src) for u, src in _read(cfg)]
    names = [c['name'] for c in L]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup: raise ValueError(f'config: acquisition names must be unique, repeated: {dup}')
    return L


def load(cfg):
    """one acquisition config (ConfigDict, dict, yaml / json path, or a Python file that defines one `config`)"""
    L = _read(cfg)
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
    """acquisition frame j of a cycle -> slot (orientation * steps + step)"""
    fo = c['pattern']['frame_order']
    return list(ORDERS[fo]) if isinstance(fo, str) else [int(v) for v in fo]


def validate(c):
    P = c['pattern']
    if P['orientations'] != 2 or P['steps'] != 3:
        raise ValueError('pattern: only 2 orientations x 3 steps (6-frame cycle) are supported by the pipeline '
                         '(docs/AUTO_PIPELINE.md section 7, proposals P1 / P2)')
    s = slot_of_frame(c)
    if sorted(s) != list(range(6)):
        raise ValueError(f'pattern.frame_order {P["frame_order"]!r} is not a permutation of the 6 slots')
    if not (0 < P['pitch_px'][0] < P['pitch_px'][1]):
        raise ValueError('pattern.pitch_px must be [lo, hi] with 0 < lo < hi')
    if c['camera']['type'] not in ('scmos', 'emccd'):
        raise ValueError('camera.type: scmos | emccd')
    if c['psf']['blur'] not in ('none', 'line') or c['psf']['photosel'] not in (0, 3):
        raise ValueError('psf.blur: none | line; psf.photosel: 0 | 3')
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
