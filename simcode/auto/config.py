"""Dataset config of the automatic pipeline (docs/AUTO_PIPELINE.md section 1): defaults = the DMD-SIMFLUX setup.
load(path) -> nested dict with every default filled in; unknown keys raise. yaml or json."""
import copy, json, math, os

DEFAULTS = {
    'name': None, 'movie': None, 'out': None,
    'beads': None,
    'camera': dict(type='scmos', gain_e_per_adu=None, offset_adu=None, readnoise_e=None, dark=None, readnoise_from=None,
                   map_frames=20000, gain_frames=12000),
    'optics': dict(pixel_um=0.10833, NA=1.49, wavelength_um=0.70, n_imm=1.518, n_med=1.33, tirf_depth_um=0.082),
    'pattern': dict(orientations=2, steps=3, frame_order='blocked', cache=None, pitch_px=[1.6, 2.8]),
    'hints': dict(photons_window=None, density_per_um2=None, bg_photons_px=None),
    'spots': dict(maxframe=30000, sum=6, thr=3.5, roi=11, track_roi=11, isolation_px=7.0, link_px=0.8,
                  chi2_max=3.6, snr_min=8.0, sigma_range=[0.75, 1.35], tracks_from='all'),
    'psf': dict(init=['zero', 'f0.1', 'f0.2', 'f0.3', 'beads'], nsamples=8000, iters=600, zmax_um=0.2, blur='line', photosel=3, lut=False, heldout=True,
                readnoise_e=1.9, offset_adu=None, lut_sub=4, lut_inner=1, heldout_nsamples=6000),
    'train': dict(model=None, quick=False, quick_steps=30,
                  steps=dict(det=40000, hd=18000, p2det=40000, p2head=24000, p2headL=50000),
                  init=dict(det=None, hd='hd_sc5nb.pt'), extra=dict(det='', hd='', p2=''),
                  gates=dict(sigma_zcal=[0.9, 1.15], rms_ratio_max=1.05), real_mult_from='m1809'),
    'process': dict(model=None, calib=None, crop=None, chunk=6000, nframes=0, chunk_list='', det_batch=64,
                    cycle_aligned_only=False, sig_u='', extra=[]),
    'gpus': [1, 2],
}
BEAD_DEFAULTS = dict(stacks=[], dz_um=0.010, sign=1.0, offset_adu=None, gain_e_per_adu=None, holdout=0, nball=32, iters=300, roi=26)
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


def load(path_or_dict):
    if isinstance(path_or_dict, dict):
        u = copy.deepcopy(path_or_dict); src = None
    else:
        src = os.path.abspath(path_or_dict)
        txt = open(src).read()
        if src.endswith(('.yaml', '.yml')):
            import yaml
            u = yaml.safe_load(txt)
        else:
            u = json.loads(txt)
    c = copy.deepcopy(DEFAULTS)
    for sec in ('train', 'process'):          # names before 2026-11-14: 'netset' (= model); process.pass2 (pass 2 always runs now)
        d = u.get(sec) or {}
        if 'netset' in d: d['model'] = d.pop('netset')
    if 'pass2' in (u.get('process') or {}):
        if not u['process'].pop('pass2'): raise ValueError('config: process.pass2 false is no longer supported (pass 2 always runs, 2026-11-14)')
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


def needs_reorder(c):
    return slot_of_frame(c) != list(range(6))


def sigma_px(c):
    """diffraction-limited Gaussian sigma estimate (px): 0.21 lambda / NA, broadened ~10 % by pixel integration"""
    o = c['optics']
    return 0.21 * o['wavelength_um'] / o['NA'] / o['pixel_um'] * 1.1


def dump(c, path):
    from simcode.fileio.work import atomic_json
    atomic_json(path, c, indent=1, default=str)
