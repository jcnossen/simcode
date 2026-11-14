"""Stage f (process): python -m simcode.run on the (blocked-order) movie with the dataset's calibration and model,
through simcode/auto/runwrap.py for the instrument overrides of docs/AUTO_PIPELINE.md section 7 (pitch range, strict crop,
detector batch, the model of stage e). With the DMD-SIMFLUX defaults the wrapper changes nothing, so the run is the
same as a plain simcode.run with the same arguments."""
import json, os
import numpy as np


def model_of(ds):
    """(name, model dict with absolute paths, or None for a model of simcode/models.json)"""
    from simcode import config
    p = ds.c['process']
    name = p['model'] or ds.c['train']['model']
    if name and not os.path.exists(str(name)):
        try:
            config.model_set(name); return name, None
        except KeyError:
            pass
    f = name if name and os.path.exists(str(name)) else os.path.join(ds.out, 'train', 'model.json')
    S = json.load(open(f))
    (n, s), = [(k, v) for k, v in S.items() if not k.startswith('_')]
    return n, s


def argv(ds, out, calib=None, nframes=None, chunk_list=None, model=None):
    p = ds.c['process']
    name, _ = model if model is not None else model_of(ds)
    calib = calib or p['calib'] or ds.summary('psf')['calib']
    a = ['--movie', ds.movie, '--calib', calib, '--out', out, '--model', name, '--chunk', p['chunk']]
    if p['crop']: a += ['--crop', ','.join(str(v) for v in p['crop'])]
    n = p['nframes'] if nframes is None else nframes
    if n: a += ['--nframes', n]
    cl = p['chunk_list'] if chunk_list is None else chunk_list
    if cl: a += ['--chunk_list', cl]
    if p['sig_u']: a += ['--sig_u', p['sig_u']]
    return a + [str(v) for v in p['extra']]


def crop_of(ds):
    p = ds.c['process']
    if p['crop']: return [float(v) for v in p['crop']]
    from simcode.fileio.movie import open_movie
    H, W = open_movie(ds.movie).shape[-2:]
    return None if (H, W) == (120, 120) else [5.0, 5.0, float(min(H, W) - 10)]


def wrap(ds, model):
    p = ds.c['process']; extra = {}
    cr = crop_of(ds)
    if cr and tuple(cr) != (5.0, 5.0, 110.0):
        extra['crop'] = cr
        if not p['crop']: p['crop'] = cr
    if p['det_batch'] != 64: extra['det_batch'] = p['det_batch']
    name, S = model
    if S is not None: extra['model'] = {name: S}
    extra['title'] = 'auto-process'
    return ds.wrapfile(extra, 'wrap_process')


def cycle_subset(out):
    """rows of windows that start at a cycle boundary (frame % 6 == 0): for interleaved acquisitions the windows at 6c + 3
    mix two acquisition cycles (assumption 1)"""
    r = {}
    for n in ('locs_p1', 'locs_p2'):
        f = os.path.join(out, n + '.npz')
        if not os.path.exists(f): continue
        T = np.load(f); k = (T['frame'] % 6) == 0
        np.savez(os.path.join(out, n + '_cyc.npz'), **{c: T[c][k] for c in T.files if T[c].shape[:1] == k.shape})
        r[n] = int(k.sum())
    return r


def run(ds):
    import hashlib
    ns = model_of(ds)
    W = wrap(ds, ns)
    a0 = argv(ds, 'X', model=ns)
    from simcode import config as SC
    calib = SC.resolve(a0[a0.index('--calib') + 1])
    key = hashlib.sha1(json.dumps([a0, open(calib).read() if os.path.exists(calib) else calib, ns[1], open(W).read()]).encode()).hexdigest()[:10]
    out = os.path.join(ds.dir('process'), 'run_' + key)     # keyed by arguments, calibration, model and overrides: chunk files are never stale
    g = ds.gpu(10)
    txt = ds.sh(['-m', 'simcode.auto.runwrap', 'simcode.run'] + argv(ds, out, model=ns), os.path.join(ds.dir('process'), 'run.log'), gpu=g, wrap=W)
    S = json.load(open(os.path.join(out, 'summary.json')))
    pe = json.load(open(os.path.join(out, 'pattern_est.json'))) if os.path.exists(os.path.join(out, 'pattern_est.json')) else {}
    r = dict(run=out, run_config=os.path.join(out, 'run_config.json'), model=ns[0], summary=S, wrap=json.load(open(W)))   # run_config.json = the complete record of the run
    if ds.c['process']['cycle_aligned_only']: r['cycle_rows'] = cycle_subset(out)
    chk = []
    pit = _pitch(pe)
    if pit:
        lo, hi = ds.c['pattern']['pitch_px']
        chk.append(dict(name='estimated pitch inside pattern.pitch_px', value=pit, ok=all(lo < v < hi for v in pit)))
    n = S.get('locs_p2', S.get('locs_p1', {})).get('n_rows', 0)
    chk.append(dict(name='rows > 0', value=n, ok=n > 0))
    r['pattern_pitch_px'] = pit
    return dict(r, checks=chk)


def _pitch(pe):
    try:
        return [float(pe['k'][a]['pitch_px']) for a in ('0', '1')]
    except Exception:
        return None
