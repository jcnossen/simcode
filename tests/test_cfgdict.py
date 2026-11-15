"""
CPU tests of ConfigDict (simcode/util/cfgdict.py) and of the config forms of the automatic pipeline (simcode/pipeline/acquisition_config.py):
access, missing keys, merge / updated, freeze, to_dict conversions, yaml / json round trip, the Python-file loader, schema
validation of derived configs, the example configs in simcode/pipeline/configs. No GPU, no data files.
Run: python tests/test_cfgdict.py (or pytest tests/test_cfgdict.py).
"""
import copy, json, os, pickle, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simcode.util.cfgdict import ConfigDict, load_python
from simcode.pipeline import acquisition_config as AC

CFG = os.path.join(ROOT, 'simcode', 'pipeline', 'configs')


def raises(exc, fn, *a, **k):
    try: fn(*a, **k)
    except exc as e: return e
    raise AssertionError(f'{exc.__name__} not raised')


def _base():
    return ConfigDict(name='a', optics=dict(pixel_um=0.1, NA=1.49), psf=dict(init=['zero', 'beads'], iters=600),
                      train=dict(steps=dict(detector=10, emitter=20)), gpus=[1, 2])


def test_access():
    c = _base()
    assert c.name == 'a' and c['name'] == 'a'
    assert isinstance(c.optics, ConfigDict) and c.optics.NA == 1.49 and c['optics']['NA'] == 1.49 and c['optics.NA'] == 1.49
    assert c.train.steps.emitter == 20 and c['train.steps.emitter'] == 20
    assert 'optics' in c and 'optics.NA' in c and 'optics.na' not in c and 'nope' not in c
    assert list(c) == ['name', 'optics', 'psf', 'train', 'gpus'] and len(c) == 5
    assert dict(c.optics) == dict(pixel_um=0.1, NA=1.49) and dict(**c.optics)['NA'] == 1.49
    c.optics.NA = 1.4; c['psf']['iters'] = 300; c['spots.thr'] = 5.0; c.new = dict(x=dict(y=1))
    assert c.optics.NA == 1.4 and c.psf.iters == 300 and c.spots.thr == 5.0 and c.new.x.y == 1
    del c.new; del c['spots']
    assert 'new' not in c and 'spots' not in c
    d = ConfigDict(dict(a=1), b=[dict(c=2), (dict(d=3),)])            # dicts inside lists / tuples
    assert d.b[0].c == 2 and d.b[1][0].d == 3
    assert ConfigDict(keys=1)['keys'] == 1                               # a key named like a method: item access only
    raises(AttributeError, setattr, ConfigDict(), 'keys', 1)
    raises(TypeError, ConfigDict, [1, 2])


def test_missing_raises():
    c = _base()
    e = raises(AttributeError, lambda: c.optics.na)
    assert "'NA'" in str(e)                                              # close match named
    raises(AttributeError, lambda: c.nope)
    raises(KeyError, lambda: c['nope'])
    raises(KeyError, lambda: c['optics.nope'])
    raises(KeyError, lambda: c['name.x'])                                # not a section
    assert c.get('nope') is None and c.get('nope', 3) == 3 and c.get('optics.nope', 4) == 4 and c.get('optics.NA') == 1.49
    assert not hasattr(c, 'nope') and getattr(c, 'nope', 7) == 7


def test_merge_updated():
    b = _base()
    c = b.updated(name='b', psf=dict(iters=300), optics=dict(wavelength_um=0.7), gpus=[0])
    assert c.name == 'b' and c.psf.iters == 300 and c.psf.init == ['zero', 'beads']        # nested keys merged
    assert c.optics == dict(pixel_um=0.1, NA=1.49, wavelength_um=0.7) and c.gpus == [0]    # lists replaced
    assert b.name == 'a' and b.psf.iters == 600 and 'wavelength_um' not in b.optics and b.gpus == [1, 2]   # base untouched
    c.psf.init.append('x'); c.train.steps.detector = 1
    assert b.psf.init == ['zero', 'beads'] and b.train.steps.detector == 10                    # no shared lists / sections
    c = b.updated({'psf.iters': 5, 'train.steps.detector': 7}, {'name': 'c'}, movie='m.tif')
    assert c.psf.iters == 5 and c.train.steps == dict(detector=7, emitter=20) and c.name == 'c' and c.movie == 'm.tif'
    assert b.updated(psf=None).psf is None                                # a non-dict replaces a section
    assert b.replaced(psf=dict(iters=1)).psf == dict(iters=1)            # whole-section replacement
    assert b.updated(b.updated(name='z')).name == 'z'                    # merging another ConfigDict
    m = _base(); assert m.merge(name='q') is m and m.name == 'q'         # in place


def test_freeze_clone():
    b = _base().freeze()
    assert b.frozen and b.optics.frozen
    raises(TypeError, setattr, b, 'name', 'x'); raises(TypeError, setattr, b.optics, 'NA', 1)
    raises(TypeError, b.__setitem__, 'optics.NA', 1); raises(TypeError, b.__delitem__, 'name'); raises(TypeError, b.merge, name='x')
    c = b.updated(name='x'); c.optics.NA = 1.2                           # derived objects are editable
    assert not c.frozen and b.optics.NA == 1.49 and b.name == 'a'
    k = b.clone(); k.optics.NA = 1.3
    assert not k.frozen and b.optics.NA == 1.49 and k == b.updated(optics=dict(NA=1.3))
    assert copy.deepcopy(b) == b and pickle.loads(pickle.dumps(b)) == b and pickle.loads(pickle.dumps(b)).frozen


def test_to_dict():
    try:
        import torch
        t = torch.tensor([1.5, 2.5])
    except ImportError:
        t = [1.5, 2.5]
    c = ConfigDict(a=np.arange(3), b=np.float32(2.5), n=np.int64(4), t=t, tup=(1, 2), sec=dict(m=np.eye(2), p=(np.float64(1.0),)),
                   path=__import__('pathlib').Path('/x/y'), lst=[dict(q=(1, 2))], s={3, 1}, none=None, flag=True)
    d = c.to_dict()
    assert d == dict(a=[0, 1, 2], b=2.5, n=4, t=[1.5, 2.5], tup=[1, 2], sec=dict(m=[[1.0, 0.0], [0.0, 1.0]], p=[1.0]), path='/x/y',
                     lst=[dict(q=[1, 2])], s=[1, 3], none=None, flag=True)
    assert type(d) is dict and type(d['sec']) is dict and type(d['lst'][0]) is dict and type(d['b']) is float and type(d['n']) is int
    json.dumps(d)                                                        # plain enough for json
    assert 'ConfigDict(' in repr(c) and 'm:' in repr(c) and repr(ConfigDict()) == 'ConfigDict()'


def test_round_trip():
    c = _base().updated(arr=np.array([1.0, 2.0]), tup=(1, 2), none=None)
    with tempfile.TemporaryDirectory() as d:
        for ext in ('yaml', 'json'):
            f = os.path.join(d, 'c.' + ext); c.save(f)
            r = ConfigDict.load(f)
            assert r == c and r.to_dict() == c.to_dict() and r.optics.NA == 1.49 and r.arr == [1.0, 2.0] and list(r) == list(c)
        raises(ValueError, c.save, os.path.join(d, 'c.txt'))
        f = os.path.join(d, 'l.yaml'); open(f, 'w').write('- 1\n'); raises(ValueError, ConfigDict.load, f)


PYFILE = '''
import os
from simcode.util.cfgdict import ConfigDict
HERE = os.path.dirname(os.path.abspath(__file__))
base = ConfigDict(optics=dict(pixel_um=0.065), spots=dict(thr=5.0), gpus=[0]).freeze()
configs = [base.updated(name=n, movie=f'{HERE}/{n}.tif', out=f'{HERE}/out/{n}', process=dict(model=m), spots=dict(maxframe=1000 * (i + 1)))
           for i, (n, m) in enumerate([('a', 'm1809'), ('b', 'm25')])]
'''


def test_python_loader_and_schema():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'setup.py'); open(f, 'w').write(PYFILE)
        L = load_python(f)
        assert [c.name for c in L] == ['a', 'b'] and all(isinstance(c, ConfigDict) for c in L)
        assert L[1].movie == os.path.join(d, 'b.tif') and L[1].spots == dict(thr=5.0, maxframe=2000) and L[0].process.model == 'm1809'
        R = AC.load_all(f)                                                # schema: defaults filled in, plain dicts
        assert [c['name'] for c in R] == ['a', 'b'] and type(R[0]) is dict and R[0]['_file'] == f
        assert R[1]['optics'] == dict(AC.DEFAULTS['optics'], pixel_um=0.065) and R[1]['spots']['thr'] == 5.0 and R[1]['spots']['roi'] == 11
        assert R[1]['process']['model'] == 'm25' and R[0]['gpus'] == [0] and R[0]['psf'] == AC.DEFAULTS['psf']
        raises(ValueError, AC.load, f)                                    # two configs: not one acquisition
        g = os.path.join(d, 'one.py'); open(g, 'w').write("config = dict(name='x', movie='m.tif', out='o', psf=dict(iters=5))\n")
        assert AC.load(g)['psf']['iters'] == 5 and ConfigDict.load(g).psf.iters == 5 and AC.load(g)['_file'] == g
        h = os.path.join(d, 'named.py'); open(h, 'w').write("configs = dict(x=dict(movie='m.tif', out='o'), y=dict(movie='n.tif', out='p'))\n")
        assert [c['name'] for c in AC.load_all(h)] == ['x', 'y']
        for bad in ('x = 1\n', 'config = 3\n', 'configs = []\n', 'configs = [1]\n'):
            open(g, 'w').write(bad); raises((ValueError, TypeError), load_python, g)
        open(g, 'w').write("c = dict(name='x', movie='m', out='o')\nconfigs = [c, c]\n"); raises(ValueError, AC.load_all, g)   # same name twice


def test_schema_of_derived_config():
    base = ConfigDict(movie='m.tif', out='o', optics=dict(pixel_um=0.065), pattern=dict(frame_order='interleaved'))
    c = base.updated(name='x', train=dict(steps=dict(detector=5)), camera=dict(offset_adu=np.float32(100.0)), process=dict(crop=(5, 5, 390)))
    r = AC.load(c)
    assert type(r) is dict and r['optics']['pixel_um'] == 0.065 and r['optics']['NA'] == 1.49 and r['_file'] is None
    assert r['camera']['offset_adu'] == 100.0 and type(r['camera']['offset_adu']) is float and r['process']['crop'] == [5, 5, 390]
    assert r['train']['steps'] == dict(AC.DEFAULTS['train']['steps'], detector=5) and AC.needs_reorder(r)
    assert r == AC.load(c.to_dict()) and c.train.steps == dict(detector=5)     # same as the dict form; the input is not changed
    assert AC.load(r) == r                                                # a resolved config resolves to itself
    with tempfile.TemporaryDirectory() as d:                              # same as the yaml / json forms
        for ext in ('yaml', 'json'):
            f = os.path.join(d, 'c.' + ext); c.save(f)
            assert dict(AC.load(f), _file=None) == r
    e = raises(KeyError, AC.load, c.updated(spots=dict(thrr=3)))          # unknown keys are errors in every form
    assert 'spots.thrr' in str(e)
    raises(KeyError, AC.load, c.updated(colour=1))
    raises(ValueError, AC.load, base)                                     # name is required
    raises(ValueError, AC.load, c.updated(pattern=dict(steps=4)))         # validate()
    raises(ValueError, AC.load, c.updated(name=1809))


def test_example_configs():
    a, b = AC.load(os.path.join(CFG, '1809_2.py')), AC.load(os.path.join(CFG, '1809_2.yaml'))
    assert dict(a, _file=None) == dict(b, _file=None)
    L = AC.load_all(os.path.join(CFG, 'example_setup.py'))
    assert [(c['name'], c['process']['model'], c['train']['model']) for c in L] == [('1809_2', 'm1809', 'm1809'), ('2.5nM_2', 'm25', 'm25'), ('20nM_2', 'm25', 'm25')]
    assert len({c['movie'] for c in L}) == 3 and all(c['out'].endswith('/' + c['name']) and c['beads']['sign'] == -1 for c in L)
    assert all(c['camera'] == L[0]['camera'] and c['optics'] == AC.DEFAULTS['optics'] for c in L)


def test_pipeline_class():
    import simcode.pipeline as A
    from simcode.pipeline import localization_pipeline as P
    assert A.LocalizationPipeline is P.LocalizationPipeline and not hasattr(P, 'Acquisition') and not hasattr(P, 'Dataset')
    assert list(P.METHODS) == P.STAGES and all(callable(getattr(P.LocalizationPipeline, m)) and hasattr(P.LocalizationPipeline, '_st_' + s)
                                               for s, m in P.METHODS.items())
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'setup.py'); open(f, 'w').write(PYFILE)
        link = [os.path.join(P.REPO, 'out', 'auto', n) for n in 'ab']
        pre = [os.path.lexists(l) for l in link]
        try:
            L = P.LocalizationPipeline.from_file(f, log=lambda *a: None)
            assert [p.c['name'] for p in L] == ['a', 'b'] and L[1].cfg.spots.maxframe == 2000 and L[1].cfg.frozen
            assert json.load(open(os.path.join(d, 'out', 'b', 'config_resolved.json')))['process']['model'] == 'm25'
            (one,) = P.LocalizationPipeline.from_file(f, only=['b'], log=lambda *a: None)
            assert one.c['name'] == 'b' and P.LocalizationPipeline(L[0].cfg.updated(gpus=[2]), log=lambda *a: None).c['gpus'] == [2]
            raises(KeyError, P.LocalizationPipeline.from_file, f, only=['zz'])
            p = L[0]
            assert p._stage_name(p.estimate_psf) == 'psf' and p._stage_name('estimate_psf') == 'psf' and p._stage_name('psf') == 'psf'
            raises(ValueError, p._stage_name, 'nope'); raises(ValueError, p.run, only=['nope'])
            ran = []
            p._st_order = lambda: ran.append('order') or dict(checks=[])
            p._st_camera = lambda: ran.append('camera') or dict(checks=[dict(name='c', value=1, ok=False)])
            assert p.determine_frame_order()['stage'] == 'order' and p.done('order')
            p.determine_frame_order(); assert ran == ['order']             # finished: skipped
            p.determine_frame_order(redo=True); assert ran == ['order', 'order']
            raises(P.StageFailed, p.calibrate_camera); assert not p.done('camera')
            p.run(only=[p.determine_frame_order, p.calibrate_camera], redo=[p.calibrate_camera], force=True)
            assert ran == ['order', 'order', 'camera', 'camera'] and p.done('camera')
            raises(FileNotFoundError, p.run_dir)
        finally:
            for l, was in zip(link, pre):
                if not was and os.path.islink(l): os.remove(l)


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('ok', k)
