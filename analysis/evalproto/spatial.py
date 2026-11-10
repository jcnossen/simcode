"""critic 37 #2 + coordinator (depth-only test): two spatial drivers of the strict product, the illumination envelope and the
pattern depth (modulation) field. For every covpool selection (same code: covpool.py up to the metrics) the kept fraction per
tercile of (a) the envelope and (b) the depth factor, relative to ours-all (SIMCODE: its row share per tercile / ours-all
share), reported as the ratio low tercile / high tercile (1 = spatially neutral).
Envelope: real = envmap.py Gaussian fit (raw sensor coordinates); sim = the movie's knobs env_* (m*pe; none in m*pd).
Depth: real 0.5 / 2.5 = a depth field_own.pt (raw coordinates); sim = the movie's true_field.pt; 20 nM: none.  same algorithm; the dataset is a datasets.py json spec
with two optional extra keys:
  "envmap": "<envmap_<tag>.json>" (+ "envmap_key": "<key in it>", default the only key)   real: envelope
  "depth_field": "<field_own.pt>"                                                         real: depth field
  sim: the envelope comes from the GT knobs (sim_gt.npz 'knobs' over.env_*) and the depth from "true_field" (default
       <dir of gt>/true_field.pt).
usage: python analysis/evalproto/spatial.py <dataset.json> [out_dir]   -> spatial_<dataset>.txt / .json"""
import os, sys, json, numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = json.load(open(sys.argv[1])); _OUTD = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = sys.argv[:2]
src = open(os.path.join(_HERE, 'covpool.py')).read().split('MKF = ')[0]
__file__ = os.path.join(_HERE, 'covpool.py')
exec(src)
from simcode2.pipeline.pattern_field import field_at


def env_fn(P):
    def f(x, y):
        dx, dy = x - P['x0'], y - P['y0']; c, s = np.cos(P['th']), np.sin(P['th'])
        u = c * dx + s * dy; v = -s * dx + c * dy
        return np.exp(-0.5 * (u ** 2 / P['s1'] ** 2 + v ** 2 / P['s2'] ** 2))
    return f


ENV = DEP = None
if mode == 'real':
    if _SPEC.get('envmap'):
        J = json.load(open(_SPEC['envmap'])); ENV = env_fn(J[_SPEC.get('envmap_key', next(iter(J)))]['med'])
    if _SPEC.get('depth_field'):
        Fd = torch.load(_SPEC['depth_field'], weights_only=False)
        DEP = lambda x, y: (field_at(Fd, x, y)[1] * np.array(Fd['depth_norm'])).mean(1)
else:
    G = np.load(_SPEC['gt']); kn = json.loads(str(G['knobs']))['over']
    if 'env_s1' in kn: ENV = env_fn({k: kn['env_' + k] for k in ('x0', 'y0', 's1', 's2', 'th')})
    Ft = torch.load(_SPEC.get('true_field', os.path.join(os.path.dirname(_SPEC['gt']), 'true_field.pt')), weights_only=False)
    DEP = lambda x, y: field_at(Ft, x, y)[1].mean(1)


def coords(D):
    # real: raw sensor coordinates (the envelope / depth field are fixed on the sensor); sim: the offset-corrected x, y (as the
    # dev simload.py rows, which had no raw columns)
    return (D['xr'], D['yr']) if mode == 'real' else (D['x'], D['y'])


def terc(D, fn):
    x, y = coords(D); return fn(np.asarray(x, float), np.asarray(y, float))


OUT = {}; LINES = []
def log(s=''):
    print(s, flush=True); LINES.append(s)


log(f'# {tag if tag.startswith(mode) else mode + " " + tag}: kept fraction low / high tercile (relative to ours-all of the same pass); env = illumination envelope, depth = '
    'pattern modulation-depth factor; 1 = spatially neutral')
log('| set | N | env low/high | depth low/high |'); log('|---|---|---|---|')
for nm, A in MS.items():
    V, info = variants(A, S)
    V = {'all': np.arange(len(A['f'])), **V}
    vals = {}
    for dn, fn in (('env', ENV), ('depth', DEP)):
        if fn is None: continue
        va = terc(A, fn); ed = np.percentile(va, [100 / 3, 200 / 3]); ra = np.digitize(va, ed); base = np.bincount(ra, minlength=3) / len(ra)
        vs = terc(S, fn); rs = np.digitize(vs, ed); sh = np.bincount(rs, minlength=3) / len(rs)
        vals[dn] = dict(SC=float((sh[0] / base[0]) / (sh[2] / base[2])))
        for lab, i in V.items():
            h = np.bincount(ra[i], minlength=3) / len(i); vals[dn][lab] = float((h[0] / base[0]) / (h[2] / base[2]))
    if nm == 'ours p1':
        log(f'| SIMCODE (b) all | {len(S["f"])} | ' + ' | '.join(f'{vals[d]["SC"]:.2f}' if d in vals else '–' for d in ('env', 'depth')) + ' |')
    for lab, i in V.items():
        log(f'| {nm} {lab} | {len(i)} | ' + ' | '.join(f'{vals[d][lab]:.2f}' if d in vals else '–' for d in ('env', 'depth')) + ' |')
    OUT[nm] = vals
if ENV is not None and DEP is not None:
    A = MS['ours p1']; log(f'corr(envelope, depth) over ours p1 rows: {np.corrcoef(terc(A, ENV), terc(A, DEP))[0, 1]:+.2f}')
fn = DS.out_path(sys.argv[1], 'spatial', _OUTD)
json.dump(OUT, open(fn + '.json', 'w'), indent=1); open(fn + '.txt', 'w').write('\n'.join(LINES) + '\n')
