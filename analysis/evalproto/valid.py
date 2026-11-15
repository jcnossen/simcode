"""Part 2: validate GT-free metrics against GT on the punctate sims (m05p / m25p).  CPU only.
Methods (G3 window starts, final positions in 5-115 px): ours p1 (pass2_est, p >= 0.63), ours p2L (pass2_est2, p >= 0.63),
SIMCODE (b) = paper nets + sign-fixed NDI, DECODE (retrained hd_sum6, prob >= 0.5).
TRUTH (GT-matched rows, one-to-one <= 1 px on the detection position, global median offset removed, |e| < 50 nm; nm / axis):
  rms        total RMS (and incl. gross rows: rms_all)
  site       per-site static  = sqrt cov(e_i, e_j), rows of DIFFERENT events at the same docking site
  event      per-event static = sqrt(cov(same event, window starts >= 12 frames apart) - site^2)
  dyn        window-to-window = sqrt(var - cov(same event, >= 12 frames))
  ctx        extra lag-6 covariance (adjacent disjoint windows) beyond the event static (temporal context)
GT-FREE (the real-data protocol, on all rows of the set):
  std / adj300 / gap300 / gap600 FRC (lib.FRCS), NN proxy (own rows at f + 6, 1-D Gaussian + uniform), event-revisit proxy
  (linked G6 events, nearest later event mean >= 300 frames after the end within 50 nm; sigma of the difference / sqrt2;
  'rev_st' = revisit^2 - nn^2 <1/n>: the per-event static estimate), doubly-shared NN on identical rows.
SENSITIVITY: the metric after adding iid N(0, a) nm per row ('+a') or a per-linked-event static N(0, a) ('+ev a').  same algorithm and loader (analysis/baselines/compare.py
load_method); the movie and the method outputs come from a datasets.py json spec (mode sim): "gt", sets of kind ours (dir, tag),
simcode (npz, NDI rows; the "reference" set, labelled SIMCODE here) and decode (npz with frame, x, y, sx, sy, phot, prob).
usage: python analysis/evalproto/valid.py <sim dataset.json> [out_dir]  -> valid_<dataset>.json / .txt"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
import datasets as DS
from simcode.evaluation import metrics as E
import compare as C       # analysis/baselines/compare.py
PX = E.PX
spec = sys.argv[1]; SPEC = json.load(open(spec))
m = SPEC.get('name', '')
G = dict(np.load(SPEC['gt'])); NF = int(G['nf'])
GW = E.gt_windows_fast(G, grid=3)
site_of = G['site'] if 'site' in G else -np.ones(len(G['x']), np.int64)
dim_of = G['dim'] if 'dim' in G else np.zeros(len(G['x']))
rng = np.random.default_rng(11)
OUT = {}; LINES = []
def log(s=''):
    print(s, flush=True); LINES.append(s)


def load(kind, path):
    if kind == 'decode':
        R = np.load(path); k = R['prob'] >= 0.5
        D = dict(w=R['frame'][k].astype(int), xd=R['x'][k], yd=R['y'][k], x=R['x'][k], y=R['y'][k], sx=R['sx'][k], sy=R['sy'][k], phot=R['phot'][k])
    else:
        D = C.load_method(kind, path, NF)
        if 'p' in D: D = {a: b[D['p'] >= 0.63] for a, b in D.items()}
    ok = (D['w'] % 3 == 0) & (D['w'] <= NF - 6) & (D['x'] >= 5) & (D['x'] < 115) & (D['y'] >= 5) & (D['y'] < 115) & np.isfinite(D['x'])
    D = {a: np.asarray(b)[ok] for a, b in D.items()}
    D['f'] = D.pop('w').astype(np.int64)
    gi = E.match_fast(GW, D['f'], D['xd'], D['yd'])
    D['gi'] = gi; g = gi >= 0
    e = np.full((len(gi), 2), np.nan); e[g] = np.c_[D['x'][g] - GW[gi[g], 2], D['y'][g] - GW[gi[g], 3]]
    P = np.where(g, GW[np.maximum(gi, 0), 4], np.nan)
    sel = g & (P >= 400) & (P < 2000)
    D['off'] = np.median(e[sel], 0); D['e'] = (e - D['off']) * PX
    D['ev'] = np.where(g, GW[np.maximum(gi, 0), 1], -1).astype(np.int64)
    D['site'] = np.where(g, site_of[np.maximum(D['ev'], 0)], -1); D['site'][~g] = -1
    D['P'] = P; D['dim'] = np.where(g, dim_of[np.maximum(D['ev'], 0)], 0)
    D['x'] = D['x'] - D['off'][0]; D['y'] = D['y'] - D['off'][1]        # same global offset removal for the GT-free metrics
    return D


REF = SPEC.get('reference', 'SIMCODE (b)')
METH = {}
for k, v in SPEC['sets'].items():
    if v['kind'] == 'ours': METH[k] = ('ours_' + v.get('tag', 'est'), v['dir'])
    elif v['kind'] == 'simcode': METH['SIMCODE' if k == REF else k] = ('sc_ndi', v['npz'])
    elif v['kind'] == 'decode': METH[k] = ('decode', v['npz'])
D = {}
for k, (kind, path) in METH.items():
    try:
        D[k] = load(kind, path)
    except FileNotFoundError as ex:
        log(f'# {k}: missing ({ex})'); continue
    g = D[k]['gi'] >= 0
    log(f'# {k}: G3 rows {len(g)}, GT-matched {g.mean():.3f}, offset nm {np.round(D[k]["off"] * PX, 2).tolist()}')


# ------------------------------------------------------------------ truth
truth = E.truth


# ------------------------------------------------------------------ GT-free metrics
nn_own, revisit_proxy, metrics, perturb = E.nn_own, E.revisit_proxy, E.metrics, E.perturb
E.RNG = rng


def fm(d, keys):
    return ' | '.join(f'{d.get(k, np.nan):.2f}' for k in keys)


TK = ('rms', 'dyn', 'event', 'site', 'ctx')
MK = ('std', 'adj300', 'gap300', 'gap600', 'nn', 'nn12', 'rev', 'rev_st')
log(f'\n## {m}: pools (each method its own full G3 set)')
log('| method | rows | truth rms | dyn | event | site | ctx | FRC std | adj300 | gap300 | gap600 | NN | NN12 | revisit | rev static | n rev pairs |')
log('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
OUT['pool'] = {}
for k, M in D.items():
    i = np.arange(len(M['x']))
    T = truth(M, i); Mt = metrics(M['x'], M['y'], M['f'])
    OUT['pool'][k] = dict(truth=T, metr=Mt)
    log(f'| {k} | {len(i)} | {fm(T, TK)} | {fm(Mt, MK)} | {Mt["n_rev"]} |')

# ------------------------------------------------------------------ identical rows vs SIMCODE, photon-matched / random pools
S = D['SIMCODE']
ia_all = {}
log(f'\n## {m}: identical rows (exact start, mutual nearest < 0.5 px on final positions), each method on the SAME windows')
log('| pair | n | truth rms A / B | dyn A / B | event A / B | site A / B | FRC std A / B | gap300 A / B | gap600 A / B | NN doubly-shared A / B | NN12 doubly-shared A / B | revisit A / B | rev static A / B |')
log('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
OUT['ident'] = {}
for k in [a for a in D if a != 'SIMCODE']:
    A = D[k]
    ia, ib = E.pair(A['x'], A['y'], A['f'], S['x'], S['y'], S['f'])
    ia_all[k] = (ia, ib)
    Ta, Tb = truth(A, ia), truth(S, ib)
    Ma = metrics(A['x'][ia], A['y'][ia], A['f'][ia]); Mb = metrics(S['x'][ib], S['y'][ib], S['f'][ib])
    sa, sb, nn2 = E.nn_proxy_shared({kk: A[kk] for kk in ('x', 'y', 'f')}, {kk: S[kk] for kk in ('x', 'y', 'f')}, ia, ib)
    sa12, sb12, n12 = E.nn_proxy_shared({kk: A[kk] for kk in ('x', 'y', 'f')}, {kk: S[kk] for kk in ('x', 'y', 'f')}, ia, ib, dt=12)
    OUT['ident'][k] = dict(n=len(ia), truth=(Ta, Tb), metr=(Ma, Mb), nn2=(sa, sb, nn2), nn12=(sa12, sb12, n12))
    ab = lambda a, b, key: f'{a.get(key, np.nan):.2f} / {b.get(key, np.nan):.2f}'
    log(f'| {k} vs SC | {len(ia)} | {ab(Ta, Tb, "rms")} | {ab(Ta, Tb, "dyn")} | {ab(Ta, Tb, "event")} | {ab(Ta, Tb, "site")} | {ab(Ma, Mb, "std")} | '
        f'{ab(Ma, Mb, "gap300")} | {ab(Ma, Mb, "gap600")} | {sa:.2f} / {sb:.2f} (n {nn2}) | {sa12:.2f} / {sb12:.2f} | {ab(Ma, Mb, "rev")} | {ab(Ma, Mb, "rev_st")} |')

log(f'\n## {m}: equal-N pools vs SIMCODE (N = SIMCODE rows): random and photon-matched (ours photons / kappa on the identical pairs)')
log('| method | pool | truth rms | dyn | event | site | FRC std | gap300 | gap600 | NN | NN12 | revisit |')
log('|---|---|---|---|---|---|---|---|---|---|---|---|')
NS = len(S['x']); EB = np.logspace(np.log10(200), np.log10(3000), 13)
OUT['pools'] = {}
for k in [a for a in D if a != 'SIMCODE']:
    A = D[k]; ia, ib = ia_all[k]
    kap = float(np.median(A['phot'][ia] / S['phot'][ib]))
    pa = A['phot'] / kap
    for lab in ('random', 'photon-matched'):
        if lab == 'random':
            i = rng.choice(len(A['x']), min(NS, len(A['x'])), replace=False)
        else:
            b = np.digitize(pa, EB) - 1; b[(pa < EB[0]) | (pa >= EB[-1])] = -1
            hs = np.bincount(b[b >= 0], minlength=12); ht = np.histogram(S['phot'], EB)[0]; ft = ht / ht.sum()
            N = min(NS, int(np.floor(np.min(hs[ft > 0] / ft[ft > 0]))))
            i = np.concatenate([rng.choice(np.nonzero(b == q)[0], int(round(N * ft[q])), replace=False) for q in range(12) if round(N * ft[q])])
        i = np.sort(i)
        T = truth(A, i); Mt = metrics(A['x'][i], A['y'][i], A['f'][i])
        OUT['pools'][f'{k} {lab}'] = dict(n=len(i), truth=T, metr=Mt)
        log(f'| {k} | {lab} (N {len(i)}) | {fm(T, ("rms", "dyn", "event", "site"))} | {fm(Mt, ("std", "gap300", "gap600", "nn", "nn12", "rev"))} |')
T = OUT['pool']['SIMCODE']['truth']; Mt = OUT['pool']['SIMCODE']['metr']
log(f'| SIMCODE | all (N {NS}) | {fm(T, ("rms", "dyn", "event", "site"))} | {fm(Mt, ("std", "gap300", "gap600", "nn", "nn12", "rev"))} |')

# ------------------------------------------------------------------ sensitivity on identical rows (ours p1 rows of the p1-SC pairs)
log(f'\n## {m}: injected-noise sensitivity, ours p1 identical rows (iid per row / static per linked event)')
log('| injection | FRC std | adj300 | gap300 | gap600 | NN | NN12 | revisit |')
log('|---|---|---|---|---|---|---|---|')
A = D['ours p1']; ia, ib = ia_all['ours p1']
x0, y0, f0 = A['x'][ia], A['y'][ia], A['f'][ia]
OUT['sens'] = {}
for mode in ('iid', 'event'):
    for a in (0, 3, 6):
        if mode == 'event' and a == 0: continue
        x, y = perturb(x0, y0, f0, a, mode)
        Mt = metrics(x, y, f0)
        OUT['sens'][f'{mode} {a}'] = Mt
        log(f'| {mode} +{a} nm | {fm(Mt, ("std", "adj300", "gap300", "gap600", "nn", "nn12", "rev"))} |')

# ------------------------------------------------------------------ synthetic error components on the real selection
log(f'\n## {m}: synthetic errors on the ours-p1 GT-matched G3 rows (GT + known components, nm / axis)')
log('| dyn / event / site | truth rms | FRC std | adj300 | gap300 | gap600 | NN | NN12 | revisit | rev static |')
log('|---|---|---|---|---|---|---|---|---|---|')
g = A['gi'] >= 0
gx, gy = GW[A['gi'][g], 2], GW[A['gi'][g], 3]; evg, stg, fg = A['ev'][g], A['site'][g], A['f'][g]
ue, ie = np.unique(evg, return_inverse=True); us, is_ = np.unique(np.where(stg >= 0, stg, -1 - evg), return_inverse=True)
OUT['synth'] = {}
for dy, evs, sts in ((10, 0, 0), (7.07, 7.07, 0), (7.07, 0, 7.07), (7.07, 5, 5), (5, 0, 0), (5, 5, 0), (5, 0, 5)):
    e = (rng.normal(0, dy, (len(fg), 2)) + rng.normal(0, evs, (len(ue), 2))[ie] + rng.normal(0, sts, (len(us), 2))[is_]) / PX
    x, y = gx + e[:, 0], gy + e[:, 1]
    Mt = metrics(x, y, fg)
    OUT['synth'][f'{dy}/{evs}/{sts}'] = Mt
    log(f'| {dy:.1f} / {evs:.1f} / {sts:.1f} | {np.sqrt((e ** 2).mean()) * PX:.2f} | {fm(Mt, ("std", "adj300", "gap300", "gap600", "nn", "nn12", "rev", "rev_st"))} |')

fn = DS.out_path(spec, 'valid', sys.argv[2] if len(sys.argv) > 2 else None)
json.dump(OUT, open(fn + '.json', 'w'), default=float, indent=1); open(fn + '.txt', 'w').write('\n'.join(LINES) + '\n')
