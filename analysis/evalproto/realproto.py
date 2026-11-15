"""Part 3: real-data comparisons re-stated with the validated GT-free protocol (lib.metrics; same code as valid.py). CPU only.
0.5 / 2.5 nM: critic35/c35lib loaders (own DME v2 drift, crop 5-115), ours pass 1 ('p1'), p2L ('p2L'), SIMCODE (b); grid G3.
20 nM: real20/lib20 loaders (each method its own DME v2), ours p1G / p2G, SIMCODE (b), SMLM / SF / DECODE summed (paper
DECODE models) / ...; grid G6 (DECODE summed exists on starts % 6 == 0 only).
Per ours method vs SIMCODE: identical rows (exact start, mutual nearest < 0.5 px), doubly-shared NN, revisit proxy, FRC std /
adj300 / gap300 / gap600; sensitivity (ours identical rows + iid 3 / 6 nm, + per-linked-event static 3 / 6 nm); pools
(random N_SC, photon-matched N <= N_SC; SIMCODE full). 20 nM baselines: identical rows vs SIMCODE + pools at N_SC.  same algorithm; the data = a datasets.py json spec
(mode real; the 20 nM baselines are sets with "baseline": true, kind "hdf5"; see datasets.py).
usage: python analysis/evalproto/realproto.py <dataset.json> [out_dir]  -> realproto_<dataset>.txt / .json"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as DS
from simcode.evaluation import metrics as E
spec = sys.argv[1]
rng = np.random.default_rng(36); E.RNG = rng
LINES = []; OUT = {}
def log(s=''):
    print(s, flush=True); LINES.append(s)


def std(D, grid):
    k = D['f'] % grid == 0
    o = dict(x=D['x'][k], y=D['y'][k], f=D['f'][k].astype(np.int64))
    o['ph'] = (D['phot'] if 'phot' in D else D.get('photons', D.get('ph')))[k]
    return o


S0, MS0, META = DS.load(spec, baselines=True); grid = META['grid']; c = META['name']
OUT['conc'] = c
BL = {k for k, v in json.load(open(spec))['sets'].items() if v.get('baseline')}
MS = {k: std(v, grid) for k, v in MS0.items() if k not in BL}
BASE = {k: std(v, grid) for k, v in MS0.items() if k in BL}
S = std(S0, grid)
NS = len(S['x'])
log(f'# {c}, grid G{grid}; rows: SIMCODE {NS}; ' + ', '.join(f'{k} {len(v["x"])}' for k, v in {**MS, **BASE}.items()))
MK = ('std', 'adj300', 'gap300', 'gap600', 'nn', 'nn12', 'rev', 'rev_st')
fm = lambda d, keys=MK: ' | '.join(f'{d.get(k, np.nan):.2f}' for k in keys)

# ------------------------------------------------------------------ identical rows
log(f'\n## identical rows vs SIMCODE (same windows): FRC (nm), NN = doubly-shared consecutive-window proxy (nm / axis), revisit proxy')
log('| method | n | FRC std A / SC | adj300 A / SC | gap300 A / SC | gap600 A / SC | NN doubly-shared A / SC (n) | NN12 doubly-shared A / SC (n) | revisit A / SC | rev static A / SC |')
log('|---|---|---|---|---|---|---|---|---|---|')
OUT['ident'] = {}; PAIRS = {}
for k, A in {**MS, **BASE}.items():
    ia, ib = E.pair(A['x'], A['y'], A['f'], S['x'], S['y'], S['f'])
    PAIRS[k] = (ia, ib)
    if len(ia) < 2000:
        log(f'| {k} | {len(ia)} | (too few) |'); continue
    Ma = E.metrics(A['x'][ia], A['y'][ia], A['f'][ia]); Mb = E.metrics(S['x'][ib], S['y'][ib], S['f'][ib])
    sa, sb, n2 = E.nn_proxy_shared(A, S, ia, ib)
    s12a, s12b, n12 = E.nn_proxy_shared(A, S, ia, ib, dt=12)
    OUT['ident'][k] = dict(n=int(len(ia)), A=Ma, SC=Mb, nn2=(sa, sb, n2), nn12=(s12a, s12b, n12))
    ab = lambda key: f'{Ma[key]:.2f} / {Mb[key]:.2f}'
    log(f'| {k} | {len(ia)} | {ab("std")} | {ab("adj300")} | {ab("gap300")} | {ab("gap600")} | {sa:.2f} / {sb:.2f} ({n2}) | {s12a:.2f} / {s12b:.2f} ({n12}) | {ab("rev")} | {ab("rev_st")} |')

# ------------------------------------------------------------------ sensitivity (ours p1 identical rows)
log(f'\n## sensitivity: ours p1 identical rows + injected noise (iid per row / static per linked G{grid} event)')
log('| injection | ' + ' | '.join(MK) + ' |'); log('|---' * (len(MK) + 1) + '|')
OUT['sens'] = {}
ia, ib = PAIRS['ours p1']; A = MS['ours p1']
x0, y0, f0 = A['x'][ia], A['y'][ia], A['f'][ia]
OUT['sens']['0'] = OUT['ident']['ours p1']['A']
log(f'| none | {fm(OUT["sens"]["0"])} |')
for mode in ('iid', 'event'):
    for a in (3, 6):
        x, y = E.perturb(x0, y0, f0, a, mode, grid=grid)
        Mt = E.metrics(x, y, f0); OUT['sens'][f'{mode} {a}'] = Mt
        log(f'| {mode} +{a} nm | {fm(Mt)} |')

# ------------------------------------------------------------------ pools
log(f'\n## pools (selection level): random N = N_SC and photon-matched (own photons / kappa from the identical pairs, 200-3000 SC ph)')
log('| method | pool | N | ' + ' | '.join(MK) + ' |'); log('|---' * (len(MK) + 3) + '|')
OUT['pools'] = {}
EB = np.logspace(np.log10(200), np.log10(3000), 13)
Mt = E.metrics(S['x'], S['y'], S['f']); OUT['pools']['SIMCODE all'] = dict(n=NS, m=Mt)
log(f'| SIMCODE | all | {NS} | {fm(Mt)} |')
for k, A in {**MS, **BASE}.items():
    ia, ib = PAIRS[k]
    for lab in ('random', 'photon-matched'):
        if lab == 'random':
            if len(A['x']) < NS: i = np.arange(len(A['x']))
            else: i = np.sort(rng.choice(len(A['x']), NS, replace=False))
        else:
            if k in BASE or len(ia) < 2000: continue
            kap = float(np.median(A['ph'][ia] / S['ph'][ib])); pa = A['ph'] / kap
            b = np.digitize(pa, EB) - 1; b[(pa < EB[0]) | (pa >= EB[-1])] = -1
            hs = np.bincount(b[b >= 0], minlength=12); ht = np.histogram(S['ph'], EB)[0]; ft = ht / ht.sum()
            N = min(NS, int(np.floor(np.min(hs[ft > 0] / ft[ft > 0]))))
            i = np.sort(np.concatenate([rng.choice(np.nonzero(b == q)[0], int(round(N * ft[q])), replace=False) for q in range(12) if round(N * ft[q])]))
        Mt = E.metrics(A['x'][i], A['y'][i], A['f'][i]); OUT['pools'][f'{k} {lab}'] = dict(n=int(len(i)), m=Mt)
        log(f'| {k} | {lab} | {len(i)} | {fm(Mt)} |')
fn = DS.out_path(spec, 'realproto', sys.argv[2] if len(sys.argv) > 2 else None)
json.dump(OUT, open(fn + '.json', 'w'), default=float, indent=1); open(fn + '.txt', 'w').write('\n'.join(LINES) + '\n')
