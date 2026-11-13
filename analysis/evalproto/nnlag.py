"""Identical-row NN, reported as the mean ratio ours / reference (SIMCODE) over lags 12-36 frames (doubly-shared NN,
lib.lagavg_shared: 0.01-nm sigma grid, event-block Poisson bootstrap over linked events, 200 replicates; sig1d uniform-weight
tail indicator). Also the per-lag table. Sim datasets (mode sim) add the GT truth dyn / rms ratios of the same rows.
usage: python analysis/evalproto/nnlag.py <dataset.json> [out_dir]   -> nnlag_<dataset>.txt / .json"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as DS
from simcode.evalproto import lib as E
spec = sys.argv[1]
LINES = []; OUT = {}
def log(s=''):
    print(s, flush=True); LINES.append(s)


S, MS, META = DS.load(spec)
mode, tag, grid = META['mode'], META['name'], META['grid']

log(f'# {mode} {tag}: grid G{grid}; NN = doubly-shared, identical rows (exact start, mutual nearest < 0.5 px); lags {E.LAGS}; '
    f'sigma grid 0.01 nm; CI = event-block Poisson bootstrap (200, events = linked G{grid} events of the method rows)')
log('| method | identical rows | NN ours | NN SC | ratio (mean over lags) | 95 % CI | uniform w ours / SC | quadrature SC excess (nm) |' + (' GT dyn ratio | GT rms ratio |' if mode == 'sim' else ''))
log('|---|---|---|---|---|---|---|---|' + ('---|---|' if mode == 'sim' else ''))
for k, A in MS.items():
    ia, ib = E.pair(A['x'], A['y'], A['f'], S['x'], S['y'], S['f'])
    if len(ia) < 2000: log(f'| {k} | {len(ia)} | too few |'); continue
    R = E.lagavg_shared(A, S, ia, ib, grid)
    q = np.sign(R['sB'] - R['sA']) * np.sqrt(abs(R['sB'] ** 2 - R['sA'] ** 2))
    extra = ''
    if mode == 'sim':
        Ta = E.truth(A, ia); Tb = E.truth(S, ib); R['truth'] = dict(A=Ta, SC=Tb)
        extra = f' {Ta["dyn"] / Tb["dyn"]:.3f} | {Ta["rms"] / Tb["rms"]:.3f} |'
    OUT[k] = dict(R, n_ident=int(len(ia)))
    log(f'| {k} | {len(ia)} | {R["sA"]:.2f} | {R["sB"]:.2f} | **{R["ratio"]:.3f}** | {R["lo"]:.3f}–{R["hi"]:.3f} | {R["uA"]:.3f} / {R["uB"]:.3f} | {q:.2f} |' + extra)
    log('    per lag: ' + '; '.join(f'{P["dt"]}: {P["sa"]:.2f}/{P["sb"]:.2f} = {P["sa"] / P["sb"]:.3f} (n {P["n"]})' for P in R['per']))
fn = DS.out_path(spec, 'nnlag', sys.argv[2] if len(sys.argv) > 2 else None)
json.dump(OUT, open(fn + '.json', 'w'), default=float, indent=1); open(fn + '.txt', 'w').write('\n'.join(LINES) + '\n')
