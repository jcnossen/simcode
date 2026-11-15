"""critic 37 #1 / #2: coverage-matched and tile-stratified pool comparisons, ours sigma-filtered vs SIMCODE (b) full output.
Coverage = distinct GT events (sim; GT-matched rows |e| < 50 nm) or occupied 20-nm bins (real; drift-corrected, 5-115 crop).
Variants of ours (G rows, p >= 0.63, sigma = mean of the calibrated axes):
  random_N_SC  random N_SC rows (selection baseline)
  topN_SC      global top N_SC by sigma (the old 'strict at matched N')
  cov          global top N by sigma, N = smallest with coverage >= SIMCODE's (first-occurrence ranks, exact)
  tileN        per 10-px tile: top N_SC(tile) by sigma (SIMCODE's row count per tile)
  tilecov      per tile: top N by sigma with the tile's coverage = SIMCODE's in that tile
  tileq        per-tile sigma quantile: keep sigma rank / n(tile) <= q in every tile, q set so that global coverage = SIMCODE's
  sigcov       a fixed sigma threshold (global) = cov (reported as the threshold)
Reported per pool: N, coverage (events, sites: sim; occupied 20-nm bins: all), centre / edge row ratio (5 x 5 central tiles vs
the edge ring), kept-fraction CV over tiles (pool / ours-all), median photons, sim GT truth (rms, dyn, event, site, zcal rms
of e / sigma), NN lag-averaged own (lags 12-36, lib.lagavg_own; ratio vs SIMCODE full with event-block bootstrap CI), FRC std.
usage: python analysis/evalproto/covpool.py <dataset.json> [out_dir]   -> covpool_<dataset>.txt / .json"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as DS
from simcode.evaluation import metrics as E
spec = sys.argv[1]
PX = E.PX
LINES = []; OUT = {}
def log(s=''):
    print(s, flush=True); LINES.append(s)


def sub(D, i):
    n = len(D['f'])
    return {k: (v[i] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in D.items()}


S, MS, META = DS.load(spec)
mode, tag, grid = META['mode'], META['name'], META['grid']

NB20 = int(110 * PX / 20)


def tile_of(D):
    ix = np.clip(((D['x'] - 5) / 10).astype(int), 0, 10); iy = np.clip(((D['y'] - 5) / 10).astype(int), 0, 10)
    return iy * 11 + ix


def bin20(D):
    ix = np.clip(((D['x'] - 5) / 110 * NB20).astype(int), 0, NB20 - 1); iy = np.clip(((D['y'] - 5) / 110 * NB20).astype(int), 0, NB20 - 1)
    return iy * NB20 + ix


def cov_key(D):
    """coverage unit per row (-1 = does not count): sim = GT event of a GT-matched row with |e| < 50 nm; real = 20-nm bin"""
    if mode == 'sim':
        ok = np.isfinite(D['e']).all(1) & (D['ev'] >= 0)
        ok[ok] &= np.hypot(D['e'][ok, 0], D['e'][ok, 1]) < 50
        return np.where(ok, D['ev'], -1)
    return bin20(D)


def coverage(D, i=None):
    k = cov_key(D) if i is None else cov_key(D)[i]
    return len(np.unique(k[k >= 0]))


def n_for_cov(keys_sorted, target):
    """smallest N such that the first N rows (in this order) cover >= target distinct keys"""
    k = keys_sorted; ok = k >= 0
    u, first = np.unique(k[ok], return_index=True)
    fi = np.sort(np.nonzero(ok)[0][first])
    return int(fi[min(target, len(fi)) - 1] + 1)


def variants(A, S):
    sig = A['sig']; o = np.argsort(sig, kind='stable'); N = len(S['f']); kA = cov_key(A); kS = cov_key(S)
    tA, tS = tile_of(A), tile_of(S)
    V = {}
    V['random_N_SC'] = np.sort(np.random.default_rng(5).choice(len(sig), min(N, len(sig)), replace=False))
    V['topN_SC'] = np.sort(o[:N])
    Nc = n_for_cov(kA[o], coverage(S)); V['cov'] = np.sort(o[:Nc])
    thr = float(sig[o[Nc - 1]])
    tN, tC = [], []
    for t in range(121):
        it = o[tA[o] == t]
        if not len(it): continue
        ns = int((tS == t).sum()); tN.append(it[:ns])
        cs = len(np.unique(kS[(tS == t) & (kS >= 0)]))
        if cs: tC.append(it[:n_for_cov(kA[it], cs)])
    V['tileN'] = np.sort(np.concatenate(tN)); V['tilecov'] = np.sort(np.concatenate(tC))
    # per-tile sigma quantile: rank within tile / n(tile)
    u = np.empty(len(sig))
    for t in range(121):
        it = o[tA[o] == t]
        if len(it): u[it] = (np.arange(len(it)) + 0.5) / len(it)
    ou = np.argsort(u, kind='stable'); Nq = n_for_cov(kA[ou], coverage(S)); V['tileq'] = np.sort(ou[:Nq])
    return V, dict(N_cov=Nc, sig_thr=thr, q=float(u[ou[Nq - 1]]))


def geo(D, i, ref=None):
    """centre / edge row ratio (5 x 5 central tiles vs the edge ring) and the kept-fraction CV over tiles vs ref (ours all)"""
    T = np.bincount(tile_of(D)[i], minlength=121).reshape(11, 11).astype(float)
    ce = T[3:8, 3:8].mean() / np.r_[T[0], T[-1], T[1:-1, 0], T[1:-1, -1]].mean()
    if ref is None: return ce, np.nan
    q = T / np.maximum(ref, 1); m = ref > 50
    return ce, float(q[m].std() / q[m].mean())


MKF = lambda D, i: E.frc_std(D['x'][i], D['y'][i], D['f'][i])
REF = {}
log(f'# {mode} {tag}: grid G{grid}; SIMCODE rows {len(S["f"])}; coverage unit = ' + ('distinct GT events (GT-matched, |e| < 50 nm)' if mode == 'sim' else 'occupied 20-nm bins'))
nnS = E.lagavg_own(S['x'], S['y'], S['f'], grid)
hdr = '| set | N | events | sites | occ 20-nm bins | centre/edge | kept-frac CV | med ph | med sigma |' + (' truth rms | dyn | event | site | zcal |' if mode == 'sim' else '') + ' NN lag-avg | NN ratio vs SC (95 % CI) | NN unif w | FRC std |'
log(hdr); log('|---' * (hdr.count('|') - 1) + '|')


def row(lab, D, i, ref=None, nn=None):
    i = np.asarray(i)
    occ = len(np.unique(bin20(D)[i]))
    ev = st = np.nan
    if mode == 'sim':
        k = cov_key(D)[i]; ev = len(np.unique(k[k >= 0])); s_ = D['site'][i][k >= 0]; st = len(np.unique(s_[s_ >= 0]))
    ce, cv = geo(D, i, ref)
    T = {}
    if mode == 'sim':
        T = E.truth(D, i)
        e = D['e'][i]; g = np.isfinite(e).all(1); g[g] &= np.hypot(e[g, 0], e[g, 1]) < 50
        z = e[g] / D['sig'][i][g, None]; T['zcal'] = float(np.sqrt((z ** 2).mean())); T['zrob'] = float(np.median(np.abs(z)) / 0.6745)
    nn = nn if nn is not None else E.lagavg_own(D['x'][i], D['y'][i], D['f'][i], grid)
    rc = E.ratio_ci(nn, nnS)
    fr = MKF(D, i)
    r = dict(n=int(len(i)), events=ev, sites=st, occ=occ, ce=ce, cv=cv, ph=float(np.median(D['phot'][i])), sig=float(np.median(D['sig'][i])),
             truth=T, nn=nn['s'], nn_u=nn['u'], nn_ratio=rc, frc=fr)
    OUT[lab] = r
    tt = (f' {T["rms"]:.2f} | {T["dyn"]:.2f} | {T["event"]:.2f} | {T["site"]:.2f} | {T["zcal"]:.2f} |' if mode == 'sim' else '')
    log(f'| {lab} | {len(i)} | {ev} | {st} | {occ} | {ce:.2f} | {cv:.2f} | {r["ph"]:.0f} | {r["sig"]:.2f} |{tt} {nn["s"]:.2f} | {rc["ratio"]:.3f} ({rc["lo"]:.3f}–{rc["hi"]:.3f}) | {nn["u"]:.3f} | {fr:.2f} |')
    return r


row('SIMCODE (b) all', S, np.arange(len(S['f'])), nn=nnS)
for nm, A in MS.items():
    ref = np.bincount(tile_of(A), minlength=121).reshape(11, 11).astype(float)
    row(f'{nm} all', A, np.arange(len(A['f'])), ref)
    V, info = variants(A, S)
    OUT[f'{nm} info'] = info
    for lab, i in V.items():
        row(f'{nm} {lab}', A, i, ref)
    log(f'    {nm}: N_cov = {info["N_cov"]} ({info["N_cov"] / len(S["f"]):.2f} x N_SC), equivalent global sigma threshold {info["sig_thr"]:.2f} nm; '
        f'per-tile quantile q = {info["q"]:.3f}')
fn = DS.out_path(spec, 'covpool', sys.argv[2] if len(sys.argv) > 2 else None)
json.dump(OUT, open(fn + '.json', 'w'), default=float, indent=1); open(fn + '.txt', 'w').write('\n'.join(LINES) + '\n')
