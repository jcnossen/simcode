"""critic 37 #3: conditional sigma calibration fixes, fitted on one sim and tested on the others (out of sample).
Fixes (ours p1 / p2L; sigma = mean of the calibrated axes, nm):
  floor   sigma' = sqrt(k^2 sigma^2 + s0^2): weighted LSQ of the per-quantile-bin truth rms^2 on mean sigma^2 (bins as zcal.py)
          ('tot': truth rms incl. static terms; 'dyn': window-to-window truth only)
  qrank   sigma' = sigma * g(q), q = sigma rank quantile within the dataset (the same per-bin ratios placed at the bins' mid
          quantiles, log-q interpolation): the winner's-curse correction as a function of selection strictness
  qbin    sigma' = sigma * g(log sigma), g = per-bin zcal / full-set zcal interpolated in log sigma (zcal.py; keeps the full-set
          zcal, removes the conditional trend only)
Test per sim: zcal = rms(e / sigma') per sigma-quantile bin (|e| < 50 nm), spread max/min over bins, strict-product zcal
(top-N_SC and coverage-matched N by sigma; both fixes are monotone in sigma so the selections do not change), and the
'sigma <= 6 nm' threshold mapped through the fix.  same algorithm; sims via datasets.py json specs
(mode sim; the dev simload.py sets), named by the spec "name".
usage: python analysis/evalproto/zcal_fix.py <fit sim specs, comma> <test sim specs, comma> [--out dir]  -> zcal_fix.txt / .json
(dev zcal_fix.txt = fits simm05pe.json,simm25pe.json, tests simm05pd.json,simm25pd.json,simm05pe.json,simm25pe.json)"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as DS
from simcode2.evalproto import lib as E
from simcode2 import config
QE = (0, .01, .02, .05, .10, .20, .35, .50, .70, 1.0)
ARGS = sys.argv[1:]
if '--out' in ARGS:
    i = ARGS.index('--out'); D_ = ARGS[i + 1]; del ARGS[i:i + 2]
else:
    D_ = config.out_dir('evalproto')
os.makedirs(D_, exist_ok=True); D_ = os.path.join(D_, '')
FSPEC, TSPEC = ARGS[0].split(','), ARGS[1].split(',')
NAME = {s: json.load(open(s))['name'] for s in FSPEC + TSPEC}
FITS, TESTS = [NAME[s] for s in FSPEC], [NAME[s] for s in TSPEC]
LINES = []; OUT = {}
def log(s=''):
    print(s, flush=True); LINES.append(s)


def qbins(sig):
    r = np.empty(len(sig)); o = np.argsort(sig, kind='stable'); r[o] = (np.arange(len(sig)) + 0.5) / len(sig)
    return np.clip(np.searchsorted(QE, r, side='right') - 1, 0, len(QE) - 2)


DATA = {}
for sp in sorted(set(FSPEC + TSPEC), key=lambda s: NAME[s]):
    S, MS, _ = DS.load(sp); m = NAME[sp]
    DATA[m] = {'N_SC': len(S['f']), 'cov_SC': None, **MS}
    k = np.isfinite(S['e']).all(1) & (S['ev'] >= 0); k[k] &= np.hypot(S['e'][k, 0], S['e'][k, 1]) < 50
    DATA[m]['cov_SC'] = len(np.unique(S['ev'][k]))


def okrows(D):
    e = D['e']; g = np.isfinite(e).all(1) & (D['ev'] >= 0); g[g] &= np.hypot(e[g, 0], e[g, 1]) < 50
    return g


def fit_floor(D, kind):
    sig = D['sig']; qb = qbins(sig); ok = okrows(D)
    X, Y, W = [], [], []
    for b in range(len(QE) - 1):
        i = np.nonzero(qb == b)[0]
        T = E.truth(D, i)
        X.append(np.mean(sig[i[ok[i]]] ** 2)); Y.append((T['rms'] if kind == 'tot' else T['dyn']) ** 2); W.append(len(i))
    X, Y, W = map(np.array, (X, Y, W)); A = np.c_[X, np.ones(len(X))] * np.sqrt(W)[:, None]
    p, *_ = np.linalg.lstsq(A, Y * np.sqrt(W), rcond=None)
    return dict(k=float(np.sqrt(max(p[0], 1e-6))), s0=float(np.sqrt(max(p[1], 0.0))))


def fit_qbin(D):
    sig = D['sig']; qb = qbins(sig); ok = okrows(D); e = D['e']
    zall = np.sqrt(np.mean((e[ok] / sig[ok, None]) ** 2))
    ls, g = [], []
    for b in range(len(QE) - 1):
        k = (qb == b) & ok
        ls.append(float(np.log(np.median(sig[qb == b])))); g.append(float(np.sqrt(np.mean((e[k] / sig[k, None]) ** 2)) / zall))
    return dict(logsig=ls, g=g)


def fit_qrank(D):
    """g(q): per-quantile-bin zcal / full-set zcal at the bin's mid quantile (q = sigma rank / n within the dataset)"""
    F = fit_qbin(D)
    return dict(q=[0.5 * (a + b) for a, b in zip(QE[:-1], QE[1:])], g=F['g'])


def rankq(sig):
    r = np.empty(len(sig)); o = np.argsort(sig, kind='stable'); r[o] = (np.arange(len(sig)) + 0.5) / len(sig); return r


def apply(fix, F, sig):
    if fix == 'none': return sig
    if fix == 'qrank': return sig * np.interp(np.log(rankq(sig)), np.log(F['q']), F['g'])
    if fix.startswith('floor'): return np.sqrt((F['k'] * sig) ** 2 + F['s0'] ** 2)
    return sig * np.interp(np.log(sig), F['logsig'], F['g'])


def zc(D, sig2, i):
    ok = okrows(D)[i]; e = D['e'][i][ok]
    return float(np.sqrt(np.mean((e / sig2[i][ok, None]) ** 2)))


log(f'# conditional sigma calibration: fits on {FITS}, tests on {TESTS}; zcal = rms(e / sigma\') per axis, |e| < 50 nm')
for nm in ('ours p1', 'ours p2L'):
    for fm in FITS:
        D = DATA[fm][nm]
        Fs = {'none': None, 'floor tot': fit_floor(D, 'tot'), 'floor dyn': fit_floor(D, 'dyn'), 'qbin': fit_qbin(D), 'qrank': fit_qrank(D)}
        OUT[f'{nm}|{fm}'] = {k: v for k, v in Fs.items() if v is not None}
        log(f'\n## {nm}, fit on {fm}: floor tot k {Fs["floor tot"]["k"]:.3f} s0 {Fs["floor tot"]["s0"]:.2f} nm; floor dyn k {Fs["floor dyn"]["k"]:.3f} '
            f's0 {Fs["floor dyn"]["s0"]:.2f} nm; "sigma <= 6 nm" -> sigma\'_tot <= {np.sqrt((Fs["floor tot"]["k"] * 6) ** 2 + Fs["floor tot"]["s0"] ** 2):.2f} nm')
        log('| test sim | fix | zcal per q bin (0–1 %, 1–2, 2–5, 5–10, 10–20, 20–35, 35–50, 50–70, 70–100 %) | max/min | full | top-N_SC | cov-matched N |')
        log('|---|---|---|---|---|---|---|')
        for tm in TESTS:
            T = DATA[tm][nm]; sig = T['sig']; qb = qbins(sig); o = np.argsort(sig, kind='stable')
            # coverage-matched N on the test sim (GT events, as covpool)
            ok = okrows(T); kk = np.where(ok, T['ev'], -1)[o]; okk = kk >= 0
            u, first = np.unique(kk[okk], return_index=True); fi = np.sort(np.nonzero(okk)[0][first])
            Nc = int(fi[min(DATA[tm]['cov_SC'], len(fi)) - 1] + 1)
            for fix, F in Fs.items():
                s2 = apply(fix, F, sig)
                zb = [zc(T, s2, np.nonzero(qb == b)[0]) for b in range(len(QE) - 1)]
                r = dict(bins=zb, spread=max(zb) / min(zb), full=zc(T, s2, np.arange(len(sig))), topN=zc(T, s2, o[:DATA[tm]['N_SC']]), cov=zc(T, s2, o[:Nc]))
                OUT[f'{nm}|{fm}'][f'test {tm} {fix}'] = r
                log(f'| {tm}{" (in-sample)" if tm == fm else ""} | {fix} | ' + ' '.join(f'{v:.2f}' for v in zb) + f' | {r["spread"]:.2f} | {r["full"]:.2f} | {r["topN"]:.2f} | {r["cov"]:.2f} |')
json.dump(OUT, open(D_ + 'zcal_fix.json', 'w'), indent=1, default=float)
open(D_ + 'zcal_fix.txt', 'w').write('\n'.join(LINES) + '\n')
