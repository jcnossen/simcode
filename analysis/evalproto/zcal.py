"""critic 37 #3: conditional sigma calibration (winner's curse of the strict product).
Sim (GT): per sigma-quantile bin of each method's rows (ours p1 / p2L: calibrated sigma = mean of the axes; SIMCODE: its CRLB):
n, median sigma, rms sigma, truth rms per axis (|e| < 50 nm) and dyn, gross > 50 nm fraction, zcal = rms(e / sigma), robust zcal,
and the own-set lag-averaged NN with BOTH rows of a pair in the bin (lib.lagavg_own sel=) -> kappa = NN / truth dyn.
Fix (fitted on the sim, ours only): sigma' = sigma * g(log sigma), g = piecewise-linear interpolation of the per-bin
zcal_true / zcal_all (relative to the full-set zcal, so the overall calibration is unchanged) against the bin's median log sigma.
Real GT-free check (real mode): per quantile bin the NN (both rows in bin) / rms sigma before and after the fix, compared
with the sim's NN / rms sigma profile; a flat real profile after the fix = the sim-fitted rescale transfers.  same algorithm; data via datasets.py json specs (the sim
spec = simload.py's sets, the real specs = critic35/c35lib (0.5 / 2.5 nM, G3) and real20/lib20 (20 nM, G6) loaders).
usage: python analysis/evalproto/zcal.py sim <sim spec.json> [...] [--out dir]   (fits -> zcal_fit_<spec>.json, zcal_sim_<specs>.txt)
       python analysis/evalproto/zcal.py real <real spec.json> <zcal_fit_<sim>.json> [--out dir]   -> zcal_<spec>_<sim>.txt / .json
(dev outputs: zcal_sim_m05pe_m25pe.txt = sim simm05pe simm25pe; zcal_real0.5_m05pe.txt = real real0.5 zcal_fit_simm05pe.json ...)"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as DS
from simcode2.evalproto import lib as E
from simcode2 import config
PX = E.PX
QE = (0, .01, .02, .05, .10, .20, .35, .50, .70, 1.0)
ARGS = sys.argv[1:]
if '--out' in ARGS:
    i = ARGS.index('--out'); D_ = ARGS[i + 1]; del ARGS[i:i + 2]
else:
    D_ = config.out_dir('evalproto')
os.makedirs(D_, exist_ok=True); D_ = os.path.join(D_, '')
mode = ARGS[0]
LINES = []; OUT = {}
def log(s=''):
    print(s, flush=True); LINES.append(s)


def qbins(sig):
    r = np.empty(len(sig)); o = np.argsort(sig, kind='stable'); r[o] = (np.arange(len(sig)) + 0.5) / len(sig)
    return np.clip(np.searchsorted(QE, r, side='right') - 1, 0, len(QE) - 2)


_PRE = {}
def nn_bin(D, m, grid, nboot=50, seed=37):
    """own lag-avg NN (lags 12-36, 0.01-nm grid) on pairs with both rows in m; NN search over all rows; event-block bootstrap"""
    key = id(D)
    if key not in _PRE:
        ev = E.link_events(D['x'], D['y'], D['f'], grid=grid, r=0.75); u, evc = np.unique(ev, return_inverse=True)
        _PRE[key] = dict(evc=evc, nev=len(u), J={dt: E.nn_next(D['x'], D['y'], D['f'], dt) for dt in E.LAGS})
    P = _PRE[key]; rng = np.random.default_rng(seed); W = rng.poisson(1.0, (nboot, P['nev'])).astype(float)
    per = []
    for dt in E.LAGS:
        j = P['J'][dt]; k = (j >= 0) & m; k[k] &= m[j[k]]; i = np.nonzero(k)[0]
        v = np.r_[D['x'][j[i]] - D['x'][i], D['y'][j[i]] - D['y'][i]] * PX; b = E._bins(v); e = P['evc'][np.r_[i, i]]
        s, w = E.sig1d_fine(h=E._hist_w(b, e, np.ones(P['nev'])))
        per.append((s, w, b, e, len(i)))
    if not np.all(np.isfinite([p_[0] for p_ in per])): return np.nan, np.nan, np.nan, 0
    B = [np.mean([E.sig1d_fine(h=E._hist_w(b, e, W[r]), s0=s, w0=w)[0] for s, w, b, e, _ in per]) for r in range(nboot)]
    return float(np.mean([p_[0] for p_ in per])), float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5)), int(np.mean([p_[4] for p_ in per]))


def g_apply(fit, sig):
    return sig * np.interp(np.log(sig), fit['logsig'], fit['g'])


if mode == 'sim':
    names = []
    for spec in ARGS[1:]:
        S, MS, META = DS.load(spec); m = META['name']; sn = os.path.splitext(os.path.basename(spec))[0]; names.append(sn)
        log(f'## sim {m} (G3): per sigma-quantile bin; truth per axis, |e| < 50 nm; NN = own lag-avg (12-36) with both rows in the bin')
        log('| method | q bin | n | med sigma | rms sigma | truth rms | truth dyn | gross | zcal | zcal rob | NN (95 % CI) | NN / truth dyn | NN / rms sigma |')
        log('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        OUT[m] = {}
        for nm, D in {'SIMCODE (b), CRLB': S, **MS}.items():
            sig = D['sig']; qb = qbins(sig)
            e = D['e']; g = np.isfinite(e).all(1); r = np.where(g, np.hypot(e[:, 0], e[:, 1]), np.inf); ok = r < 50
            zall = np.sqrt(np.mean((e[ok] / sig[ok, None]) ** 2))
            rows = []
            for b in range(len(QE) - 1):
                m_ = qb == b; k = m_ & ok
                z = e[k] / sig[k, None]; zc = float(np.sqrt((z ** 2).mean())); zr = float(np.median(np.abs(z)) / 0.6745)
                T = E.truth(D, np.nonzero(m_)[0])
                nn, lo, hi, nnn = nn_bin(D, m_, 3)
                rs = float(np.sqrt(np.mean(sig[m_] ** 2)))
                rows.append(dict(q0=QE[b], q1=QE[b + 1], n=int(m_.sum()), med_sig=float(np.median(sig[m_])), rms_sig=rs, rms=T['rms'], dyn=T['dyn'],
                                 gross=float(1 - ok[m_ & g].mean()), zcal=zc, zrob=zr, nn=nn, nn_lo=lo, nn_hi=hi, nn_n=nnn))
                log(f'| {nm} | {QE[b]:.2f}–{QE[b + 1]:.2f} | {m_.sum()} | {np.median(sig[m_]):.2f} | {rs:.2f} | {T["rms"]:.2f} | {T["dyn"]:.2f} | {rows[-1]["gross"]:.4f} | **{zc:.2f}** | {zr:.2f} | {nn:.2f} ({lo:.2f}–{hi:.2f}) | {nn / T["dyn"]:.2f} | {nn / rs:.2f} |')
            fit = None
            if nm.startswith('ours'):
                ls = np.array([np.log(r_['med_sig']) for r_ in rows]); gg = np.array([r_['zcal'] / zall for r_ in rows])
                fit = dict(logsig=ls.tolist(), g=gg.tolist(), zall=float(zall))
                # validation of the fix on this sim: zcal per bin after the rescale + the strict-product zcal (top 10 % / 20 %)
                s2 = g_apply(fit, sig)
                zb = [float(np.sqrt(np.mean((e[(qb == b) & ok] / s2[(qb == b) & ok, None]) ** 2))) for b in range(len(QE) - 1)]
                log(f'    {nm}: full-set zcal {zall:.2f}; after the sigma rescale per bin: ' + ' '.join(f'{v:.2f}' for v in zb))
                fit['zcal_after'] = zb
            OUT[m][nm] = dict(rows=rows, zall=float(zall), fit=fit)
        json.dump(OUT[m], open(D_ + f'zcal_fit_{sn}.json', 'w'), indent=1, default=float)
    open(D_ + f'zcal_sim_{"_".join(names)}.txt', 'w').write('\n'.join(LINES) + '\n')
else:
    spec, fitf = ARGS[1], ARGS[2]
    FIT = json.load(open(fitf))
    S, MS, META = DS.load(spec); grid = META['grid']; c = META['name']
    sn = os.path.splitext(os.path.basename(spec))[0]; fm = os.path.splitext(os.path.basename(fitf))[0].replace('zcal_fit_', '')
    log(f'## {c} (G{grid}): NN (own lag-avg, both rows in the bin) / rms sigma per sigma-quantile bin, before / after the '
        f'sim-fitted rescale (fit: {fm}); sim reference = the same ratio in {fm}')
    log('| method | q bin | n | med sigma | rms sigma | rms sigma\' | NN (95 % CI) | NN / rms sigma | NN / rms sigma\' | sim NN / rms sigma | sim NN / rms sigma\' |')
    log('|---|---|---|---|---|---|---|---|---|---|---|')
    OUT = {}
    for nm, D in {'SIMCODE (b), CRLB': S, **MS}.items():
        sig = D['sig']; qb = qbins(sig); F = FIT[nm]
        s2 = g_apply(F['fit'], sig) if F['fit'] else sig
        rows = []
        for b in range(len(QE) - 1):
            m_ = qb == b
            nn, lo, hi, nnn = nn_bin(D, m_, grid)
            rs = float(np.sqrt(np.mean(sig[m_] ** 2))); rs2 = float(np.sqrt(np.mean(s2[m_] ** 2)))
            sr = F['rows'][b]
            # the sim's ratio after the fix: sim rms sigma' from the fit at the sim bin's sigma
            sim2 = sr['nn'] / (sr['rms_sig'] * (np.interp(np.log(sr['med_sig']), F['fit']['logsig'], F['fit']['g']) if F['fit'] else 1))
            rows.append(dict(n=int(m_.sum()), med_sig=float(np.median(sig[m_])), rms_sig=rs, rms_sig2=rs2, nn=nn, lo=lo, hi=hi, r=nn / rs, r2=nn / rs2,
                             sim_r=sr['nn'] / sr['rms_sig'], sim_r2=float(sim2)))
            log(f'| {nm} | {QE[b]:.2f}–{QE[b + 1]:.2f} | {m_.sum()} | {np.median(sig[m_]):.2f} | {rs:.2f} | {rs2:.2f} | {nn:.2f} ({lo:.2f}–{hi:.2f}) | '
                f'{nn / rs:.2f} | {nn / rs2:.2f} | {sr["nn"] / sr["rms_sig"]:.2f} | {sim2:.2f} |')
        OUT[nm] = rows
    json.dump(OUT, open(D_ + f'zcal_{sn}_{fm}.json', 'w'), indent=1, default=float)
    open(D_ + f'zcal_{sn}_{fm}.txt', 'w').write('\n'.join(LINES) + '\n')
