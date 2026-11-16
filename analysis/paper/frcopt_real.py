"""Selection sweep on real data (no ground truth): each method's threshold on its OWN reported precision, chosen by the fair
half-split FRC (frc_fair.py), for the experimental figures (fig_exp_new.py).

In the simulations (analysis/papersim/frcopt.py, out/papersim/frcopt/table_*.md) the half-split FRC minimum lands within ~1 nm
GT-FRC of the ground-truth optimum for SIMCODE and SIMFLUX (the half-split curve is flat near its minimum), which is what
justifies this rule on real data.

Rows per method = the fair-FRC set of the figure (disjoint windows, max(sx, sy) < 0.15 px, crop [5, 115) px), drift-corrected
with the figure's own DME v2 drift caches (<prefix>_<kind>_drift.npy; never re-estimated: a stricter subset of the same rows
keeps the drift of the full set). Sweep: thresholds at quantiles Q of max(sx, sy) of that set (100 % ... 10 % kept); at each the
half-split FRC (first 1/7 crossing = frc_fair, the figure's number; the sustained crossing and the 0.025-0.075 / nm band mean as
robustness checks).
SIMCODE: 'sig_p' = the production rule (p >= 0.63; its 12 nm point = the figure's set; must reproduce the figure's n and FRC),
'sig_all' = calibrated sigma only, no p condition (= p_agg >= 0.63, pixel-bias study). Rows rebuilt from the run's pass2_est2
files (simcode.table.load_rows; sigma rescale g(u) ranked within each family's own rows, as build_table does for p >= 0.63).
Optimum = argmin of the half-split FRC (ties: more rows); also the loosest threshold within --tol nm of that minimum.
usage: python analysis/paper/frcopt_real.py 1809_2 --prefix out/paper_v2psf/baselines_psfv2/1809_2_fig3_psfv2 --v2 <run dir>
         --smlm .. --sf .. --decode .. [--nframes 120000]
output: <prefix>_frcopt.json (sweep rows, optima, thresholds for fig_exp_new.py --select)"""
import os, sys, json, argparse, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'analysis', 'papersim')]
import frc_fair as FF

Q = np.round(np.arange(1.0, 0.0999, -0.025), 4)
PX = FF.PX


def sust(freq, curve, width=0.01):
    f = np.asarray(freq); c = np.asarray(curve); w = max(int(width / max(f[1] - f[0], 1e-12)), 1)
    for i in range(3, len(c) - w):
        if (c[i:i + w] < 1 / 7).all(): return float(1 / f[i])
    return float('nan')


def band(freq, curve, lo=0.025, hi=0.075):
    f = np.asarray(freq); k = (f >= lo) & (f <= hi); return float(np.asarray(curve)[k].mean())


def score(x, y, f):
    res, fr, c = FF.frc_halves(x, y, f)
    return dict(n=int(len(x)), hs_first=float(res), hs_sust=sust(fr, c), hs_band=band(fr, c))


def sweep(x, y, f, key, label):
    rows = []; t0 = time.time()
    for q in Q:
        thr = float(np.quantile(key, q)); m = key <= thr
        r = dict(q=float(q), thr_px=thr, **score(x[m], y[m], f[m])); rows.append(r)
    print(f'{label}: {len(key)} rows, {time.time() - t0:.0f} s; FRC ' + ' '.join(f"{r['q']:.2f}:{r['hs_first']:.1f}" for r in rows[::4]), flush=True)
    return rows


def optimum(rows, crit='hs_first', tol=0.5):
    ok = [i for i, r in enumerate(rows) if np.isfinite(r[crit])]
    j = min(ok, key=lambda i: (round(rows[i][crit], 6), -rows[i]['n']))
    jl = max((i for i in ok if rows[i][crit] <= rows[j][crit] + tol), key=lambda i: rows[i]['n'])
    return j, jl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset'); ap.add_argument('--prefix', required=True); ap.add_argument('--v2', required=True)
    for k in ('smlm', 'sf', 'decode'): ap.add_argument('--' + k, required=True)
    ap.add_argument('--nframes', type=int, default=120000); ap.add_argument('--tol', type=float, default=0.5)
    a = ap.parse_args(); F = a.nframes
    vals = json.load(open(a.prefix + '_values.json'))
    out = dict(dataset=a.dataset, prefix=a.prefix, nframes=F, Q=Q.tolist(), tol_nm=a.tol, methods={})
    for i, k in enumerate(('smlm', 'sf', 'decode')):
        T = FF.load(k, getattr(a, k)); keep = T['f'] + 6 <= F; T = {kk: v[keep] for kk, v in T.items()}
        dc = f'{a.prefix}_{k}_drift.npy'; assert os.path.exists(dc), dc
        R = FF.evaluate(T, F, drift_cache=dc, secondary=False); m = R['mask']
        assert R['n_frc'] == vals['n_frc'][i] and abs(R['frc_nm'] - vals['frc_nm'][i]) < 1e-6, (k, R['n_frc'], R['frc_nm'])
        key = np.maximum(T['sx'][m], T['sy'][m])
        rows = sweep(R['x_dc'][m], R['y_dc'][m], T['f'][m], key, k)
        j, jl = optimum(rows, tol=a.tol)
        out['methods'][k] = dict(n_base=int(m.sum()), frc_current=R['frc_nm'], fam={'sig': rows}, opt={'sig': dict(min=j, loose=jl)})
        del T, R
    # SIMCODE from the pass-2 rows
    import frcopt as FO
    from simcode.table import in_crop
    Tv = FO.v2_rows(a.v2, None, crop=FF.CROP)
    drift = np.load(f'{a.prefix}_v2_drift.npy'); ti = np.clip(Tv['f'] + 3, 0, len(drift) - 1)
    xd, yd = Tv['x'] - drift[ti, 0], Tv['y'] - drift[ti, 1]
    base = (Tv['f'] % 6 == 0) & (Tv['f'] + 6 <= F) & FF.in_crop(Tv['x'], Tv['y']) & np.isfinite(Tv['x']) & np.isfinite(Tv['y'])
    M = dict(fam={}, opt={}, variants={})
    for fam, vb in (('sig_p', Tv['p'] >= 0.63), ('sig_all', np.ones(len(Tv['x']), bool))):
        sx, sy = Tv['sigma'](vb & in_crop(Tv['x'], Tv['y'], FF.CROP))
        key = np.maximum(sx, sy); m = base & vb & np.isfinite(key) & (key < FF.SIGMA_MAX)
        m12 = m & (key * PX <= 12.0)
        r12 = score(xd[m12], yd[m12], Tv['f'][m12]); r12['thr_px'] = 12.0 / PX
        if fam == 'sig_p':
            i = 3
            assert r12['n'] == vals['n_frc'][i] and abs(r12['hs_first'] - vals['frc_nm'][i]) < 1e-6, (r12, vals['n_frc'][i], vals['frc_nm'][i])
        rows = sweep(xd[m], yd[m], Tv['f'][m], key[m], 'v2 ' + fam)
        j, jl = optimum(rows, tol=a.tol)
        M['fam'][fam] = rows; M['opt'][fam] = dict(min=j, loose=jl); M['variants'][fam] = dict(n_base=int(m.sum()), at12=r12)
    M['n_base'] = M['variants']['sig_p']['n_base']; M['frc_current'] = M['variants']['sig_p']['at12']['hs_first']
    out['methods']['v2'] = M
    # thresholds for fig_exp_new.py --select (px; on top of the figure's own rows): the FRC minimum of the 'sig' / 'sig_p' family
    sel = {}
    for k, Mk in out['methods'].items():
        fam = 'sig_p' if k == 'v2' else 'sig'; r = Mk['fam'][fam][Mk['opt'][fam]['min']]
        sel[k] = dict(sigma_max_px=r['thr_px'], n=r['n'], frc_nm=r['hs_first'], frc_current=Mk['frc_current'],
                      n_current=(Mk['variants']['sig_p']['at12']['n'] if k == 'v2' else Mk['n_base']))
    out['select'] = sel
    fn = a.prefix + '_frcopt.json'; json.dump(out, open(fn, 'w'), indent=1)
    for k, s in sel.items(): print(k, s, flush=True)
    print('written', fn)


if __name__ == '__main__':
    main()
