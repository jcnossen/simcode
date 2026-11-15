"""Check 3 (2026-11-10): pattern depth and phase field maps from the pass-1 network intensities (pattern_est.py machinery).

Rebuilds the pattern-estimation input R exactly as run_uncalibrated (chunk de-dup, prior_off, crop 'out'), selects the
clean on-through emitter-windows (pattern_est.select, nn >= 3, >= 300 ph), and per 10 px cell of the common sensor frame
(0.5 nM pixel = 2.5 nM pixel + (dx 2, dy 22)) computes, per angle a with the run's own global pattern_est.pt:
  phase residual delta_a = arg sum w Z exp(-i k_a.r) exp(+i eps_a(t))  ->  nm along k_a (= delta / |k| * 108.33; the
      pattern says r_true = r_prior + delta/|k| k_hat, i.e. the same quantity as check 1's d, gauge-equivalent),
  cross-cycle depth m_a (photon-disjoint consecutive windows, fit_depth estimator),
with Poisson bootstrap over 2x2 px sub-cells (100 reps); and the same for the first / second half of the movie.
Outputs: check3_<ds>.npz, check3_summary.txt
"""
import glob, json, math, os, sys
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcode import config
from simcode.processing.pattern_estimation import select, angle_Z, _interp_eps, T
from field_maps import G0, CS, NC, SHIFT, smooth_fit, wcorr
from check1_offsets import DS, PX

OUT = config.out_dir('field')
CAL = config.model(config.CAL_P1_REAL)
CALSIM = config.model(config.CAL_P1_SIM)
NB = 100


def build_R(run, cal):
    fs = sorted(glob.glob(f'{run}/pass1_*.npz'), key=lambda s: int(s.split('_')[-1][:-4]))
    cols = ('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc')
    Ls = [np.load(f) for f in fs]; chunks = [int(f.split('_')[-1][:-4]) for f in fs]
    keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (f0 == chunks[-1]))
                           for L, f0 in zip(Ls, chunks)])
    R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}
    po = json.load(open(cal)).get('prior_off')
    if po is not None: R['x'] = R['x'] - po[0]; R['y'] = R['y'] - po[1]
    x0c, y0c, szc = [int(v) for v in Ls[0]['crop']]
    R['out'] = (R['tc'] >= 1.0) & (R['tc'] < 4.0) & (R['x'] >= x0c) & (R['x'] < x0c + szc) & (R['y'] >= y0c) & (R['y'] < y0c + szc)
    return R


def cell_index(x, y, sh):
    cx = np.clip(((x + sh[0] - G0) // CS).astype(int), 0, NC - 1); cy = np.clip(((y + sh[1] - G0) // CS).astype(int), 0, NC - 1)
    sub = np.clip((x + sh[0] - G0) // 2, 0, 80).astype(int) * 128 + np.clip((y + sh[1] - G0) // 2, 0, 80).astype(int)
    return cy * NC + cx, sub


def boot_sums(cid, sub, vals, rng):
    """vals: dict name -> per-row weights to sum per cell. Returns point sums [NC*NC] and bootstrap sums [NB, NC*NC]."""
    us, sinv = np.unique(sub, return_inverse=True)
    gid = sinv.astype(np.int64) * NC * NC + cid
    ug, ginv = np.unique(gid, return_inverse=True); gs, gc = ug // (NC * NC), ug % (NC * NC)
    GS = {k: np.bincount(ginv, v, len(ug)) for k, v in vals.items()}
    P = {k: np.bincount(gc, g, NC * NC) for k, g in GS.items()}
    W = rng.poisson(1.0, (NB, len(us))).astype(float)[:, gs]
    B = {k: np.stack([np.bincount(gc, W[b] * g, NC * NC) for b in range(NB)]) for k, g in GS.items()}
    return P, B


def analyse(ds, rng):
    run = DS[ds]; sh = SHIFT.get(ds, (0.0, 0.0))
    R = build_R(run, CAL if ds.startswith('real') else CALSIM)
    P = torch.load(f'{run}/pattern_est.pt')
    K = P['k'].numpy().astype(np.float64); psi = P['phi'].numpy().astype(np.float64); rel = np.exp(P['logrel'].numpy().astype(np.float64))
    m = P['m'].numpy()[[0, 3]].astype(np.float64); eps = P['dphi'].numpy().astype(np.float64); kf = int(P['knot_frames'])
    D = select(R)
    sel = D['clean']; idx = np.nonzero(sel)[0]
    w = D['w'][idx]; f = w[:, None] + np.arange(T)[None]; st = f % 6
    half = (w >= np.median(w)).astype(int)
    out = {}
    cid, sub = cell_index(D['x'][idx], D['y'][idx], sh)
    for a in (0, 1):
        A, Z, cols = angle_Z(D['I'][idx], st, psi, rel, a)
        ft = np.take_along_axis(f, cols, 1).mean(1)
        e = _interp_eps(eps, kf, ft)[a]
        kn = np.hypot(*K[a])
        u = Z * np.exp(-1j * (K[a, 0] * D['x'][idx] + K[a, 1] * D['y'][idx])) * np.exp(1j * e)
        u = u / np.maximum(np.abs(u), 1e-9)
        var = kn ** 2 * D['s'][idx] ** 2 + 2.0 / (max(m[a], 0.3) ** 2 * np.maximum(A, 1.0))
        wt = np.where(A > 0, 1.0 / var, 0.0)
        for hname, hm in (('all', np.ones(len(idx), bool)), ('h0', half == 0), ('h1', half == 1)):
            Pp, Bp = boot_sums(cid[hm], sub[hm], dict(re=wt[hm] * u.real[hm], im=wt[hm] * u.imag[hm], n=np.ones(hm.sum())), rng)
            ph = np.angle(Pp['re'] + 1j * Pp['im']); phb = np.angle(Bp['re'] + 1j * Bp['im'])
            se = np.std(np.angle(np.exp(1j * (phb - ph[None]))), 0)
            nm = ph / kn * PX; nm_se = se / kn * PX
            bad = Pp['n'] < 30; nm[bad] = np.nan; nm_se[bad] = np.nan
            out[f'ph{a}_{hname}'] = nm.reshape(NC, NC); out[f'ph{a}_se_{hname}'] = nm_se.reshape(NC, NC); out[f'n_{hname}'] = Pp['n'].reshape(NC, NC)
    # depth per cell: cross-cycle pairs (w, w + 6) of the same clean emitter (fit_depth estimator)
    Pt = np.c_[D['x1'][idx], D['y1'][idx], D['w'][idx] * 1e4]
    dd, jj = cKDTree(Pt).query(np.c_[D['x1'][idx], D['y1'][idx], (D['w'][idx] + 6) * 1e4], distance_upper_bound=1.0)
    ok = np.isfinite(dd); i1, i2 = idx[ok], idx[jj[ok]]
    c1, s1 = cell_index(D['x'][i1], D['y'][i1], sh)
    for a in (0, 1):
        res = []
        for ii in (i1, i2):
            ww = D['w'][ii]; ff = ww[:, None] + np.arange(T)[None]; stt = ff % 6
            A, Z, cols = angle_Z(D['I'][ii], stt, psi, rel, a)
            e = _interp_eps(eps, kf, np.take_along_axis(ff, cols, 1).mean(1))[a]
            res.append((A, Z * np.exp(1j * e)))
        (A1, Z1), (A2, Z2) = res
        g = (A1 > 0) & (A2 > 0)
        cr = np.where(g, np.real(Z1 * np.conj(Z2)), 0); aa = np.where(g, A1 * A2, 0)
        Pp, Bp = boot_sums(c1, s1, dict(cr=cr, aa=aa, n=g * 1.0), rng)
        md = np.sqrt(np.maximum(Pp['cr'] / np.maximum(Pp['aa'], 1e-9), 0)); mb = np.sqrt(np.maximum(Bp['cr'] / np.maximum(Bp['aa'], 1e-9), 0))
        se = np.std(mb, 0); bad = Pp['n'] < 30; md[bad] = np.nan; se[bad] = np.nan
        out[f'm{a}'] = md.reshape(NC, NC); out[f'm{a}_se'] = se.reshape(NC, NC)
        out[f'm{a}_global'] = float(Pp['cr'].sum() / Pp['aa'].sum())
    out['n_clean'] = int(sel.sum())
    return out


def main():
    rng = np.random.default_rng(0)
    L = []; pr = lambda s='': (print(s, flush=True), L.append(s))
    C = {}
    for ds in sys.argv[1:] or ('real05', 'real25', 'sim05', 'sim25'):
        C[ds] = analyse(ds, rng)
        np.savez_compressed(f'{OUT}/check3_{ds}.npz', **{k: v for k, v in C[ds].items()}, G0=G0, CS=CS, NC=NC)
    pr('## check 3: pattern phase residual (nm along k_a, vs the global estimated pattern, pass-1 prior reference) and cross-cycle depth per 10 px sensor cell')
    for ds, o in C.items():
        pr(f'### {ds} (clean emitter-windows {o["n_clean"]})')
        for a in (0, 1):
            f = smooth_fit(o[f'ph{a}_all'], o[f'ph{a}_se_all'])
            v = o[f'ph{a}_all']; s = o[f'ph{a}_se_all']; okk = np.isfinite(v)
            ex = math.sqrt(max(np.nanvar(v) - np.nanmean(s ** 2), 0))
            rh = wcorr(o[f'ph{a}_h0'], o[f'ph{a}_h1'], o[f'ph{a}_se_h0'], o[f'ph{a}_se_h1'])
            pr(f'  angle {a} phase: cell rms {np.nanstd(v):.2f} nm, noise {math.sqrt(np.nanmean(s ** 2)):.2f}, excess {ex:.2f} nm; chi2 flat {f["chi2_flat"]:.1f} -> L2 resid {f["chi2_res"]:.1f} (R2 {f["R2"]:.2f}); '
               f'L2 coef [1,u,v,P2u,uv,P2v] nm {np.round(f["coef"], 2).tolist()}; first vs second half r {rh[0]:+.2f}')
            md = o[f'm{a}']; ms = o[f'm{a}_se']; fm = smooth_fit(md, ms)
            pr(f'  angle {a} depth: global {o[f"m{a}_global"] ** 0.5 if False else np.sqrt(o[f"m{a}_global"]):.3f}; cell p5/p50/p95 {np.nanpercentile(md, 5):.3f} / {np.nanpercentile(md, 50):.3f} / {np.nanpercentile(md, 95):.3f}; '
               f'median se {np.nanmedian(ms):.3f}; excess sd {math.sqrt(max(np.nanvar(md) - np.nanmean(ms ** 2), 0)):.3f}; chi2 flat {fm["chi2_flat"]:.1f} -> L2 resid {fm["chi2_res"]:.1f} (R2 {fm["R2"]:.2f}); '
               f'L2 coef {np.round(fm["coef"], 3).tolist()}')
    pr('### consistency: phase map vs check-1 d map (same dataset; angle 0 ~ dx, angle 1 ~ dy), and across datasets on the sensor frame')
    for ds in C:
        try: M = np.load(f'{OUT}/maps_{ds}.npz')
        except FileNotFoundError: continue
        for a, c in ((0, 0), (1, 1)):
            r = wcorr(C[ds][f'ph{a}_all'], M[f'd{c}_4'], C[ds][f'ph{a}_se_all'], M[f'd{c}_se_4'])
            ok = np.isfinite(C[ds][f'ph{a}_all']) & np.isfinite(M[f'd{c}_4'])
            sl = np.polyfit(C[ds][f'ph{a}_all'][ok], M[f'd{c}_4'][ok], 1)[0] if ok.sum() > 5 else np.nan
            pr(f'  {ds} angle {a} phase vs d{"xy"[c]} (>=700 ph): r {r[0]:+.2f} (cells {r[1]}), slope {sl:.2f}')
    for a_, b_ in (('real05', 'real25'), ('sim05', 'sim25'), ('real05', 'sim05')):
        if a_ in C and b_ in C:
            for a in (0, 1):
                r = wcorr(C[a_][f'ph{a}_all'], C[b_][f'ph{a}_all'], C[a_][f'ph{a}_se_all'], C[b_][f'ph{a}_se_all'])
                rm = wcorr(C[a_][f'm{a}'], C[b_][f'm{a}'], C[a_][f'm{a}_se'], C[b_][f'm{a}_se'])
                pr(f'  {a_} vs {b_} angle {a}: phase r {r[0]:+.2f}, depth r {rm[0]:+.2f} (cells {r[1]})')
    open(f'{OUT}/check3_summary.txt', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
