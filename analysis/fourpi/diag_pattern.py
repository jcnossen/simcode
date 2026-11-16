"""Step 5 pattern diagnostics on the refit tables (p2_*.npz). usage: python diag_pattern.py <cell> --work <4Pi work folder>
Writes <work>/smlm/diag/pattern_<cell>.json and figures fig/pattern_<cell>_*.png."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import load_locs
from pattern import AnglePattern, wrap, cmean, emitter_phase, refine_k, slot_z
from register import feats

K0 = (np.array([3.84, -0.026]), np.array([0.0, 3.852]))


def select(L, nmin=1000):
    th = L['pfit']
    ok = L['on'] & (L['nn'] >= 6) & (th[:, 3] > nmin) & (L['pcrlb'][:, 0] < 0.1) & (np.abs(th[:, 0]) < 1.5) & (np.abs(th[:, 1]) < 1.5)
    ok &= np.isfinite(L['A']).all((1, 2))
    return ok


def R_of(d):
    return float(np.abs(cmean(d))) if len(d) else float('nan')


def poly_phase_fit(d, X, order=2, nit=6):
    """fit a smooth phase map phi(x, y) = arg of a complex polynomial fit to exp(i delta); returns corrected delta"""
    F = feats(X[:, 0], X[:, 1], order)
    e = np.exp(1j * d)
    c = np.linalg.lstsq(F, e, rcond=None)[0]
    ph = np.angle(F @ c)
    return wrap(d - ph), ph


def run(cell, P4):
    """P4: the 4Pi work folder"""
    OUT = f'{P4}/smlm'
    os.makedirs(f'{OUT}/diag', exist_ok=True)
    L = load_locs(cell, P4)
    s = select(L)
    A = L['A'][s]; sA = L['sA'][s]                       # [n,6,4]
    I = A.sum(2); sI = np.sqrt((sA ** 2).sum(2))
    X = np.c_[L['x'][s], L['y'][s]]; z = L['pfit'][s, 2]; N = L['pfit'][s, 3]
    t = L['frame'][s]; fi = L['fi'][s]; crx = L['pcrlb'][s, 0]
    res = dict(cell=cell, n=int(s.sum()), n_all=int(len(s)))
    pats = []
    figs = {}
    for a in range(2):
        I3, s3 = I[:, 3 * a:3 * a + 3], sI[:, 3 * a:3 * a + 3]
        P = AnglePattern(I3, s3, X, K0[a])
        m_med, m_proj = P.depth()
        R = P.coherence()
        # noise-limited coherence for a perfect plane wave with depth m_true; scan m_true to match the observed slot |Z|
        m_true = np.clip(np.nanmedian(P.m), 0.05, 1.0)
        Rn = P.noise_R(m_true)
        r = dict(k=P.k.tolist(), period_px=float(2 * np.pi / np.linalg.norm(P.k)), psi_deg=np.rad2deg(P.psi).round(1).tolist(),
                 psi_steps_deg=np.rad2deg(wrap(np.diff(P.psi))).round(1).tolist(), slot_abs=np.abs(P.Z).round(3).tolist(),
                 m_median=float(m_med), m_proj=float(m_proj), R=R, R_noise=Rn, R_over_noise=R / Rn)
        # ---- tests ----
        d0 = P.delta.copy()
        tests = {}
        # bright subset
        for nm, sel in (('N>3000', N > 3000), ('N>6000', N > 6000), ('crlb<0.03', crx < 0.03)):
            tests[nm] = dict(n=int(sel.sum()), R=R_of(d0[sel]), R_noise=P.noise_R(m_true, sel))
        # positions from the free-sigma Gaussian fit instead of the PSF fit (pass-1 style)
        Xg = np.c_[L['gx'][s], L['gy'][s]]
        tests['gauss_positions'] = dict(R=R_of(wrap(P.phi - Xg @ P.k)))
        # clean single-emitter subset: central 80 % of z, N < 5000, chi2 below its 80th percentile, Gaussian widths
        # below their 80th percentile (wide / poorly fitted spots are mostly blends)
        chi = L['pchi2'][s]; gs = L['gfit'][s]
        clean = ((z > np.quantile(z, 0.1)) & (z < np.quantile(z, 0.9)) & (N < 5000) & (chi < np.quantile(chi, 0.8))
                 & (gs[:, 4] < np.quantile(gs[:, 4], 0.8)) & (gs[:, 5] < np.quantile(gs[:, 5], 0.8)))
        tests['clean_subset'] = dict(n=int(clean.sum()), R=R_of(d0[clean]), m_median=float(np.nanmedian(P.m[clean])))
        dcl, phcl = poly_phase_fit(d0[clean], X[clean], 2)
        tests['clean_subset+field_quadratic'] = dict(R=R_of(dcl), map_ptp_deg=float(np.rad2deg(np.ptp(phcl))))
        tests['complement_of_clean'] = dict(n=int((~clean).sum()), R=R_of(d0[~clean]))
        for nm_, v_ in (('gauss_sx', gs[:, 4]), ('gauss_sy', gs[:, 5]), ('chi2_px', chi / 225), ('N', N)):
            e_ = np.quantile(v_, np.linspace(0, 1, 7))
            r.setdefault('vs', {})[nm_] = [(float(e_[i]), float(e_[i + 1]), R_of(d0[(v_ >= e_[i]) & (v_ <= e_[i + 1])])) for i in range(6)]
        # (b) time drift: per file and per 1200-frame chunk mean phase; coherence after removing it
        ch = (t // 1200).astype(int)
        mu_t = {}
        d1 = d0.copy()
        for c in np.unique(ch):
            q = ch == c; mu = np.angle(cmean(d0[q])); mu_t[int(c)] = (float(np.rad2deg(mu)), float(R_of(d0[q])), int(q.sum()))
            d1[q] = wrap(d0[q] - mu)
        tests['time_chunk1200'] = dict(R=R_of(d1), R_within_chunk_median=float(np.median([v[1] for v in mu_t.values()])))
        r['phase_vs_time'] = mu_t
        per_file = {int(f): (float(np.rad2deg(np.angle(cmean(d0[fi == f])))), R_of(d0[fi == f])) for f in np.unique(fi)}
        r['phase_per_file'] = per_file
        # (c) smooth field map (quadratic), after time correction
        d2, ph2 = poly_phase_fit(d1, X, 2)
        tests['+field_quadratic'] = dict(R=R_of(d2), map_ptp_deg=float(np.rad2deg(np.ptp(ph2))))
        d2c, ph2c = poly_phase_fit(d1, X, 4)
        tests['+field_quartic'] = dict(R=R_of(d2c), map_ptp_deg=float(np.rad2deg(np.ptp(ph2c))))
        # tile map 6 x 6 (for the figure)
        xe = np.quantile(X[:, 0], np.linspace(0, 1, 7)); ye = np.quantile(X[:, 1], np.linspace(0, 1, 7))
        tile = np.full((6, 6, 2), np.nan)
        for i in range(6):
            for j in range(6):
                q = (X[:, 0] >= xe[j]) & (X[:, 0] <= xe[j + 1]) & (X[:, 1] >= ye[i]) & (X[:, 1] <= ye[i + 1])
                if q.sum() > 20:
                    c_ = cmean(d1[q]); tile[i, j] = (np.rad2deg(np.angle(c_)), abs(c_))
        r['tile_phase_deg'] = tile[..., 0].round(0).tolist(); r['tile_R'] = tile[..., 1].round(3).tolist()
        # (d) z: phase and depth per z bin, after time + field correction
        ze = np.quantile(z, np.linspace(0, 1, 9))
        zb = []
        d3 = d2.copy()
        for i in range(8):
            q = (z >= ze[i]) & (z <= ze[i + 1])
            c_ = cmean(d2[q]); zb.append(dict(z=[float(ze[i]), float(ze[i + 1])], n=int(q.sum()), phase_deg=float(np.rad2deg(np.angle(c_))),
                                              R=float(abs(c_)), R_noise=P.noise_R(m_true, q), m_median=float(np.nanmedian(P.m[q])),
                                              m_proj=float(np.nanmean(P.m[q] * np.cos(d2[q])))))
            d3[q] = wrap(d2[q] - np.angle(c_))
        r['zbins'] = zb
        tests['+z_binned_phase'] = dict(R=R_of(d3))
        # linear kappa z
        best = (0, 0)
        for kap in np.linspace(-20, 20, 401):
            Rk = R_of(wrap(d2 - kap * z))
            if Rk > best[0]: best = (Rk, kap)
        tests['+z_linear'] = dict(R=best[0], kappa_rad_per_um=best[1])
        # (e) per-channel: pattern phase from each channel / pair alone, same k, psi
        chres = {}
        for nm, cs in (('ch0', [0]), ('ch1', [1]), ('ch2', [2]), ('ch3', [3]), ('pair02', [0, 2]), ('pair13', [1, 3])):
            Ic = A[:, 3 * a:3 * a + 3][:, :, cs].sum(2)
            ph, mm, _ = emitter_phase(Ic, P.psi)
            dc = wrap(ph - X @ P.k)
            ok = np.isfinite(dc)
            chres[nm] = dict(R=R_of(dc[ok]), mean_vs_all_deg=float(np.rad2deg(np.angle(cmean(wrap(dc[ok] - d0[ok]))))),
                             m_median=float(np.nanmedian(mm)))
        r['per_channel'] = chres
        # (g) localization error: coherence vs crlb
        qb = np.quantile(crx, np.linspace(0, 1, 6))
        r['vs_crlb'] = [dict(crlb=[float(qb[i]), float(qb[i + 1])], R=R_of(d0[(crx >= qb[i]) & (crx <= qb[i + 1])]),
                             R_noise=P.noise_R(m_true, (crx >= qb[i]) & (crx <= qb[i + 1]))) for i in range(5)]
        # m distribution
        r['m_hist'] = np.histogram(np.clip(P.m, 0, 1.6), bins=16, range=(0, 1.6))[0].tolist()
        r['tests'] = tests
        res[f'angle{a}'] = r
        pats.append((P, d0, d1, d2, d3))
        print(f'{cell} angle {a}: k {np.round(P.k, 4)}, period {r["period_px"]:.4f} px, psi {r["psi_deg"]}, m_med {m_med:.2f}, '
              f'R {R:.3f} (noise-limited {Rn:.3f}); tests:', json.dumps(tests), flush=True)
        print('   per channel', json.dumps(chres), flush=True)
        print('   z bins', json.dumps([(round(b['z'][0], 2), round(b['phase_deg']), round(b['R'], 2), round(b['R_noise'], 2), round(b['m_median'], 2)) for b in zb]), flush=True)
    json.dump(res, open(f'{OUT}/diag/pattern_{cell}.json', 'w'), indent=1)
    # figures
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for a in range(2):
        P, d0, d1, d2, d3 = pats[a]; r = res[f'angle{a}']
        ax[a, 0].hist(np.clip(P.m, 0, 1.6), bins=40, range=(0, 1.6)); ax[a, 0].set_title(f'angle {a}: per-emitter depth m (median {r["m_median"]:.2f})')
        ax[a, 1].hist(np.rad2deg(d0), bins=72, range=(-180, 180), histtype='step', label=f'plane wave R={r["R"]:.2f}')
        ax[a, 1].hist(np.rad2deg(d2), bins=72, range=(-180, 180), histtype='step', label=f'+time+field R={r["tests"]["+field_quadratic"]["R"]:.2f}')
        ax[a, 1].set_title(f'phase residual (noise-limited R={r["R_noise"]:.2f})'); ax[a, 1].legend(fontsize=7)
        im = ax[a, 2].imshow(np.array(r['tile_phase_deg'], float), cmap='twilight', vmin=-180, vmax=180); plt.colorbar(im, ax=ax[a, 2])
        ax[a, 2].set_title('phase residual per field tile (deg, after time corr.)')
        zc = [np.mean(b['z']) for b in r['zbins']]
        ax[a, 3].plot(zc, [b['phase_deg'] for b in r['zbins']], 'o-', label='phase (deg)')
        ax3 = ax[a, 3].twinx(); ax3.plot(zc, [b['R'] for b in r['zbins']], 's--', c='C1', label='R'); ax3.plot(zc, [b['R_noise'] for b in r['zbins']], 'x:', c='C2', label='R noise-limited')
        ax3.plot(zc, [b['m_median'] for b in r['zbins']], 'd-.', c='C3', label='median m'); ax3.legend(fontsize=7); ax[a, 3].set_xlabel('z (um, PSF fit)')
        ax[a, 3].set_title('phase / coherence / depth vs z')
    fig.suptitle(f'{cell}: pattern diagnostics ({res["n"]} emitters on through the window)')
    fig.savefig(f'{OUT}/fig/pattern_{cell}.png', dpi=75, bbox_inches='tight')
    # time
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    for a in range(2):
        mt = res[f'angle{a}']['phase_vs_time']; kk = sorted(mt)
        ax.plot([k * 1200 * 5.575e-3 for k in kk], [mt[k][0] for k in kk], 'o-', ms=3, label=f'angle {a}')
    ax.set_xlabel('time in subset (s, files concatenated)'); ax.set_ylabel('mean pattern phase (deg)'); ax.legend()
    fig.savefig(f'{OUT}/fig/pattern_time_{cell}.png', dpi=75, bbox_inches='tight')
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('cell'); ap.add_argument('--work', required=True, help='4Pi work folder')
    A = ap.parse_args()
    run(A.cell, A.work)
