"""Step 5, 4Pi diagnostics: bead channel calibration, per-emitter interference phase vs PSF-fit z, fringe period,
phase-z precision, fringe assignment, per-pattern-angle phase difference. usage: python diag_4pi.py <cell>"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import load_locs
from fourpi import chan_phase, chan_phase_err, fit_period, bead_channel_fit
from pattern import wrap, cmean
from diag_pattern import select
from register import feats

OUT = f'{P4}/smlm'


def joint_model(Phi, z, x, y, t, Lams=np.arange(0.18, 0.361, 0.005), order=3, tblk=300, nmax=120000, seed=0):
    """Phi = s 2 pi z / Lam + map(x, y) [poly order] + drift(t) [per tblk frames]; alternating fit for each Lam, s.
    Returns best (R, Lam, s), the R curve, and the residual for all emitters."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Phi), min(nmax, len(Phi)), replace=False)
    F = feats(x, y, order); tb = (t // tblk).astype(int)
    def fit(L, sg, idx, nit=4):
        d = wrap(Phi[idx] - sg * 2 * np.pi * z[idx] / L); Fi = F[idx]; ti = tb[idx]
        pt = np.zeros(tb.max() + 1)
        for _ in range(nit):
            c = np.linalg.lstsq(Fi, np.exp(1j * wrap(d - pt[ti])), rcond=None)[0]; pxy = np.angle(Fi @ c)
            r = wrap(d - pxy)
            for b in np.unique(ti): pt[b] = np.angle(cmean(r[ti == b]))
        return c, pt
    curve = []; best = None
    for sg in (1, -1):
        for L in Lams:
            c, pt = fit(L, sg, sub)
            res = wrap(Phi[sub] - sg * 2 * np.pi * z[sub] / L - np.angle(F[sub] @ c) - pt[tb[sub]])
            R = abs(cmean(res)); curve.append((sg, L, R))
            if best is None or R > best[0]: best = (R, L, sg)
    R, L, sg = best
    c, pt = fit(L, sg, np.arange(len(Phi)))
    res = wrap(Phi - sg * 2 * np.pi * z / L - np.angle(F @ c) - pt[tb])
    return dict(R=float(abs(cmean(res))), Lam=float(L), sign=int(sg), drift_deg=np.rad2deg(pt).round(0).tolist(),
                map_ptp_deg=float(np.rad2deg(np.ptp(np.angle(F @ c))))), np.array(curve), res


def run(cell):
    zc = json.load(open(f'{OUT}/zcal/zcal_{cell}.json'))
    zst = np.array(zc['zstage']); good = np.array(zc['good'])
    res = dict(cell=cell)
    for ap in ('r1', 'r3', 'r6'):
        Ib = np.array(zc[f'I_{ap}'])
        res[f'bead_{ap}'] = bead_channel_fit(zst[good], Ib[good])
    print(cell, 'bead r3:', json.dumps(res['bead_r3']), '\n bead r1 V', res['bead_r1']['V'], 'r6 V', res['bead_r6']['V'], flush=True)
    phc = np.deg2rad([0, 270, 180, 90])
    bead_ph = np.array(res['bead_r3']['phc'])           # measured, relative to ch0
    L = load_locs(cell)
    s = select(L, nmin=800)
    A = L['A'][s]; sA = L['sA'][s]
    At = A.sum(1); sAt = np.sqrt((sA ** 2).sum(1))       # [n,4] summed over 6 frames
    z = L['pfit'][s, 2]; sz = L['pcrlb'][s, 2]; N = L['pfit'][s, 3]
    f = At / At.sum(1, keepdims=True)
    tau = np.median(f, 0)
    res['tau_movie'] = tau.round(4).tolist()
    Phi, V, Q = chan_phase(At, tau, bead_ph)
    sPhi = chan_phase_err(At, sAt, tau, bead_ph)
    Lam, sgn, Phi0, Rz, curve = fit_period(Phi, z)
    res['fringe'] = dict(Lam_um=float(Lam), sign=int(sgn), Phi0_deg=float(np.rad2deg(Phi0)), R=float(Rz), n=int(len(z)))
    # residual phase vs z model -> z precision from phase, fringe-assignment test
    resid = wrap(Phi - sgn * 2 * np.pi * z / Lam - Phi0)
    sz4 = sPhi * Lam / (2 * np.pi)
    res['V_median'] = float(np.median(V)); res['sPhi_median_deg'] = float(np.rad2deg(np.median(sPhi)))
    res['sz_4pi_median_nm'] = float(np.median(sz4) * 1e3); res['sz_astig_median_nm'] = float(np.median(sz) * 1e3)
    # expected residual sd if only noise: sqrt(sPhi^2 + (2pi sz/Lam)^2)
    pred = np.sqrt(sPhi ** 2 + (2 * np.pi * sz / Lam) ** 2)
    res['resid_sd_deg_robust'] = float(np.rad2deg(1.4826 * np.median(np.abs(resid))))
    res['resid_sd_pred_deg_median'] = float(np.rad2deg(np.median(pred)))
    res['frac_resid_gt_90deg'] = float(np.mean(np.abs(resid) > np.pi / 2))
    # fringe pick: probability that the astigmatic z picks the right fringe, P(|err| < Lam/2) with err ~ N(0, sz)
    from scipy.special import erf
    pc = erf((Lam / 2) / (np.sqrt(2) * np.sqrt(sz ** 2 + sz4 ** 2)))
    res['fringe_pick_prob_median'] = float(np.median(pc)); res['fringe_pick_prob_mean'] = float(np.mean(pc))
    # by brightness
    res['by_N'] = []
    for lo, hi in ((800, 2000), (2000, 4000), (4000, 8000), (8000, 1e9)):
        q = (N >= lo) & (N < hi)
        if q.sum() < 20: continue
        res['by_N'].append(dict(N=[lo, hi], n=int(q.sum()), sz_astig_nm=float(np.median(sz[q]) * 1e3), sz_4pi_nm=float(np.median(sz4[q]) * 1e3),
                                V=float(np.median(V[q])), resid_sd_deg=float(np.rad2deg(1.4826 * np.median(np.abs(resid[q])))),
                                pick_prob=float(np.mean(pc[q]))))
    # contrast vs z (phase-aligned projection, robust to noise bias)
    ze = np.quantile(z, np.linspace(0, 1, 11)); res['V_vs_z'] = []
    for i in range(10):
        q = (z >= ze[i]) & (z <= ze[i + 1])
        res['V_vs_z'].append(dict(z=float(np.mean(z[q])), V_median=float(np.median(V[q])), V_proj=float(np.mean(V[q] * np.cos(resid[q]))),
                                  R=float(abs(cmean(resid[q])))))
    # per pattern angle 4Pi phase
    A0 = A[:, :3].sum(1); A1 = A[:, 3:].sum(1)
    p0, _, _ = chan_phase(A0, tau, bead_ph); p1, _, _ = chan_phase(A1, tau, bead_ph)
    B0 = A[:, [0, 1, 3]].sum(1); B1 = A[:, [2, 4, 5]].sum(1)
    q0, _, _ = chan_phase(B0, tau, bead_ph); q1, _, _ = chan_phase(B1, tau, bead_ph)
    sel = V > 0.15
    d = wrap(p1 - p0)[sel]; e = wrap(q1 - q0)[sel]
    bs = np.random.default_rng(0)
    bse = np.std([np.angle(cmean(d[bs.integers(0, len(d), len(d))])) for _ in range(200)])
    res['angle_phase_diff'] = dict(mean_deg=float(np.rad2deg(np.angle(cmean(d)))), err_deg=float(np.rad2deg(bse)), R=float(abs(cmean(d))),
                                   control_mean_deg=float(np.rad2deg(np.angle(cmean(e)))), n=int(sel.sum()))
    # vs z
    res['angle_phase_diff_vs_z'] = []
    zq = np.quantile(z[sel], np.linspace(0, 1, 6))
    for i in range(5):
        q = (z[sel] >= zq[i]) & (z[sel] <= zq[i + 1])
        res['angle_phase_diff_vs_z'].append((float(np.mean(z[sel][q])), float(np.rad2deg(np.angle(cmean(d[q]))))))
    jm, jcurve, jres = joint_model(Phi, z, L['x'][s], L['y'][s], L['frame'][s])
    res['joint_model'] = jm
    res['joint_resid_sd_deg_robust'] = float(np.rad2deg(1.4826 * np.median(np.abs(jres))))
    res['joint_R_by_N'] = [(lo, float(abs(cmean(jres[(N >= lo) & (N < hi)])))) for lo, hi in ((800, 2000), (2000, 4000), (4000, 8000), (8000, 1e9))]
    # local consistency: emitter pairs close in space (< 1.5 px) and time (< 450 frames), |dz| < 20 nm
    from scipy.spatial import cKDTree
    P3 = np.c_[L['x'][s], L['y'][s], L['frame'][s] / 300.0]
    pr = cKDTree(P3).query_pairs(1.5, output_type='ndarray')
    pr = pr[np.abs(z[pr[:, 0]] - z[pr[:, 1]]) < 0.02]
    res['close_pairs'] = dict(n=int(len(pr)), R=float(abs(cmean(wrap(Phi[pr[:, 0]] - Phi[pr[:, 1]])))))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith('bead') and k not in ('by_N', 'V_vs_z')}, indent=0)[:3500], flush=True)
    json.dump(res, open(f'{OUT}/diag/fourpi_{cell}.json', 'w'), indent=1)
    np.savez(f'{OUT}/diag/fourpi_{cell}_emitters.npz', jres=jres, Phi=Phi, V=V, z=z, sz=sz, sPhi=sPhi, N=N, x=L['x'][s], y=L['y'][s], frame=L['frame'][s], resid=resid)
    fig, ax = plt.subplots(1, 5, figsize=(25, 4.5))
    Ib = np.array(zc['I_r3']); fb = Ib / Ib.sum(1, keepdims=True)
    for c in range(4): ax[0].plot(zst[good], fb[good, c], '.-', label=f'ch{c}')
    ax[0].set_xlabel('stage z (um)'); ax[0].set_title(f'bead channel fractions (r=3px), period {res["bead_r3"]["Lam"]*1e3:.0f} nm'); ax[0].legend()
    hb = ax[1].hist2d(z, np.rad2deg(Phi), bins=(80, 72), range=((np.percentile(z, 1), np.percentile(z, 99)), (-180, 180)), cmap='magma')
    zz = np.linspace(np.percentile(z, 1), np.percentile(z, 99), 200)
    for kk in range(-3, 4):
        ax[1].plot(zz, np.rad2deg(wrap(sgn * 2 * np.pi * zz / Lam + Phi0)) , ',', c='c')
    ax[1].set_xlabel('z (um, PSF fit)'); ax[1].set_ylabel('4Pi phase (deg)'); ax[1].set_title(f'phase vs z: period {Lam*1e3:.0f} nm, R={Rz:.2f}')
    ax[2].plot(curve[curve[:, 0] == 1, 1] * 1e3, curve[curve[:, 0] == 1, 2], label='+'); ax[2].plot(curve[curve[:, 0] == -1, 1] * 1e3, curve[curve[:, 0] == -1, 2], label='-')
    ax[2].set_xlabel('fringe period (nm)'); ax[2].set_ylabel('coherence'); ax[2].legend()
    ax[3].plot([b['z'] for b in res['V_vs_z']], [b['V_median'] for b in res['V_vs_z']], 'o-', label='median |V|')
    ax[3].plot([b['z'] for b in res['V_vs_z']], [b['V_proj'] for b in res['V_vs_z']], 's-', label='phase-aligned V')
    ax[3].set_xlabel('z (um)'); ax[3].set_title('interference contrast vs z (PSF-weighted amplitudes)'); ax[3].legend()
    for sg_ in (1, -1):
        cc = jcurve[jcurve[:, 0] == sg_]; ax[4].plot(cc[:, 1] * 1e3, cc[:, 2], label=f'sign {sg_:+d}')
    ax[4].set_xlabel('fringe period (nm)'); ax[4].set_ylabel('R after field map + drift'); ax[4].legend(); ax[4].set_title(f'joint model: {jm["Lam"]*1e3:.0f} nm, R={jm["R"]:.2f}')
    fig.savefig(f'{OUT}/fig/fourpi_{cell}.png', dpi=75, bbox_inches='tight')
    return res


if __name__ == '__main__':
    run(sys.argv[1])
