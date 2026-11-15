"""Pre-registered smoke-test metrics (critic 30 #4; revised iteration 32 for critic 31 #3) for a run_uncalibrated
--pass2_mode solve run on a real chunk, against SIMCODE's localizations of the same frames (read-only hdf5). Expectations are
fixed before the run in smoke_expectations.txt (same directory).
  M0 global pre-fit: loose one-to-one match (|frame diff| <= 3, 1.5 px) of ours vs SIMCODE and vs SIMCODE with x <-> y
     swapped; the configuration with more pairs is kept, its median offset is removed from SIMCODE, then re-match at 0.5 px.
  M1 match rate / offset vs SIMCODE per SIMCODE photon bin (fractions of SIMCODE locs and of ours matched); the offset
     median and the robust sd of the offset (1.4826 MAD, nm = combined precision) on EXACT window-start pairs
     (our wstart == SIMCODE frame); the non-exact pairs (start differs by 1-3 frames) are reported separately.
  M2 photon-unit cross-calibration: our window photons (sum of the network I_t, label units) / SIMCODE photons on exact
     pairs: median ratio per SIMCODE bin and the Theil-Sen slope over 200-2000 SIMCODE photons (expect ~ kappa).
  M3 q/dof (intensity-domain chi2 / dof of the pattern solve) per our-photon bin (label units): p50 / p90 / p99, frac > 2 /
     > 4, next to the sim reference (gate scene --ref_scene with the SAME network + calibration incl. PM, matched targets,
     binned by its own network photons = the same units).
  M4 consecutive-window precision vs s_cal: the same site in photon-disjoint windows w and w + 6 (< 1 px, mutual
     nearest), z = (r_w - r_w+6) / sqrt(s_w^2 + s_w+6^2) per axis: 1.4826 med |z| (expect ~1; > 1.2 = sigma too
     small), frac |z|_radial > 4, per photon bin (mean of the two windows).
  M5 FRC (1/7) of ours vs SIMCODE: halves = alternating 300-frame blocks, 5 nm render. Sampling-limited on a 6000-frame
     chunk (indicative only, not a resolution claim).
  M6 counts in SIMCODE photon units: (a) SIMCODE locs per SIMCODE bin and the fraction matched; (b) ours binned by our
     photons / kappa vs SIMCODE binned by its photons; (c) our matched fraction per (our photons / kappa) bin.
  M7 pattern: our pattern_est.pt (pitch, angle per angle, depth per angle, step relint) vs the in-situ calibration (models/insitu/calib_0.5nM.json) and
     SIMCODE's mp_ndi.pickle (read-only); depth reported as an estimator output, not a physical depth (critic 31 #7).
usage: python smoke_metrics.py --run <run_dir> --simcode <sfhd-ndi.hdf5> --f0 12000 --f1 18000 [--tag est] [--kappa 1.95]
       [--ref_scene rs05qm --ref_tag _hd_sc5nb15pm] [--mp <mp_ndi.pickle>] -> <run_dir>/smoke_metrics.txt / .json"""
import argparse, glob, json, os, pickle
import numpy as np
from scipy.spatial import cKDTree
from simcode import config

# SIMCODE's published localizations of the 0.5 nM movie (read-only, next to the movie under the data root)
SIMCODE_05 = 'dnapaint/0.5nM/cover1/1/results/simflux_5.5ms_500mW/sfhd-ndi.hdf5'

PX = 108.33
PB = [(200, 300), (300, 350), (350, 400), (400, 450), (450, 600), (600, 1000), (1000, 2000), (2000, 1e9)]
PB3 = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9)]


def load_ours(run, tag):
    Q = {}
    for f in sorted(glob.glob(f'{run}/pass2_{tag}_*.npz')):
        d = np.load(f)
        for k in d.files:
            if d[k].ndim: Q.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in Q.items()}


def load_simcode(path, f0, f1):
    import h5py
    L = h5py.File(path, 'r')['locs'][:]
    L = L[(L['frame'] >= f0) & (L['frame'] < f1)]
    return dict(frame=L['frame'].astype(np.int64), x=L['x'].astype(np.float64), y=L['y'].astype(np.float64), phot=L['photons'].astype(np.float64))


def match(A, B, dt=3, r=0.5):
    """one-to-one greedy of A (frame w, x, y) to B within |frame diff| <= dt and r px: exact-start pairs first, then by distance"""
    tb = cKDTree(np.c_[B['x'], B['y'], B['frame'] / dt * r * 0.999])
    ta = np.c_[A['x'], A['y'], A['frame'] / dt * r * 0.999]
    pairs = tb.query_ball_point(ta, r * np.sqrt(2) + 1e-6)
    cand = []
    for i, js in enumerate(pairs):
        for j in js:
            d = np.hypot(A['x'][i] - B['x'][j], A['y'][i] - B['y'][j])
            df = abs(A['frame'][i] - B['frame'][j])
            if d <= r and df <= dt: cand.append((df > 0, d, i, j))
    cand.sort(); ua, ub = set(), set(); M = []      # exact window starts first, then by distance
    for _, d, i, j in cand:
        if i in ua or j in ub: continue
        ua.add(i); ub.add(j); M.append((i, j))
    return np.array(M, int).reshape(-1, 2)


def theil_sen(x, y, n=4000, seed=0):
    rng = np.random.default_rng(seed); i = rng.integers(0, len(x), n); j = rng.integers(0, len(x), n)
    ok = np.abs(x[i] - x[j]) > 50
    return float(np.median((y[i] - y[j])[ok] / (x[i] - x[j])[ok]))


from simcode.evaluation.frc import frc, frc_halves as frc_split  # noqa (the one FRC implementation)


mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) else np.nan


def pattern_summary(k, m, rel=None):
    k = np.asarray(k, float)
    out = {}
    for a in (0, 1):
        out[f'pitch_px_{a}'] = float(2 * np.pi / np.hypot(*k[a][:2])); out[f'angle_deg_{a}'] = float(np.degrees(np.arctan2(k[a][1], k[a][0])))
        out[f'depth_{a}'] = float(np.mean(m[3 * a:3 * a + 3]))
    if rel is not None: out['relint'] = [float(v) for v in np.asarray(rel) / np.mean(rel)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--simcode', default=SIMCODE_05)                # relative: under the data root
    ap.add_argument('--f0', type=int, required=True)
    ap.add_argument('--f1', type=int, required=True)
    ap.add_argument('--tag', default='est')
    ap.add_argument('--crop', default='5,5,110')
    ap.add_argument('--kappa', type=float, default=1.95)       # pre-registered (smoke_expectations.txt, sc7/kappa.txt)
    ap.add_argument('--ref_dir', default=os.path.join(config.root('out'), 'gate'))
    ap.add_argument('--ref_scene', default='rs05qm')
    ap.add_argument('--ref_tag', default='_hd_sc5nb15pm')
    ap.add_argument('--calib', default=config.INSITU_05)
    ap.add_argument('--mp', default='')                         # default: mp_ndi.pickle next to --simcode
    ap.add_argument('--no_prefit', type=int, default=0)
    # 2026-11-09 (duplicates worker): output-mode options; defaults reproduce the pre-registered run exactly
    ap.add_argument('--mode', default='window')                # 'window' (per-window list) | 'disjoint' (wstart % 6 == 0 only)
    ap.add_argument('--pmin', type=float, default=0.0)         # detection-probability cut on our list (critic 32 #2: 0.63)
    ap.add_argument('--suffix', default='')                    # output file suffix (smoke_metrics<suffix>.txt / .json)
    ap.add_argument('--simcode_npz', default='')               # sim test: run_simcode.py npz (ndi rows) instead of the hdf5
    # 2026-11-09 (critic 33 #8, metric hygiene): like-for-like M3 sim reference = a sim-movie run of the SAME pipeline and
    # calibration procedure (pass2_est rows, the same --mode / --pmin selection, binned by its network photons); default off
    ap.add_argument('--ref_run', default='')
    a = ap.parse_args()
    a.simcode = config.resolve(a.simcode, 'data'); a.calib = config.model(a.calib)
    L = []; pr = lambda s: (L.append(s), print(s, flush=True)); J = {}
    K = a.kappa
    O = load_ours(a.run, a.tag)
    O_all = dict(O)
    if a.mode == 'disjoint' or a.pmin > 0:
        mk = (O['wstart'] % 6 == 0) if a.mode == 'disjoint' else np.ones(len(O['wstart']), bool)
        mk &= O['p'] >= a.pmin
        O = {k: v[mk] for k, v in O.items()}
    x0c, y0c, szc = (int(v) for v in a.crop.split(','))
    if a.simcode_npz:
        Z = np.load(a.simcode_npz); r_ = Z['ndi_idx']; fr_ = Z['frame'][r_].astype(np.int64); mm = (fr_ >= a.f0) & (fr_ < a.f1)
        S = dict(frame=fr_[mm], x=Z['ndi_x'][mm].astype(np.float64), y=Z['ndi_y'][mm].astype(np.float64), phot=Z['ndi_photons'][mm].astype(np.float64))
    else:
        S = load_simcode(a.simcode, a.f0, a.f1)
    ins = (S['x'] >= x0c) & (S['x'] < x0c + szc) & (S['y'] >= y0c) & (S['y'] < y0c + szc)
    S = {k: v[ins] for k, v in S.items()}
    keep = (O['wstart'] >= a.f0) & (O['wstart'] < a.f1) & np.isfinite(O['x_hy0'])
    O = {k: v[keep] for k, v in O.items()}
    A = dict(frame=O['wstart'].astype(np.int64), x=O['x_hy0'].astype(np.float64), y=O['y_hy0'].astype(np.float64))
    pr(f'ours {len(A["x"])} localizations, SIMCODE {len(S["x"])} in frames [{a.f0}, {a.f1}) inside the crop; kappa (pre-registered) {K}')
    # ---- M0 global pre-fit (offset / x-y swap)
    J['M0'] = {}
    if not a.no_prefit:
        best = None
        for swap in (False, True):
            Sb = dict(S, x=S['y'], y=S['x']) if swap else S
            M0 = match(A, Sb, r=1.5)
            off = (np.median(A['x'][M0[:, 0]] - Sb['x'][M0[:, 1]]), np.median(A['y'][M0[:, 0]] - Sb['y'][M0[:, 1]])) if len(M0) else (0.0, 0.0)
            pr(f'M0 pre-fit swap={swap}: {len(M0)} loose pairs (1.5 px), median offset ours - SIMCODE ({off[0] * PX:+.1f}, {off[1] * PX:+.1f}) nm')
            if best is None or len(M0) > best[1]: best = (swap, len(M0), off)
        swap, _, off = best
        if swap: S = dict(S, x=S['y'], y=S['x'])
        S = dict(S, x=S['x'] + off[0], y=S['y'] + off[1])
        J['M0'] = dict(swap=bool(swap), off_nm=[float(off[0] * PX), float(off[1] * PX)])
        pr(f'   -> kept swap={swap}, SIMCODE shifted by ({off[0] * PX:+.1f}, {off[1] * PX:+.1f}) nm before the 0.5 px match')
    M = match(A, S)
    ex = A['frame'][M[:, 0]] == S['frame'][M[:, 1]]
    pr(f'M1/M2 matched pairs {len(M)} (ours {len(M) / max(len(A["x"]), 1):.3f}, SIMCODE {len(M) / max(len(S["x"]), 1):.3f}); '
       f'exact window start {ex.mean():.3f}; |start diff| 1/2/3: ' + ' / '.join(f'{np.mean(np.abs(A["frame"][M[:, 0]] - S["frame"][M[:, 1]]) == d):.3f}' for d in (1, 2, 3)))
    dx = (A['x'][M[:, 0]] - S['x'][M[:, 1]]) * PX; dy = (A['y'][M[:, 0]] - S['y'][M[:, 1]]) * PX
    ps = S['phot'][M[:, 1]]; po = O['phot'][M[:, 0]]
    mS = np.zeros(len(S['x']), bool); mS[M[:, 1]] = True
    mO = np.zeros(len(A['x']), bool); mO[M[:, 0]] = True
    J['M1'] = {}
    for lo, hi in PB:
        ms = (S['phot'] >= lo) & (S['phot'] < hi)
        if ms.sum() < 10: continue
        mo = (O['phot'] / K >= lo) & (O['phot'] / K < hi)
        r = dict(n_simcode=int(ms.sum()), frac_simcode_matched=float(mS[ms].mean()), n_ours=int(mo.sum()),
                 frac_ours_matched=float(mO[mo].mean()) if mo.any() else np.nan)
        for nm, sel in (('exact', ex), ('nonexact', ~ex)):
            m = (ps >= lo) & (ps < hi) & sel
            r[nm] = dict(n=int(m.sum()), dx_med=float(np.median(dx[m])) if m.any() else np.nan, dy_med=float(np.median(dy[m])) if m.any() else np.nan,
                         sd_x=float(mad(dx[m])), sd_y=float(mad(dy[m])), phot_ratio=float(np.median(po[m] / ps[m])) if m.any() else np.nan)
        J['M1'][f'{lo}-{hi:.0f}'] = r
        e, n_ = r['exact'], r['nonexact']
        pr(f'   SIMCODE ph {lo:4d}-{hi:<5.0f} n {r["n_simcode"]:6d} matched {r["frac_simcode_matched"]:.3f} | ours (ph/kappa bin) n {r["n_ours"]:6d} '
           f'matched {r["frac_ours_matched"]:.3f} | EXACT n {e["n"]:5d} offset ({e["dx_med"]:+.2f}, {e["dy_med"]:+.2f}) nm sd ({e["sd_x"]:.2f}, {e["sd_y"]:.2f}) nm '
           f'| non-exact n {n_["n"]:5d} sd ({n_["sd_x"]:.2f}, {n_["sd_y"]:.2f}) nm')
    m = (ps >= 200) & (ps < 2000) & ex
    J['M2'] = dict(theil_sen=theil_sen(ps[m], po[m]) if m.sum() > 50 else np.nan, median_ratio=float(np.median(po[m] / ps[m])) if m.any() else np.nan,
                   per_bin={f'{lo}-{hi:.0f}': r['exact']['phot_ratio'] for (lo, hi), r in zip(PB, J['M1'].values())}, kappa_prereg=K)
    pr(f'M2 photon units (ours label / SIMCODE, exact pairs, 200-2000): Theil-Sen {J["M2"]["theil_sen"]:.3f}, median ratio {J["M2"]["median_ratio"]:.3f} '
       f'(pre-registered kappa {K}); per SIMCODE bin: ' + ' '.join(f'{k}:{v:.2f}' for k, v in J['M2']['per_bin'].items()))
    # ---- M3 q/dof vs the sim reference (same network + calibration, same units)
    from simcode.benchmark.gate_report import load, match as gmatch
    J['M3'] = {}
    rq = rp = None
    if a.ref_run:
        Rr = load_ours(a.ref_run, 'est')
        mk = np.isfinite(Rr['q_dof']) & (Rr['p'] >= a.pmin)
        if a.mode == 'disjoint': mk &= Rr['wstart'] % 6 == 0
        rq, rp = Rr['q_dof'][mk], Rr['phot'][mk]
        pr(f'M3 sim reference: run {a.ref_run} ({len(rq)} rows, mode {a.mode}, p >= {a.pmin})'); J['M3']['ref'] = a.ref_run
    for scn in ((a.ref_scene, 'rs05pm') if not a.ref_run else ()):
        try:
            R, _ = load(a.ref_dir, scn, a.ref_tag)
            idx = np.nonzero(R['sc_score'] >= 0.5)[0]
            og = gmatch(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], R['sc_x_det'][idx], R['sc_y_det'][idx], 1.0)
            rq, rp = R['sc_q'][idx][og >= 0], R['sc_phot'][idx][og >= 0]
            pr(f'M3 sim reference: {scn}{a.ref_tag} ({len(rq)} matched targets)'); J['M3']['ref'] = scn + a.ref_tag
            break
        except Exception as e:      # noqa
            pr(f'   (no sim reference {scn}{a.ref_tag}: {e})')
    qs = lambda q: (np.percentile(q, 50), np.percentile(q, 90), np.percentile(q, 99), np.mean(q > 2), np.mean(q > 4)) if len(q) > 10 else (np.nan,) * 5
    pr('M3 q/dof per our-photon bin (label units): p50 / p90 / p99 / >2 / >4   real | sim reference (same bins, its network photons)')
    for lo, hi in PB3:
        m = (O['phot'] >= lo) & (O['phot'] < hi)
        a_ = qs(O['q_dof'][m]); b_ = qs(rq[(rp >= lo) & (rp < hi)]) if rq is not None else (np.nan,) * 5
        nr = int(((rp >= lo) & (rp < hi)).sum()) if rq is not None else 0
        J['M3'][f'{lo}-{hi:.0f}'] = dict(real=list(map(float, a_)), sim=list(map(float, b_)), n=int(m.sum()), n_sim=nr)
        pr(f'   {lo:4d}-{hi:<5.0f} n {m.sum():6d} real {a_[0]:.2f} / {a_[1]:.2f} / {a_[2]:.2f} / {a_[3]:.3f} / {a_[4]:.3f}   '
           f'| sim n {nr:5d} {b_[0]:.2f} / {b_[1]:.2f} / {b_[2]:.2f} / {b_[3]:.3f} / {b_[4]:.3f}')
    # ---- M4 consecutive-window precision
    w = O['wstart']; x, y = O['x_hy0'].astype(np.float64), O['y_hy0'].astype(np.float64)
    s = np.c_[O['sx_cal'], O['sy_cal']].astype(np.float64)
    T1 = cKDTree(np.c_[x, y, w * 1e3])
    d, j = T1.query(np.c_[x, y, (w + 6) * 1e3], distance_upper_bound=1.0)
    ok = np.isfinite(d); i = np.nonzero(ok)[0]; j = j[ok]
    d2, i2 = T1.query(np.c_[x[j], y[j], (w[j] - 6) * 1e3], distance_upper_bound=1.0)
    mut = i2 == i; i, j = i[mut], j[mut]
    z = np.c_[x[i] - x[j], y[i] - y[j]] / np.sqrt(s[i] ** 2 + s[j] ** 2)
    pm = 0.5 * (O['phot'][i] + O['phot'][j])
    J['M4'] = {}
    pr(f'M4 consecutive photon-disjoint windows (w, w+6), mutual nearest < 1 px: {len(i)} pairs (bins: our photons, label units)')
    for lo, hi in PB3 + [(200, 2000)]:
        m = (pm >= lo) & (pm < hi)
        if m.sum() < 20: continue
        r = dict(n=int(m.sum()), zcal=float(1.4826 * np.median(np.abs(z[m]))), z4=float(np.mean(np.hypot(*z[m].T) > 4)),
                 sd_nm=float(1.4826 * np.median(np.abs(np.c_[x[i] - x[j], y[i] - y[j]][m])) * PX / np.sqrt(2)),
                 s_cal_nm=float(np.median(s[i][m]) * PX))
        J['M4'][f'{lo}-{hi:.0f}'] = r
        pr(f'   {lo:4d}-{hi:<5.0f} n {r["n"]:6d} zcal {r["zcal"]:.2f} |z|>4 {100 * r["z4"]:.2f}% robust per-window sd {r["sd_nm"]:.2f} nm vs median s_cal {r["s_cal_nm"]:.2f} nm')
    # ---- M5 FRC (indicative)
    m = (O['phot'] >= 200)
    J['M5'] = dict(ours=frc_split(x[m], y[m], w[m]), simcode=frc_split(S['x'], S['y'], S['frame']))
    pr(f'M5 FRC (1/7, 300-frame alternating halves, 5 nm px; sampling-limited, indicative only): ours {J["M5"]["ours"]:.1f} nm, SIMCODE {J["M5"]["simcode"]:.1f} nm')
    # ---- M6 counts in SIMCODE photon units
    J['M6'] = {}
    pr(f'M6 counts per SIMCODE-photon bin: SIMCODE n / matched | ours by (our photons / kappa {K}) n / ratio to SIMCODE / matched')
    for lo, hi in PB:
        ms = (S['phot'] >= lo) & (S['phot'] < hi); mo = (O['phot'] / K >= lo) & (O['phot'] / K < hi)
        r = dict(simcode=int(ms.sum()), simcode_matched=float(mS[ms].mean()) if ms.any() else np.nan, ours=int(mo.sum()),
                 ours_matched=float(mO[mo].mean()) if mo.any() else np.nan, pairs=int(((ps >= lo) & (ps < hi)).sum()))
        J['M6'][f'{lo}-{hi:.0f}'] = r
        pr(f'   {lo:4d}-{hi:<5.0f} SIMCODE {r["simcode"]:7d} / {r["simcode_matched"]:.3f} | ours {r["ours"]:7d} ratio {r["ours"] / max(r["simcode"], 1):.2f} / {r["ours_matched"]:.3f}'
           + ('   <-- 300-450' if 300 <= lo < 450 else ''))
    # ---- M6x (2026-11-09): our counts per output mode (per-window / photon-disjoint / merged chains) x detection cut, binned by
    #      our photons / kappa AND by our photons / the measured M2 ratio (empirical cross-calibration), vs SIMCODE
    from simcode.processing import dedup as DD
    J['M6x'] = {}
    Oa = {k: v[(O_all['wstart'] >= a.f0) & (O_all['wstart'] < a.f1) & np.isfinite(O_all['x_hy0'])] for k, v in O_all.items()}
    k2 = J['M2']['median_ratio']
    modes = {}
    for pm in (0.5, 0.63):
        q = {k: v[Oa['p'] >= pm] for k, v in Oa.items()}
        modes[f'window p>={pm}'] = q['phot']
        modes[f'disjoint p>={pm}'] = q['phot'][q['wstart'] % 6 == 0]
        Mg = DD.merge(q)
        modes[f'merged p>={pm}'] = Mg['phot']
    pr(f'M6x counts per SIMCODE-photon bin; ours binned by photons / kappa {K} | by photons / M2 ratio {k2:.3f}: n (ratio to SIMCODE)')
    pr('   bin        SIMCODE | ' + ' | '.join(f'{m:>22s}' for m in modes))
    for lo, hi in PB:
        ms = int(((S['phot'] >= lo) & (S['phot'] < hi)).sum())
        cells = []
        J['M6x'][f'{lo}-{hi:.0f}'] = dict(simcode=ms)
        for m, ph_ in modes.items():
            n1 = int(((ph_ / K >= lo) & (ph_ / K < hi)).sum()); n2 = int(((ph_ / k2 >= lo) & (ph_ / k2 < hi)).sum())
            J['M6x'][f'{lo}-{hi:.0f}'][m] = dict(n_kappa=n1, n_m2=n2)
            cells.append(f'{n1:6d} ({n1 / max(ms, 1):5.2f}) {n2 / max(ms, 1):5.2f}')
        pr(f'   {lo:4d}-{hi:<7.0f} {ms:6d} | ' + ' | '.join(f'{c:>22s}' for c in cells))
    pr('   totals: SIMCODE ' + str(len(S['phot'])) + ' | ' + ' | '.join(f'{m} {len(v)}' for m, v in modes.items()))
    # ---- M7 pattern vs calibration and SIMCODE
    J['M7'] = {}
    try:
        import torch
        P = torch.load(f'{a.run}/pattern_est.pt', weights_only=False)
        ours = pattern_summary(P['k'].numpy(), P['m'].numpy(), np.exp(P['logrel'].numpy()))
        c = json.load(open(a.calib))['pattern']
        cal = pattern_summary(c['k_rad_per_px'], c['depth'], c['relint'])
        mpf = a.mp or os.path.join(os.path.dirname(a.simcode), 'mp_ndi.pickle')
        mp = None
        if os.path.exists(mpf):
            q = pickle.load(open(mpf, 'rb'))['mod']
            mp = pattern_summary([q['k'][0], q['k'][3]], q['depth'], q['relint'])
        J['M7'] = dict(ours=ours, calib=cal, mp_ndi=mp)
        pr('M7 pattern (ours | calib_0.5nM.json | SIMCODE mp_ndi):')
        for kk in ('pitch_px_0', 'pitch_px_1', 'angle_deg_0', 'angle_deg_1', 'depth_0', 'depth_1'):
            fm = '.4f' if 'pitch' in kk else ('.3f' if 'angle' in kk else '.3f')
            pr(f'   {kk:12s} {ours[kk]:{fm}} | {cal[kk]:{fm}} | ' + (f'{mp[kk]:{fm}}' if mp else 'n/a') +
               f'   (ours - calib {ours[kk] - cal[kk]:+{fm}}' + (f', ours - mp {ours[kk] - mp[kk]:+{fm}})' if mp else ')'))
        pr(f'   relint ours {np.round(ours["relint"], 3).tolist()} | calib {np.round(cal["relint"], 3).tolist()}' + (f' | mp {np.round(mp["relint"], 3).tolist()}' if mp else ''))
        pr('   (depth = estimator output on the network intensities, photon-dependent; not a physical depth)')
    except Exception as e:      # noqa
        pr(f'M7 failed: {e}')
    json.dump(J, open(f'{a.run}/smoke_metrics{a.suffix}.json', 'w'), indent=1, default=float)
    open(f'{a.run}/smoke_metrics{a.suffix}.txt', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
