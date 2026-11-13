"""
Part 4: per-pattern phases of the earlier processings of this movie versus the in-situ shared / per-step fits.

Conventions. Earlier processings: excitation relint (1 + depth sin(k . r - phase)), patterns in ACQUISITION order
(pattern p: orientation p % 2, phase step p // 2). Here: relint (1 + m cos(k . r - phi_s - D_s(f))), stored step
s = 3 (p % 2) + p // 2, so phi = phase + pi / 2 and stored order = patterns [0, 2, 4, 1, 3, 5]; a per-frame phase offset o that
is SUBTRACTED from the phase (phase - o) corresponds to D = -o.
Phases are compared at the field centre r_c = (200, 200) px (phi_c = phi - k . r_c), which removes the lever of the small
differences in k; phase steps are origin-free.
Output: out/nanoruler/perstep/compare.json
"""
import sys, json, math, pickle, warnings
import numpy as np, torch, scipy.io as sio
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L
warnings.filterwarnings('ignore')

SUB = '/mnt/data/SIMFLUX/Submission Data/80 nm rulers/Sim4_1_MMStack_Pos0.omeMod0.12Gauss10/'
F2 = '/mnt/data/SIMFLUX/data/fig2/80nm-panel-a-e/'
ORD = [0, 2, 4, 1, 3, 5]
RC = np.array([200.0, 200.0])
wrap = lambda v: (np.asarray(v) + math.pi) % (2 * math.pi) - math.pi


def entry(k6, phase_sin, depth=None, relint=None, note=''):
    k6 = np.asarray(k6, float)[ORD][:, :2]; phi = np.asarray(phase_sin, float)[ORD] + math.pi / 2
    e = dict(note=note, k=k6[[0, 3]].tolist(), phi=wrap(phi).tolist(), phi_centre=wrap(phi - k6 @ RC).tolist())
    if depth is not None: e['depth'] = np.asarray(depth, float)[ORD].tolist()
    if relint is not None:
        r = np.asarray(relint, float)[ORD]; e['relint'] = (r / r.mean()).tolist()
    return e


def steps(phi):
    phi = np.asarray(phi)
    return [[float(wrap(phi[3 * a + j + 1] - phi[3 * a + j])) for j in (0, 1)] for a in (0, 1)]


if __name__ == '__main__':
    cs, cp = sys.argv[1], sys.argv[2]
    R = {}
    im = sio.loadmat(SUB + 'info.mat')['mod']                       # [6, 5] = kx, ky, depth, phase, power
    R['paper_info_mat'] = entry(im[:, :2], im[:, 3], im[:, 2], im[:, 4], 'submission info.mat (estimated depths; SIMFLUX fit used depth 0.85)')
    rep_phase = np.deg2rad([-43.80, 39.34, -164.10, -68.65, 75.38, -179.57])
    rep_k = [[-0.0023, 1.8573], [1.8605, 0.0017]] * 3
    R['paper_report_txt'] = entry(rep_k, rep_phase, [0.82, 0.88, 0.90, 0.90, 0.87, 0.93], [0.183, 0.152, 0.183, 0.145, 0.184, 0.152],
                                  'submission report.txt (k rounded to 4 decimals)')
    for nm, fn in (('cache_2023_sf_fits_cfg', F2 + 'cache/sim4_1_MMStack_Pos0.ome/sf_fits_cfg.pickle'),):
        m, off = pickle.load(open(fn, 'rb'))['mod']
        R[nm] = entry(m['k'], m['phase'], m['depth'], m['relint'], 'cache pickle: mod[0] per pattern + mod[1] per-frame phase offsets (first 5000 frames)')
        off = np.asarray(off)
    p = pickle.load(open(F2 + 'results/sim4_1_MMStack_Pos0.ome/pattern.pickle', 'rb'))
    R['results_2022_pattern_pickle'] = entry(p['mod']['k'], p['mod']['phase'], p['mod']['depth'], p['mod']['relint'],
                                             'results/pattern.pickle (ModulationPattern; phase_offset_per_frame [30006], identical for the 3 steps of an orientation)')
    po22 = p['phase_offset_per_frame']
    bl = json.load(open(L.ROOT + '/out/nanoruler/baselines/pattern_bl.json'))['estimated']
    ph = np.asarray(bl['phase_sin_rad']); kb = np.asarray(bl['k_rad_per_px'])
    R['baseline_rerun'] = dict(note='analysis/baselines/paper_pipeline.py rerun (already in stored step order)', k=kb[[0, 3]].tolist(),
                               phi=wrap(ph + math.pi / 2).tolist(), phi_centre=wrap(ph + math.pi / 2 - kb @ RC).tolist(), depth=bl['depth'],
                               relint=bl['relint_mean1'])
    obl = np.load(L.ROOT + '/out/nanoruler/baselines/pattern_bl_phase_offset.npy')
    F = torch.load(L.OUT + 'fits.pt', weights_only=False)
    fr6 = np.arange(0, L.NFRAMES, 6)
    mine = {}
    for nm, c in (('insitu_shared', cs), ('insitu_perstep', cp)):
        P = L.Pattern.from_state(F[c]['full']['state'])
        Dm = P.total_drift_np(fr6)                                                # [6, ncyc]
        phi = P.phi.numpy() + Dm.mean(1); k = P.k.numpy()
        R[nm] = dict(note=f'this work, {c}, time-mean phase', k=k.tolist(), phi=wrap(phi).tolist(),
                     phi_centre=wrap(phi - np.repeat(k, 3, 0) @ RC).tolist(), depth=P.m.tolist(),
                     relint=np.exp(P.logrel.numpy() - P.logrel.numpy().mean()).tolist())
        mine[nm] = Dm - Dm.mean(1, keepdims=True)
    ref = np.asarray(R['insitu_perstep']['phi_centre'])
    for nm, e in R.items():
        e['phase_steps_rad'] = steps(e['phi']); e['phase_steps_deg'] = np.rad2deg(steps(e['phi'])).tolist()
        d = wrap(np.asarray(e['phi_centre']) - ref)
        e['dphi_centre_vs_perstep_rad'] = d.tolist()
        # per-orientation common offset (a translation of that orientation's coordinate) and the step-relative remainder
        e['common_offset_rad'] = [float(d[:3].mean()), float(d[3:].mean())]
        e['step_relative_rad'] = np.concatenate([d[:3] - d[:3].mean(), d[3:] - d[3:].mean()]).tolist()
        e['step_relative_nm'] = (np.asarray(e['step_relative_rad']) * np.repeat([219.9, 219.5], 3) / (2 * math.pi)).tolist()

    # ---- drift models of the earlier processings
    C = {}
    o6 = off[:len(off) // 6 * 6].reshape(-1, 6)[:, ORD]                           # [833, 6 stored steps]
    C['cache_offsets'] = dict(length=int(len(off)), per_step_sd_rad=o6.std(0).tolist(), per_step_mean=o6.mean(0).tolist(),
                              step_relative_sd_rad=np.concatenate([(o6[:, :3] - o6[:, :3].mean(1, keepdims=True)).std(0),
                                                                   (o6[:, 3:] - o6[:, 3:].mean(1, keepdims=True)).std(0)]).tolist(),
                              frame_to_frame_rms=float(np.sqrt(np.mean(np.diff(o6, axis=0) ** 2))))
    # compare the cache's per-pattern offsets (first 5000 frames) with the per-step fit, both as D (= -offset), mean removed
    # over the same frames; total per step and step-relative (within orientation)
    nc = len(o6); Dc = -(o6 - o6.mean(0)).T                                       # [6, 833]
    Dp = mine['insitu_perstep'][:, :nc]; Dp = Dp - Dp.mean(1, keepdims=True)
    Ds = mine['insitu_shared'][:, :nc]; Ds = Ds - Ds.mean(1, keepdims=True)
    rel = lambda X: np.concatenate([X[:3] - X[:3].mean(0), X[3:] - X[3:].mean(0)])
    cc = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    C['cache_vs_perstep_first5000'] = dict(
        corr_total=[cc(Dc[s], Dp[s]) for s in range(6)], corr_total_flipped_sign=[cc(-Dc[s], Dp[s]) for s in range(6)],
        corr_step_relative=[cc(rel(Dc)[s], rel(Dp)[s]) for s in range(6)],
        rms_cache_rel=rel(Dc).std(1).tolist(), rms_perstep_rel=rel(Dp).std(1).tolist(),
        corr_cache_vs_shared_total=[cc(Dc[s], Ds[s]) for s in range(6)])
    np.savez(L.OUT + 'compare_cache_drift.npz', D_cache=Dc, D_perstep=Dp, D_shared=Ds, frames=fr6[:nc])
    q = po22[:30000].reshape(-1, 6)[:, ORD]
    C['results_2022_offsets'] = dict(length=int(len(po22)), per_step_sd_rad=q.std(0).tolist(),
                                     max_abs_diff_between_steps_of_an_orientation=float(max(np.abs(q[:, :3] - q[:, :1]).max(), np.abs(q[:, 3:] - q[:, 3:4]).max())))
    qb = obl.reshape(-1, 6)
    C['baseline_offsets'] = dict(length=int(len(obl)), per_step_sd_rad=qb.std(0).tolist(),
                                 max_abs_diff_between_steps_of_an_orientation=float(max(np.abs(qb[:, :3] - qb[:, :1]).max(), np.abs(qb[:, 3:] - qb[:, 3:4]).max())),
                                 corr_with_shared_fit=[cc(-qb[:, 3 * a], mine['insitu_shared'][3 * a]) for a in (0, 1)],
                                 corr_with_shared_fit_flipped=[cc(qb[:, 3 * a], mine['insitu_shared'][3 * a]) for a in (0, 1)])
    R['drift_models'] = C
    json.dump(R, open(L.OUT + 'compare.json', 'w'), indent=1)
    for nm, e in R.items():
        if nm == 'drift_models': continue
        print(f"{nm:28s} phi_c {np.round(e['phi_centre'], 3).tolist()} steps deg {np.round(e['phase_steps_deg'], 1).tolist()} rel nm {np.round(e['step_relative_nm'], 2).tolist()}")
    print(json.dumps(C, indent=1))
