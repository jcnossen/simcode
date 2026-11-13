"""Classical-baseline run (bl_run.py) -> nanoruler-analysis tables, pattern json and sanity report (CPU only).

Tables (nra_lib format, npz): frame = CYCLE index (window start // 6; only windows whose start is a multiple of 6 are kept
and this is asserted / counted), x_nm, y_nm (65 nm px, x = column, pixel centres at integer px coordinates, plus the
correction --shift if the measured offset to the published tables calls for one; default: applied automatically only if the
measured median offset is within 0.1 px of +-0.5 px), sx_nm, sy_nm = crlb_x_nm, crlb_y_nm = the fit's CRLB, photons (per
6-frame window), bg (photons / px per 6-frame WINDOW for both methods, the unit of the published tables: the SMLM fit's bg as
is, the SIMFLUX fit's per-frame bg x 6). NOT drift corrected. Extra arrays: SMLM chisq, moderr (modulation error against
the SIMFLUX pattern), in_sf (row passed to the SIMFLUX table).

Sanity (json + text): counts vs published, acceptance at each filter, mutual-nearest-neighbour matched pairs per cycle
(radius --rmatch px) vs the published g2d-nodriftc / simflux-nodriftc tables (median offset, robust sd = 1.4826 MAD), CRLB
and photon percentiles, pattern vs the calib json and vs the published report.txt, run time.
Pattern conventions: the pipeline uses I_k = relint_k (1 + depth_k sin(k.r - phase_k)) with relint summing to 1 over the 6
steps; the calib json uses relint (mean 1) (1 + depth cos(k.r - phase)): phase_cos = phase_sin + pi / 2. report.txt lists the
patterns in ACQUISITION order (orientation alternating); reordered step s = acquisition pattern [0, 2, 4, 1, 3, 5][s].

usage: python analysis/nanoruler/bl_tables.py --tag run_roi10 [--suffix ''] [--log out/nanoruler/baselines/logs/<tag>.log]
outputs in out/nanoruler/baselines/: tab_bl_smlm<suffix>.npz, tab_bl_simflux<suffix>.npz, pattern_bl<suffix>.json,
  pattern_bl<suffix>_phase_offset.npy (per-frame phase offsets used by the SIMFLUX fit), sanity_bl<suffix>.json / .txt"""
import os, sys, json, re, argparse, pickle
import numpy as np, h5py
from scipy.spatial import cKDTree
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.join(REPO, 'analysis', 'nanoruler'))
import nra_lib as L

PX = 65.0
OUT = f'{REPO}/out/nanoruler/baselines/'
PUB = '/mnt/data/SIMFLUX/Submission Data/80 nm rulers/Sim4_1_MMStack_Pos0.omeMod0.12Gauss10/'
CALIB = f'{REPO}/out/nanoruler/calib_nr80.json'
ACQ = [0, 2, 4, 1, 3, 5]          # reordered step -> acquisition pattern index of report.txt


def h5(fn): return h5py.File(fn, 'r')['locs'][:]
def pct(v): return [float(x) for x in np.percentile(v, [5, 25, 50, 75, 95])]
def wrap(a): return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def match(a, p, r):
    """mutual nearest neighbours within r px in the same cycle; a, p: dicts with frame (cycle), x, y. Returns index pairs."""
    A = np.c_[a['x'], a['y'], a['frame'] * 1e4]; P = np.c_[p['x'], p['y'], p['frame'] * 1e4]
    da, ja = cKDTree(P).query(A, distance_upper_bound=r); dp, jp = cKDTree(A).query(P, distance_upper_bound=r)
    i = np.nonzero(np.isfinite(da))[0]; i = i[jp[ja[i]] == i]
    return i, ja[i]


def cmp_pub(t, p, r):
    i, j = match(t, p, r)
    dx = t['x'][i] - p['x'][j]; dy = t['y'][i] - p['y'][j]
    tb = np.minimum(t['frame'][i] // 1000, 4)
    return dict(n_ours=int(len(t['x'])), n_published=int(len(p['x'])), n_matched=int(len(i)),
                frac_of_ours_matched=float(len(i) / len(t['x'])), frac_of_published_matched=float(len(i) / len(p['x'])),
                median_offset_px=[float(np.median(dx)), float(np.median(dy))], rsd_px=[L.rsd(dx), L.rsd(dy)],
                median_offset_nm=[float(np.median(dx)) * PX, float(np.median(dy)) * PX], rsd_nm=[L.rsd(dx) * PX, L.rsd(dy) * PX],
                median_offset_px_per_1000_cycles=[[float(np.median(dx[tb == k])), float(np.median(dy[tb == k]))] for k in range(5)],
                photon_ratio_ours_over_published_median=float(np.median(t['photons'][i] / p['photons'][j])),
                crlb_ratio_ours_over_published_median=[float(np.median(t['lpx'][i] / p['lpx'][j])), float(np.median(t['lpy'][i] / p['lpy'][j]))],
                bg_ratio_ours_over_published_median=float(np.median(t['bg'][i] / p['bg'][j])),
                matched_ours=dict(photons=pct(t['photons'][i]), crlb_x_px=pct(t['lpx'][i]), crlb_y_px=pct(t['lpy'][i])),
                matched_published=dict(photons=pct(p['photons'][j]), crlb_x_px=pct(p['lpx'][j]), crlb_y_px=pct(p['lpy'][j])))


def dist(t): return dict(n=int(len(t['x'])), photons=pct(t['photons']), crlb_x_px=pct(t['lpx']), crlb_y_px=pct(t['lpy']),
                         crlb_x_nm=pct(t['lpx'] * PX), crlb_y_nm=pct(t['lpy'] * PX), bg_per_px_window=pct(t['bg']))


def parse_report():
    k = {}; ph = {}; dep = {}; pw = {}
    for ln in open(PUB + 'report.txt'):
        m = re.match(r'Pattern (\d): Phase\s+([-\d.]+) \(std=\s*([\d.]+)\) Depth=\s*([\d.]+) \(std=([\d.]+)\) Power=([\d.]+)', ln)
        if m: ph[int(m[1])] = float(m[2]); dep[int(m[1])] = float(m[4]); pw[int(m[1])] = float(m[6])
        m = re.match(r'Pattern (\d): kx=([-\d.]+) ky=([-\d.]+)', ln)
        if m: k[int(m[1])] = [float(m[2]), float(m[3])]
    s = ACQ
    return dict(k=[k[i] for i in s], phase_sin_deg=[ph[i] for i in s], depth_est=[dep[i] for i in s], power=[pw[i] for i in s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--suffix', default='')
    ap.add_argument('--log', default=None); ap.add_argument('--rmatch', type=float, default=1.0)
    ap.add_argument('--shift', type=float, nargs=2, default=None, help='px added to x, y (default: automatic, see docstring)')
    a = ap.parse_args()
    d = OUT + a.tag + '/'
    from smlmtorch.simflux.dataset import SFDataset
    from smlmtorch.simflux.pattern_estimator import ModulationPattern
    W = json.load(open(d + 'wrapper.json')); run = json.load(open(d + 'run.json'))
    si = json.load(open(d + 'smlm/info.json')); fi = json.load(open(d + 'sf/info.json'))
    roi = W['args']['roi']
    ds = SFDataset.load(d + 'smlm/smlm.npy'); f = h5(d + 'sf/sf.hdf5')
    mp_sf = ModulationPattern.load(d + 'smlm/mp_sf.pickle')
    resdir = d + 'work/' + run['run'] + '/results/sim4_1_reord/'
    mp_est = ModulationPattern.load(resdir + 'mp_smlm.pickle')          # saved before the depth is fixed
    me = mp_sf.mod_error(ds)
    S = dict(frame=ds.frame.astype(np.int64), x=ds.pos[:, 0].astype(float), y=ds.pos[:, 1].astype(float), photons=ds.photons.astype(float),
             bg=ds.background.astype(float), lpx=ds.crlb.pos[:, 0].astype(float), lpy=ds.crlb.pos[:, 1].astype(float))
    F = dict(frame=f['frame'].astype(np.int64), x=f['x'].astype(float), y=f['y'].astype(float), photons=f['photons'].astype(float),
             bg=f['bg'].astype(float) * 6, lpx=f['lpx'].astype(float), lpy=f['lpy'].astype(float))
    R = dict(tag=a.tag, wrapper=W['args'], settings=W['PAPER'], limits=dict(lo=W['limits_lo'], hi=W['limits_hi']))
    # cycle-aligned windows only
    al = {}
    for nm, t in (('smlm', S), ('simflux', F)):
        g = t['frame'] % 6 == 0; al[nm] = dict(n_all=int(len(g)), n_cycle_aligned=int(g.sum()))
        for k in t: t[k] = t[k][g]
        if nm == 'smlm': me_al = me[g]; chisq_al = ds.chisq[g]
        t['frame'] = t['frame'] // 6
    R['cycle_alignment'] = al
    # which SMLM rows are in the SIMFLUX table (same ROI: nearest SMLM row of the cycle within the 0.5 px distance filter)
    i, j = match(S, F, 0.5); in_sf = np.zeros(len(S['x']), bool); in_sf[i] = True
    # acceptance
    acc = dict(rois_detected=si['numrois'], smlm_after_limits_and_chisq=si['n_after_chisq'],
               chisq_threshold_per_px=4.0, chisq_per_px_median=si['chisq_per_px_median'],
               moderr_lt_0p1=fi['n_modsel'], simflux_out=fi['n_out'], simflux_dist_rejected=fi['lm'].get('n_dist_rejected'),
               simflux_at_limit=fi['lm'].get('atlim'), lm_smlm=si['lm'], lm_simflux=fi['lm'],
               moderr_percentiles_5_25_50_75_95=pct(me), moderr_lt=dict((str(v), float(np.mean(me < v))) for v in (0.012, 0.05, 0.1, 0.2)))
    if a.log and os.path.exists(a.log):
        txt = open(a.log).read()
        m = re.search(r'Remaining localizations: (\d+)/(\d+)', txt)
        if m: acc['smlm_after_nan_and_limits'] = int(m[1])
        m = re.search(r'Chi-sq filtered: (\d+)/(\d+)', txt)
        if m: acc['smlm_after_chisq'] = int(m[1]); acc['smlm_before_chisq'] = int(m[2])
    n0 = acc['rois_detected']
    acc['fractions'] = dict(limits=(acc.get('smlm_before_chisq', np.nan) / n0), chisq_of_previous=si['n_after_chisq'] / acc.get('smlm_before_chisq', np.nan),
                            moderr_of_previous=fi['n_modsel'] / si['n_after_chisq'], simflux_limits_dist_of_previous=fi['n_out'] / fi['n_modsel'],
                            smlm_of_detected=si['n_after_chisq'] / n0, simflux_of_detected=fi['n_out'] / n0, simflux_of_smlm=fi['n_out'] / si['n_after_chisq'])
    R['acceptance'] = acc
    # published
    P = {}
    for nm, fn in (('smlm', 'g2d-nodriftc.hdf5'), ('simflux', 'simflux-nodriftc.hdf5')):
        p = h5(PUB + fn); P[nm] = dict(frame=p['frame'].astype(np.int64), x=p['x'].astype(float), y=p['y'].astype(float),
                                       photons=p['photons'].astype(float), bg=p['bg'].astype(float), lpx=p['lpx'].astype(float), lpy=p['lpy'].astype(float))
    R['vs_published'] = dict(smlm=cmp_pub(S, P['smlm'], a.rmatch), simflux=cmp_pub(F, P['simflux'], a.rmatch),
                             published_simflux_vs_published_smlm=cmp_pub(P['simflux'], P['smlm'], a.rmatch),
                             ours_simflux_vs_ours_smlm=cmp_pub(F, S, a.rmatch))
    R['distributions_percentiles_5_25_50_75_95'] = dict(ours_smlm=dist(S), ours_simflux=dist(F), published_smlm=dist(P['smlm']), published_simflux=dist(P['simflux']))
    R['per_cycle'] = dict(ours_smlm=len(S['x']) / 5000, ours_simflux=len(F['x']) / 5000, published_smlm=len(P['smlm']['x']) / 5000,
                          published_simflux=len(P['simflux']['x']) / 5000, n_cycles_with_locs_smlm=int(len(np.unique(S['frame']))))
    # half-pixel convention
    off = np.array(R['vs_published']['smlm']['median_offset_px'])
    if a.shift is not None: sh = np.array(a.shift, float); why = 'given'
    else:
        sh = np.where(np.abs(np.abs(off) - 0.5) < 0.1, -np.sign(off) * 0.5, 0.0); why = 'automatic (0.5 px applied only if |measured offset| is within 0.1 px of 0.5 px)'
    R['coordinate_convention'] = dict(measured_offset_ours_minus_published_px=off.tolist(), shift_applied_px=sh.tolist(), rule=why)
    for nm, t, extra in (('smlm', S, dict(chisq=chisq_al, moderr=me_al, in_sf=in_sf)), ('simflux', F, {})):
        L.save_table(OUT + f'tab_bl_{nm}{a.suffix}.npz', frame=t['frame'], x_nm=(t['x'] + sh[0]) * PX, y_nm=(t['y'] + sh[1]) * PX,
                     sx_nm=t['lpx'] * PX, sy_nm=t['lpy'] * PX, photons=t['photons'], crlb_x_nm=t['lpx'] * PX, crlb_y_nm=t['lpy'] * PX,
                     bg=t['bg'], **extra)
    # pattern
    C = json.load(open(CALIB))['pattern']; rep = parse_report()
    def pat(mp):
        m = mp.mod; k = m['k'][:, :2].astype(float)
        return dict(k_rad_per_px=k.tolist(), pitch_px=(2 * np.pi / np.linalg.norm(k, axis=1)).tolist(),
                    pitch_nm=(2 * np.pi / np.linalg.norm(k, axis=1) * PX).tolist(), angle_deg=np.rad2deg(np.arctan2(k[:, 1], k[:, 0])).tolist(),
                    phase_sin_rad=m['phase'].astype(float).tolist(), phase_cos_rad=wrap(m['phase'].astype(float) + np.pi / 2).tolist(),
                    depth=m['depth'].astype(float).tolist(), relint_sum1=m['relint'].astype(float).tolist(),
                    relint_mean1=(m['relint'].astype(float) * 6).tolist())
    pe_, pu = pat(mp_est), pat(mp_sf)
    po = np.asarray(mp_sf.phase_offset_per_frame, float)
    np.save(OUT + f'pattern_bl{a.suffix}_phase_offset.npy', po)
    kc = np.repeat(np.array(C['k_rad_per_px'], float), 3, axis=0)
    steps = lambda ph: [np.rad2deg(wrap(np.diff(np.array(ph)[o * 3:o * 3 + 3]))).tolist() for o in (0, 1)]
    dphi_cal = np.rad2deg(wrap(np.array(pe_['phase_cos_rad']) - np.array(C['phase'])))
    dphi_rep = wrap(np.deg2rad(np.rad2deg(np.array(pe_['phase_sin_rad'])) - np.array(rep['phase_sin_deg']))) * 180 / np.pi
    pj = dict(
        note='convention I_k = relint_k (1 + depth_k sin(k . r - phase_sin_k)), r in px (pixel centres at integers, x = column), '
             'reordered step k = frame % 6, orientation = k // 3; phase_cos = phase_sin + pi/2 (calib json convention). '
             'phase used at window start frame f, step k: phase_sin_k - phase_offset[f + k] (pattern_bl_phase_offset.npy; mean 0 per step; '
             'the SIMFLUX pattern has one common drift per orientation = mp_smlm.const_phase_offsets()). Coordinates WITHOUT the table shift.',
        estimated=pe_, used_for_simflux=pu, fix_depth=W['PAPER']['fix_depth'],
        phase_offset_file=f'pattern_bl{a.suffix}_phase_offset.npy',
        phase_offset_range_deg_per_orientation=[[float(np.rad2deg(po[o * 3::6].min())), float(np.rad2deg(po[o * 3::6].max()))] for o in (0, 1)],
        phase_steps_deg=dict(ours=steps(pe_['phase_sin_rad']), calib=steps(C['phase']), report=steps(np.deg2rad(rep['phase_sin_deg']))),
        vs_calib=dict(k_calib=kc.tolist(), dk_rad_per_px=(np.array(pe_['k_rad_per_px']) - kc).tolist(),
                      pitch_nm_calib=C['pitch_nm'], dphase_cos_deg=dphi_cal.tolist(), depth_calib=C['depth'],
                      depth_calib_regression_diluted=C.get('depth_regression_diluted'), relint_calib=C['relint']),
        vs_report=dict(k_report=rep['k'], pitch_nm_report=(2 * np.pi / np.linalg.norm(np.array(rep['k']), axis=1) * PX).tolist(),
                       angle_deg_report=np.rad2deg(np.arctan2(np.array(rep['k'])[:, 1], np.array(rep['k'])[:, 0])).tolist(),
                       phase_sin_deg_report=rep['phase_sin_deg'], dphase_deg=dphi_rep.tolist(), depth_report_est=rep['depth_est'],
                       depth_report_used=0.85, power_report=rep['power']))
    json.dump(pj, open(OUT + f'pattern_bl{a.suffix}.json', 'w'), indent=1)
    R['pattern'] = pj; R['timing_s'] = run['timing']; R['wall_s'] = W.get('wall_s')
    json.dump(R, open(OUT + f'sanity_bl{a.suffix}.json', 'w'), indent=1)
    # text
    o = []; p = o.append
    p(f'run {a.tag}: ROI {roi}, threshold {W["args"]["thr"]}, det sigma {W["args"]["det_sigma"]}, fix_depth {W["PAPER"]["fix_depth"]}, bg_min {W["args"]["bg_min"]}, psf {os.path.basename(W["args"]["psf"])}')
    p(f'cycle alignment: {al}')
    f_ = acc['fractions']
    p(f'acceptance: detected ROIs {n0}; after NaN/limits {acc.get("smlm_before_chisq")} ({f_["limits"]:.4f}); chi2 <= 4/px {si["n_after_chisq"]} ({f_["chisq_of_previous"]:.4f} of previous; median chi2/px {si["chisq_per_px_median"]:.2f}) = SMLM table;')
    p(f'            mod error < 0.1 {fi["n_modsel"]} ({f_["moderr_of_previous"]:.4f}); SIMFLUX limits + |SF - SMLM| < 0.5 px {fi["n_out"]} ({f_["simflux_limits_dist_of_previous"]:.4f}; distance-rejected {fi["lm"].get("n_dist_rejected")}) = SIMFLUX table ({f_["simflux_of_smlm"]:.4f} of SMLM)')
    p(f'            mod error percentiles 5/25/50/75/95: {np.round(acc["moderr_percentiles_5_25_50_75_95"], 4).tolist()}; fraction below thresholds {acc["moderr_lt"]}')
    for nm in ('smlm', 'simflux'):
        c = R['vs_published'][nm]
        p(f'{nm}: ours {c["n_ours"]} vs published {c["n_published"]}; matched {c["n_matched"]} ({c["frac_of_ours_matched"]:.3f} of ours, {c["frac_of_published_matched"]:.3f} of published)')
        p(f'   ours - published: median {c["median_offset_px"][0]:+.4f} {c["median_offset_px"][1]:+.4f} px ({c["median_offset_nm"][0]:+.2f} {c["median_offset_nm"][1]:+.2f} nm), robust sd {c["rsd_px"][0]:.4f} {c["rsd_px"][1]:.4f} px ({c["rsd_nm"][0]:.2f} {c["rsd_nm"][1]:.2f} nm)')
        p(f'   per 1000 cycles median offset px: {np.round(c["median_offset_px_per_1000_cycles"], 3).tolist()}')
        p(f'   matched pairs: photon ratio {c["photon_ratio_ours_over_published_median"]:.3f}, CRLB ratio {c["crlb_ratio_ours_over_published_median"][0]:.3f} {c["crlb_ratio_ours_over_published_median"][1]:.3f}, bg ratio {c["bg_ratio_ours_over_published_median"]:.3f}')
    for k in ('published_simflux_vs_published_smlm', 'ours_simflux_vs_ours_smlm'):
        c = R['vs_published'][k]
        p(f'{k}: matched {c["n_matched"]}, median {c["median_offset_px"][0]:+.4f} {c["median_offset_px"][1]:+.4f} px, robust sd {c["rsd_px"][0]:.4f} {c["rsd_px"][1]:.4f} px')
    p('distributions (percentiles 5/25/50/75/95):')
    for k, v in R['distributions_percentiles_5_25_50_75_95'].items():
        p(f'   {k:18s} n {v["n"]:6d} photons {np.round(v["photons"]).astype(int).tolist()} crlb x px {np.round(v["crlb_x_px"], 4).tolist()} y {np.round(v["crlb_y_px"], 4).tolist()} bg/px/window {np.round(v["bg_per_px_window"], 2).tolist()}')
    p(f'coordinate convention: {R["coordinate_convention"]}')
    p('pattern (estimated; reordered steps 0-5):')
    p(f'   k rad/px {np.round(pe_["k_rad_per_px"], 4).tolist()}')
    p(f'   pitch nm ours {np.round(pe_["pitch_nm"][::3], 2).tolist()} calib {np.round(C["pitch_nm"], 2).tolist()} report {np.round(pj["vs_report"]["pitch_nm_report"][::3], 2).tolist()}')
    p(f'   angle deg ours {np.round(pe_["angle_deg"][::3], 3).tolist()} calib {np.round(np.rad2deg(np.arctan2(kc[::3, 1], kc[::3, 0])), 3).tolist()} report {np.round(pj["vs_report"]["angle_deg_report"][::3], 3).tolist()}')
    p(f'   phase (sin conv) deg ours {np.round(np.rad2deg(pe_["phase_sin_rad"]), 2).tolist()} report {rep["phase_sin_deg"]}; ours - report {np.round(dphi_rep, 2).tolist()}')
    p(f'   phase (cos conv) deg ours {np.round(np.rad2deg(pe_["phase_cos_rad"]), 2).tolist()} calib {np.round(np.rad2deg(wrap(C["phase"])), 2).tolist()}; ours - calib {np.round(dphi_cal, 2).tolist()}')
    p(f'   phase steps deg (within orientation) {pj["phase_steps_deg"]}')
    p(f'   depth estimated {np.round(pe_["depth"], 3).tolist()} (used: {pu["depth"][0]:.2f}); calib {C["depth"]}; report est {rep["depth_est"]} (used 0.85)')
    p(f'   relint (mean 1) ours {np.round(pe_["relint_mean1"], 3).tolist()} calib {np.round(C["relint"], 3).tolist()} report power x 6 {np.round(np.array(rep["power"]) * 6, 3).tolist()}')
    p(f'   phase-offset (drift) range deg per orientation {np.round(pj["phase_offset_range_deg_per_orientation"], 1).tolist()}')
    p(f'timing s: {run["timing"]}')
    txt = '\n'.join(o); open(OUT + f'sanity_bl{a.suffix}.txt', 'w').write(txt + '\n'); print(txt)


if __name__ == '__main__':
    main()
