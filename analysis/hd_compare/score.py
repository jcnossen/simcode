"""High-density sim comparison (2026-11-10): SMLM / SF (classical, run_classic.py) vs SIMCODE (paper L64_5, sign fixed,
sfhd-ndi+smlm merged output; + its unfiltered network) vs ours pass 1 (det_n1 + sc5nb + pattern solve) vs ours pass 2 (p2L),
on the same sim movie, scored against the GT with the protocol of simcode_base2/compare.py (imported):
  W3 windows (6-frame windows starting at frame % 3 == 0; the stride-1 methods are evaluated on those windows only),
  GT targets (event, window) inside the crop with the photon time-centroid in [1, 4), 1 px greedy one-to-one matching per
  window on the detection point (SMLM: its fit; SF: the SMLM fit of its row; SIMCODE: network xy; ours: x_prior), global
  offset (median error, 400-2000 label ph) removed, CRLB_ref = gate rs25pm CRLB_SF_multi at the target's photons.
  All outputs are restricted to the crop (5..115 px) on the detection point first (ours were run with that crop).
Extra metrics here:
  recall at matched FP: each method ranked by its own GT-free confidence (ours: detection p; others: -reported sigma);
     recall of the largest prefix with FP fraction <= 5 % / 1 % (FP = unmatched outputs).
  merged (as n20/merged_gross.py): targets >= 200 label ph whose nearest GT neighbour (>= 50 window ph) in the same window
     is at d <= 1 px: matched / merged (unmatched but the neighbour matched) / undetected.
  counts: outputs per W3 window vs GT emitters on per frame / per window, inside the crop.
  FRC: common window-start grid for all methods (critic 35 #1): grid A = starts % 6 == 0 (disjoint windows), check grid B =
     starts % 6 == 3; halves = alternating windows of the grid, 5 nm pixels, 1/7 threshold, each method subsampled at random
     to a COMMON N (several N levels; 5 draws) and top-N by own sigma, crop region.
2026-11-10 addendum (critic 35 #6): methods 'SMLM tuned' (run_smlm_tuned.py: ROI 6, detection spacing 4, chi^2/px <= 2),
  'DECODE' (decode/: trained on DECODE's simulator with the sim's vectorial PSF; 6-frame sums starting at frame 0 and 3,
  ranked by DECODE prob) and 'DECODE per-frame' (per-frame DECODE, linked into W3 windows by link_windows); the classical
  columns are relabelled 'SMLM (paper pipeline)' / 'SF (paper pipeline)'. Pre-addendum version: scratchpad
  score_backup_pre_decode.py.  same protocol and code; the movie, the method outputs
and the output dir come from a json spec instead of hard-coded paths:
  {"name": "m25qh", "sim": "<sim dir with sim_gt.npz>", "gate_scene": "rs25pm" (default rs05pm for m05*, else rs25pm),
   "out": "<dir>" (default <simcode out>/hd_compare),
   "methods": [["SMLM (paper pipeline)", "smlm", "<classic_<m>.npz>"], ["SMLM tuned", "smlmt:2.0", "<smlmt_<m>_r6_m4_dt3.5.npz>"],
               ["SF (paper pipeline)", "sf", "<classic_<m>.npz>"], ["DECODE", "decode_sum6", "<m>_hd_sum6.npz"],
               ["DECODE per-frame", "decode_frame", "<m>_hd_frame.npz"], ["SIMCODE", "sc_mrg", "<m>_L64_5c3_paper_fix.npz"],
               ["SIMCODE-net", "sc_net", "<same npz>"], ["ours p1", "ours_est", "<pass-1 run dir>"], ["ours p2", "ours_p2L", "<p2L dir>"]]}
  (methods whose path does not exist are skipped). Inputs: the movie from analysis/simreal/recipes.sh m25qh | m20qh; the
  classical inputs from analysis/baselines/run_classic.py <sim> <work> classic_<m>.npz (SMLM + SF), the tuned SMLM from
  analysis/baselines/run_smlm_tuned.py <sim> <work> smlmt_<m>_r6_m4_dt3.5.npz 6 4; DECODE from analysis/baselines/decode;
  SIMCODE from analysis/baselines/run_simcode.py --model L64_5 --cam paper --fix_sign 1. Renders: analysis/hd_compare/render.py
  <spec>. crlb_ref: see analysis/baselines/compare.py (SIMCODE_GATE_DIR).
usage: python analysis/hd_compare/score.py <spec.json>   -> <out>/score_<name>.txt / .json
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
import compare as C       # analysis/baselines/compare.py

PX = 108.33
SPEC = json.load(open(sys.argv[1]))
nm = SPEC['name']
sd = SPEC['sim']
X0, X1 = 5.0, 115.0
METHODS = [tuple(m) for m in SPEC['methods']]
if SPEC.get('out'):
    HC = SPEC['out']; os.makedirs(HC, exist_ok=True)
else:
    from simcode import config
    HC = config.out_dir('hd_compare')
LINK_R = 1.0     # DECODE per-frame -> window linking radius (px)


def link_windows(R, NF):
    """DECODE per-frame localizations -> one output per (track, W3 window): mutual-nearest-neighbour linking of consecutive
    frames within LINK_R px (chains with <= 1 loc per frame), then each chain's locs inside window [w, w+5] are merged with
    inverse-variance weights (sigma_comb = 1/sqrt(sum 1/sigma^2), photons summed, prob = max)."""
    from scipy.spatial import cKDTree
    f = R['frame'].astype(int); o = np.argsort(f, kind='stable')
    f, x, y = f[o], R['x'][o], R['y'][o]; sx, sy, ph, pr = R['sx'][o], R['sy'][o], R['phot'][o], R['prob'][o]
    st = np.searchsorted(f, np.arange(NF + 1))
    nxt = np.full(len(f), -1)
    for t in range(NF - 1):
        a, b = slice(st[t], st[t + 1]), slice(st[t + 1], st[t + 2])
        if st[t + 1] == st[t] or st[t + 2] == st[t + 1]: continue
        A = np.c_[x[a], y[a]]; B = np.c_[x[b], y[b]]
        dab, jab = cKDTree(B).query(A, distance_upper_bound=LINK_R); dba, jba = cKDTree(A).query(B, distance_upper_bound=LINK_R)
        ok = np.isfinite(dab); i = np.nonzero(ok)[0]; j = jab[ok]
        mut = jba[j] == i
        nxt[st[t] + i[mut]] = st[t + 1] + j[mut]
    has_prev = np.zeros(len(f), bool); has_prev[nxt[nxt >= 0]] = True
    chain = np.full(len(f), -1); c = 0
    for s0 in np.nonzero(~has_prev)[0]:
        k = s0
        while k >= 0:
            chain[k] = c; k = nxt[k]
        c += 1
    # each frame belongs to the W3 windows starting at w1 = f - f % 3 and w1 - 3
    rows = np.r_[np.arange(len(f)), np.arange(len(f))]
    w = np.r_[f - f % 3, f - f % 3 - 3]
    ok = (w >= 0) & (w <= NF - 6); rows, w = rows[ok], w[ok]
    key = chain[rows].astype(np.int64) * (NF + 8) + w
    u, inv = np.unique(key, return_inverse=True)
    wx = 1 / np.maximum(sx[rows], 1e-4) ** 2; wy = 1 / np.maximum(sy[rows], 1e-4) ** 2
    Sx = np.bincount(inv, wx); Sy = np.bincount(inv, wy)
    X = np.bincount(inv, wx * x[rows]) / Sx; Y = np.bincount(inv, wy * y[rows]) / Sy
    P = np.zeros(len(u)); np.maximum.at(P, inv, pr[rows])
    return dict(w=(u % (NF + 8)).astype(int), xd=X, yd=Y, x=X, y=Y, sx=1 / np.sqrt(Sx), sy=1 / np.sqrt(Sy),
                phot=np.bincount(inv, ph[rows]), prob=P, nfr=np.bincount(inv))


def load(kind, path, NF):
    if kind in ('smlm', 'sf') or kind.startswith('smlmt'):
        R = np.load(path); fr = R['frame'].astype(int)
        if kind == 'smlm' or kind.startswith('smlmt'):
            k = np.ones(len(fr), bool)
            if kind.startswith('smlmt:') and kind.split(':')[1] != 'none': k = R['chisq_px'] <= float(kind.split(':')[1])
            M = dict(w=fr[k], xd=R['x'][k], yd=R['y'][k], x=R['x'][k], y=R['y'][k], sx=R['sx'][k], sy=R['sy'][k], phot=R['photons'][k])
        else:
            r = R['sf_idx']
            M = dict(w=fr[r], xd=R['x'][r], yd=R['y'][r], x=R['sf_x'], y=R['sf_y'], sx=R['sf_sx'], sy=R['sf_sy'], phot=R['sf_photons'])
        M['p'] = -np.hypot(M['sx'], M['sy'])
    elif kind == 'decode_sum6':
        # DECODE on non-overlapping 6-frame sums starting at frame 0 and 3: frame = window start; ranked by DECODE's
        # probability, ties (prob saturates at 1) broken by its sigma
        R = np.load(path)
        M = dict(w=R['frame'].astype(int), xd=R['x'], yd=R['y'], x=R['x'], y=R['y'], sx=R['sx'], sy=R['sy'], phot=R['phot'])
        M['p'] = R['prob'] - 1e-3 * np.hypot(R['sx'], R['sy'])
    elif kind == 'decode_frame':
        R = dict(np.load(path)); M = link_windows(R, NF)
        M['p'] = -np.hypot(M['sx'], M['sy']); M.pop('prob'); M.pop('nfr')
    else:
        M = C.load_method(kind, path, NF)
        if 'p' not in M: M['p'] = -np.hypot(M['sx'], M['sy'])
    ok = (M['xd'] >= X0) & (M['xd'] < X1) & (M['yd'] >= X0) & (M['yd'] < X1) & np.isfinite(M['x']) & np.isfinite(M['y'])
    return {k: v[ok] for k, v in M.items()}


def frc_res(x, y, half, pix_nm=5.0):
    from simcode.evaluation.frc import frc_fov
    return frc_fov(x, y, half, X0, X1, PX, pix_nm)


def main():
    G = dict(np.load(f'{sd}/sim_gt.npz')); NF = int(G['nf'])
    GW = C.gt_windows(G, np.arange(0, NF - 5, 3))
    tg = GW[:, 5] > 0; P = GW[:, 4]
    crf, cinfo = C.crlb_ref(SPEC.get('gate_scene', 'rs05pm' if nm.startswith('m05') else 'rs25pm'))
    REP = dict(movie=nm, sim=sd, crlb_ref=cinfo)
    # ---- density / photons of the GT
    inc = (G['x'] >= X0) & (G['x'] < X1) & (G['y'] >= X0) & (G['y'] < X1)
    t0, t1 = G['t0'][inc], G['t1'][inc]
    on_pf = np.array([np.sum((t0 < f + 1) & (t1 > f)) for f in range(0, NF, 7)])
    nwin = len(np.unique(GW[:, 0]))
    on_pw = np.bincount((GW[:, 0] // 3).astype(int), (GW[:, 2] >= X0) & (GW[:, 2] < X1) & (GW[:, 3] >= X0) & (GW[:, 3] < X1))
    tpw = np.bincount((GW[:, 0] // 3).astype(int), tg)
    Pt = P[tg]
    REP['gt'] = dict(on_per_frame=float(on_pf.mean()), on_per_window=float(on_pw[on_pw > 0].mean()), targets_per_window=float(tpw[tpw > 0].mean()),
                     on_per_frame_per_um2=float(on_pf.mean() / ((X1 - X0) * PX / 1000) ** 2),
                     target_phot_q=[float(v) for v in np.percentile(Pt, [10, 25, 50, 75, 90])], n_windows=int(nwin), n_targets=int(tg.sum()))
    # nearest visible GT neighbour per window (for 'merged')
    nnd = np.full(len(GW), 99.0); nnj = np.full(len(GW), -1)
    o = np.argsort(GW[:, 0], kind='stable'); ws = GW[o, 0]; cut = np.flatnonzero(np.diff(ws)) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(ws)]):
        ii = o[a:b]
        if len(ii) < 2: continue
        D = np.hypot(GW[ii, 2][:, None] - GW[ii, 2][None], GW[ii, 3][:, None] - GW[ii, 3][None]); np.fill_diagonal(D, 1e9)
        D[:, P[ii] < 50] = 1e9
        j = D.argmin(1); nnd[ii] = D[np.arange(len(ii)), j]; nnj[ii] = ii[j]
    kappa = None
    Ms = {}; ROWS = {}; FRC_SETS = {}
    import os
    METHODS[:] = [m for m in METHODS if os.path.exists(m[2])]
    for name, kind, path in METHODS:
        M = load(kind, path, NF); Ms[name] = M
        r = C.evaluate(G, M, GW, crf, 1.45, kind not in ('ours_est', 'ours_p2L', 'decode_sum6', 'decode_frame'))
        R_ = r.pop('_rows'); ROWS[name] = R_
        # matched FP recall levels
        od = np.argsort(-R_['det_p'], kind='stable')
        ctp = np.cumsum(R_['det_tp'][od]); cfp = np.cumsum(R_['det_fp'][od]); fpf = cfp / np.maximum(ctp + cfp, 1)
        r['recall_at_fpfrac'] = {}
        for lev in (0.05, 0.01):
            ok = np.nonzero(fpf <= lev)[0]
            r['recall_at_fpfrac'][str(lev)] = float(ctp[ok[-1]] / R_['n_t200']) if len(ok) else 0.0
        # merged
        w3 = (M['w'] % 3 == 0) & (M['w'] <= NF - 6)
        gi = C.match(GW, M['w'][w3], M['xd'][w3], M['yd'][w3])
        rec = np.zeros(len(GW), bool); rec[gi[gi >= 0]] = True
        mg = {}
        for lo, hi in ((0, 0.5), (0.5, 1.0), (0, 1.0), (1.0, 1.5), (1.5, 99)):
            t = tg & (P >= 200) & (nnd >= lo) & (nnd < hi); nb = nnj[t]
            mg[f'{lo}-{hi}'] = dict(n=int(t.sum()), matched=float(rec[t].mean()), merged=float((~rec[t] & rec[nb]).mean()),
                                    undet=float((~rec[t] & ~rec[nb]).mean()))
        r['merged'] = mg
        # gross > 50 nm on own matched targets
        er = np.hypot(*R_['e'].T) * PX
        r['gross50'] = {f'{lo:.0f}-{hi:.0f}': float(100 * np.mean(er[(R_['P'] >= lo) & (R_['P'] < hi)] > 50)) if ((R_['P'] >= lo) & (R_['P'] < hi)).sum() else np.nan
                        for lo, hi in ((200, 700), (700, 2000), (2000, 4000), (4000, 1e9), (200, 2000), (200, 1e9))}
        r['outputs_per_W3_window'] = float(w3.sum() / nwin)
        # common window-start grid for every method (critic 35 #1): grid A = starts % 6 == 0 (disjoint windows), grid B =
        # starts % 6 == 3; halves = alternating windows of the grid
        FRC_SETS[name] = {}
        for gname, g0 in (('A', 0), ('B', 3)):
            w6 = (M['w'] % 6 == g0) & (M['w'] <= NF - 6)
            FRC_SETS[name][gname] = (M['x'][w6] - r['offset_px'][0], M['y'][w6] - r['offset_px'][1], (((M['w'][w6] - g0) // 6) % 2) == 0,
                                     np.hypot(M['sx'][w6], M['sy'][w6]))
        REP[name] = r
        print(name, 'recall', round(r['det']['recall'], 3), 'FP/win', round(r['det']['FP_per_window'], 2), 'out/win', round(r['outputs_per_W3_window'], 1), flush=True)
    # ---- paired vs ours p2 (targets matched by both)
    A = ROWS.get('ours p2')
    for name in Ms:
        if name == 'ours p2' or A is None: continue
        B_ = ROWS[name]; com, ia, ib = np.intersect1d(A['g'], B_['g'], return_indices=True)
        REP[name]['paired_vs_p2'] = {}
        for lo, hi in C.HBINS + [(200, 1e9)]:
            m = (A['P'][ia] >= lo) & (A['P'][ia] < hi)
            REP[name]['paired_vs_p2'][f'{lo:.0f}-{hi:.0f}'] = dict(n=int(m.sum()), method=C._rc(B_, ib[m])[0], p2=C._rc(A, ia[m])[0])
    # ---- FRC at common N on a common window-start grid (grid A; grid B as a check), random subsampling (5 seeds) and top-N
    names = [n for n, _, _ in METHODS]
    cnt = {n: len(FRC_SETS[n]['A'][0]) for n in names}
    cntB = {n: len(FRC_SETS[n]['B'][0]) for n in names}
    levels = sorted(set([cnt[n] for n in names if cnt[n] >= 3000] + [min(c for c in cnt.values() if c >= 3000)]))
    REP['frc'] = dict(n_available=cnt, n_available_gridB=cntB, levels={}, levels_gridB={}, levels_topN={}, grid='A: window starts % 6 == 0, B: % 6 == 3')
    rng = np.random.default_rng(0)
    for N in levels:
        row = {}; rowB = {}; rowt = {}
        for n in names:
            for gname, dst in (('A', row), ('B', rowB)):
                x, y, h, sg = FRC_SETS[n][gname]
                if len(x) < N: dst[n] = None; continue
                v = []
                for rep in range(5 if len(x) > N else 1):
                    s = rng.choice(len(x), N, replace=False) if len(x) > N else np.arange(len(x))
                    v.append(frc_res(x[s], y[s], h[s]))
                dst[n] = [float(np.nanmean(v)), float(np.nanstd(v))]
            x, y, h, sg = FRC_SETS[n]['A']
            if len(x) < N: rowt[n] = None; continue
            s = np.argsort(sg, kind='stable')[:N]            # top-N by the method's own reported sigma
            rowt[n] = [frc_res(x[s], y[s], h[s]), 0.0]
        REP['frc']['levels'][str(N)] = row; REP['frc']['levels_gridB'][str(N)] = rowB; REP['frc']['levels_topN'][str(N)] = rowt
        print('FRC N', N, 'random A', {k: (round(v[0], 1) if v else None) for k, v in row.items()}, 'B', {k: (round(v[0], 1) if v else None) for k, v in rowB.items()},
              'topN', {k: (round(v[0], 1) if v else None) for k, v in rowt.items()}, flush=True)
    json.dump(REP, open(f'{HC}/score_{nm}.json', 'w'), indent=1)
    write_txt(REP, names)


def write_txt(REP, names):
    g = REP['gt']
    L = [f'# hd_compare {nm}: {REP["sim"]}',
         f'GT (crop {X0:.0f}-{X1:.0f} px): emitters on per frame {g["on_per_frame"]:.1f} ({g["on_per_frame_per_um2"]:.3f} /um2), on per 6-frame window '
         f'{g["on_per_window"]:.1f}, targets per W3 window {g["targets_per_window"]:.1f}; target window photons (label) p10/25/50/75/90 '
         + '/'.join(f'{v:.0f}' for v in g['target_phot_q']) + f' (SIMCODE-net units ~ /1.45)']
    f = lambda v, d=2: '-' if v is None or not np.isfinite(v) else f'{v:.{d}f}'
    L.append('\n== detection (targets >= 200 label ph; W3 windows)')
    L.append(f'{"method":22s} {"out/win":>7s} {"recall":>6s} {"rec_any":>7s} {"FP/win":>7s} {"FPfrac":>6s} {"JI":>5s} | recall @ FPfrac<=5% / 1% | offset nm')
    for n in names:
        d = REP[n]['det']; q = REP[n]['recall_at_fpfrac']
        L.append(f'{n:22s} {REP[n]["outputs_per_W3_window"]:7.1f} {d["recall"]:6.3f} {d["recall_any"]:7.3f} {d["FP_per_window"]:7.2f} {d["FP_frac"]:6.3f} {d["JI"]:5.3f} | '
                 f'{q["0.05"]:.3f} / {q["0.01"]:.3f} | {np.array(REP[n]["offset_px"]) * PX}')
    L.append('\n== localization error, own matched targets, label-photon bins: n | recall | RMS nm | RMS/CRLB_ref | trim/C | gross>50nm % | zcal')
    for lo, hi in C.HBINS + [(200, 1e9)]:
        b = f'{lo:.0f}-{hi:.0f}'
        for n in names:
            q = REP[n]['label'].get(b, {})
            if 'n' not in q:
                L.append(f'  {b:>12s} {n:22s} n {"<15":>6s} rec {f(q.get("recall"), 3)}'); continue
            L.append(f'  {b:>12s} {n:22s} n {q["n"]:6d} rec {q["recall"]:.3f} rms {q["rms_nm"]:6.2f} nm  rms/C {q["rms_over_crlb"]:5.2f} trim/C {q["trim_over_crlb"]:5.2f} '
                     f'gross50 {q["gross50_pct"]:5.2f}% zcal {q["zcal"]:.2f}')
        L.append('')
    L.append('== paired with ours p2 (targets matched by both): n | method RMS/C | p2 RMS/C')
    for n in names:
        if 'paired_vs_p2' not in REP[n]: continue
        L.append(f'  {n:22s} ' + ' | '.join(f'{b} n {v["n"]} {f(v["method"])} vs {f(v["p2"])}' for b, v in REP[n]['paired_vs_p2'].items()))
    L.append('\n== nearest-neighbour bins (targets >= 200 ph): matched / merged / undetected')
    for b in REP[names[0]]['merged']:
        L.append(f'  d {b:>8s} n {REP[names[0]]["merged"][b]["n"]:6d} ' + ' | '.join(
            f'{n} {REP[n]["merged"][b]["matched"]:.2f}/{REP[n]["merged"][b]["merged"]:.2f}/{REP[n]["merged"][b]["undet"]:.2f}' for n in names))
    L.append('\n== FRC (nm), common window-start grid A (starts % 6 == 0, disjoint windows), alternating-window halves, common N (mean +- sd over 5 draws); available N: ' +
             ', '.join(f'{n} {c}' for n, c in REP['frc']['n_available'].items()))
    for N, row in REP['frc']['levels'].items():
        L.append(f'  random N {int(N):7d}: ' + ' | '.join(f'{n} ' + (f'{v[0]:.1f}+-{v[1]:.1f}' if v else '-') for n, v in row.items()))
    L.append('  (grid B, window starts % 6 == 3: random subsampling, 5 draws)')
    for N, row in REP['frc']['levels_gridB'].items():
        L.append(f'  random N {int(N):7d}: ' + ' | '.join(f'{n} ' + (f'{v[0]:.1f}+-{v[1]:.1f}' if v else '-') for n, v in row.items()))
    L.append('  (top-N by own reported sigma, grid A)')
    for N, row in REP['frc']['levels_topN'].items():
        L.append(f'  top    N {int(N):7d}: ' + ' | '.join(f'{n} ' + (f'{v[0]:.1f}' if v else '-') for n, v in row.items()))
    open(f'{HC}/score_{nm}.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
