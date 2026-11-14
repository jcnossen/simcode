"""Paired gate comparison: targets matched (1 px on x_det, score >= 0.5) by BOTH SIMCODE (tag A) and ours (tag B); RMS / CRLB_SF_multi,
trimmed RMS (|e| / CRLB <= 5), zmed, zcal, gross |e|/s > 4 per photon bin.
--offset (critic 32 #2): each method's own global (x, y) offset (simcode.eff.simcode_gate.fit_offset: median est - GT over its
matched targets 400-2000 ph) is fitted per scene / seed and removed before the statistics (the offsets are printed). 
same code on simcode.hd.gate_report.match / simcode.eff.simcode_gate.fit_offset; the gate npz directory is
$SIMCODE_GATE_DIR (default <simcode out>/gate). SIMCODE gate npz: simcode/eff/simcode_gate.py --model ni6_3 | L64_5
(tags _scode_<model> / _scodenf_<model>); ours: simcode/hd/gate.py (analysis/gate/run_gate.sh; dev tag _hd_sc5nbb15pmb10).
usage: python analysis/gate/gate_paired.py tagA tagB scen1,scen2 [--offset]   (RESULTS: _scode_ni6_3 _hd_sc5nbb15pmb10
       iso,pair2,rs05qm --offset > gate_paired_scode_ni6_3_vs_b10.txt)"""
import sys, glob, os, re, numpy as np
from simcode.hd.gate_report import match
from simcode.eff.simcode_gate import fit_offset
D = os.environ.get('SIMCODE_GATE_DIR')
if not D:
    from simcode import config
    D = config.out_dir('gate', make=False)
tA, tB = sys.argv[1], sys.argv[2]
OFF = '--offset' in sys.argv
BINS = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 1e9), (200, 2000)]


def seeds(sc, tag):
    pat = re.compile(re.escape(sc) + r'__s(\d+)' + re.escape(tag) + r'\.npz$')
    return sorted(int(m.group(1)) for f in glob.glob(f'{D}/{sc}__s*.npz') if (m := pat.match(os.path.basename(f))))


def rows(R):
    idx = np.nonzero(R['sc_score'] >= 0.5)[0]
    og = match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], R['sc_x_det'][idx], R['sc_y_det'][idx], 1.0)
    ti, oi = og[og >= 0], idx[og >= 0]; k = R['gt_target'][ti] > 0
    return dict(zip(ti[k], oi[k]))


def st(R, ti, oi):
    e = np.c_[R['sc_x_sc'][oi] - R['gt_x'][ti], R['sc_y_sc'][oi] - R['gt_y'][ti]]; c = R['gt_sf_m'][ti]; s = R['sc_scal_sc'][oi]
    z = np.hypot(*(e / c).T)
    return (np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(c ** 2)), np.sqrt(np.mean(e[z <= 5] ** 2)) / np.sqrt(np.mean(c[z <= 5] ** 2)),
            1.4826 * np.median(np.abs(e / c)), 1.4826 * np.median(np.abs(e / s)), 100 * np.mean(np.hypot(*(e / s).T) > 4))


print("(targets with CRLB_SF_multi < 25 nm, as gate_report_xdet)" + (' ; per-method global offset removed per scene/seed' if OFF else ''))
print(f'paired {tA} (SIMCODE) vs {tB} (ours): rms / trim / zmed / zcal / gross4s%')
for sc in sys.argv[3].split(','):
    sd = sorted(set(seeds(sc, tA)) & set(seeds(sc, tB)))
    if not sd: print(f'{sc}: no common seeds'); continue
    Sa, Sb, offs = [], [], []
    for s_ in sd:      # per seed: load, pair, (optionally) remove each method's own offset
        A = dict(np.load(f'{D}/{sc}__s{s_}{tA}.npz')); B = dict(np.load(f'{D}/{sc}__s{s_}{tB}.npz'))
        assert np.array_equal(A['gt_x'], B['gt_x'])
        ra, rb = rows(A), rows(B)
        if OFF:
            for R_, r_ in ((A, ra), (B, rb)):
                ti = np.array(list(r_.keys())); oi = np.array([r_[i] for i in ti])
                o = fit_offset(R_, ti, oi); R_['sc_x_sc'] = R_['sc_x_sc'] - o[0]; R_['sc_y_sc'] = R_['sc_y_sc'] - o[1]
                offs.append((s_, 'A' if R_ is A else 'B', o * 108.33))
        com = np.array(sorted(set(ra) & set(rb)))
        Sa.append((A, com, np.array([ra[i] for i in com]))); Sb.append((B, com, np.array([rb[i] for i in com])))
    if OFF:
        print(f'{sc:7s} offsets nm: ' + '; '.join(f's{s_} {w} ({o[0]:+.2f},{o[1]:+.2f})' for s_, w, o in offs))
    for lo, hi in BINS:
        acc = {k: [] for k in ('ea', 'eb', 'c', 'sa', 'sb')}
        for (A, com, oa), (B, _, ob) in zip(Sa, Sb):
            ph = A['gt_phot'][com]
            ok = np.isfinite(A['gt_sf_m'][com]).all(1) & (np.hypot(*A['gt_sf_m'][com].T) / np.sqrt(2) * 108.33 < 25)
            m = (ph >= lo) & (ph < hi) & ok
            t = com[m]
            acc['ea'].append(np.c_[A['sc_x_sc'][oa[m]] - A['gt_x'][t], A['sc_y_sc'][oa[m]] - A['gt_y'][t]])
            acc['eb'].append(np.c_[B['sc_x_sc'][ob[m]] - B['gt_x'][t], B['sc_y_sc'][ob[m]] - B['gt_y'][t]])
            acc['c'].append(A['gt_sf_m'][t]); acc['sa'].append(A['sc_scal_sc'][oa[m]]); acc['sb'].append(B['sc_scal_sc'][ob[m]])
        a_ = {k: np.concatenate(v) for k, v in acc.items()}
        n = len(a_['c'])
        if n < 20: continue
        def stat(e, c, s):
            z = np.hypot(*(e / c).T)
            return (np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(c ** 2)), np.sqrt(np.mean(e[z <= 5] ** 2)) / np.sqrt(np.mean(c[z <= 5] ** 2)),
                    1.4826 * np.median(np.abs(e / c)), 1.4826 * np.median(np.abs(e / s)), 100 * np.mean(np.hypot(*(e / s).T) > 4))
        a = stat(a_['ea'], a_['c'], a_['sa']); b = stat(a_['eb'], a_['c'], a_['sb'])
        print(f'{sc:7s} {lo:5.0f}-{hi:<6.0f} n {n:5d} | SIMCODE ' + ' '.join(f'{v:5.2f}' for v in a) + ' | ours ' + ' '.join(f'{v:5.2f}' for v in b))
