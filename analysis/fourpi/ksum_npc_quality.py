"""NPC pore quality per acquired frame: the K = 4 summed real movie (network v11b, pattern + phase field of the K = 1 run
as calibration) against the K = 1 tables (network v10, classical published-style and pixel MLE) restricted to a fraction f
of the raw windows ((w % 100) < 100 f per file; f = 1/4 is what a 4x shorter sparse acquisition would have given).

Machinery of select_curve_npc.py: the same pores (found once on the classical published-style table, fixed centres),
per-pore z centring on the pore mean (r < 95 nm), residual per-block pore z drift per table, free-mean two-Gaussian EM.
Per row: localizations per pore, ring radial width, ring radius, two-ring z separation, z sd per ring, z dip, the fraction
of ring localizations with |dz| > 60 nm and > 80 nm about the pore mean (pedestal from fringe errors), FRC (central
20 x 20 um, halves = alternating 500-window blocks).
Registration of the K = 4 table (its own DME and z drift; no per-window match with the K = 1 rows is possible) to the
classical frame: median offset of the per-pore circle-fit centres from the classical pore centres. The K = 1 network is
registered on matched emitters (select_curve_npc.load).
Usage: python analysis/fourpi/ksum_npc_quality.py --k4 out/4pi/net/net_NPC_v11b_k4cal_all.npz --net out/4pi/net/net_NPC_v10_all.npz
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
REPO = '/home/jc/simcode2'
sys.path.insert(0, f'{REPO}/analysis/fourpi')
import select_curve_npc as SN
from select_curve_npc import sub, comb_sigma, pore_metrics, pore_zdrift, avgpore, FC, PX, FPF, OUTD, lg
import showcase_real_mt as SR
import matplotlib.pyplot as plt

FR = [0.125, 0.25, 0.5, 1.0]


def frac_rows(T, f):
    w = (T['frame'] % FPF) // 6
    return sub(T, np.nonzero((w % 100) < round(100 * f))[0])


def metrics(T, P, frc_box):
    out, V = pore_metrics(T, P)
    out['n'] = int(len(T['x']))
    if V is not None:
        ring = (V['r'] > 25) & (V['r'] < 85)
        out['out_of_ring_60'] = float(np.mean(np.abs(V['dz'][ring]) > 60)); out['out_of_ring_80'] = float(np.mean(np.abs(V['dz'][ring]) > 80))
    m = SR.inrect(T['x'], T['y'], frc_box); h = ((T['frame'][m] // 6) // 500) % 2 == 0
    out['frc'] = FC.frc(T['x'][m][h], T['y'][m][h], T['x'][m][~h], T['y'][m][~h]) if h.sum() > 200 and (~h).sum() > 200 else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cl', default=f'{REPO}/out/4pi/classical/real/cl_NPC_wv_all.npz')
    ap.add_argument('--net', default=f'{REPO}/out/4pi/net/net_NPC_v10_all.npz')
    ap.add_argument('--k4', default=f'{REPO}/out/4pi/net/net_NPC_v11b_k4cal_all.npz')
    ap.add_argument('--tag', default='ksum4')
    a = ap.parse_args()
    M, info = SN.load(argparse.Namespace(cl=a.cl, net=a.net))
    C = M['cl']; fb = (np.percentile(C['x'], 0.5), np.percentile(C['x'], 99.5), np.percentile(C['y'], 0.5), np.percentile(C['y'], 99.5))
    cx, cy = 0.5 * (fb[0] + fb[1]), 0.5 * (fb[2] + fb[3]); frc_box = (cx - 10000, cx + 10000, cy - 10000, cy + 10000)
    # K = 4 table
    K = np.load(a.k4)
    cnt = {}
    base = K['good'] & (K['N'] >= 1000) & (K['sx'] < 0.12) & (K['sy'] < 0.12) & np.isfinite(K['zd']) & np.isfinite(K['xd'])
    zmed = np.median(K['zd'][base])
    s = base & (K['sz'] < 0.05) & (np.abs(K['zd'] - zmed) <= 0.3)
    cnt['k4_rows'] = int(len(K['x'])); cnt['k4_good'] = int(K['good'].sum()); cnt['k4_good_N1000_sxy'] = int(base.sum())
    cnt['k4_cut_sz_0.05'] = int(s.sum()); cnt['k4_cut_sz_0.025'] = int((base & (K['sz'] < 0.025) & (np.abs(K['zd'] - zmed) <= 0.3)).sum())
    k4 = dict(x=K['xd'][s] * PX, y=K['yd'][s] * PX, z=K['zd'][s] * 1000, frame=K['frame'][s], f=K['fidx'][s], N=K['N'][s],
              sx=K['sx'][s], sy=K['sy'][s], sz=K['sz'][s])
    k4_files = int(len(np.unique(K['fidx'])))
    del K
    M['k4'] = k4
    M = {k: sub(T, np.nonzero(SR.inrect(T['x'], T['y'], fb))[0]) for k, T in M.items()}
    P = SN.find_pores(M['cl']); lg(f'{len(P)} pores; counts {cnt}')
    # register K = 4 laterally via pore centres
    tr = cKDTree(np.c_[M['k4']['x'], M['k4']['y']]); off = []
    for (px_, py_, _) in P:
        idx = tr.query_ball_point((px_, py_), 100.0)
        if len(idx) < 30: continue
        fx, fy, r = FC.fit_circle(M['k4']['x'][idx], M['k4']['y'][idx])
        if 30 < r < 80 and np.hypot(fx - px_, fy - py_) < 40: off.append((fx - px_, fy - py_))
    off = np.array(off); o = np.median(off, 0)
    M['k4']['x'] -= o[0]; M['k4']['y'] -= o[1]
    info['k4_registration'] = dict(n_pores=int(len(off)), dx_nm=float(o[0]), dy_nm=float(o[1]),
                                   robust_sd_nm=[SN.rsd(off[:, 0]), SN.rsd(off[:, 1])])
    # same check for the K = 1 network (already registered on matched emitters): residual pore-centre offset
    tr = cKDTree(np.c_[M['net']['x'], M['net']['y']]); off1 = []
    for (px_, py_, _) in P:
        idx = tr.query_ball_point((px_, py_), 100.0)
        if len(idx) < 30: continue
        fx, fy, r = FC.fit_circle(M['net']['x'][idx], M['net']['y'][idx])
        if 30 < r < 80 and np.hypot(fx - px_, fy - py_) < 40: off1.append((fx - px_, fy - py_))
    info['k1_net_pore_centre_offset_nm'] = np.median(np.array(off1), 0).tolist()
    info['pore_zdrift_ptp_nm'] = {}
    for k, T in M.items():
        d = pore_zdrift(T, P); T['z'] = T['z'] - d[np.clip(T['frame'] // 6 // 500, 0, len(d) - 1)]; info['pore_zdrift_ptp_nm'][k] = float(np.ptp(d))
    lg(f"k4 registration {info['k4_registration']}; K=1 net residual {info['k1_net_pore_centre_offset_nm']}; z drift {info['pore_zdrift_ptp_nm']}")
    nfiles = int(len(np.unique(M['cl']['f']))); frames_all = nfiles * FPF
    R = {}
    def row(name, T, f_frames):
        r = metrics(T, P, frc_box); r['acquired_frames'] = int(round(f_frames * frames_all)); r['frame_fraction'] = f_frames; R[name] = r
        lg(f"{name}: n {r['n']} /pore {r['locs_per_pore']:.0f} radw {r.get('radial_width_nm', np.nan):.2f} R {r.get('ring_radius_median_nm', np.nan):.1f} "
           f"sep {r.get('two_ring_sep_em2_nm', np.nan):.1f} sd {r.get('ring_z_sd_nm', np.nan):.1f} dip {r.get('z_dip', np.nan):.2f} "
           f"out60 {r.get('out_of_ring_60', np.nan):.3f} frc {r['frc']:.1f}")
    q = frac_rows(M['net'], 0.25)
    row('(a) K=4 summed, network v11b, all kept', M['k4'], 0.25)
    o4 = np.argsort(comb_sigma(M['k4'], 'net'), kind='stable')
    row('(a) K=4 summed, network v11b, combined sigma to the count of (c)', sub(M['k4'], np.sort(o4[:min(len(q['x']), len(o4))])), 0.25)
    row('(b) K=1 network v10, all frames, all kept', M['net'], 1.0)
    on = np.argsort(comb_sigma(M['net'], 'net'), kind='stable')
    row('(b) K=1 network v10, all frames, matched to classical count', sub(M['net'], np.sort(on[:len(M['cl']['x'])])), 1.0)
    row('(c) K=1 network v10, 1/4 frames, all kept', q, 0.25)
    row('(d) K=1 classical published-style, all frames', M['cl'], 1.0)
    row('(d) K=1 classical pixel MLE, all frames', M['clp'], 1.0)
    row('(e) K=1 classical published-style, 1/4 frames', frac_rows(M['cl'], 0.25), 0.25)
    row('(e) K=1 classical pixel MLE, 1/4 frames', frac_rows(M['clp'], 0.25), 0.25)
    # same-count comparison with selection on both sides: the classical published-style count of 1/4 of the frames
    n_e = len(frac_rows(M['cl'], 0.25)['x'])
    oq = np.argsort(comb_sigma(q, 'net'), kind='stable')
    row('(c) K=1 network v10, 1/4 frames, combined sigma to the classical 1/4 count', sub(q, np.sort(oq[:n_e])), 0.25)
    row('(a) K=4 summed, network v11b, combined sigma to the classical 1/4 count', sub(M['k4'], np.sort(o4[:n_e])), 0.25)
    curves = {}
    for k in ('net', 'cl', 'clp'):
        curves[k] = []
        for f in FR:
            r = metrics(frac_rows(M[k], f), P, frc_box); r['frame_fraction'] = f; r['acquired_frames'] = int(round(f * frames_all)); curves[k].append(r)
    # equivalent sparse acquisition length of the K = 4 result per metric (interpolated on the K = 1 curves, log f)
    keys = ['locs_per_pore', 'z_dip', 'ring_z_sd_nm', 'radial_width_nm', 'frc', 'out_of_ring_60']
    equiv = {}
    for k, ka_name in (('net', '(a) K=4 summed, network v11b, all kept'), ('cl', '(a) K=4 summed, network v11b, all kept'), ('clp', '(a) K=4 summed, network v11b, all kept'),
                       ('net|sel', '(a) K=4 summed, network v11b, combined sigma to the count of (c)')):
        ka = R[ka_name]; equiv[k] = {}
        k_ = k.split('|')[0]
        for key in keys:
            fs = np.log2([r['frame_fraction'] for r in curves[k_]]); v = np.array([r.get(key, np.nan) for r in curves[k_]], float)
            t = ka.get(key, np.nan)
            if not np.isfinite(t) or not np.all(np.isfinite(v)) or not (np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0)):
                equiv[k][key] = None; continue
            d = v[-1] - v[0]
            if (t - v[0]) / d < 0: equiv[k][key] = '<1/8'
            elif (t - v[-1]) / d > 0: equiv[k][key] = '>1'
            else:
                o_ = np.argsort(v); equiv[k][key] = float(2 ** np.interp(t, v[o_], fs[o_]))
    out = dict(tag=a.tag, counts=cnt, k4_files=k4_files, n_files=nfiles, frames_all=frames_all, n_pores=int(len(P)), info=info, rows=R,
               curves=curves, equivalent_frame_fraction=equiv,
               notes=['K = 4 run uses the SIM pattern and the 4Pi phase field of the K = 1 run as calibration',
                      'the summed data contain the same photons as the K = 1 data (not an independent acquisition)',
                      'K = 4 frame = centre time of its 4 sources; (w % 100) < 100 f selects the K = 1 windows of a fraction f',
                      'equivalent fraction: interpolation in log2 f on the K = 1 curve of that method (all kept rows), None if the curve is not monotonic; key net|sel = the selected K = 4 row on the K = 1 network curve',
                      'with the K = 4 sigma typical sz ~ 0.007 um, the cuts sz < 0.05 and sz < 0.025 keep the same rows',
                      'FRC on the central 20 x 20 um, halves = alternating 500-window blocks'])
    fn = f'{OUTD}/NPC_{a.tag}_quality'
    json.dump(out, open(fn + '.json', 'w'), indent=1, default=float)
    plot(out, fn + '.png')
    avgpore({'K = 4 summed, network v11b': M['k4'], 'K = 1 network v10, 1/4 frames': q, 'K = 1 network v10, all frames': M['net'],
             'K = 1 pixel MLE, 1/4 frames': frac_rows(M['clp'], 0.25), 'K = 1 pixel MLE, all frames': M['clp']},
            P, f'{OUTD}/NPC_{a.tag}_avgpore.png', dict(n_pores=len(P)))
    lg(f'-> {fn}.png, {OUTD}/NPC_{a.tag}_avgpore.png')


def plot(out, fn):
    cols = dict(net='#2a78d6', cl='#eb6834', clp='#1baf7a')
    names = dict(net='K = 1 network v10', cl='K = 1 classical published-style', clp='K = 1 classical pixel MLE')
    panels = [('locs_per_pore', 'localizations per pore'), ('z_dip', 'two-ring z dip (valley / peak)'), ('ring_z_sd_nm', 'z sd per ring (nm)'),
              ('radial_width_nm', 'ring radial width (nm)'), ('two_ring_sep_em2_nm', 'two-ring z separation (nm)'), ('out_of_ring_60', 'ring locs with |dz| > 60 nm'),
              ('frc', 'FRC, central 20 µm (nm)'), ('ring_radius_median_nm', 'ring radius (nm)')]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.2))
    ka = out['rows']['(a) K=4 summed, network v11b, all kept']; kb = out['rows']['(a) K=4 summed, network v11b, combined sigma to the count of (c)']
    for ax, (key, lab) in zip(axs.ravel(), panels):
        for k in ('net', 'cl', 'clp'):
            c = out['curves'][k]
            ax.plot([r['frame_fraction'] for r in c], [r.get(key, np.nan) for r in c], color=cols[k], lw=1.8, marker='o', ms=4, label=names[k] + ', fraction f of the frames')
        ax.plot(0.25, ka.get(key, np.nan), '*', color='#0b0b0b', ms=11, label='K = 4 summed, network v11b, all kept (f = 1/4)')
        ax.plot(0.25, kb.get(key, np.nan), 'D', color='#0b0b0b', mfc='none', ms=6, label='K = 4 summed, combined sigma to the count of K = 1 network 1/4')
        ax.set_xscale('log', base=2); ax.set_xticks(FR); ax.set_xticklabels(['1/8', '1/4', '1/2', '1'])
        ax.set_title(lab, fontsize=8); ax.grid(alpha=0.25, lw=0.5); ax.set_xlabel('fraction of the acquired frames')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"Real NPC: pore quality per acquired frame; K = 4 summed (4x density, same photons; pattern + phase field from the K = 1 run) vs K = 1 "
                 f"tables restricted to a fraction of the windows; {out['n_pores']} pores, {out['n_files']} files", fontsize=9)
    plt.tight_layout(rect=(0, 0.09, 1, 0.96)); fig.savefig(fn, dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
