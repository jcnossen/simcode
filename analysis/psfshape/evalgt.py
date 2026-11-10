"""[2026-11-11 PSF-shape mismatch, (b)/(c)] GT readouts on identical rows, per axis: ours pass-1 prior / final (pattern solve)
and SIMCODE sign-fixed net / NDI vs truth, plus the GT-free (prior - final) and (SC net - NDI) robust sds (the critic-38 readout),
binned by GT window photons.  Identical rows = GT target windows (window start w % 3 == 0, tc in [1, 4), inside the crop) matched
(1 px, greedy, per window) by ours (p >= 0.63) AND by SIMCODE's NDI row at the same window start.
usage: evalgt.py <name> sim:<sim_dir>:<ours_dir>:<ours_tag>:<sc_fix.npz>   |   inj:<inj_dir>:<ours_tag>   [more ...]
For inj, also the REAL (non-injected) rows of the same movie: ours (prior - final) and SC (net - NDI) on doubly-matched rows.
[2026-11-11 critic 39, opt-in env EVALGT_BINBY = gt (default) | ours | sc]: photon variable of the bins.  Injected / sim rows: GT window
photons | our row's phot | SIMCODE NDI row's photons; the REAL rows of an inj movie (no GT): SC photons for gt / sc, ours for ours.
Output json gets the suffix _by<binby> when not gt.  same algorithm; changes: compare.py from
analysis/baselines, pattern_frames from simcode2.pipeline.common, the inj-mode pattern (e.g. the real 0.5 nM pass-1 run's
pattern_est.pt) is an optional 4th field inj:<inj_dir>:<ours_tag>:<pattern_est.pt> (default <inj_dir>/pattern_est.pt), and the
json goes to $EVALGT_OUT (default <simcode2 out>/psfshape). The report is printed to stdout.
The injection movies and the PSF-diversity heads (sc5pd_*) are not part of simcode2.
usage: python analysis/psfshape/evalgt.py <name>=sim:<sim_dir>:<ours_dir>:<ours_tag>:<sc_fix.npz> [...] > evalgt_<...>.txt"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
import compare as C       # analysis/baselines/compare.py
from scipy.spatial import cKDTree
PX = 108.33
rs = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 5 else np.nan
rms = lambda v: float(np.sqrt(np.mean(np.clip(v, -50, 50) ** 2))) if len(v) > 5 else np.nan
BINBY = os.environ.get('EVALGT_BINBY', 'gt')
BINS = ((200, 400), (400, 800), (800, 1500), (1500, 1e9), (400, 1e9))
OUT = {}
for arg in sys.argv[1:]:
    nm, spec = arg.split('=', 1); kind, *pp = spec.split(':')
    if kind == 'sim':
        sd, od, tag, scf = pp
        G = dict(np.load(f'{sd}/sim_gt.npz')); NF = int(G['nf'])
    else:
        sd, tag = pp[:2]; od = f'{sd}/ours'; scf = f'{sd}/sc_fix.npz'
        I = np.load(f'{sd}/inj_gt.npz'); NF = int(I['nf']); N = len(I['x'])
        import torch
        from simcode2.pipeline.common import pattern_frames
        Pt = torch.load(pp[2] if len(pp) > 2 else f'{sd}/pattern_est.pt', weights_only=False)
        patt = pattern_frames(Pt, np.arange(NF), 0, 0, dev='cpu').numpy().astype(np.float64)
        G = dict(x=I['x'], y=I['y'], bright=I['bright'], t0=I['t0'], t1=I['t1'], ea=np.ones((N, 2)), es=np.ones((N, 6)), patt=patt, nf=NF)
    GW = C.gt_windows(G, np.arange(0, NF - 5, 3))
    O = C.load_method(f'ours_{tag}', od, NF); O = {k: v[O['p'] >= 0.63] for k, v in O.items()}
    Sn = C.load_method('sc_ndi', scf, NF)          # NDI rows: x/y = NDI, xd/yd = the net position of the same row
    go = C.match(GW, O['w'], O['xd'], O['yd']); gs = C.match(GW, Sn['w'], Sn['xd'], Sn['yd'])
    io = np.full(len(GW), -1); io[go[go >= 0]] = np.nonzero(go >= 0)[0]
    js = np.full(len(GW), -1); js[gs[gs >= 0]] = np.nonzero(gs >= 0)[0]
    tg = (GW[:, 5] > 0) & (io >= 0) & (js >= 0)
    g = np.nonzero(tg)[0]; i, j = io[g], js[g]
    gx, gy, P = GW[g, 2], GW[g, 3], GW[g, 4]
    if BINBY == 'ours': P = O['phot'][i]
    elif BINBY == 'sc': P = Sn['phot'][j]
    if kind == 'sim' and 'shift_dx' in G:   # the GT of every method is the illumination position r (imaging shift field only moves the PSF)
        pass
    E = {'ours_prior': np.c_[O['xd'][i] - gx, O['yd'][i] - gy] * PX, 'ours_final': np.c_[O['x'][i] - gx, O['y'][i] - gy] * PX,
         'sc_net': np.c_[Sn['xd'][j] - gx, Sn['yd'][j] - gy] * PX, 'sc_ndi': np.c_[Sn['x'][j] - gx, Sn['y'][j] - gy] * PX}
    E['ours_prior-final'] = E['ours_prior'] - E['ours_final']; E['sc_net-ndi'] = E['sc_net'] - E['sc_ndi']
    R = dict(n_gt_targets=int((GW[:, 5] > 0).sum()), n_identical=int(len(g)), recall_ours=float(((GW[:, 5] > 0) & (io >= 0)).sum() / max((GW[:, 5] > 0).sum(), 1)),
             recall_sc=float(((GW[:, 5] > 0) & (js >= 0)).sum() / max((GW[:, 5] > 0).sum(), 1)), bins={})
    for lo, hi in BINS:
        m = (P >= lo) & (P < hi); b = {'n': int(m.sum())}
        for k, e in E.items():
            b[k] = dict(rsd_x=rs(e[m, 0]), rsd_y=rs(e[m, 1]), rms_x=rms(e[m, 0]), rms_y=rms(e[m, 1]), bias_x=float(np.median(e[m, 0])) if m.sum() else np.nan,
                        bias_y=float(np.median(e[m, 1])) if m.sum() else np.nan)
        b['prior/net_rms'] = dict(x=b['ours_prior']['rms_x'] / b['sc_net']['rms_x'], y=b['ours_prior']['rms_y'] / b['sc_net']['rms_y'])
        R['bins'][f'{lo}-{hi:.0f}'] = b
    # event-static part (lag-6 covariance of the same GT event's identical rows, P >= 400, |e| < 40 nm), per axis
    key = {(int(GW[gg, 1]), int(GW[gg, 0])): n for n, gg in enumerate(g)}
    pr = [(n, key[(a_, w_ + 6)]) for (a_, w_), n in key.items() if (a_, w_ + 6) in key]
    R['static'] = {}
    if pr:
        p0, p1 = np.array(pr).T; mP = (P[p0] >= 400) & (P[p1] >= 400)
        for k in ('ours_prior', 'ours_final', 'ours_prior-final', 'sc_net', 'sc_ndi', 'sc_net-ndi'):
            v = []
            for ax in (0, 1):
                a_, b_ = E[k][p0[mP], ax], E[k][p1[mP], ax]; ok_ = (np.abs(a_) < 40) & (np.abs(b_) < 40)
                v.append(float(np.sqrt(max(np.cov(a_[ok_], b_[ok_])[0, 1], 0))) if ok_.sum() > 10 else np.nan)
            R['static'][k] = v
        R['static']['n_pairs'] = int(mP.sum())
    if kind == 'inj':   # real (non-injected) rows: ours rows and SC NDI rows at the same window start within 0.5 px, > 3 px from any injected GT row
        gwt = cKDTree(np.c_[GW[:, 2], GW[:, 3], GW[:, 0] * 1e3])
        far_o = gwt.query(np.c_[O['xd'], O['yd'], O['w'] * 1e3])[0] > 3; far_s = gwt.query(np.c_[Sn['xd'], Sn['yd'], Sn['w'] * 1e3])[0] > 3
        ts = cKDTree(np.c_[Sn['xd'], Sn['yd'], Sn['w'] * 1e3]); d, jj = ts.query(np.c_[O['xd'], O['yd'], O['w'] * 1e3], distance_upper_bound=0.5)
        ok = np.isfinite(d) & far_o & (O['w'] % 3 == 0); ok[ok] &= far_s[jj[ok]]
        ii = np.nonzero(ok)[0]; jj = jj[ok]; ph = O['phot'][ii] if BINBY == 'ours' else Sn['phot'][jj]
        po = np.c_[O['xd'][ii] - O['x'][ii], O['yd'][ii] - O['y'][ii]] * PX; ps = np.c_[Sn['xd'][jj] - Sn['x'][jj], Sn['yd'][jj] - Sn['y'][jj]] * PX
        R['real_rows'] = {}
        for lo, hi in BINS:
            m = (ph >= lo) & (ph < hi)
            R['real_rows'][f'{lo}-{hi:.0f}'] = dict(n=int(m.sum()), ours_pf_x=rs(po[m, 0]), ours_pf_y=rs(po[m, 1]), sc_nn_x=rs(ps[m, 0]), sc_nn_y=rs(ps[m, 1]))
    OUT[nm] = R
    print(f'### {nm}: GT targets {R["n_gt_targets"]}, identical rows {R["n_identical"]}, recall ours {R["recall_ours"]:.3f} SC {R["recall_sc"]:.3f}')
    for bn, b in R['bins'].items():
        print(f'  {bn:>10} n {b["n"]:6d} | rms x/y: prior {b["ours_prior"]["rms_x"]:.2f}/{b["ours_prior"]["rms_y"]:.2f} final {b["ours_final"]["rms_x"]:.2f}/{b["ours_final"]["rms_y"]:.2f} '
              f'SCnet {b["sc_net"]["rms_x"]:.2f}/{b["sc_net"]["rms_y"]:.2f} NDI {b["sc_ndi"]["rms_x"]:.2f}/{b["sc_ndi"]["rms_y"]:.2f} | prior/net {b["prior/net_rms"]["x"]:.3f}/{b["prior/net_rms"]["y"]:.3f} '
              f'| rsd prior-final {b["ours_prior-final"]["rsd_x"]:.2f}/{b["ours_prior-final"]["rsd_y"]:.2f} net-NDI {b["sc_net-ndi"]["rsd_x"]:.2f}/{b["sc_net-ndi"]["rsd_y"]:.2f} '
              f'| prior bias {b["ours_prior"]["bias_x"]:+.2f}/{b["ours_prior"]["bias_y"]:+.2f}')
    if R['static']:
        print('  event-static (lag-6 cov, P >= 400, n pairs %d): ' % R['static']['n_pairs'] + ' | '.join(f'{k} {v[0]:.2f}/{v[1]:.2f}' for k, v in R['static'].items() if k != 'n_pairs'))
    if 'real_rows' in R:
        for bn, b in R['real_rows'].items():
            print(f'  real rows {bn:>10} n {b["n"]:6d} | ours prior-final rsd x/y {b["ours_pf_x"]:.2f}/{b["ours_pf_y"]:.2f} | SC net-NDI {b["sc_nn_x"]:.2f}/{b["sc_nn_y"]:.2f}')
_od = os.environ.get('EVALGT_OUT')
if not _od:
    from simcode2 import config
    _od = config.out_dir('psfshape')
os.makedirs(_od, exist_ok=True)
json.dump(OUT, open(os.path.join(_od, f'evalgt_{"_".join(OUT)}{"" if BINBY == "gt" else "_by" + BINBY}.json'), 'w'), indent=1)
