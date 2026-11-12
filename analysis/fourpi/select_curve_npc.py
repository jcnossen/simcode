"""Selection curves on the real 4Pi NPC (Nup96) movie: quality versus number of localizations kept, network (head v10,
post_w4pi.py table) against the classical published-style and pixel-MLE rows, each ranked by its own combined reported
sigma sqrt(sx^2 + sy^2 + sz^2) (the rule that won on MT, select_curve_mt.py); the network also by lateral sigma, photons
and combined sigma x sqrt(chi2/dof) to check that the ranking holds here.

Tables: network rows good & N >= 1000 (xd, yd: its own DME; z: zd, its own structure-based z drift removed); classical
published-style keep and pixel MLE keep_pix (xd, yd / xpd, ypd: classical DME; z with the classical structure-based z drift
fig_cl.zdrift_struct removed, the pixel-MLE row with the published-style curve, as fig_cl.load). The network is shifted by
the median offset to the classical rows on matched emitters (same window, < 64 nm on raw positions; dz too).
Pores: found ONCE on the classical published-style table (fig_cl.npc_figures recipe: 10 nm render, sigma 4 smoothing,
local maxima in 13 x 13 above the 90th percentile, circle fit within 100 nm, 30 < r < 80 nm, centre shift < 40 nm), the
same pore list and centres for every method and selection level. Per selection: ring radius (peak of the area-normalised
radial density, and median 25-85 nm), radial width (robust sd of r, 25 < r < 85 nm), z about each pore's mean z (r < 95 nm):
two-Gaussian separation and common sd (EM with free means: per-ring z spread), histogram peak separation and dip (valley / lower
peak, as fig_cl), localizations per pore; DAFL x / y / z (adjacent windows of the same file, all pairs and 2-4k e),
fringe-hop rate (|dz| > 80 nm), FRC (central 10 x 10 um, halves = alternating 500-window blocks).
Usage: python analysis/fourpi/select_curve_npc.py --net out/4pi/net/net_NPC_v10_all.npz --tag v10
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
sys.path.insert(0, f'{REPO}/analysis/fourpi')
from simcode.fourpi.setup import P4
from select_curve_mt import pairs, rsd, photon_match
import showcase_real_mt as SR
from showcase_real_mt import FC, PX, FPF, OUTD, lg
import matplotlib.pyplot as plt

FRACS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1]
_G = {}


def sub(T, idx): return {k: v[idx] for k, v in T.items()}


def comb_sigma(T, kind):
    if kind == 'clp': return np.sqrt((T['sxp'] * PX) ** 2 + (T['syp'] * PX) ** 2 + (T['szp'] * 1000) ** 2)
    return np.sqrt((T['sx'] * PX) ** 2 + (T['sy'] * PX) ** 2 + (T['sz'] * 1000) ** 2)


def load(a):
    Z = np.load(a.cl)
    cols = ('x', 'y', 'xd', 'yd', 'z', 'keep', 'xpd', 'ypd', 'zp', 'keep_pix', 'frame', 'file_index', 'sx', 'sy', 'sz', 'sxp', 'syp', 'szp', 'N', 'chi2px')
    C = {k: Z[k] for k in cols}
    m = C['keep'] | C['keep_pix']; C = {k: v[m] for k, v in C.items()}
    mk = C['keep'] & np.isfinite(C['z']) & np.isfinite(C['xd'])
    zd = FC.zdrift_struct(C['xd'][mk] * PX, C['yd'][mk] * PX, C['z'][mk] * 1000, C['frame'][mk] // 6)
    zc = lambda z, fr: z * 1000 - zd[np.clip(fr // 6 // 500, 0, len(zd) - 1)]
    base = dict(frame=C['frame'], f=C['file_index'], ux=C['x'] * PX, uy=C['y'] * PX, N=C['N'], chi2px=C['chi2px'])
    cl = dict(base, x=C['xd'] * PX, y=C['yd'] * PX, z=zc(C['z'], C['frame']), sx=C['sx'], sy=C['sy'], sz=C['sz'])
    clp = dict(base, x=C['xpd'] * PX, y=C['ypd'] * PX, z=zc(C['zp'], C['frame']), sxp=C['sxp'], syp=C['syp'], szp=C['szp'])
    cl = sub(cl, np.nonzero(mk)[0]); mp_ = C['keep_pix'] & np.isfinite(C['zp']) & np.isfinite(C['xpd']); clp = sub(clp, np.nonzero(mp_)[0])
    N = np.load(a.net)
    s = N['good'] & (N['N'] >= 1000) & np.isfinite(N['zd']) & np.isfinite(N['xd'])
    first = int(np.min(N['fidx']))
    net = dict(x=N['xd'][s] * PX, y=N['yd'][s] * PX, z=N['zd'][s] * 1000, frame=(N['fidx'][s] - first) * FPF + 6 * N['w'][s], f=N['fidx'][s],
               ux=N['x'][s] * PX, uy=N['y'][s] * PX, N=N['N'][s], sx=N['sx'][s], sy=N['sy'][s], sz=N['sz'][s],
               chi2=N['chi2'][s], dof=N['dof'][s], sel=N['sel'][s] if 'sel' in N.files else np.ones(s.sum(), bool))
    cfirst = int(np.min(C['file_index']))
    assert cfirst == first, (cfirst, first)
    # registration on matched emitters
    K = 1e7; tr = cKDTree(np.c_[cl['ux'], cl['uy'], (cl['frame'] // 6) * K])
    d, k = tr.query(np.c_[net['ux'], net['uy'], (net['frame'] // 6) * K], distance_upper_bound=64.0)
    ok = np.isfinite(d); ia, ib = np.nonzero(ok)[0], k[ok]
    dx, dy, dz = net['x'][ia] - cl['x'][ib], net['y'][ia] - cl['y'][ib], net['z'][ia] - cl['z'][ib]
    reg = dict(n_matched=int(len(ia)), frac_of_net=float(len(ia) / len(net['x'])), frac_of_cl=float(len(np.unique(ib)) / len(cl['x'])),
               dx_median_nm=float(np.median(dx)), dy_median_nm=float(np.median(dy)), dz_median_nm=float(np.median(dz)),
               dx_robust_sd_nm=rsd(dx), dy_robust_sd_nm=rsd(dy), dz_robust_sd_nm=rsd(dz),
               dz_fringe_disagree=float(np.mean(np.abs(dz - np.median(dz)) > 120)))
    net['x'] -= reg['dx_median_nm']; net['y'] -= reg['dy_median_nm']; net['z'] -= reg['dz_median_nm']
    zdn = N['zdrift'] if 'zdrift' in N.files else None
    info = dict(registration=reg, zdrift_classical_ptp_nm=float(np.ptp(zd)), zdrift_network_own_ptp_nm=float(np.ptp(zdn)) if zdn is not None else None,
                n_files=int(len(np.unique(net['f']))), first_file=first)
    return dict(net=net, cl=cl, clp=clp), info


def find_pores(T):
    px = 10.0
    x0, y0 = np.percentile(T['x'], 0.5), np.percentile(T['y'], 0.5)
    h, xe, ye = np.histogram2d(T['x'], T['y'], bins=[np.arange(x0, np.percentile(T['x'], 99.5), px), np.arange(y0, np.percentile(T['y'], 99.5), px)])
    hs = gaussian_filter(h, 4.0)
    pk = (hs == maximum_filter(hs, size=13)) & (hs > np.percentile(hs[hs > 0], 90))
    ci, cj = np.nonzero(pk); cen = np.c_[xe[ci] + px / 2, ye[cj] + px / 2]
    tr = cKDTree(np.c_[T['x'], T['y']]); out = []
    for c in cen:
        idx = tr.query_ball_point(c, 100.0)
        if len(idx) < 30: continue
        cx, cy, r = FC.fit_circle(T['x'][idx], T['y'][idx])
        if not (30 < r < 80) or np.hypot(cx - c[0], cy - c[1]) > 40: continue
        out.append((cx, cy, r))
    return np.array(out)


def em2(z, n_it=200):
    """two-Gaussian mixture with free means and weights, common sd (EM): -> separation, sd"""
    z = z[np.abs(z - np.median(z)) < 100]
    mu = np.array([np.median(z) - 20.0, np.median(z) + 20.0]); s = 15.0; w = np.array([0.5, 0.5])
    for _ in range(n_it):
        L = w[None] * np.exp(-0.5 * ((z[:, None] - mu[None]) / s) ** 2)
        R = L / np.maximum(L.sum(1, keepdims=True), 1e-300)
        nk = R.sum(0); mu = (R * z[:, None]).sum(0) / np.maximum(nk, 1e-9); w = nk / nk.sum()
        s = np.sqrt((R * (z[:, None] - mu[None]) ** 2).sum() / len(z))
    return float(abs(mu[1] - mu[0])), float(s)


def pore_zdrift(T, P, n_it=6):
    """residual sample z drift from the pores (fig_cl.npc_figures recipe, on the fixed pore list): per 500-window block,
    median of z minus the pore mean, smoothed over blocks (sd 1 block), iterated. Returns drift (nm) per block."""
    tr = cKDTree(P[:, :2]); d, pid = tr.query(np.c_[T['x'], T['y']], distance_upper_bound=85.0)
    m = np.isfinite(d); zz = T['z'][m]; pid = pid[m]; blk = (T['frame'][m] // 6 // 500).astype(int); nb = int(T['frame'].max() // 6 // 500) + 1
    import pandas as pd
    drift = np.zeros(nb)
    for _ in range(n_it):
        zc = np.bincount(pid, zz - drift[blk], minlength=len(P)) / np.maximum(np.bincount(pid, minlength=len(P)), 1)
        r = zz - drift[blk] - zc[pid]
        g = pd.Series(r).groupby(blk); md = g.median().reindex(range(nb)).fillna(0).to_numpy().copy()
        md[g.size().reindex(range(nb)).fillna(0).to_numpy() < 50] = 0
        drift = drift + gaussian_filter(md, 1.0); drift -= drift.mean()
    return drift


def pore_metrics(T, P):
    tr = cKDTree(np.c_[T['x'], T['y']]); L = tr.query_ball_point(P[:, :2], 95.0)
    dx, dy, dz, rr, npore = [], [], [], [], []
    for (cx, cy, _), idx in zip(P, L):
        idx = np.asarray(idx, int)
        if len(idx) == 0: npore.append(0); continue
        px_, py_ = T['x'][idx] - cx, T['y'][idx] - cy; r = np.hypot(px_, py_); ring = (r > 25) & (r < 85)
        npore.append(int(ring.sum()))
        if ring.sum() < 4: continue
        zz = T['z'][idx] - np.mean(T['z'][idx])          # pore mean over r < 95 nm (as fig_cl; a median sits in one ring and smears the two)
        dx.append(px_); dy.append(py_); dz.append(zz); rr.append(r)
    out = dict(locs_per_pore=float(np.mean(npore)), pores_with_ge4=int(len(rr)))
    if not rr: return out, None
    dx, dy, dz, rr = map(np.concatenate, (dx, dy, dz, rr)); ring = (rr > 25) & (rr < 85)
    hr, er = np.histogram(rr, bins=np.arange(0, 96, 2)); rc = 0.5 * (er[1:] + er[:-1])
    hn = gaussian_filter(hr / (2 * np.pi * rc), 1.0)
    out.update(ring_radius_peak_nm=float(rc[np.argmax(hn * (rc > 20))]), ring_radius_median_nm=float(np.median(rr[ring])),
               radial_width_nm=rsd(rr[ring]))
    zr = dz[ring & (np.abs(dz) < 120)]
    if len(zr) > 100:
        d, s = em2(zr); out.update(two_ring_sep_em2_nm=float(d), ring_z_sd_nm=float(s))
        hz, ez = np.histogram(dz[ring], bins=np.arange(-120, 121, 2)); zc = 0.5 * (ez[1:] + ez[:-1]); hzs = gaussian_filter(hz.astype(float), 1.5)
        lm = np.nonzero((hzs[1:-1] > hzs[:-2]) & (hzs[1:-1] >= hzs[2:]))[0] + 1; lm = lm[np.argsort(-hzs[lm])][:2]
        if len(lm) == 2:
            a_, b_ = sorted(lm); out.update(z_peak_sep_nm=float(zc[b_] - zc[a_]), z_dip=float(hzs[a_:b_ + 1].min() / min(hzs[a_], hzs[b_])))
        else: out.update(z_peak_sep_nm=float('nan'), z_dip=1.0)
    return out, dict(dx=dx, dy=dy, dz=dz, r=rr)


def metrics(T, P, frc_box, pore_only=False):
    out = dict(n=int(len(T['x'])))
    pm, _ = pore_metrics(T, P); out.update(pm)
    if pore_only: return out
    i, j = pairs(T)
    Np = 0.5 * (T['N'][i] + T['N'][j]); b = (Np >= 2000) & (Np < 4000)
    d = {q: T[q][i] - T[q][j] for q in ('x', 'y', 'z')}
    out.update(n_pairs=int(len(i)), dafl=[rsd(d[q]) / np.sqrt(2) for q in 'xyz'], n_pairs_2_4k=int(b.sum()),
               dafl_2_4k=[rsd(d[q][b]) / np.sqrt(2) for q in 'xyz'],
               fringe_hop=float(np.mean(np.abs(d['z'] - np.median(d['z'])) > 80)) if len(i) > 20 else float('nan'))
    m = SR.inrect(T['x'], T['y'], frc_box); h = ((T['frame'][m] // 6) // 500) % 2 == 0
    out['frc'] = FC.frc(T['x'][m][h], T['y'][m][h], T['x'][m][~h], T['y'][m][~h]) if h.sum() > 200 and (~h).sum() > 200 else float('nan')
    return out


def _job(args):
    k, sn, n = args
    o = _G['order'][(k, sn)]; n = int(min(n, len(o)))
    r = metrics(sub(_G['M'][k], np.sort(o[:n])), _G['P'], _G['frc_box']); r['n_keep'] = n
    return k, sn, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cl', default=f'{P4}/classical/real/cl_NPC_wv_all.npz')
    ap.add_argument('--net', required=True); ap.add_argument('--tag', default='v10')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    M, info = load(a); lg(f"rows: { {k: len(v['x']) for k, v in M.items()} }; registration {info['registration']}")
    # common footprint: classical kept 0.5-99.5 % box
    C = M['cl']; fb = (np.percentile(C['x'], 0.5), np.percentile(C['x'], 99.5), np.percentile(C['y'], 0.5), np.percentile(C['y'], 99.5))
    M = {k: sub(T, np.nonzero(SR.inrect(T['x'], T['y'], fb))[0]) for k, T in M.items()}
    cx, cy = 0.5 * (fb[0] + fb[1]), 0.5 * (fb[2] + fb[3]); frc_box = (cx - 5000, cx + 5000, cy - 5000, cy + 5000)
    P = find_pores(M['cl']); lg(f'{len(P)} pores on the classical published-style table')
    info['pore_zdrift_ptp_nm'] = {}
    for k, T in M.items():                                 # residual z drift from the pores, each table on its own (as fig_cl)
        d = pore_zdrift(T, P); T['z'] = T['z'] - d[np.clip(T['frame'] // 6 // 500, 0, len(d) - 1)]
        info['pore_zdrift_ptp_nm'][k] = float(np.ptp(d))
    lg(f"residual pore z drift removed (ptp nm): {info['pore_zdrift_ptp_nm']}")
    n = {k: len(T['x']) for k, T in M.items()}
    SC = {('net', 'combined sigma'): comb_sigma(M['net'], 'net'),
          ('net', 'lateral sigma'): np.hypot(M['net']['sx'], M['net']['sy']) * PX,
          ('net', 'photons N'): -M['net']['N'],
          ('net', 'combined sigma x chi2'): comb_sigma(M['net'], 'net') * np.sqrt(np.maximum(M['net']['chi2'] / M['net']['dof'], 1.0)),
          ('cl', 'combined sigma'): comb_sigma(M['cl'], 'cl'), ('clp', 'combined sigma'): comb_sigma(M['clp'], 'clp')}
    order = {k: np.argsort(np.where(np.isfinite(v), v, np.inf), kind='stable') for k, v in SC.items()}
    _G.update(M=M, P=P, frc_box=frc_box, order=order)
    counts_net = sorted(set([int(round(f * n['net'])) for f in FRACS] + [int(round(f * n['cl'])) for f in FRACS]), reverse=True)
    jobs = [(k, sn, c) for (k, sn) in SC for c in (counts_net if k == 'net' else [int(round(f * n[k])) for f in FRACS])]
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(16, mp_context=mp.get_context('fork')) as ex:
        res = list(ex.map(_job, jobs))
    curves = {}
    for k, sn, r in res: curves.setdefault(k, {}).setdefault(sn, []).append(r)
    for k in curves:
        for sn in curves[k]: curves[k][sn].sort(key=lambda r: -r['n_keep'])
    lg('curves done')
    # operating points
    on = order[('net', 'combined sigma')]
    OP = {}
    def op(name, T, nk=None):
        r = metrics(T, P, frc_box); r['n_keep'] = int(len(T['x']) if nk is None else nk); OP[name] = r
    op('network, all kept (good & N >= 1000)', M['net'])
    op('network, sel rows', sub(M['net'], np.nonzero(M['net']['sel'])[0]))
    n_m = n['cl']; sel_m = sub(M['net'], np.sort(on[:n_m])); op('network, matched count (combined sigma)', sel_m)
    # matched localizations per pore (ring locs, all pores)
    trP = cKDTree(P[:, :2]); dd, _ = trP.query(np.c_[M['net']['x'], M['net']['y']]); inpore = dd < 85
    dc, _ = trP.query(np.c_[M['cl']['x'], M['cl']['y']]); n_cl_pore = int((dc < 85).sum())
    cz = np.cumsum(inpore[on]); n_pp = int(np.searchsorted(cz, n_cl_pore) + 1)
    op('network, matched localizations per pore (combined sigma)', sub(M['net'], np.sort(on[:n_pp])))
    op('classical published-style', M['cl']); op('classical pixel MLE', M['clp'])
    for nm_, A_, ref in (('classical published, photon-matched to the matched network', M['cl'], sel_m['N']),
                         ('classical pixel MLE, photon-matched to the matched network', M['clp'], sel_m['N'])):
        pm, cover = photon_match(A_, ref, rng); op(nm_, sub(A_, pm)); OP[nm_]['ref_photon_mass_covered'] = cover
    # matched count, other network scores (ranking check)
    rank = {}
    for (k, sn), o in order.items():
        if k != 'net': continue
        r = metrics(sub(M['net'], np.sort(o[:n_m])), P, frc_box); r['n_keep'] = n_m; rank[sn] = r
    # sampling control: random subsets of the classical rows at the network's operating counts and fractions
    samp = []
    for c in sorted(set([int(round(f * n['cl'])) for f in FRACS] + [min(n_pp, n['cl'])])):
        for rep in range(2):
            r = metrics(sub(M['cl'], np.sort(rng.choice(n['cl'], c, replace=False))), P, frc_box, pore_only=True); r['n_keep'] = c; samp.append(r)
    for k_, r in OP.items():
        lg(f"OP {k_}: n {r['n_keep']} per pore {r['locs_per_pore']:.1f} radw {r.get('radial_width_nm', np.nan):.1f} R {r.get('ring_radius_median_nm', np.nan):.1f} "
           f"sep {r.get('two_ring_sep_em2_nm', np.nan):.1f} sd {r.get('ring_z_sd_nm', np.nan):.1f} dip {r.get('z_dip', np.nan):.2f} dafl {np.round(r['dafl'], 2).tolist()} "
           f"2-4k {np.round(r['dafl_2_4k'], 2).tolist()} hop {r['fringe_hop']:.4f} frc {r['frc']:.1f}")
    for sn, r in rank.items(): lg(f"rank {sn}: dafl {np.round(r['dafl'], 2).tolist()} radw {r.get('radial_width_nm', np.nan):.1f} sd {r.get('ring_z_sd_nm', np.nan):.1f} dip {r.get('z_dip', np.nan):.2f}")
    out = dict(tag=a.tag, info=info, footprint_nm=fb, frc_box_nm=frc_box, n_pores=int(len(P)), n_rows=n, n_cl_in_pores=n_cl_pore,
               n_net_matched_per_pore=n_pp, curves=curves, operating_points=OP, network_scores_at_matched_count=rank,
               classical_random_subsets=samp, photon_median=dict(net_all=float(np.median(M['net']['N'])), net_matched=float(np.median(sel_m['N'])),
                                                                 cl=float(np.median(M['cl']['N'])), clp=float(np.median(M['clp']['N']))),
               notes=['pores found once on the classical published-style table, fixed centres for all methods and levels',
                      'z: network zd (its own structure-based z drift); classical rows: the classical structure-based curve (pixel MLE: the published-style curve); plus the constant network-classical dz from matched emitters; then a residual per-block z drift from the fixed pores, each table on its own (fig_cl.npc_figures recipe)',
                      'FRC on the central 10 x 10 um, halves = alternating 500-window blocks'])
    fn = f'{OUTD}/NPC_selection_curves_{a.tag}'
    json.dump(out, open(fn + '.json', 'w'), indent=1, default=float)
    plot(out, fn + '.png')
    avgpore({'network, all kept': M['net'], f'network, matched count ({n_m:,})': sel_m, 'classical published-style': M['cl'], 'classical pixel MLE': M['clp']},
            P, f'{OUTD}/NPC_avgpore_selection_{a.tag}.png', out)
    lg(f'-> {fn}.png, {OUTD}/NPC_avgpore_selection_{a.tag}.png')


def plot(out, fn):
    C = out['curves']
    cols = dict(net='#2a78d6', cl='#eb6834', clp='#1baf7a')
    lines = [('net', 'combined sigma', '-', 2.0, 'network, combined sigma'), ('net', 'lateral sigma', ':', 1.2, 'network, lateral sigma'),
             ('net', 'photons N', '--', 1.2, 'network, photons N'), ('net', 'combined sigma x chi2', '-.', 1.2, 'network, combined sigma x chi2'),
             ('cl', 'combined sigma', '-', 2.0, 'classical published-style, combined sigma'), ('clp', 'combined sigma', '-', 2.0, 'classical pixel MLE, combined sigma')]
    panels = [('radial_width_nm', 'ring radial width (robust sd, nm)'), ('ring_radius_median_nm', 'ring radius, median (nm)'),
              ('two_ring_sep_em2_nm', 'two-ring z separation (nm)'), ('ring_z_sd_nm', 'per-ring z spread (sd, nm)'),
              ('z_dip', 'two-ring z dip (valley / peak)'), ('locs_per_pore', 'localizations per pore (25-85 nm)'),
              ('dafl_lat', 'DAFL lateral, mean x/y (nm)'), ('dafl_z', 'DAFL axial (nm)'), ('dafl_lat_24', 'DAFL lateral, 2–4k photons (nm)'),
              ('dafl_z_24', 'DAFL axial, 2–4k photons (nm)'), ('fringe_hop', 'fringe hops, |Δz| > 80 nm'), ('frc', 'FRC, central 10 µm (nm)')]
    def val(r, key):
        if key == 'dafl_lat': return np.nanmean(r['dafl'][:2]) if 'dafl' in r else np.nan
        if key == 'dafl_z': return r['dafl'][2] if 'dafl' in r else np.nan
        if key == 'dafl_lat_24': return np.nanmean(r['dafl_2_4k'][:2]) if 'dafl_2_4k' in r else np.nan
        if key == 'dafl_z_24': return r['dafl_2_4k'][2] if 'dafl_2_4k' in r else np.nan
        return r.get(key, np.nan)
    fig, axs = plt.subplots(3, 4, figsize=(15, 10.5))
    for ax, (key, lab) in zip(axs.ravel(), panels):
        for k, sn, ls, lw, lab_ in lines:
            c = C[k][sn]; ax.plot([r['n_keep'] for r in c], [val(r, key) for r in c], color=cols[k], ls=ls, lw=lw, marker='o' if ls == '-' else None, ms=3, label=lab_)
        S = out['classical_random_subsets']
        ax.plot([r['n_keep'] for r in S], [val(r, key) for r in S], 's', color=cols['cl'], mfc='none', ms=5, label='classical published, random subsets')
        for nm_, mk in (('network, all kept (good & N >= 1000)', 'o'), ('network, sel rows', 'P'), ('network, matched count (combined sigma)', '*'),
                        ('network, matched localizations per pore (combined sigma)', 'D'),
                        ('classical published, photon-matched to the matched network', 's'), ('classical pixel MLE, photon-matched to the matched network', '^')):
            r = out['operating_points'][nm_]
            ax.plot(r['n_keep'], val(r, key), mk, color='#0b0b0b', ms=8 if mk == '*' else 5, mfc='#0b0b0b' if mk == '*' else 'none', label=nm_)
        ax.set_xscale('log'); ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.set_title(lab, fontsize=8); ax.grid(alpha=0.25, lw=0.5); ax.set_xlabel('localizations in the field')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"Real NPC (Nup96), quality vs number kept, each method ranked by its own reported sigma; {out['info']['n_files']} files, "
                 f"{out['n_pores']} pores found once on the classical table", fontsize=9)
    plt.tight_layout(rect=(0, 0.11, 1, 0.97)); fig.savefig(fn, dpi=140); plt.close(fig)


def avgpore(D, P, fn, out):
    fig, axs = plt.subplots(len(D), 4, figsize=(13, 3.0 * len(D) + 0.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]))
    for i, (nm, T) in enumerate(D.items()):
        pm, V = pore_metrics(T, P)
        ax = axs[i, 0]; h = np.histogram2d(V['dy'], V['dx'], bins=np.arange(-100, 101, 2.5))[0]; h = gaussian_filter(h, 0.7)
        ax.imshow(h / np.percentile(h, 99.7), cmap='magma', vmin=0, vmax=1, extent=[-100, 100, 100, -100]); ax.set_ylabel(f'{nm}\n{len(V["dx"]):,} locs, {pm["locs_per_pore"]:.1f} / pore\ny (nm)', fontsize=8)
        ring = (V['r'] > 25) & (V['r'] < 85)
        ax = axs[i, 1]; s = np.abs(V['dy']) < 30
        h = gaussian_filter(np.histogram2d(V['dz'][s], V['dx'][s], bins=[np.arange(-100, 101, 2.5), np.arange(-100, 101, 2.5)])[0], 0.7)
        ax.imshow(h[::-1] / np.percentile(h, 99.7), cmap='magma', vmin=0, vmax=1, extent=[-100, 100, -100, 100]); ax.set_ylabel('z (nm)', fontsize=7)
        ax = axs[i, 2]
        h = gaussian_filter(np.histogram2d(V['dz'], V['r'], bins=[np.arange(-100, 101, 2.5), np.arange(0, 95.1, 2.5)])[0], 0.7)
        ax.imshow(h[::-1] / np.percentile(h, 99.7), cmap='magma', vmin=0, vmax=1, extent=[0, 95, -100, 100], aspect=95 / 200); ax.set_ylabel('z (nm)', fontsize=7)
        ax = axs[i, 3]
        hz, ez = np.histogram(V['dz'][ring], bins=np.arange(-100, 101, 2)); ax.plot(0.5 * (ez[1:] + ez[:-1]), gaussian_filter(hz.astype(float), 1.0) / max(hz.max(), 1), color='#2a78d6', lw=1.6)
        ax.set_ylim(0, 1.25); ax.grid(alpha=0.25, lw=0.5)
        ax.text(0.02, 0.98, f"sep {pm.get('two_ring_sep_em2_nm', np.nan):.0f} nm, sd {pm.get('ring_z_sd_nm', np.nan):.1f}\ndip {pm.get('z_dip', np.nan):.2f}; "
                f"radial w {pm.get('radial_width_nm', np.nan):.1f} nm", transform=ax.transAxes, va='top', fontsize=7)
        if i == len(D) - 1:
            axs[i, 0].set_xlabel('x (nm)'); axs[i, 1].set_xlabel('x (nm), |y| < 30 nm'); axs[i, 2].set_xlabel('radius (nm)'); axs[i, 3].set_xlabel('z about pore mean (nm), 25 < r < 85')
    for ax, t in zip(axs[0], ('averaged pore, x-y', 'x-z side view', 'radius-z', 'z histogram of the ring')): ax.set_title(t)
    fig.suptitle(f"Averaged NPC pore ({out['n_pores']} pores found on the classical table, same centres for all rows); 2.5 nm bins, same rendering per row "
                 f"(normalised to the 99.7th percentile)", fontsize=8)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(fn, dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
