"""Plain-4Pi SIMCODE on the public 4Pi-SMS Cell11 (Li & Ries, Zenodo 4022827) vs their published localizations:
counts, matched localizations, the z relation (scale / sign / offset of our z against theirs), precision (reported and
empirical from emitters on in consecutive frames), the Nup96 ring radius and the two-layer z separation (same site
finder on both tables), and a figure.

Their tables: Cell11_642_tmpresult_20200910.mat (all raw localizations: x, y px of their 168 x 168 registered crop,
frame, photons, CRLB) and Cell11_DCresult.mat (the published set, x / y / z nm; pr_out = 1-based row of tmpresult; no
drift shift applied: x_out = 128 x_result). Ours: <run>/process/locs_<file>.npz (setup fourpi_plain: x, y ch0 crop px,
z um, frame within the file). Their crop of channel 0 starts 15 camera columns right of ours; the remaining offset
(0 / 1-based, sub-pixel) is estimated from the localizations themselves.

  python analysis/fourpi_plain/compare_published.py <run dir> <out dir> [--files 0 1 2 3]
"""
import argparse, json, os
import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree

D = '/mnt/data/dmdsf/public_4pi_sms/zenodo_4022827/extracted/'
PX = 128.0
NF = 3000          # frames per file


def load_ours(run, files):
    rows = []
    for fi in files:
        f = os.path.join(run, 'process', f'locs_{fi:03d}.npz')
        if not os.path.exists(f): continue
        d = np.load(f)
        n = len(d['x'])
        r = dict(x=(d['x'] - 15.0) * PX, y=d['y'] * PX, z=d['z'] * 1000.0, sx=d['sx'] * PX, sy=d['sy'] * PX, sz=d['sz'] * 1000.0,
                 N=d['N'], t=d['frame'] + NF * fi, good=d['good'], chi2=d['chi2'] / d['dof'], nn=d['nn'], pdet=d['pdet'],
                 bg=d['bg'].sum(1) if d['bg'].ndim == 2 else d['bg'], file=np.full(n, fi),
                 dzref=(d['z'] - d['zref']) * 1000.0 if 'zref' in d.files else np.full(n, np.nan))
        rows.append(r)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def load_theirs():
    m = sio.loadmat(D + 'Cell11_642_tmpresult_20200910.mat', squeeze_me=True)
    d = sio.loadmat(D + 'Cell11_DCresult.mat', squeeze_me=True)
    raw = dict(x=m['xresult'] * PX, y=m['yresult'] * PX, t=m['tresult'].astype(int), N=m['Iresult'], sx=m['CRLBresult'][:, 0] * PX,
               sy=m['CRLBresult'][:, 1] * PX, bg=m['bgresult'], ll=m['llresult'])
    i = d['pr_out'].astype(int) - 1
    pub = dict(x=d['xout'].astype(float), y=d['yout'].astype(float), z=d['zout'].astype(float), t=d['t_out'].astype(int), N=m['Iresult'][i],
               sx=raw['sx'][i], sy=raw['sy'][i], zerr=d['z_err_out'].astype(float))
    return raw, pub


def sel(d, m):
    return {k: v[m] for k, v in d.items()}


def match(A, B, r, dxy=(0.0, 0.0)):
    """one-to-one nearest matches per frame within r (nm): -> ia, ib"""
    ia, ib = [], []
    tb = np.unique(B['t']); order_b = np.argsort(B['t'], kind='stable'); sb = np.searchsorted(B['t'][order_b], tb); sb = np.r_[sb, len(order_b)]
    pos = dict(zip(tb, range(len(tb))))
    order_a = np.argsort(A['t'], kind='stable')
    ta, sa = np.unique(A['t'][order_a], return_index=True); sa = np.r_[sa, len(order_a)]
    for j, t in enumerate(ta):
        if t not in pos: continue
        a = order_a[sa[j]:sa[j + 1]]; k = pos[t]; b = order_b[sb[k]:sb[k + 1]]
        pa = np.c_[A['x'][a] + dxy[0], A['y'][a] + dxy[1]]; pb = np.c_[B['x'][b], B['y'][b]]
        d, q = cKDTree(pb).query(pa, distance_upper_bound=r)
        ok = np.isfinite(d)
        if not ok.any(): continue
        # mutual nearest
        d2, q2 = cKDTree(pa).query(pb, distance_upper_bound=r)
        for i_ in np.nonzero(ok)[0]:
            if q2[q[i_]] == i_: ia.append(a[i_]); ib.append(b[q[i_]])
    return np.array(ia, int), np.array(ib, int)


def consecutive_pairs(d, r=60.0):
    """pairs of localizations in frames t, t+1 within r nm (the same emitter on in both frames)"""
    A = d; B = dict(d, t=d['t'] - 1)
    ia, ib = match(A, B, r)
    return ia, ib


def robust_sd(v):
    v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) > 5 else np.nan


def circle_fit(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]; b = x ** 2 + y ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c[0], c[1], np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 0))


def npc_sites(d, eps=40.0, min_pts=8, nmin=20, nmax=600):
    """NPC candidates: DBSCAN on xy, circle fit; keep clusters with ring radius 30 - 80 nm and a centre hole"""
    from sklearn.cluster import DBSCAN
    X = np.c_[d['x'], d['y']]
    lab = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)
    out = []
    for l in np.unique(lab[lab >= 0]):
        i = np.nonzero(lab == l)[0]
        if not (nmin <= len(i) <= nmax): continue
        cx, cy, R = circle_fit(d['x'][i], d['y'][i])
        rr = np.hypot(d['x'][i] - cx, d['y'][i] - cy)
        if not (30 < R < 80) or np.mean(rr < 20) > 0.1 or np.median(rr) > 90: continue
        out.append((cx, cy, R, len(i)))
    return np.array(out)


def site_stats(d, sites, rmax=110.0):
    """per site: locs within rmax of the centre (refined by a circle fit), radius (circle fit), z of the two layers
    (2-component Gaussian mixture, equal widths), and the pooled radial distances / centred z"""
    from sklearn.mixture import GaussianMixture
    tr = cKDTree(np.c_[d['x'], d['y']])
    S = []; rad = []; zc = []
    for cx, cy, _, _ in sites:
        i = np.array(tr.query_ball_point([cx, cy], rmax), int)
        if len(i) < 15: continue
        cx2, cy2, R = circle_fit(d['x'][i], d['y'][i])
        i = np.array(tr.query_ball_point([cx2, cy2], rmax), int)
        if len(i) < 15: continue
        rr = np.hypot(d['x'][i] - cx2, d['y'][i] - cy2)
        z = d['z'][i]
        g = GaussianMixture(2, covariance_type='tied', random_state=0).fit(z[:, None])
        mu = np.sort(g.means_[:, 0]); sep = mu[1] - mu[0]
        S.append(dict(x=cx2, y=cy2, R=R, Rmed=float(np.median(rr)), n=len(i), zsep=float(sep), zsd=float(np.sqrt(g.covariances_.ravel()[0])),
                      w=float(min(g.weights_)), zmid=float(mu.mean())))
        rad.append(rr); zc.append(z - mu.mean())
    return S, (np.concatenate(rad) if rad else np.zeros(0)), (np.concatenate(zc) if zc else np.zeros(0))


def two_gauss_sep(zc, rng=(-120, 120)):
    """pooled centred z: separation of a symmetric two-Gaussian fit (mu = +- s / 2, equal sd)"""
    from scipy.optimize import least_squares
    h, e = np.histogram(zc, bins=96, range=rng); c = 0.5 * (e[1:] + e[:-1])
    f = lambda p: p[0] * (np.exp(-0.5 * ((c - p[1] / 2) / p[2]) ** 2) + np.exp(-0.5 * ((c + p[1] / 2) / p[2]) ** 2)) - h
    p = least_squares(f, [h.max(), 50.0, 15.0], bounds=([0, 0, 2], [np.inf, 150, 80])).x
    return float(p[1]), float(p[2]), (c, h, p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('out'); ap.add_argument('--files', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--chi2_max', type=float, default=None); ap.add_argument('--sig_max', type=float, default=25.0, help='ours: max(sx, sy) nm')
    ap.add_argument('--nmin', type=float, default=500.0, help='photon cut of both sets (their published set: I > 500)')
    ap.add_argument('--zref_max', type=float, default=120.0, help='ours: max |z - structure reference z| (nm; 0: no cut)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    O = load_ours(a.run, a.files)
    raw, pub = load_theirs()
    tmax = NF * len(a.files)
    pub = sel(pub, pub['t'] < tmax); raw = sel(raw, raw['t'] < tmax)
    good = O['good'] if a.chi2_max is None else O['chi2'] < a.chi2_max
    keep = good & (np.maximum(O['sx'], O['sy']) < a.sig_max) & (O['N'] >= a.nmin)
    if a.zref_max > 0: keep &= ~(np.abs(O['dzref']) > a.zref_max)          # the structure contradicts the 4Pi period (NaN: kept)
    Os = sel(O, keep)
    R = dict(n_ours_all=int(len(O['x'])), n_ours_sel=int(keep.sum()), n_theirs_raw=int(len(raw['x'])), n_theirs_pub=int(len(pub['x'])),
             selection=dict(good='chi2/dof < fourpi.chi2_max' if a.chi2_max is None else f'chi2/dof < {a.chi2_max}', sig_max_nm=a.sig_max, nmin=a.nmin,
                            zref_max_nm=a.zref_max),
             frames=tmax)
    # ---- offset of our frame against theirs (lateral), then one-to-one matches within 100 nm
    ia, ib = match(Os, pub, 300.0)
    dxy = (float(np.median(pub['x'][ib] - Os['x'][ia])), float(np.median(pub['y'][ib] - Os['y'][ia])))
    ia, ib = match(Os, pub, 100.0, dxy)
    Os['x'] = Os['x'] + dxy[0]; Os['y'] = Os['y'] + dxy[1]; O['x'] = O['x'] + dxy[0]; O['y'] = O['y'] + dxy[1]
    ex, ey = pub['x'][ib] - Os['x'][ia], pub['y'][ib] - Os['y'][ia]
    R['match'] = dict(offset_nm=dxy, n=int(len(ia)), frac_of_pub=float(len(ib) / len(pub['x'])), frac_of_ours=float(len(ia) / len(Os['x'])),
                      dx_sd_nm=robust_sd(ex), dy_sd_nm=robust_sd(ey), photon_ratio_median=float(np.median(Os['N'][ia] / pub['N'][ib])))
    # ---- z: our z against theirs on the matched pairs (robust line), per 1000-frame block offsets
    zo, zt = Os['z'][ia], pub['z'][ib]
    from scipy.stats import theilslopes
    sub = np.random.default_rng(0).choice(len(zo), min(len(zo), 4000), replace=False)
    sl, ic, lo, hi = theilslopes(zt[sub], zo[sub])
    res = zt - (sl * zo + ic)
    blk = {}
    for b0 in range(0, tmax, 1000):
        s = (Os['t'][ia] >= b0) & (Os['t'][ia] < b0 + 1000)
        if s.sum() > 20: blk[b0] = float(np.median(zt[s] - sl * zo[s]))
    R['z_relation'] = dict(slope=float(sl), slope_ci95=[float(lo), float(hi)], offset_nm=float(ic), resid_sd_nm=robust_sd(res),
                           frac_resid_gt_100nm=float(np.mean(np.abs(res) > 100)), block_offsets_nm=blk,
                           note='their z = slope x our z + offset (nm); |slope| != 1: z scale difference, slope < 0: opposite z axis')
    # ---- precision: reported, and empirical from consecutive frames (same emitter)
    prec = {}
    for nm, d, cols in (('ours', Os, ('sx', 'sy', 'sz')), ('theirs', pub, ('sx', 'sy', None))):
        i, j = consecutive_pairs(d)
        Nm = np.sqrt(d['N'][i] * d['N'][j])
        bins = [500, 1000, 2000, 4000, 8000, 30000]
        rows = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            s = (Nm >= b0) & (Nm < b1)
            if s.sum() < 30: continue
            row = dict(N=[b0, b1], pairs=int(s.sum()), emp_x=robust_sd(d['x'][i][s] - d['x'][j][s]) / np.sqrt(2),
                       emp_y=robust_sd(d['y'][i][s] - d['y'][j][s]) / np.sqrt(2), emp_z=robust_sd(d['z'][i][s] - d['z'][j][s]) / np.sqrt(2),
                       rep_x=float(np.median(d[cols[0]][i][s])), rep_y=float(np.median(d[cols[1]][i][s])))
            if cols[2]: row['rep_z'] = float(np.median(d[cols[2]][i][s]))
            rows.append(row)
        prec[nm] = dict(pairs=int(len(i)), bins=rows, median_N=float(np.median(d['N'])),
                        all=dict(emp_x=robust_sd(d['x'][i] - d['x'][j]) / np.sqrt(2), emp_y=robust_sd(d['y'][i] - d['y'][j]) / np.sqrt(2),
                                 emp_z=robust_sd(d['z'][i] - d['z'][j]) / np.sqrt(2)))
    R['precision'] = prec
    # paired: the same emitters in the same consecutive frames in both sets (their consecutive pairs whose two
    # localizations both have a matched SIMCODE localization), binned by THEIR photons (our photon scale is 1.7 x theirs)
    m2 = dict(zip(ib.tolist(), ia.tolist()))
    pi, pj = consecutive_pairs(pub)
    ok = np.array([(u in m2) and (v in m2) for u, v in zip(pi, pj)], bool)
    pi, pj = pi[ok], pj[ok]; oi = np.array([m2[u] for u in pi], int); oj = np.array([m2[v] for v in pj], int)
    Nm = np.sqrt(pub['N'][pi] * pub['N'][pj]); sz = abs(R['z_relation']['slope'])
    rows = []
    for b0, b1 in zip([500, 1000, 2000, 4000, 8000], [1000, 2000, 4000, 8000, 30000]):
        s = (Nm >= b0) & (Nm < b1)
        if s.sum() < 30: continue
        rows.append(dict(N=[b0, b1], pairs=int(s.sum()),
                         ours=[robust_sd(Os[k][oi][s] - Os[k][oj][s]) / np.sqrt(2) * (sz if k == 'z' else 1) for k in 'xyz'],
                         theirs=[robust_sd(pub[k][pi][s] - pub[k][pj][s]) / np.sqrt(2) for k in 'xyz']))
    R['precision_paired'] = dict(pairs=int(len(pi)), bins=rows, note='robust sd / sqrt 2 of consecutive-frame differences on the same emitters; z in their units',
                                 all=dict(ours=[robust_sd(Os[k][oi] - Os[k][oj]) / np.sqrt(2) * (sz if k == 'z' else 1) for k in 'xyz'],
                                          theirs=[robust_sd(pub[k][pi] - pub[k][pj]) / np.sqrt(2) for k in 'xyz']))
    # matched pairs: precision of each set on the same emitters / frames (pairs of matched localizations in consecutive frames)
    # ---- NPC: sites found on each set with the same finder; stats on each set's own sites and on the common sites
    sites = {nm: npc_sites(d) for nm, d in (('ours', Os), ('theirs', pub))}
    npc = {}
    pooled = {}
    for nm, d in (('ours', Os), ('theirs', pub)):
        S, rad, zc = site_stats(d, sites[nm])
        sep, sd, fit = two_gauss_sep(zc)
        pooled[nm] = (rad, zc, fit, S)
        npc[nm] = dict(sites=len(S), R_median_nm=float(np.median([s['R'] for s in S])) if S else None,
                       Rmed_median_nm=float(np.median([s['Rmed'] for s in S])) if S else None,
                       zsep_site_median_nm=float(np.median([s['zsep'] for s in S])) if S else None,
                       zsep_pooled_nm=sep, zsd_pooled_nm=sd, locs_per_site=float(np.median([s['n'] for s in S])) if S else None)
    # common sites: both finders, centres within 40 nm
    if len(sites['ours']) and len(sites['theirs']):
        d_, q = cKDTree(sites['theirs'][:, :2]).query(sites['ours'][:, :2], distance_upper_bound=40.0)
        ok = np.isfinite(d_)
        common = sites['ours'][ok]
        for nm, d in (('ours', Os), ('theirs', pub)):
            S, rad, zc = site_stats(d, common)
            if nm == 'theirs': pooled['common_theirs'] = S
            sep, sd, _ = two_gauss_sep(zc)
            npc[nm]['common'] = dict(sites=len(S), R_median_nm=float(np.median([s['R'] for s in S])) if S else None,
                                     zsep_site_median_nm=float(np.median([s['zsep'] for s in S])) if S else None, zsep_pooled_nm=sep,
                                     zsd_pooled_nm=sd, locs_per_site=float(np.median([s['n'] for s in S])) if S else None)
    R['npc'] = npc
    json.dump(R, open(os.path.join(a.out, 'compare.json'), 'w'), indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))
    figure(a.out, Os, pub, R, pooled, sites)


C_OURS, C_THEIRS = '#2a78d6', '#eb6834'      # categorical slots 1 / 2 (validated pair)


def render(ax, d, ext, px=10.0, title='', zr=(0, 450), bar_nm=None):
    """density x z-colour render (turbo for z, brightness for density) of the localizations inside ext (nm)"""
    import matplotlib.cm as cm
    x0, x1, y0, y1 = ext
    nx, ny = max(1, int((x1 - x0) / px)), max(1, int((y1 - y0) / px))
    m = (d['x'] >= x0) & (d['x'] < x1) & (d['y'] >= y0) & (d['y'] < y1)
    H, _, _ = np.histogram2d(d['y'][m], d['x'][m], bins=(ny, nx), range=((y0, y1), (x0, x1)))
    Z, _, _ = np.histogram2d(d['y'][m], d['x'][m], bins=(ny, nx), range=((y0, y1), (x0, x1)), weights=d['z'][m])
    zz = np.where(H > 0, Z / np.maximum(H, 1), np.nan)
    v = np.clip(H / max(np.percentile(H[H > 0], 99.0), 1), 0, 1) ** 0.7 if (H > 0).any() else H
    rgb = cm.turbo(np.clip((np.nan_to_num(zz, nan=zr[0]) - zr[0]) / (zr[1] - zr[0]), 0, 1))[..., :3] * v[..., None]
    ax.imshow(rgb, extent=(0, (x1 - x0) / 1000, (y1 - y0) / 1000, 0), interpolation='nearest')
    ax.set_title(title, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    if bar_nm:
        L = bar_nm / 1000; W, Hh = (x1 - x0) / 1000, (y1 - y0) / 1000
        ax.plot([W * 0.95 - L, W * 0.95], [Hh * 0.93] * 2, color='w', lw=2)
        ax.text(W * 0.95 - L / 2, Hh * 0.89, f'{bar_nm:g} nm' if bar_nm < 1000 else f'{bar_nm / 1000:g} um', color='w', ha='center', fontsize=7)


def figure(out, Os, pub, R, pooled, sites):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    zl = R['z_relation']; Oz = dict(Os, z=zl['slope'] * Os['z'] + zl['offset_nm'])       # our z in their frame for the colour scale
    zr = (np.percentile(pub['z'], 2), np.percentile(pub['z'], 98))
    fig = plt.figure(figsize=(12.5, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1, 1], hspace=0.38, wspace=0.32)
    ext = (np.percentile(pub['x'], 0.3), np.percentile(pub['x'], 99.7), np.percentile(pub['y'], 0.3), np.percentile(pub['y'], 99.7))
    render(fig.add_subplot(gs[0, 0]), Oz, ext, 25, f'A  SIMCODE plain 4Pi ({R["n_ours_sel"]:,} locs)', zr, 2000)
    render(fig.add_subplot(gs[0, 1]), pub, ext, 25, f'B  published ({R["n_theirs_pub"]:,} locs)', zr, 2000)
    S_o = pooled.get('common_theirs') or pooled['theirs'][3]       # zooms: the published-set sites (common ones first) with most localizations
    sub = gs[0, 2:].subgridspec(2, 3, wspace=0.05, hspace=0.18)
    if S_o:
        top = sorted(S_o, key=lambda s: -s['n'])[:3]
        for k, s in enumerate(top):
            e2 = (s['x'] - 140, s['x'] + 140, s['y'] - 140, s['y'] + 140)
            render(fig.add_subplot(sub[0, k]), Oz, e2, 4, ('C  SIMCODE, ' if k == 0 else '') + f'site {k + 1}', zr, 100 if k == 0 else None)
            render(fig.add_subplot(sub[1, k]), pub, e2, 4, ('published, ' if k == 0 else '') + f'site {k + 1}', zr)
    lab = {'ours': 'SIMCODE', 'theirs': 'published'}; col = {'ours': C_OURS, 'theirs': C_THEIRS}
    ax = fig.add_subplot(gs[1, 0])
    for nm in ('ours', 'theirs'):
        ax.hist(pooled[nm][0], bins=55, range=(0, 110), histtype='step', lw=1.5, density=True, color=col[nm],
                label=f'{lab[nm]}: R {R["npc"][nm]["R_median_nm"]:.1f} nm ({R["npc"][nm]["sites"]} sites)')
    ax.axvline(53.7, color='0.4', lw=0.8, ls=':'); ax.set_xlabel('distance from site centre (nm)'); ax.set_ylabel('density')
    ax.legend(fontsize=7, frameon=False); ax.set_title('D  Nup96 ring radius', fontsize=8.5, loc='left')
    ax = fig.add_subplot(gs[1, 1])
    for nm in ('ours', 'theirs'):
        zc = pooled[nm][1] * (abs(zl['slope']) if nm == 'ours' else 1.0)
        sep = R['npc'][nm]['zsep_pooled_nm'] * (abs(zl['slope']) if nm == 'ours' else 1.0)
        ax.hist(zc, bins=60, range=(-120, 120), histtype='step', lw=1.5, density=True, color=col[nm], label=f'{lab[nm]}: {sep:.1f} nm')
    ax.set_xlabel('z - site centre (nm, published z units)'); ax.legend(fontsize=7, frameon=False)
    ax.set_title('E  two layers (pooled fit)', fontsize=8.5, loc='left')
    ax = fig.add_subplot(gs[1, 2])
    i, j = match(Os, pub, 100.0)
    ax.hexbin(Os['z'][i], pub['z'][j], gridsize=60, bins='log', cmap='Greys', mincnt=1)
    zz = np.linspace(np.percentile(Os['z'][i], 1), np.percentile(Os['z'][i], 99), 10); ax.plot(zz, zl['slope'] * zz + zl['offset_nm'], color=C_OURS, lw=1.2)
    ax.set_xlabel('SIMCODE z (nm, model z)'); ax.set_ylabel('published z (nm)')
    ax.set_title(f'F  z, matched locs (slope {zl["slope"]:.3f})', fontsize=8.5, loc='left')
    PP = R['precision_paired']['bins']; Nc = [np.sqrt(b['N'][0] * b['N'][1]) for b in PP]
    ax = fig.add_subplot(gs[1, 3])
    for nm in ('ours', 'theirs'):
        ax.plot(Nc, [b[nm][0] for b in PP], 'o-', color=col[nm], ms=4, label=f'{lab[nm]} x')
        ax.plot(Nc, [b[nm][1] for b in PP], 's--', color=col[nm], ms=4, mfc='none', label=f'{lab[nm]} y')
    ax.set_xscale('log'); ax.set_xlabel('photons (published scale)'); ax.set_ylabel('nm'); ax.legend(fontsize=7, frameon=False)
    ax.set_title('G  lateral precision, same emitters', fontsize=8.5, loc='left')
    ax = fig.add_subplot(gs[2, 0])
    for nm in ('ours', 'theirs'):
        ax.plot(Nc, [b[nm][2] for b in PP], 'o-', color=col[nm], ms=4, label=lab[nm])
    ax.set_xscale('log'); ax.set_xlabel('photons (published scale)'); ax.set_ylabel('nm (published z units)'); ax.legend(fontsize=7, frameon=False)
    ax.set_title('H  z precision, same emitters', fontsize=8.5, loc='left')
    ax = fig.add_subplot(gs[2, 1])
    for nm, d in (('ours', Os), ('theirs', pub)):
        ax.hist(np.log10(np.maximum(d['N'], 1)), bins=60, range=(2.5, 4.6), histtype='step', lw=1.5, color=col[nm], label=lab[nm])
    ax.set_xlabel('log10 photons per localization'); ax.set_ylabel('localizations'); ax.legend(fontsize=7, frameon=False)
    ax.set_title('I  photons (selected sets)', fontsize=8.5, loc='left')
    ax = fig.add_subplot(gs[2, 2:]); ax.axis('off')
    M = R['match']; N_ = R['npc']; P = R['precision']
    txt = [f'frames {R["frames"]:,}; SIMCODE {R["n_ours_all"]:,} solved, {R["n_ours_sel"]:,} selected; published {R["n_theirs_pub"]:,} (their raw {R["n_theirs_raw"]:,})',
           f'selection: N >= {R["selection"]["nmin"]:.0f} photons, max(sx, sy) < {R["selection"]["sig_max_nm"]:.0f} nm, |z - zref| < {R["selection"]["zref_max_nm"]:.0f} nm',
           f'matched (<= 100 nm, same frame): {M["n"]:,} = {100 * M["frac_of_pub"]:.1f} % of published, {100 * M["frac_of_ours"]:.1f} % of SIMCODE',
           f'z: published = {zl["slope"]:.3f} x SIMCODE {zl["offset_nm"]:+.0f} nm (resid sd {zl["resid_sd_nm"]:.1f} nm)',
           f'photons: SIMCODE / published on matched localizations {M["photon_ratio_median"]:.2f} (median)',
           f'precision, same emitters in consecutive frames ({R["precision_paired"]["pairs"]:,} pairs), x / y / z nm:',
           '   SIMCODE   ' + ' / '.join(f'{v:.1f}' for v in R['precision_paired']['all']['ours']),
           '   published ' + ' / '.join(f'{v:.1f}' for v in R['precision_paired']['all']['theirs']),
           f'precision, each full set (all its own pairs), x / y / z nm: SIMCODE {P["ours"]["all"]["emp_x"]:.1f} / {P["ours"]["all"]["emp_y"]:.1f} / {P["ours"]["all"]["emp_z"] * abs(zl["slope"]):.1f},'
           f' published {P["theirs"]["all"]["emp_x"]:.1f} / {P["theirs"]["all"]["emp_y"]:.1f} / {P["theirs"]["all"]["emp_z"]:.1f}',
           f'NPC sites: SIMCODE {N_["ours"]["sites"]}, published {N_["theirs"]["sites"]}; ring radius {N_["ours"]["R_median_nm"]:.1f} / {N_["theirs"]["R_median_nm"]:.1f} nm',
           f'two-layer separation (pooled): {N_["ours"]["zsep_pooled_nm"] * abs(zl["slope"]):.1f} / {N_["theirs"]["zsep_pooled_nm"]:.1f} nm (published z units)']
    if 'common' in N_['ours']:
        c_o, c_t = N_['ours']['common'], N_['theirs']['common']
        txt.append(f'common sites ({c_o["sites"]}): R {c_o["R_median_nm"]:.1f} / {c_t["R_median_nm"]:.1f} nm, '
                   f'z sep {c_o["zsep_pooled_nm"] * abs(zl["slope"]):.1f} / {c_t["zsep_pooled_nm"]:.1f} nm, locs per site {c_o["locs_per_site"]:.0f} / {c_t["locs_per_site"]:.0f}')
    ax.text(0, 1, '\n'.join(txt), va='top', fontsize=7.3, family='monospace')
    fig.suptitle('Plain 4Pi (4Pi-SMS), Nup96-AF647, Li & Ries Cell11: SIMCODE (network read-outs + explicit solve) vs published localizations', fontsize=10)
    fig.savefig(os.path.join(out, 'compare_cell11.png'), dpi=140, bbox_inches='tight')


if __name__ == '__main__':
    main()
