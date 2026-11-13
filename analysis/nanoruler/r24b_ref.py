"""Reference tables of the 40 nm / 20 nm nanoruler movies (nr40, nr20) in the common analysis format (CPU only).

Common frame = frame of the quick fit on the reordered cache (pixel centres at integer pixel coordinates), nm = px x 65.
Table format = nra_lib (npz: frame = 6-frame CYCLE index, x_nm, y_nm, sx_nm, sy_nm = published lpx / lpy x 65, photons,
crlb_x_nm, crlb_y_nm, bg per px per 6-frame block).

  python analysis/nanoruler/r24b_ref.py nr40|nr20

Steps (all results also in <out>/ref.json, text in <out>/ref_report.txt):
  1. shift check: quick 6-frame-sum fit (rows with frame % 6 == 0, cycle = frame // 6, first 5000 cycles) against every
     published stage (mutual nearest neighbours in the same cycle within --rmatch px): median (quick - published), robust sd,
     medians per 500 cycles. The pixel-convention shift is the one of the NON-drift-corrected stage.
  2. drift files: drift.txt (RCC) and the second-stage Picasso files against the table differences
     (convention tested: corrected = raw - drift[cycle]); cluster sharpness of raw / raw - drift / raw + drift.
  3. tables: tab_pub_{smlm,simflux}.npz (final), tab_pub_{smlm,simflux}_nodc.npz, drift_rcc.npy, drift_pub_total_{smlm,simflux}.npy
     ([n_cycles, 2] px, x then y; corrected = raw - drift[cycle]).
  4. picks -> picks_*.npz (x_nm, y_nm in the frame of the FINAL tables, n = localizations of the method's final table within
     the pick radius), counts per pick in every stage, pick-pair distances.
  5. quick-look of the final tables.
Outputs: out/nanoruler/<movie>/ref/."""
import os, sys, json, re, base64, struct, argparse
import numpy as np, h5py
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.join(REPO, 'analysis', 'nanoruler'))
import nra_lib as L

PX = 65.0
B = '/mnt/data/SIMFLUX/data/fig2/'
SETS = dict(
    nr40=dict(pub=B + '40nm-panel-q-r/results/3Phase_3_MMStack_Pos0.ome/', ncyc=6666,
              final=dict(smlm='g2d.hdf5', simflux='simflux.hdf5'),
              stage2=dict(smlm='g2d_190824_202451_drift.txt', simflux='simflux_190824_202454_drift.txt'),
              render=dict(smlm='g2d_render.hdf5'),
              picks=dict(g2d_init=('g2d_picksinit.yaml', 'smlm'), simflux_init=('simflux_picksinit.yaml', 'simflux'),
                         simflux_1=('simflux_picks1.yaml', 'simflux'))),
    nr20=dict(pub=B + '20nm-panel-s-t/results/1_2_MMStack_Pos0.ome/', ncyc=16666,
              final=dict(smlm='g2d_render.hdf5', simflux='simflux_render.hdf5'),
              stage2=dict(smlm='g2d_190823_155542_drift.txt', simflux='simflux_190823_154951_drift.txt'),
              render=dict(smlm='g2d_render.hdf5', simflux='simflux_render.hdf5'),
              picks=dict(g2d=('g2d_picksfull.yaml', 'smlm'), simflux=('simflux_picksfull.yaml', 'simflux'))))
NODC = dict(smlm='g2d-nodriftc.hdf5', simflux='simflux-nodriftc.hdf5')
RCC = dict(smlm='g2d.hdf5', simflux='simflux.hdf5')


def h5(fn): return h5py.File(fn, 'r')['locs'][:]
def pct(v, q=(5, 25, 50, 75, 95)): return [float(x) for x in np.percentile(v, q)]


def load_picks(p):
    """pick yaml of Picasso Render; numpy scalars are stored as tagged binary and are decoded without unpickling."""
    t = open(p).read()
    vals = [struct.unpack('<d', base64.b64decode(m))[0] for m in re.findall(r'!!binary \|\s+(\S+=)\s', t) if len(base64.b64decode(m)) == 8]
    y = dict(centers=np.array(vals).reshape(-1, 2))
    for k in ('Diameter', 'Shape'):
        m = re.search(r'^%s: (.*)$' % k, t, re.M)
        if m: y[k] = m.group(1)
    return y


def mmatch(ax, ay, ac, px_, py_, pc, r):
    A = np.c_[ax, ay, ac * 1e4]; P = np.c_[px_, py_, pc * 1e4]
    da, ja = cKDTree(P).query(A, distance_upper_bound=r); dp, jp = cKDTree(A).query(P, distance_upper_bound=r)
    i = np.nonzero(np.isfinite(da))[0]; i = i[jp[ja[i]] == i]
    return i, ja[i]


def sharp(x, y, px=4.0):
    n = int(400 * PX / px); h = np.histogram2d(y * PX / px, x * PX / px, bins=n, range=[[0, n], [0, n]])[0]
    return float((gaussian_filter(h, 1.0) ** 2).sum() / len(x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('movie', choices=list(SETS)); ap.add_argument('--rmatch', type=float, default=0.6)
    a = ap.parse_args(); S = SETS[a.movie]; pub = S['pub']; ncyc = S['ncyc']
    out = f'{REPO}/out/nanoruler/{a.movie}/ref/'; os.makedirs(out, exist_ok=True)
    R = dict(movie=a.movie, published_dir=pub, n_cycles=ncyc); o = []
    def p(s): o.append(s); print(s, flush=True)

    T = {}
    for m in ('smlm', 'simflux'):
        T[m, 'nodc'] = h5(pub + NODC[m]); T[m, 'rcc'] = h5(pub + RCC[m]); T[m, 'final'] = h5(pub + S['final'][m])
        if m in S['render']: T[m, 'render'] = h5(pub + S['render'][m])
        for k in ('rcc', 'final', 'render'):
            if (m, k) in T: assert len(T[m, k]) == len(T[m, 'nodc']) and (T[m, k]['frame'] == T[m, 'nodc']['frame']).all()
        f = T[m, 'nodc']['frame']; assert f.min() >= 0 and f.max() < ncyc
    p(f'{a.movie}: published tables in {pub}')
    p(f'  final tables: smlm = {S["final"]["smlm"]}, simflux = {S["final"]["simflux"]}; rows smlm {len(T["smlm", "nodc"])}, simflux {len(T["simflux", "nodc"])}; '
      f'frame = cycle index 0..{ncyc - 1} (max in tables {int(T["smlm", "nodc"]["frame"].max())})')

    # ---- 1. shift against the quick fit
    q = h5(f'{REPO}/out/nanoruler/insitu_{a.movie}/quick_w6_good.hdf5'); q = q[q['frame'] % 6 == 0]
    qc = (q['frame'] // 6).astype(np.int64); qx = q['x'].astype(float); qy = q['y'].astype(float)
    p(f'1. shift check: quick fit, cycle-aligned rows {len(q)} in cycles {qc.min()}..{qc.max()}; mutual nearest neighbours within {a.rmatch} px, same cycle')
    R['shift'] = {}
    for (m, st), t in T.items():
        if st == 'render' and S['final'][m] == S['render'][m]: continue
        g = t['frame'] <= qc.max()
        i, j = mmatch(qx, qy, qc, t['x'][g].astype(float), t['y'][g].astype(float), t['frame'][g].astype(np.int64), a.rmatch)
        dx = qx[i] - t['x'][g][j]; dy = qy[i] - t['y'][g][j]; tb = np.minimum(qc[i] // 500, 9)
        r = dict(n_matched=int(len(i)), n_published_in_range=int(g.sum()), median_quick_minus_pub_px=[float(np.median(dx)), float(np.median(dy))],
                 rsd_px=[L.rsd(dx), L.rsd(dy)], sem_px=[L.rsd(dx) / np.sqrt(len(i)), L.rsd(dy) / np.sqrt(len(i))],
                 per500_dx=[float(np.median(dx[tb == k])) for k in range(10)], per500_dy=[float(np.median(dy[tb == k])) for k in range(10)])
        R['shift'][f'{m}_{st}'] = r
        p(f'  {m:7s} {st:6s}: matched {len(i):6d} of {g.sum():6d}; quick - published median {np.median(dx):+.4f} {np.median(dy):+.4f} px '
          f'({np.median(dx) * PX:+.2f} {np.median(dy) * PX:+.2f} nm), robust sd {L.rsd(dx):.4f} {L.rsd(dy):.4f} px ({L.rsd(dx) * PX:.2f} {L.rsd(dy) * PX:.2f} nm)')
        p(f'      medians per 500 cycles dx {np.round(r["per500_dx"], 3).tolist()}')
        p(f'                             dy {np.round(r["per500_dy"], 3).tolist()}')
    sh = np.array(R['shift']['smlm_nodc']['median_quick_minus_pub_px'])
    assert np.abs(sh).max() < 0.05, sh
    R['shift_applied_px'] = [0.0, 0.0]
    p(f'  pixel-convention shift published -> common frame (non-drift-corrected g2d): {sh[0]:+.4f} {sh[1]:+.4f} px = {sh[0] * PX:+.2f} {sh[1] * PX:+.2f} nm; applied: 0')

    # ---- 2. drift files and convention: corrected = raw - drift[cycle]
    d_rcc = np.loadtxt(pub + 'drift.txt'); assert d_rcc.shape == (ncyc, 2), d_rcc.shape
    d2 = {m: np.loadtxt(pub + S['stage2'][m]) for m in ('smlm', 'simflux')}
    for m in d2: assert d2[m].shape == (ncyc, 2), d2[m].shape
    p('2. drift (convention tested: corrected = raw - drift[cycle], columns x, y, px)')
    R['drift'] = dict(rcc_range_px=[[float(d_rcc[:, k].min()), float(d_rcc[:, k].max())] for k in (0, 1)])
    tot = {}
    for m in ('smlm', 'simflux'):
        f = T[m, 'nodc']['frame'].astype(int)
        def diff(k): return np.c_[T[m, 'nodc']['x'].astype(float) - T[m, k]['x'], T[m, 'nodc']['y'].astype(float) - T[m, k]['y']]
        e_rcc = diff('rcc') - d_rcc[f]
        r = dict(max_abs_err_rcc_minus_px=float(np.abs(e_rcc).max()), max_abs_err_rcc_plus_px=float(np.abs(diff('rcc') + d_rcc[f]).max()))
        p(f'  {m}: nodriftc - {RCC[m]} = +drift.txt[cycle]: max |err| {r["max_abs_err_rcc_minus_px"]:.2e} px (opposite sign: {r["max_abs_err_rcc_plus_px"]:.3f} px)')
        if (m, 'render') in T:
            e2 = np.c_[T[m, 'rcc']['x'].astype(float) - T[m, 'render']['x'], T[m, 'rcc']['y'].astype(float) - T[m, 'render']['y']]
            r['max_abs_err_stage2_minus_px'] = float(np.abs(e2 - d2[m][f]).max()); r['max_abs_err_stage2_plus_px'] = float(np.abs(e2 + d2[m][f]).max())
            r['stage2_range_px'] = [[float(d2[m][:, k].min()), float(d2[m][:, k].max())] for k in (0, 1)]
            p(f'      {RCC[m]} - {S["render"][m]} = +{S["stage2"][m]}[cycle]: max |err| {r["max_abs_err_stage2_minus_px"]:.2e} px (opposite sign: {r["max_abs_err_stage2_plus_px"]:.3f} px); '
              f'stage-2 range x {r["stage2_range_px"][0]}, y {r["stage2_range_px"][1]}')
        # total published correction per cycle from the tables themselves (difference nodriftc -> final), all cycles
        D = diff('final'); n = np.bincount(f, minlength=ncyc)
        t_ = np.full((ncyc, 2), np.nan)
        for k in (0, 1):
            s1 = np.bincount(f, D[:, k], minlength=ncyc); t_[n > 0, k] = s1[n > 0] / n[n > 0]
        spread = float(np.abs(D - t_[f]).max())
        model = d_rcc + (d2[m] if S['final'][m] == S['render'].get(m) else 0)
        r['n_cycles_without_locs'] = int((n == 0).sum()); r['total_within_cycle_spread_px'] = spread
        r['total_vs_files_max_abs_px'] = float(np.nanmax(np.abs(t_ - model)))
        t_[n == 0] = model[n == 0]; tot[m] = t_
        r['total_range_px'] = [[float(t_[:, k].min()), float(t_[:, k].max())] for k in (0, 1)]
        r['total_first_last_px'] = [t_[0].tolist(), t_[-1].tolist()]
        p(f'      total published correction (nodriftc - final, per cycle): within-cycle spread {spread:.2e} px, vs sum of the drift files max |diff| {r["total_vs_files_max_abs_px"]:.2e} px, '
          f'cycles without localizations {r["n_cycles_without_locs"]} (filled from the files); range x {np.round(r["total_range_px"][0], 4).tolist()} y {np.round(r["total_range_px"][1], 4).tolist()} px; '
          f'first {np.round(t_[0], 4).tolist()} last {np.round(t_[-1], 4).tolist()}')
        # sharpness
        x0 = T[m, 'nodc']['x'].astype(float); y0 = T[m, 'nodc']['y'].astype(float)
        sp = {'raw': sharp(x0, y0), 'raw - drift_rcc': sharp(x0 - d_rcc[f, 0], y0 - d_rcc[f, 1]), 'raw + drift_rcc': sharp(x0 + d_rcc[f, 0], y0 + d_rcc[f, 1]),
              'raw - total': sharp(x0 - t_[f, 0], y0 - t_[f, 1]), 'raw + total': sharp(x0 + t_[f, 0], y0 + t_[f, 1])}
        if m in S['render'] and S['final'][m] != S['render'][m]:
            sp['raw - rcc - stage2 (render table, not final)'] = sharp(T[m, 'render']['x'].astype(float), T[m, 'render']['y'].astype(float))
        r['sharpness'] = sp
        p('      cluster sharpness (4 nm bins, sum of squared blurred counts / n): ' + ', '.join(f'{k} {v:.3f}' for k, v in sp.items()))
        R['drift'][m] = r
    # sharpness of the independent quick fit with the published drift (first 5000 cycles)
    sq = {'raw': sharp(qx, qy)}
    for nm, d in (('drift_rcc', d_rcc), ('total_smlm', tot['smlm'])):
        for s in (-1, 1): sq[f'raw {"-" if s < 0 else "+"} {nm}'] = sharp(qx + s * d[qc, 0], qy + s * d[qc, 1])
    R['drift']['sharpness_quick_fit'] = sq
    p('  quick fit (independent of the published tables, first 5000 cycles): ' + ', '.join(f'{k} {v:.3f}' for k, v in sq.items()))
    np.save(out + 'drift_rcc.npy', d_rcc)
    np.save(out + 'drift_pub_total_smlm.npy', tot['smlm']); np.save(out + 'drift_pub_total_simflux.npy', tot['simflux'])
    p(f'  drift over the movie: RCC x {np.round(R["drift"]["rcc_range_px"][0], 3).tolist()} y {np.round(R["drift"]["rcc_range_px"][1], 3).tolist()} px')

    # ---- 3. tables
    R['tables'] = {}
    for m in ('smlm', 'simflux'):
        for st, sfx in (('final', ''), ('nodc', '_nodc')):
            t = T[m, st]; fn = out + f'tab_pub_{m}{sfx}.npz'
            L.save_table(fn, frame=t['frame'].astype(int), x_nm=t['x'].astype(float) * PX, y_nm=t['y'].astype(float) * PX, sx_nm=t['lpx'] * PX, sy_nm=t['lpy'] * PX,
                         photons=t['photons'], crlb_x_nm=t['lpx'] * PX, crlb_y_nm=t['lpy'] * PX, bg=t['bg'])
            R['tables'][os.path.basename(fn)] = int(len(t))
    # check: nodc table - drift = final table
    for m in ('smlm', 'simflux'):
        a_ = np.load(out + f'tab_pub_{m}_nodc.npz'); b_ = np.load(out + f'tab_pub_{m}.npz'); d = np.load(out + f'drift_pub_total_{m}.npy')
        e = max(np.abs(a_['x_nm'] - d[a_['frame'], 0] * PX - b_['x_nm']).max(), np.abs(a_['y_nm'] - d[a_['frame'], 1] * PX - b_['y_nm']).max())
        R['tables'][f'check_{m}_nodc_minus_total_drift_vs_final_max_nm'] = float(e)
        p(f'3. written tab_pub_{m}.npz / _nodc.npz ({len(a_["frame"])} rows); nodc - drift_pub_total_{m}[cycle] x 65 = final: max |err| {e:.2e} nm')

    # ---- 4. picks
    R['picks'] = {}
    p('4. picks (centres as stored = camera px of the table; x_nm = x px x 65)')
    for nm, (fn, m) in S['picks'].items():
        y = load_picks(pub + fn); c = y['centers']; rad = float(y['Diameter']) / 2
        cnt = {}
        for st in ('final', 'rcc', 'nodc', 'render'):
            if (m, st) not in T: continue
            t = T[m, st]; tr = cKDTree(np.c_[t['x'], t['y']])
            cnt[st] = np.array([len(v) for v in tr.query_ball_point(c, rad)])
        other = 'simflux' if m == 'smlm' else 'smlm'
        t = T[other, 'final']; cnt_other = np.array([len(v) for v in cKDTree(np.c_[t['x'], t['y']]).query_ball_point(c, rad)])
        t = T[m, 'final']; tr = cKDTree(np.c_[t['x'], t['y']]); nb = tr.query_ball_point(c, rad)
        # centre check: offset of the mean of the localizations inside the pick from the pick centre; spread inside
        off = np.array([[t['x'][v].mean() - c[k, 0], t['y'][v].mean() - c[k, 1]] if len(v) else [np.nan, np.nan] for k, v in enumerate(nb)])
        # spread within 2 radii (does the pick hold one site or more?)
        nb2 = tr.query_ball_point(c, 2 * rad); n2 = np.array([len(v) for v in nb2])
        np.savez(out + f'picks_{nm}.npz', x_nm=c[:, 0] * PX, y_nm=c[:, 1] * PX, n=cnt['final'], n_other_method=cnt_other, radius_nm=rad * PX,
                 method=m, source=fn)
        r = dict(file=fn, n=int(len(c)), diameter_px=2 * rad, radius_nm=rad * PX, method_table=m,
                 counts={st: dict(sum=int(v.sum()), percentiles_5_25_50_75_95=pct(v), mean=float(v.mean()), min=int(v.min()), max=int(v.max()), n_empty=int((v == 0).sum())) for st, v in cnt.items()},
                 counts_other_method_final=dict(sum=int(cnt_other.sum()), mean=float(cnt_other.mean()), median=float(np.median(cnt_other))),
                 frac_of_final_table_in_picks=float(cnt['final'].sum() / len(t)),
                 centre_offset_px=dict(median=[float(np.nanmedian(off[:, 0])), float(np.nanmedian(off[:, 1]))], rsd=[L.rsd(off[np.isfinite(off[:, 0]), 0]), L.rsd(off[np.isfinite(off[:, 1]), 1])]),
                 count_within_2_radii_over_within_radius_median=float(np.median(n2[cnt['final'] > 0] / cnt['final'][cnt['final'] > 0])),
                 x_range_px=[float(c[:, 0].min()), float(c[:, 0].max())], y_range_px=[float(c[:, 1].min()), float(c[:, 1].max())])
        p(f'  {nm}: {fn}, {len(c)} centres, diameter {2 * rad} px (radius {rad * PX:.2f} nm), table {m}')
        for st, v in cnt.items():
            p(f'      locs within the radius, {st:6s} table: sum {v.sum():6d}, mean {v.mean():.1f}, percentiles 5/25/50/75/95 {np.round(pct(v), 1).tolist()}, min {v.min()}, max {v.max()}, empty picks {(v == 0).sum()}')
        p(f'      other method ({other}) final table: mean {cnt_other.mean():.1f}, median {np.median(cnt_other):.0f}; fraction of the {m} final table inside picks {r["frac_of_final_table_in_picks"]:.3f}')
        p(f'      mean of the localizations in the pick - pick centre: median {np.round(r["centre_offset_px"]["median"], 4).tolist()} px, robust sd {np.round(r["centre_offset_px"]["rsd"], 4).tolist()} px; '
          f'count within 2 radii / within the radius, median {r["count_within_2_radii_over_within_radius_median"]:.2f}')
        if len(c) > 20:
            cn = c * PX; tr = cKDTree(cn)
            d1, j1 = tr.query(cn, k=2); nn = d1[:, 1]
            pr = tr.query_pairs(60.0, output_type='ndarray'); dd = np.linalg.norm(cn[pr[:, 0]] - cn[pr[:, 1]], axis=1)
            g = np.abs(dd - 20) <= 8; pg = pr[g]; dg = dd[g]
            deg = np.bincount(pg.ravel(), minlength=len(c))
            # connected groups of picks linked by 12..28 nm pairs
            lab = np.arange(len(c))
            for it in range(20):
                new = lab.copy(); mn = np.minimum(lab[pg[:, 0]], lab[pg[:, 1]]); np.minimum.at(new, pg[:, 0], mn); np.minimum.at(new, pg[:, 1], mn)
                if np.array_equal(new, lab): break
                lab = new
            gs = np.bincount(np.unique(lab, return_inverse=True)[1]);
            hist, edges = np.histogram(dd, bins=np.arange(0, 62, 2))
            ang = np.rad2deg(np.arctan2(cn[pg[:, 1], 1] - cn[pg[:, 0], 1], cn[pg[:, 1], 0] - cn[pg[:, 0], 0])) % 180
            # second-neighbour pairs (40 +- 8 nm) among picks: 3 picked sites of one ruler
            g40 = np.abs(dd - 40) <= 8
            r['pairs'] = dict(nearest_neighbour_nm_percentiles_5_25_50_75_95=pct(nn), n_nn_lt_12=int((nn < 12).sum()), n_nn_12_28=int((np.abs(nn - 20) <= 8).sum()), n_nn_gt_28=int((nn > 28).sum()),
                              n_pairs_12_28=int(g.sum()), n_picks_in_a_pair=int((deg > 0).sum()), n_picks_degree=np.bincount(deg).tolist(),
                              group_sizes_count=np.bincount(gs).tolist(),
                              pair_dist_nm=dict(mean=float(dg.mean()), median=float(np.median(dg)), sd=float(dg.std(ddof=1)), rsd=L.rsd(dg), sem=float(dg.std(ddof=1) / np.sqrt(len(dg))), percentiles_5_25_50_75_95=pct(dg)),
                              n_pairs_32_48=int(g40.sum()), pair_dist_40_nm=dict(mean=float(dd[g40].mean()), median=float(np.median(dd[g40])), sd=float(dd[g40].std(ddof=1))) if g40.sum() > 1 else None,
                              hist_all_pairs_0_60_nm_bin2=dict(edges=edges.tolist(), counts=hist.tolist()), pair_angle_deg_hist_bin30=np.histogram(ang, bins=np.arange(0, 181, 30))[0].tolist())
            q_ = r['pairs']
            p(f'      nearest-neighbour pick distance nm, percentiles 5/25/50/75/95 {np.round(q_["nearest_neighbour_nm_percentiles_5_25_50_75_95"], 1).tolist()}; nearest < 12 nm: {q_["n_nn_lt_12"]}, 12-28 nm: {q_["n_nn_12_28"]}, > 28 nm: {q_["n_nn_gt_28"]}')
            p(f'      pick pairs at 20 +- 8 nm: {q_["n_pairs_12_28"]} pairs, {q_["n_picks_in_a_pair"]} picks in at least one pair (picks with 0, 1, 2, ... partners: {q_["n_picks_degree"]}); '
              f'linked groups by size (index = number of picks): {q_["group_sizes_count"]}')
            d_ = q_['pair_dist_nm']
            p(f'      distance of those pairs: mean {d_["mean"]:.2f} nm, median {d_["median"]:.2f}, sd {d_["sd"]:.2f}, robust sd {d_["rsd"]:.2f}, sem {d_["sem"]:.2f}, percentiles {np.round(d_["percentiles_5_25_50_75_95"], 1).tolist()}')
            p(f'      pairs at 40 +- 8 nm: {q_["n_pairs_32_48"]}' + (f' (mean {q_["pair_dist_40_nm"]["mean"]:.2f}, sd {q_["pair_dist_40_nm"]["sd"]:.2f} nm)' if q_['pair_dist_40_nm'] else ''))
            p(f'      all pick-pair distances, 2 nm bins 0..60 nm: {hist.tolist()}')
            p(f'      orientation of the 20 nm pairs, 30 deg bins 0..180: {q_["pair_angle_deg_hist_bin30"]}')
        R['picks'][nm] = r

    # ---- 5. quick-look of the final tables
    p('5. quick-look, final published tables')
    R['quicklook'] = {}
    for m in ('smlm', 'simflux'):
        t = T[m, 'final']; n = np.bincount(t['frame'].astype(int), minlength=ncyc)
        thirds = [float(n[k * ncyc // 3:(k + 1) * ncyc // 3].mean()) for k in range(3)]
        r = dict(n=int(len(t)), per_cycle_mean=float(n.mean()), per_cycle_percentiles_5_50_95=pct(n, (5, 50, 95)), per_cycle_thirds_of_movie=thirds,
                 photons_percentiles=pct(t['photons']), bg_percentiles=pct(t['bg']), lpx_nm_percentiles=pct(t['lpx'] * PX), lpy_nm_percentiles=pct(t['lpy'] * PX),
                 x_range_px=[float(t['x'].min()), float(t['x'].max())], y_range_px=[float(t['y'].min()), float(t['y'].max())])
        R['quicklook'][m] = r
        p(f'  {m:7s}: {len(t)} rows, per cycle mean {n.mean():.2f} (p5/50/95 {np.round(r["per_cycle_percentiles_5_50_95"], 0).tolist()}; thirds of the movie {np.round(thirds, 2).tolist()}); '
          f'photons per 6-frame block p5/25/50/75/95 {np.round(r["photons_percentiles"]).astype(int).tolist()}; bg per px per block {np.round(r["bg_percentiles"], 2).tolist()}')
        p(f'           reported precision nm x {np.round(r["lpx_nm_percentiles"], 2).tolist()} y {np.round(r["lpy_nm_percentiles"], 2).tolist()}; x range {np.round(r["x_range_px"], 1).tolist()} y range {np.round(r["y_range_px"], 1).tolist()} px')
    for m in ('smlm', 'simflux'):
        t_ = tot[m]; p(f'  published total drift {m}: x {t_[:, 0].min():+.3f}..{t_[:, 0].max():+.3f} px, y {t_[:, 1].min():+.3f}..{t_[:, 1].max():+.3f} px '
                       f'({(t_[:, 0].max() - t_[:, 0].min()) * PX:.1f} / {(t_[:, 1].max() - t_[:, 1].min()) * PX:.1f} nm peak to peak)')
    # ---- 6. extras: published SIMFLUX - published SMLM over the movie (same raw frame), photon floor, re-centred pick pairs
    p('6. extras')
    s_, f_ = T['smlm', 'nodc'], T['simflux', 'nodc']
    i, j = mmatch(f_['x'].astype(float), f_['y'].astype(float), f_['frame'].astype(np.int64), s_['x'].astype(float), s_['y'].astype(float), s_['frame'].astype(np.int64), a.rmatch)
    dx = (f_['x'][i].astype(float) - s_['x'][j]) * PX; dy = (f_['y'][i].astype(float) - s_['y'][j]) * PX; tb = np.minimum(f_['frame'][i].astype(int) * 10 // ncyc, 9)
    R['pub_simflux_minus_smlm_nm'] = dict(n=int(len(i)), median=[float(np.median(dx)), float(np.median(dy))], rsd=[L.rsd(dx), L.rsd(dy)],
                                          per_tenth_dx=[float(np.median(dx[tb == k])) for k in range(10)], per_tenth_dy=[float(np.median(dy[tb == k])) for k in range(10)])
    e = R['pub_simflux_minus_smlm_nm']
    p(f'  published simflux - published smlm (nodriftc, same cycle, {len(i)} pairs): median {e["median"][0]:+.2f} {e["median"][1]:+.2f} nm, robust sd {e["rsd"][0]:.2f} {e["rsd"][1]:.2f} nm')
    p(f'      medians per tenth of the movie, nm: dx {np.round(e["per_tenth_dx"], 2).tolist()} dy {np.round(e["per_tenth_dy"], 2).tolist()}')
    R['photon_floor'] = {m: dict(min=float(T[m, 'final']['photons'].min()), p1=float(np.percentile(T[m, 'final']['photons'], 1))) for m in ('smlm', 'simflux')}
    p(f'  photons of the published tables: {R["photon_floor"]} (quick fit, Gaussian units, cycle-aligned rows: p5/50/95 {np.round(pct(q["photons"], (5, 50, 95))).astype(int).tolist()}, {len(q) / (qc.max() + 1):.1f} per cycle)')
    for nm, (fn, m) in S['picks'].items():
        c = load_picks(pub + fn)['centers']
        if len(c) < 20: continue
        rad = R['picks'][nm]['diameter_px'] / 2; t = T[m, 'final']; P = np.c_[t['x'], t['y']].astype(float); tr = cKDTree(P)
        mc = np.array([P[v].mean(0) for v in tr.query_ball_point(c, rad)]) * PX
        sd = np.array([np.sqrt(0.5 * (P[v].var(0, ddof=1).sum())) * PX if len(v) > 2 else np.nan for v in tr.query_ball_point(c, rad)])
        pr = cKDTree(c * PX).query_pairs(28.0, output_type='ndarray'); d0 = np.linalg.norm((c[pr[:, 0]] - c[pr[:, 1]]) * PX, axis=1); pr = pr[d0 >= 12]
        d1 = np.linalg.norm(mc[pr[:, 0]] - mc[pr[:, 1]], axis=1)
        r = dict(n_pairs=int(len(pr)), dist_of_mean_positions_nm=dict(mean=float(d1.mean()), median=float(np.median(d1)), sd=float(d1.std(ddof=1)), percentiles_5_25_50_75_95=pct(d1)),
                 sd_within_pick_nm_percentiles=pct(sd[np.isfinite(sd)]))
        R['picks'][nm]['pairs_recentred'] = r
        p(f'  {nm}: the same pick pairs with each pick replaced by the mean of its localizations (final {m} table, inside the pick circle): distance mean {d1.mean():.2f}, median {np.median(d1):.2f}, sd {d1.std(ddof=1):.2f} nm, '
          f'percentiles {np.round(pct(d1), 1).tolist()}; sd per axis of the localizations inside a pick (truncated by the 9.75 nm circle) p5/25/50/75/95 {np.round(r["sd_within_pick_nm_percentiles"], 2).tolist()} nm')
    json.dump(R, open(out + 'ref.json', 'w'), indent=1)
    open(out + 'ref_report.txt', 'w').write('\n'.join(o) + '\n')


if __name__ == '__main__':
    main()
