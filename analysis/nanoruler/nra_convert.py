"""Converters into the method-table npz format of nra_lib and the published -> common-frame coordinate transform.

Common frame = frame of the per-raw-frame quick fit (pixel centres at integer coordinates), nm = px x 65.

  transform   compare the quick fit (merged per 6-frame cycle) with the published tables (all drift stages), write
              transform.json (constant shift published -> common frame, residuals, drift-sign test by cluster sharpness)
  published   Picasso hdf5 -> npz (x_common = x + shift; sx/sy = lpx/lpy x px = the published CRLB; frame = cycle)
  quick       quick fit -> npz, filtered, optionally merged per cycle and drift corrected (x - sign x drift[cycle])
  picks       group means of a picked Picasso hdf5 -> npz (x_nm, y_nm, n) in the common frame
"""
import argparse, json, os, numpy as np, h5py
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import nra_lib as L

PUB = '/mnt/data/SIMFLUX/Submission Data/80 nm rulers/Sim4_1_MMStack_Pos0.omeMod0.12Gauss10/'
OUT = '/home/jc/simcode2/out/nanoruler/analysis/'
QUICK = '/home/jc/simcode2/out/nanoruler/quick_smlm.hdf5'
PX = 65.0


def h5(fn): return h5py.File(fn, 'r')['locs'][:]


def load_quick(fn=QUICK):
    q = h5(fn); g = (q['chi2'] < 1.6) & (q['snr'] > 7) & (q['sx'] > 1.3) & (q['sx'] < 2.5)
    return q[g]


def merge_cycles(q, rlink=1.0, min_frames=4):
    """Average the per-frame fits of one emitter within one 6-frame cycle (linking radius rlink px, >= min_frames frames).
    Returns cycle, x, y (photon-weighted mean, px), photons (sum over the detected frames x 6 / n frames), lpx, lpy
    (sqrt(sum w^2 lp^2) with the photon weights), n frames."""
    cyc = (q['frame'] // 6).astype(np.int64); o = np.argsort(cyc, kind='stable'); q = q[o]; cyc = cyc[o]
    P = np.c_[q['x'], q['y'], cyc * 1e4]; T = cKDTree(P); pairs = T.query_pairs(rlink, output_type='ndarray')
    lab = np.arange(len(q))
    for it in range(8):                                   # label propagation (components are tiny)
        m = np.minimum(lab[pairs[:, 0]], lab[pairs[:, 1]]); new = lab.copy()
        np.minimum.at(new, pairs[:, 0], m); np.minimum.at(new, pairs[:, 1], m)
        if np.array_equal(new, lab): break
        lab = new
    u, inv, n = np.unique(lab, return_inverse=True, return_counts=True)
    w = q['photons'].astype(float); sw = np.bincount(inv, w)
    x = np.bincount(inv, w * q['x']) / sw; y = np.bincount(inv, w * q['y']) / sw
    lpx = np.sqrt(np.bincount(inv, (w * q['lpx']) ** 2)) / sw; lpy = np.sqrt(np.bincount(inv, (w * q['lpy']) ** 2)) / sw
    g = (n >= min_frames) & (n <= 6)
    return dict(cycle=cyc[u][g], x=x[g], y=y[g], photons=(sw / n * 6)[g], lpx=lpx[g], lpy=lpy[g], n=n[g])


def sharp(x, y, px=4.0):
    n = int(400 * PX / px); h = np.histogram2d(y * PX / px, x * PX / px, bins=n, range=[[0, n], [0, n]])[0]
    return float((gaussian_filter(h, 1.0) ** 2).sum() / len(x))


def match(m, p):
    T = cKDTree(np.c_[p['x'], p['y'], p['frame'].astype(float) * 1e4])
    d, j = T.query(np.c_[m['x'], m['y'], m['cycle'] * 1e4], distance_upper_bound=1.5); g = np.isfinite(d)
    return g, j


def cmd_transform(a):
    q = load_quick(a.quick); m = merge_cycles(q); print('quick rows %d -> %d cycle-merged emitters' % (len(q), len(m['x'])))
    R = dict(n_quick=len(q), n_merged=len(m['x']), stages={})
    dr = dict(drift_txt=np.loadtxt(PUB + 'drift.txt'), g2d_render_drift=np.loadtxt(PUB + 'g2d-nodriftc_190824_132639_drift.txt'),
              simflux_render_drift=np.loadtxt(PUB + 'simflux-nodriftc_190824_132643_drift.txt'))
    for st in ('g2d-nodriftc', 'g2d', 'g2d-nodriftc_render', 'simflux-nodriftc', 'simflux-nodriftc_render'):
        p = h5(PUB + st + '.hdf5'); g, j = match(m, p)
        dx = m['x'][g] - p['x'][j[g]]; dy = m['y'][g] - p['y'][j[g]]; c = m['cycle'][g]
        tb = np.minimum(c // 500, 9)
        r = dict(n_matched=int(g.sum()), shift_px=[float(np.median(dx)), float(np.median(dy))], resid_rsd_px=[L.rsd(dx), L.rsd(dy)],
                 timebin_median_dx_px=[float(np.median(dx[tb == i])) for i in range(10)],
                 timebin_median_dy_px=[float(np.median(dy[tb == i])) for i in range(10)])
        for dn, d in dr.items():        # is (quick - published) explained by a published drift trace?
            r['resid_rsd_px_minus_' + dn] = [L.rsd(dx - d[c, 0]), L.rsd(dy - d[c, 1])]
            r['corr_with_' + dn] = [float(np.corrcoef(dx, d[c, 0])[0, 1]), float(np.corrcoef(dy, d[c, 1])[0, 1])]
        R['stages'][st] = r
        print('%-26s n %6d shift (quick - pub) %+.4f %+.4f px, resid rsd %.4f %.4f px (%.2f %.2f nm)' % (
            st, g.sum(), r['shift_px'][0], r['shift_px'][1], r['resid_rsd_px'][0], r['resid_rsd_px'][1],
            r['resid_rsd_px'][0] * PX, r['resid_rsd_px'][1] * PX))
        print('     time-bin medians dx', np.round(r['timebin_median_dx_px'], 3), '\n     dy', np.round(r['timebin_median_dy_px'], 3))
        for dn in dr: print('     minus %-22s rsd %.4f %.4f  corr %.2f %.2f' % (dn, *r['resid_rsd_px_minus_' + dn], *r['corr_with_' + dn]))
    # drift sign: which correction of the (raw) quick table sharpens the clusters
    R['sharpness_quick'] = {}
    for dn, d in [('none', None)] + list(dr.items()):
        for s in ((0,) if d is None else (-1, 1)):
            dd = 0 if d is None else s * d[m['cycle']]
            v = sharp(m['x'] + (dd[:, 0] if d is not None else 0), m['y'] + (dd[:, 1] if d is not None else 0))
            R['sharpness_quick']['%s_%+d' % (dn, s)] = v; print('sharpness quick merged, x %+d * %-22s %.4f' % (s, dn, v))
    for st, dn in (('g2d-nodriftc', 'g2d_render_drift'), ('simflux-nodriftc', 'simflux_render_drift')):
        p = h5(PUB + st + '.hdf5'); f = p['frame'].astype(int)
        for nm, dd in (('as is', 0 * dr[dn]), ('- drift.txt', -dr['drift_txt']), ('+ drift.txt', dr['drift_txt']), ('- ' + dn, -dr[dn]), ('+ ' + dn, dr[dn])):
            v = sharp(p['x'] + dd[f, 0], p['y'] + dd[f, 1]); R['sharpness_%s %s' % (st, nm)] = v; print('sharpness %s %s: %.4f' % (st, nm, v))
    sh = [R['stages'][s]['shift_px'] for s in ('g2d-nodriftc_render', 'simflux-nodriftc_render')]
    R['shift_px_published_to_common'] = dict(g2d=sh[0], simflux=sh[1])
    os.makedirs(OUT, exist_ok=True); json.dump(R, open(OUT + 'transform.json', 'w'), indent=1)


def cmd_published(a):
    p = h5(a.h5); sx, sy = a.shift
    L.save_table(a.out, frame=p['frame'].astype(int), x_nm=(p['x'] + sx) * PX, y_nm=(p['y'] + sy) * PX, sx_nm=p['lpx'] * PX,
                 sy_nm=p['lpy'] * PX, photons=p['photons'], crlb_x_nm=p['lpx'] * PX, crlb_y_nm=p['lpy'] * PX, bg=p['bg'])
    print(a.out, len(p))


def cmd_quick(a):
    q = load_quick(a.quick)
    if a.merge:
        m = merge_cycles(q); cyc, x, y, ph, lx, ly = m['cycle'], m['x'], m['y'], m['photons'], m['lpx'], m['lpy']
    else:
        cyc, x, y, ph, lx, ly = (q['frame'] // 6).astype(int), q['x'].astype(float), q['y'].astype(float), q['photons'] * 6.0, q['lpx'], q['lpy']
    if a.drift:
        d = np.loadtxt(a.drift); x = x + a.sign * d[cyc, 0]; y = y + a.sign * d[cyc, 1]
    L.save_table(a.out, frame=cyc, x_nm=x * PX, y_nm=y * PX, sx_nm=lx * PX, sy_nm=ly * PX, photons=ph, crlb_x_nm=lx * PX, crlb_y_nm=ly * PX)
    print(a.out, len(x))


def cmd_picks(a):
    p = h5(a.h5); g = p['group'].astype(int); n = np.bincount(g); k = n > 0
    x = (np.bincount(g, p['x'])[k] / n[k] + a.shift[0]) * PX; y = (np.bincount(g, p['y'])[k] / n[k] + a.shift[1]) * PX
    np.savez(a.out, x_nm=x, y_nm=y, n=n[k]); print(a.out, len(x), 'picks; median locs / pick', np.median(n[k]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('transform'); s.add_argument('--quick', default=QUICK); s.set_defaults(fn=cmd_transform)
    s = sp.add_parser('published'); s.add_argument('h5'); s.add_argument('out'); s.add_argument('--shift', type=float, nargs=2, default=[0.0, 0.0]); s.set_defaults(fn=cmd_published)
    s = sp.add_parser('quick'); s.add_argument('out'); s.add_argument('--quick', default=QUICK); s.add_argument('--merge', type=int, default=1)
    s.add_argument('--drift', default=''); s.add_argument('--sign', type=float, default=-1.0); s.set_defaults(fn=cmd_quick)
    s = sp.add_parser('picks'); s.add_argument('h5'); s.add_argument('out'); s.add_argument('--shift', type=float, nargs=2, default=[0.0, 0.0]); s.set_defaults(fn=cmd_picks)
    a = ap.parse_args(); a.fn(a)
