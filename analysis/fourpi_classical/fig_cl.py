"""Classical 4Pi-SIMFLUX vs the network ("4Pi-SIMCODE") on the real data: same regions, same rendering, same z colour scale.

MT: the dense boxes of <work>/patches/regions_MT.json (candidate pixel inside the box, raw per-file coordinates;
both tables restricted the same way). NPC: whole field overview, zooms on single pores, averaged pore (ring radius,
two-ring z separation).
Numbers per region: localizations kept, median reported sigma, DAFL (adjacent-window pairs, analysis/fourpi/dafl4pi.py)
for each method alone and on the SAME emitters (rows of both tables matched per window within 0.5 px), FRC (two
halves of 3000-frame blocks).
Usage: python analysis/fourpi_classical/fig_cl.py --work <4Pi work folder> --cell MT --net <work>/net/net_MT_v2_all.npz [--tag _short --files 5]
"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
sys.path.insert(0, f'{REPO}/analysis/fourpi')
from dafl4pi import analyse
from simcode.pipeline.configs.fourpi import PIXEL_UM; PX = PIXEL_UM * 1000   # 4Pi pixel (nm; the same for MT and NPC)
OUT = None      # output folder of the figure functions: <work>/classical (set by main(); callers that import them set FC.OUT)


def frc(xa, ya, xb, yb, px=4.0):
    """FRC (nm in, nm out), 1/7 threshold: simcode/evaluation/frc.py"""
    from simcode.evaluation.frc import frc as _frc
    return _frc(xa, ya, xb, yb, px=px, units='nm')


def render(ax, x, y, z, box, zr, sr, title, vmax=None):
    from matplotlib.colors import hsv_to_rgb
    x0, x1, y0, y1 = box
    W, H = int((x1 - x0) / sr), int((y1 - y0) / sr)
    ok = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    ix = np.clip(((x[ok] - x0) / sr).astype(int), 0, W - 1); iy = np.clip(((y[ok] - y0) / sr).astype(int), 0, H - 1)
    c = np.zeros((H, W)); zs = np.zeros((H, W)); np.add.at(c, (iy, ix), 1); np.add.at(zs, (iy, ix), z[ok])
    cs = gaussian_filter(c, 0.8); zz = gaussian_filter(zs, 0.8) / np.maximum(cs, 1e-9)
    vm = vmax if vmax is not None else max(np.percentile(cs[cs > 0], 99.5), 1e-9) if (cs > 0).any() else 1
    v = np.clip(cs / vm, 0, 1)
    ax.imshow(hsv_to_rgb(np.stack([np.clip((zz - zr[0]) / (zr[1] - zr[0]), 0, 1) * 0.75, np.ones_like(v), v], -1)),
              extent=[0, (x1 - x0) / 1000, (y1 - y0) / 1000, 0])
    ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6)
    return vm


def render_xz(ax, x, z, xr, zr, sr, title):
    W, Hz = int((xr[1] - xr[0]) / sr), int((zr[1] - zr[0]) / sr)
    ix = ((x - xr[0]) / sr).astype(int); iz = ((z - zr[0]) / sr).astype(int)
    ok = (ix >= 0) & (ix < W) & (iz >= 0) & (iz < Hz)
    c = np.zeros((Hz, W)); np.add.at(c, (iz[ok], ix[ok]), 1); c = gaussian_filter(c, 0.8)
    ax.imshow(c[::-1], cmap='magma', vmax=max(np.percentile(c[c > 0], 99.5), 1e-9) if (c > 0).any() else 1,
              extent=[0, (xr[1] - xr[0]) / 1000, zr[0], zr[1]], aspect='auto')
    ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6); ax.set_ylabel('z (nm)', fontsize=7)


from simcode.util.drift import zdrift_struct  # noqa (structure-based sample z drift; used by the scripts that import this module)


def load(cl_fn, net_fn, files=None, wmax=None, zdrift=False):
    C = dict(np.load(cl_fn))
    N = dict(np.load(net_fn))
    if files is not None:
        s = np.isin(N['fidx'], files)
        if wmax is not None: s &= N['w'] < wmax
        N = {k: (v[s] if (hasattr(v, 'shape') and v.shape[:1] == s.shape) else v) for k, v in N.items()}
    fi_ = np.clip(C['frame'] + 3, 0, len(C['drift']) - 1)
    cl = dict(x=C['xd'] * PX, y=C['yd'] * PX, z=C['z'] * 1000, sx=C['sx'], sy=C['sy'], sz=C['sz'], N=C['N'], w=C['frame'] // 6,
              ok=C['keep'], rx=C['cx'], ry=C['cy'], ux=C['x'] * PX, uy=C['y'] * PX, f=C['file_index'],
              candx=(C['cx'] - C['drift'][fi_, 0]) * PX, candy=(C['cy'] - C['drift'][fi_, 1]) * PX, flags=C['flags'].astype(bool))
    clp = dict(x=C['xpd'] * PX, y=C['ypd'] * PX, z=C['zp'] * 1000, sx=C['sxp'], sy=C['syp'], sz=C['szp'], N=C['N'], w=C['frame'] // 6,
               ok=C['keep_pix'], rx=C['cx'], ry=C['cy'], ux=C['xp'] * PX, uy=C['yp'] * PX, f=C['file_index'])
    good = N['good'] if 'good' in N else np.ones(len(N['x']), bool)
    net = dict(x=N['xd'] * PX, y=N['yd'] * PX, z=N['z'] * 1000, sx=N['sx'], sy=N['sy'], sz=N['sz'], N=N['N'], w=N['frame'] // 6,
               ok=good, rx=np.round(N['x']), ry=np.round(N['y']), on=N['on'] if 'on' in N else np.ones(len(good), bool),
               ux=N['x'] * PX, uy=N['y'] * PX, f=N['fidx'])
    D = dict(cl=cl, clp=clp, net=net)
    # global translation between the two drift references (DME's arbitrary offset): median over matched rows
    ia, ib = same_emitters(sel(net, net['ok']), sel(cl, cl['ok']))
    A, B = sel(net, net['ok']), sel(cl, cl['ok'])
    if len(ia) > 50:
        off = [float(np.median((A['x'][ia] - A['ux'][ia]) - (B['x'][ib] - B['ux'][ib]))), float(np.median((A['y'][ia] - A['uy'][ia]) - (B['y'][ib] - B['uy'][ib])))]
        for k in ('cl', 'clp'): D[k]['x'] = D[k]['x'] + off[0]; D[k]['y'] = D[k]['y'] + off[1]
        D['cl']['candx'] = D['cl']['candx'] + off[0]; D['cl']['candy'] = D['cl']['candy'] + off[1]
        D['drift_ref_offset_nm'] = off
    if zdrift:      # sample z drift, the same estimator on each table from its own kept rows; pixel MLE takes the classical curve
        D['zdrift'] = {}
        for k in ('net', 'cl'):
            T = D[k]; m = T['ok'] & np.isfinite(T['z']) & np.isfinite(T['x'])
            d = zdrift_struct(T['x'][m], T['y'][m], T['z'][m], T['w'][m]); D['zdrift'][k] = d
        nb = max(len(v) for v in D['zdrift'].values())
        for k, src in (('net', 'net'), ('cl', 'cl'), ('clp', 'cl')):
            d = np.r_[D['zdrift'][src], np.zeros(nb - len(D['zdrift'][src]))]
            D[k]['z'] = D[k]['z'] - d[np.clip(D[k]['w'] // 500, 0, nb - 1)]
    return D


def sel(T, m):
    n = len(T['x'])
    return {k: (v[m] if hasattr(v, 'shape') and v.shape[:1] == (n,) else v) for k, v in T.items()}


def same_emitters(A, B, r_nm=64.0):
    """indices (ia, ib) of rows of A and B in the same window within r_nm"""
    ia, ib = [], []
    oa, ob = np.argsort(A['w'], kind='stable'), np.argsort(B['w'], kind='stable')
    wa, wb = A['w'][oa], B['w'][ob]
    for wv in np.intersect1d(np.unique(wa), np.unique(wb)):          # w = global window: file and window together
        a = oa[np.searchsorted(wa, wv):np.searchsorted(wa, wv, 'right')]; b = ob[np.searchsorted(wb, wv):np.searchsorted(wb, wv, 'right')]
        d, k = cKDTree(np.c_[B['ux'][b], B['uy'][b]]).query(np.c_[A['ux'][a], A['uy'][a]], distance_upper_bound=r_nm)   # raw positions
        ok = np.isfinite(d); ia.append(a[ok]); ib.append(b[k[ok]])
    return (np.concatenate(ia), np.concatenate(ib)) if ia else (np.zeros(0, int), np.zeros(0, int))


def dafl(T, name):
    r = analyse(name, T['w'], T['x'] / PX, T['y'] / PX, T['sx'], T['sy'], T['N'], T['z'] / 1000, T['sz'])
    return r


def numbers(D, name):
    """per-method and same-emitter numbers for one region"""
    out = dict(region=name, methods={})
    for k in ('net', 'cl', 'clp'):
        if k not in D: continue
        T = sel(D[k], D[k]['ok'] & np.isfinite(D[k]['z']))
        if len(T['x']) < 50: continue
        h = (T['w'] // 500) % 2 == 0
        out['methods'][k] = dict(n=int(len(T['x'])), med_sigma_nm=[float(np.median(T['sx']) * PX), float(np.median(T['sy']) * PX), float(np.median(T['sz']) * 1000)],
                                 med_N=float(np.median(T['N'])), frc_nm=frc(T['x'][h], T['y'][h], T['x'][~h], T['y'][~h]) if h.sum() > 100 and (~h).sum() > 100 else None,
                                 dafl=dafl(T, k)['bins'])
    A = sel(D['net'], D['net']['ok']); B = sel(D['cl'], D['cl']['ok']); Bp = sel(D['clp'], D['clp']['ok'] & np.isfinite(D['clp']['z']))
    ia, ib = same_emitters(A, B)
    out['same'] = dict(n=int(len(ia)), frac_of_cl=float(len(ib) / max(len(B['x']), 1)), frac_of_net=float(len(ia) / max(len(A['x']), 1)))
    if len(ia) > 200:
        An = sel(A, ia); An['N'] = B['N'][ib]                     # bin both by the classical photon count
        o_ = np.argsort(ia); An = sel(An, o_)
        out['same']['net'] = dafl(An, 'net same')['bins']
        out['same']['cl'] = dafl(sel(B, np.isin(np.arange(len(B['x'])), ib)), 'cl same')['bins']
        ia2, ib2 = same_emitters(A, Bp)
        out['same']['clp'] = dafl(sel(Bp, np.isin(np.arange(len(Bp['x'])), ib2)), 'clp same')['bins']
        dz = A['z'][ia] - B['z'][ib]
        out['same']['dz_median_nm'] = float(np.median(dz)); out['same']['dz_fringe_frac'] = float(np.mean(np.abs(dz - np.median(dz)) > 120))
        out['same']['dxy_median_nm'] = [float(np.median(A['x'][ia] - B['x'][ib])), float(np.median(A['y'][ia] - B['y'][ib]))]
    return out


def fmt_numbers(R):
    L = [f"\n## {R['region']}"]
    nm = dict(net='network', cl='classical published-style', clp='classical pixel MLE')
    L.append('| method | locs | median sigma x / y / z (nm) | median N | FRC (nm) | DAFL x / y / z at 2-4k e (nm) |')
    L.append('|---|---|---|---|---|---|')
    for k, m in R['methods'].items():
        b = [q for q in m['dafl'] if q['lo'] == 2000]
        ds = f"{b[0]['x']['dafl_nm']:.2f} / {b[0]['y']['dafl_nm']:.2f} / {b[0]['z']['dafl_nm']:.2f}" if b else '-'
        fr = f"{m['frc_nm']:.1f}" if m['frc_nm'] else '-'
        L.append(f"| {nm[k]} | {m['n']} | {m['med_sigma_nm'][0]:.1f} / {m['med_sigma_nm'][1]:.1f} / {m['med_sigma_nm'][2]:.1f} | {m['med_N']:.0f} | {fr} | {ds} |")
    s = R['same']
    L.append(f"\nSame emitters (network good and classical kept, same window, < 64 nm): {s['n']} ({s['frac_of_cl']:.2f} of classical, {s['frac_of_net']:.2f} of network)")
    if 'net' in s:
        L.append(f"median network - classical: dx {s['dxy_median_nm'][0]:.1f}, dy {s['dxy_median_nm'][1]:.1f}, dz {s['dz_median_nm']:.1f} nm; |dz - median| > 120 nm (fringe disagreement) {s['dz_fringe_frac']:.3f}")
        L.append('| same emitters | N bin | DAFL x / y / z (nm) | n pairs |'); L.append('|---|---|---|---|')
        for k in ('net', 'cl', 'clp'):
            for b in s[k]:
                L.append(f"| {nm[k]} | {b['lo']}-{b['hi']} | {b['x']['dafl_nm']:.2f} / {b['y']['dafl_nm']:.2f} / {b['z']['dafl_nm']:.2f} | {b['n']} |")
    return L


def mt_figures(D, boxes, tag, sr=5.0):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    L, res = [], []
    for b in boxes:
        bid = b['id']
        inb = lambda T: (T['rx'] >= b['x0']) & (T['rx'] < b['x1']) & (T['ry'] >= b['y0']) & (T['ry'] < b['y1'])
        R = {k: sel(T, inb(T)) for k, T in D.items() if k in ('net', 'cl', 'clp')}
        S = {k: sel(T, T['ok'] & np.isfinite(T['z'])) for k, T in R.items()}
        if len(S['net']['x']) < 100 or len(S['cl']['x']) < 100: continue
        allx = np.r_[S['net']['x'], S['cl']['x']]; ally = np.r_[S['net']['y'], S['cl']['y']]
        bx = (np.percentile(allx, 1), np.percentile(allx, 99), np.percentile(ally, 1), np.percentile(ally, 99))
        zr = tuple(np.percentile(S['net']['z'], [2, 98]))
        # zoom: the densest 1.5 um square of the network render
        h, xe, ye = np.histogram2d(S['net']['x'], S['net']['y'], bins=[np.arange(bx[0], bx[1], 250), np.arange(bx[2], bx[3], 250)])
        hs = gaussian_filter(h, 2.0); i, j = np.unravel_index(np.argmax(hs), hs.shape)
        zx, zy = xe[i], ye[j]; zb = (zx - 750, zx + 750, zy - 750, zy + 750)
        fig, axs = plt.subplots(3, 3, figsize=(15, 14), gridspec_kw=dict(height_ratios=[3, 2, 1]))
        names = [('net', 'network v2 head (prior bug, not at CRLB)'), ('cl', 'classical 4Pi-SIMFLUX (published-style)'), ('clp', 'classical pixel MLE (run in boxes 0, 1 only)')]
        vm = None
        for c_, (k, nm) in enumerate(names):
            T = S[k]
            render(axs[0, c_], T['x'], T['y'], T['z'], bx, zr, sr, f'{nm}: {len(T["x"])} locs\nhue z {zr[0]:.0f}..{zr[1]:.0f} nm, {sr:.0f} nm px')
            axs[0, c_].add_patch(matplotlib.patches.Rectangle(((zb[0] - bx[0]) / 1000, (zb[2] - bx[2]) / 1000), 1.5, 1.5, fill=False, ec='w', lw=0.7))
            render(axs[1, c_], T['x'], T['y'], T['z'], zb, zr, 3.0, f'zoom 1.5 x 1.5 um (3 nm px)')
            ys = (T['y'] >= zy - 100) & (T['y'] < zy + 100)
            render_xz(axs[2, c_], T['x'][ys], T['z'][ys], (zb[0], zb[1]), (zr[0] - 60, zr[1] + 60), 4.0, 'x-z, 200 nm strip through the zoom centre')
        fig.suptitle(f'MT dense box {bid} ({b["x1"] - b["x0"]} x {b["y1"] - b["y0"]} px = {(b["x1"] - b["x0"]) * 0.128:.1f} um), lateral + sample z drift corrected; '
                     f'classical: wavelet candidates, network: its own. {NETNAME}. {PIXNOTE}.', fontsize=9)
        plt.tight_layout(); fn = f'{OUT}/fig/MT_box{bid}{tag}.png'; plt.savefig(fn, dpi=140); plt.close()
        R_ = numbers(R, f'MT box {bid}'); R_['fig'] = fn; res.append(R_); L += fmt_numbers(R_)
    return res, L


NETNAME = 'network v2 head (4Pi-SIMCODE; known prior bug, not at the CRLB; provisional)'
PIXNOTE = 'classical pixel MLE: run only in boxes 0 and 1'


def mt_overview(D, boxes, tag, sr=10.0):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    S = {k: sel(D[k], D[k]['ok'] & np.isfinite(D[k]['z'])) for k in ('net', 'cl')}
    T = S['net']
    bx = (np.percentile(T['x'], 0.2), np.percentile(T['x'], 99.8), np.percentile(T['y'], 0.2), np.percentile(T['y'], 99.8))
    zr = tuple(np.percentile(T['z'], [2, 98]))
    asp = (bx[3] - bx[2]) / (bx[1] - bx[0])
    fig, axs = plt.subplots(1, 2, figsize=(24, 12 * asp + 1.5))
    for i, (k, nm) in enumerate((('net', NETNAME), ('cl', 'classical 4Pi-SIMFLUX (published-style, whole field)'))):
        render(axs[i], S[k]['x'], S[k]['y'], S[k]['z'], bx, zr, sr, f'{nm}\n{len(S[k]["x"])} locs, hue z {zr[0]:.0f}..{zr[1]:.0f} nm (same scale), {sr:.0f} nm px')
        for b in boxes:   # box outlines in drift-corrected nm (approximate: raw px * 128 + median drift offset)
            sb = (D['net']['rx'] >= b['x0']) & (D['net']['rx'] < b['x1']) & (D['net']['ry'] >= b['y0']) & (D['net']['ry'] < b['y1'])
            if sb.sum() < 10: continue
            xs, ys = D['net']['x'][sb], D['net']['y'][sb]
            x0, x1, y0, y1 = np.percentile(xs, 1), np.percentile(xs, 99), np.percentile(ys, 1), np.percentile(ys, 99)
            axs[i].add_patch(matplotlib.patches.Rectangle(((x0 - bx[0]) / 1000, (y0 - bx[2]) / 1000), (x1 - x0) / 1000, (y1 - y0) / 1000, fill=False, ec='w', lw=0.8))
            axs[i].text((x0 - bx[0]) / 1000, (y0 - bx[2]) / 1000, f'box {b["id"]}', color='w', fontsize=8)
        for j, c in enumerate(D.get('crossings', [])):
            axs[i].add_patch(matplotlib.patches.Rectangle(((c[0] - 750 - bx[0]) / 1000, (c[1] - 750 - bx[2]) / 1000), 1.5, 1.5, fill=False, ec='c', lw=0.8))
            axs[i].text((c[0] - 750 - bx[0]) / 1000, (c[1] + 950 - bx[2]) / 1000, f'X{j}', color='c', fontsize=8)
    fig.suptitle(f'MT whole field, lateral drift DME v2, sample z drift removed (same estimator, each table). White: dense boxes; cyan: crossings. '
                 f'{PIXNOTE} (not shown here).', fontsize=10)
    plt.tight_layout(); fn = f'{OUT}/fig/MT_overview{tag}.png'; plt.savefig(fn, dpi=110); plt.close()
    return fn


def pick_xings(D, n=6, size_um=1.5):
    sys.path.insert(0, f'{REPO}/analysis/figures'); from zoompick import pick_crossings
    S = {k: sel(D[k], D[k]['ok'] & np.isfinite(D[k]['z'])) for k in ('net', 'cl')}
    x = np.r_[S['net']['x'], S['cl']['x']] / 1000; y = np.r_[S['net']['y'], S['cl']['y']] / 1000
    lim = (np.percentile(x, 0.5), np.percentile(x, 99.5), np.percentile(y, 0.5), np.percentile(y, 99.5))
    bs = pick_crossings(x, y, lim=lim, n=n, size=size_um, px=0.02, margin=1.0)
    return [((b[0] + b[1]) / 2 * 1000, (b[2] + b[3]) / 2 * 1000) for b in bs]


def mt_crossings(D, tag):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    S = {k: sel(D[k], D[k]['ok'] & np.isfinite(D[k]['z'])) for k in ('net', 'cl', 'clp')}
    X = D['crossings']; zr = tuple(np.percentile(S['net']['z'], [2, 98]))
    fig, axs = plt.subplots(len(X), 6, figsize=(24, 4.2 * len(X)), squeeze=False)
    out = []
    for i, c in enumerate(X):
        zb = (c[0] - 750, c[0] + 750, c[1] - 750, c[1] + 750)
        for j, (k, nm) in enumerate((('net', 'network v2'), ('cl', 'classical published-style'))):
            T = S[k]
            render(axs[i, j], T['x'], T['y'], T['z'], zb, zr, 3.0, f'X{i} {nm} (3 nm px, hue z {zr[0]:.0f}..{zr[1]:.0f})')
            axs[i, j].axhline(0.75, color='w', lw=0.4, ls=':'); axs[i, j].axvline(0.75, color='w', lw=0.4, ls=':')
            ys = np.abs(T['y'] - c[1]) < 100; xs = np.abs(T['x'] - c[0]) < 100
            render_xz(axs[i, 2 + j], T['x'][ys], T['z'][ys], (zb[0], zb[1]), (zr[0] - 60, zr[1] + 60), 4.0, f'X{i} {nm}: x-z, 200 nm strip along x (dotted h-line)')
            render_xz(axs[i, 4 + j], T['y'][xs], T['z'][xs], (zb[2], zb[3]), (zr[0] - 60, zr[1] + 60), 4.0, f'X{i} {nm}: y-z, 200 nm strip along y (dotted v-line)')
        # local counts within 300 nm of the crossing centre
        r = {k: int((np.hypot(S[k]['x'] - c[0], S[k]['y'] - c[1]) < 300).sum()) for k in ('net', 'cl', 'clp')}
        cand = np.hypot(D['cl']['candx'] - c[0], D['cl']['candy'] - c[1]) < 300
        nall = np.hypot(D['net']['x'] - c[0], D['net']['y'] - c[1]) < 300
        out.append(dict(id=i, centre_nm=[float(c[0]), float(c[1])], n_net=r['net'], n_cl=r['cl'], n_clp=r['clp'],
                        cl_kept_frac_of_cand=float(D['cl']['ok'][cand].mean()) if cand.any() else None,
                        net_good_frac_of_rows=float(D['net']['ok'][nall].mean()) if nall.any() else None))
    fig.suptitle(f'MT crossings (zoompick junctions, 1.5 x 1.5 um): {NETNAME} vs classical published-style; same z scale. '
                 f'{PIXNOTE} (not shown).', fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.985)); fn = f'{OUT}/fig/MT_crossings{tag}.png'; plt.savefig(fn, dpi=110); plt.close()
    return fn, out


def fwhm(v, lo, hi, bw=4.0):
    h, e = np.histogram(v, bins=np.arange(lo, hi + bw, bw)); c = 0.5 * (e[1:] + e[:-1])
    h = gaussian_filter(h.astype(float), 1.0); n = len(h); base = np.median(np.r_[h[:n // 8], h[-n // 8:]])
    h = h - base; i = int(np.argmax(h)); hm = h[i] / 2
    if h[i] <= 0: return float('nan')
    L = np.nonzero(h[:i] < hm)[0]; R = np.nonzero(h[i:] < hm)[0]
    if len(L) == 0 or len(R) == 0: return float('nan')
    a = L[-1]; b = i + R[0]
    xl = c[a] + (hm - h[a]) / (h[a + 1] - h[a]) * bw; xr = c[b - 1] + (h[b - 1] - hm) / (h[b - 1] - h[b]) * bw
    return float(xr - xl)


def filament_widths(D, tag, px=20.0):
    """isolated straight filament segments on the joint (network + classical) render: skeleton minus 500 nm around
    junctions, components >= 1 um, perpendicular rms of the skeleton <= 15 nm, no other structure 120-400 nm off the line.
    Per method: lateral FWHM of the perpendicular profile, axial FWHM of z about a linear z(s) fit of the segment."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sys.path.insert(0, f'{REPO}/analysis/figures'); from skel import skeletonize
    from scipy.ndimage import convolve, label, distance_transform_edt
    S = {k: sel(D[k], D[k]['ok'] & np.isfinite(D[k]['z'])) for k in ('net', 'cl', 'clp')}
    x0, x1 = np.percentile(S['net']['x'], [0.5, 99.5]); y0, y1 = np.percentile(S['net']['y'], [0.5, 99.5])
    bins = [np.arange(y0, y1, px), np.arange(x0, x1, px)]
    H = sum(np.histogram2d(S[k]['y'], S[k]['x'], bins=bins)[0] / len(S[k]['x']) for k in ('net', 'cl'))
    Hs = gaussian_filter(H, 2.0); m = Hs > np.percentile(Hs[Hs > 0], 70)
    sk = skeletonize(m); nb = convolve(sk.astype(int), np.ones((3, 3), int), mode='constant') - sk
    J = sk & (nb >= 3); dj = distance_transform_edt(~J)
    seg = sk & (dj > 500 / px)
    lab, nl = label(seg, structure=np.ones((3, 3)))
    trees = {k: cKDTree(np.c_[S[k]['x'], S[k]['y']]) for k in S}
    my, mx = np.nonzero(m); mxy = np.c_[x0 + (mx + 0.5) * px, y0 + (my + 0.5) * px]; mtree = cKDTree(mxy)
    segs = []
    for i in range(1, nl + 1):
        yy, xx = np.nonzero(lab == i)
        if len(xx) < 1000 / px: continue
        P = np.c_[x0 + (xx + 0.5) * px, y0 + (yy + 0.5) * px]; c = P.mean(0)
        u, s_, vt = np.linalg.svd(P - c, full_matrices=False); d = vt[0]; nrm = np.r_[-d[1], d[0]]
        al = (P - c) @ d; pe = (P - c) @ nrm
        if np.sqrt(np.mean(pe ** 2)) > 15 or np.ptp(al) < 1000: continue
        a0, a1 = al.min() + 100, al.max() - 100
        q = mxy[mtree.query_ball_point(c, np.ptp(al) / 2 + 400)]; qa = (q - c) @ d; qp = np.abs((q - c) @ nrm)
        inr = (qa > a0) & (qa < a1)
        if np.mean((qp[inr] > 120) & (qp[inr] < 400)) > 0.02: continue
        rec = dict(c=c.tolist(), d=d.tolist(), len_nm=float(a1 - a0), methods={})
        for k, T in S.items():
            idx = np.array(trees[k].query_ball_point(c, np.ptp(al) / 2 + 300), int)
            if len(idx) == 0: continue
            v = np.c_[T['x'][idx], T['y'][idx]] - c; sa = v @ d; sp = v @ nrm
            w = (sa > a0) & (sa < a1) & (np.abs(sp) < 250)
            if w.sum() < 150: continue
            sp, sa, zz = sp[w], sa[w], T['z'][idx][w]
            sp = sp - np.median(sp[np.abs(sp) < 80])
            core = np.abs(sp) < 60
            p = np.polyfit(sa[core], zz[core], 1); zr_ = zz - np.polyval(p, sa)
            zr_ = zr_ - np.median(zr_[core])
            rec['methods'][k] = dict(n=int(w.sum()), fwhm_lat=fwhm(sp, -250, 250), fwhm_ax=fwhm(zr_[core], -200, 200),
                                     z_slope_nm_per_um=float(p[0] * 1000), sp=sp[np.abs(sp) < 250], zr=zr_[core])
        if 'net' in rec['methods'] and 'cl' in rec['methods']: segs.append(rec)
    out = dict(n_segments=len(segs), methods={})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = dict(net='C0', cl='C3', clp='k'); nm = dict(net='network v2', cl='classical published-style', clp='classical pixel MLE (boxes 0, 1)')
    for k in ('net', 'cl', 'clp'):
        R = [s['methods'][k] for s in segs if k in s['methods']]
        if not R: continue
        fl = np.array([r['fwhm_lat'] for r in R]); fa = np.array([r['fwhm_ax'] for r in R])
        sp = np.concatenate([r['sp'] for r in R]); zr_ = np.concatenate([r['zr'] for r in R])
        out['methods'][k] = dict(n_segments=len(R), n_locs=int(len(sp)), fwhm_lat_median_nm=float(np.nanmedian(fl)),
                                 fwhm_lat_iqr_nm=np.nanpercentile(fl, [25, 75]).tolist(), fwhm_ax_median_nm=float(np.nanmedian(fa)),
                                 fwhm_ax_iqr_nm=np.nanpercentile(fa, [25, 75]).tolist(),
                                 fwhm_lat_pooled_nm=fwhm(sp, -250, 250), fwhm_ax_pooled_nm=fwhm(zr_, -200, 200))
        h, e = np.histogram(sp, bins=np.arange(-150, 152, 4)); axs[0].plot(0.5 * (e[1:] + e[:-1]), h / h.max(), color=cols[k],
                                                                         label=f"{nm[k]}: FWHM {out['methods'][k]['fwhm_lat_pooled_nm']:.0f} nm (median/seg {out['methods'][k]['fwhm_lat_median_nm']:.0f}, n seg {len(R)})")
        h, e = np.histogram(zr_, bins=np.arange(-150, 152, 4)); axs[1].plot(0.5 * (e[1:] + e[:-1]), h / h.max(), color=cols[k],
                                                                          label=f"{nm[k]}: FWHM {out['methods'][k]['fwhm_ax_pooled_nm']:.0f} nm (median/seg {out['methods'][k]['fwhm_ax_median_nm']:.0f})")
        axs[2].hist(fl, bins=np.arange(0, 150, 5), histtype='step', color=cols[k], label=nm[k])
    axs[0].set_title('lateral cross-section, pooled over isolated straight segments', fontsize=8); axs[0].set_xlabel('perpendicular offset (nm)', fontsize=7)
    axs[1].set_title('axial cross-section (z about the segment z(s) line)', fontsize=8); axs[1].set_xlabel('z (nm)', fontsize=7)
    axs[2].set_title('lateral FWHM per segment', fontsize=8); axs[2].set_xlabel('FWHM (nm)', fontsize=7)
    for ax in axs: ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    fig.suptitle(f'MT filament width ({len(segs)} isolated straight segments >= 1 um). {NETNAME}. {PIXNOTE}.', fontsize=9)
    plt.tight_layout(); fn = f'{OUT}/fig/MT_filament_width{tag}.png'; plt.savefig(fn, dpi=130); plt.close()
    out['fig'] = fn
    return out


def density_emptying(D, tag, bin_nm=100.0):
    """filter emptying vs local structure density: per 100 nm bin, density = network rows (all, before its filter);
    classical kept fraction of its candidates and network good fraction of its rows, per density decile;
    and the kept-classical / good-network ratio"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    N = D['net']; C = D['cl']
    x0, y0 = np.nanpercentile(N['x'], 0.1), np.nanpercentile(N['y'], 0.1)
    nx = int((np.nanpercentile(N['x'], 99.9) - x0) / bin_nm) + 1; ny = int((np.nanpercentile(N['y'], 99.9) - y0) / bin_nm) + 1
    def bid(x, y):
        ix = np.floor((x - x0) / bin_nm); iy = np.floor((y - y0) / bin_nm)
        ok = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        return np.where(ok, iy * nx + ix, -1).astype(np.int64)
    bn = bid(N['x'], N['y']); bc = bid(C['candx'], C['candy'])
    M = nx * ny
    cnt = lambda b, w=None: np.bincount(b[b >= 0], None if w is None else w[b >= 0].astype(float), minlength=M)
    nall, ngood = cnt(bn), cnt(bn, N['ok']); call, ckeep = cnt(bc), cnt(bc, C['ok'])
    occ = nall >= 20
    edges = np.percentile(nall[occ], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100])
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = occ & (nall >= a) & (nall <= b)
        rows.append(dict(dens_lo=float(a), dens_hi=float(b), bins=int(s.sum()), net_good_frac=float(ngood[s].sum() / nall[s].sum()),
                         cl_kept_frac=float(ckeep[s].sum() / max(call[s].sum(), 1)), cl_over_net=float(ckeep[s].sum() / max(ngood[s].sum(), 1))))
    # area emptied: occupied bins where the method keeps < 10 % of what it keeps per row on a median bin
    out = dict(deciles=rows)
    top = occ & (nall >= np.percentile(nall[occ], 95)); mid = occ & (nall >= np.percentile(nall[occ], 40)) & (nall <= np.percentile(nall[occ], 60))
    out['top5'] = dict(net_good_frac=float(ngood[top].sum() / nall[top].sum()), cl_kept_frac=float(ckeep[top].sum() / call[top].sum()),
                       cl_over_net=float(ckeep[top].sum() / ngood[top].sum()))
    out['mid'] = dict(net_good_frac=float(ngood[mid].sum() / nall[mid].sum()), cl_kept_frac=float(ckeep[mid].sum() / call[mid].sum()),
                      cl_over_net=float(ckeep[mid].sum() / ngood[mid].sum()))
    # which classical filter removes at the top-5 % bins (alone), vs the median bins
    fl = C['flags']; nmf = D.get('flag_names', [f'f{i}' for i in range(fl.shape[1])])
    rt = top[np.clip(bc, 0, M - 1)] & (bc >= 0); rm = mid[np.clip(bc, 0, M - 1)] & (bc >= 0)
    out['filters_alone_top5'] = dict(zip(nmf, fl[rt].mean(0).round(4).tolist())); out['filters_alone_mid'] = dict(zip(nmf, fl[rm].mean(0).round(4).tolist()))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    xc = [0.5 * (r['dens_lo'] + r['dens_hi']) for r in rows]
    ax[0].semilogx(xc, [r['net_good_frac'] for r in rows], 'o-', label='network v2: good / rows')
    ax[0].semilogx(xc, [r['cl_kept_frac'] for r in rows], 's-', color='C3', label='classical: kept / candidates')
    ax[0].set_xlabel('network rows per 100 nm bin (local density)', fontsize=7); ax[0].set_ylabel('fraction passing filters', fontsize=7); ax[0].legend(fontsize=7)
    ax[1].semilogx(xc, [r['cl_over_net'] for r in rows], 'ko-'); ax[1].set_xlabel('network rows per 100 nm bin', fontsize=7)
    ax[1].set_ylabel('classical kept / network good', fontsize=7)
    for a_ in ax: a_.tick_params(labelsize=6)
    fig.suptitle(f'MT: filter emptying vs local density (deciles; last bins 90-95, 95-99, 99-100 %). {NETNAME}', fontsize=8)
    plt.tight_layout(); fn = f'{OUT}/fig/MT_density_filters{tag}.png'; plt.savefig(fn, dpi=130); plt.close()
    out['fig'] = fn
    return out


def filter_table(C, names):
    fl = C['flags']; keep = C['ok']; n = len(keep)
    seq, rem = {}, np.zeros(n, bool)
    for i, nm in enumerate(names):
        f = fl[:, i] & ~rem; seq[nm] = float(f.mean()); rem |= fl[:, i]
    return dict(n_cand=int(n), n_keep=int(keep.sum()), kept_frac=float(keep.mean()), alone={nm: float(fl[:, i].mean()) for i, nm in enumerate(names)},
                sequential=seq, other_removed=float((~keep & ~rem).mean()))


def fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]; s = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    return s[0], s[1], np.sqrt(max(s[2] + s[0] ** 2 + s[1] ** 2, 0))


def em2(z, n_it=50):
    """symmetric two-Gaussian mixture (means m +- d/2, common sd) -> d, sd"""
    m, d, s = np.median(z), 40.0, max(np.std(z) / 2, 5); p = 0.5
    for _ in range(n_it):
        a = p * np.exp(-0.5 * ((z - m - d / 2) / s) ** 2); b = (1 - p) * np.exp(-0.5 * ((z - m + d / 2) / s) ** 2)
        r = a / np.maximum(a + b, 1e-300)
        m = np.mean(z - (r - 0.5) * d)
        d = 2 * np.sum((2 * r - 1) * (z - m)) / max(np.sum((2 * r - 1) ** 2), 1e-9)
        s = np.sqrt(np.mean(r * (z - m - d / 2) ** 2 + (1 - r) * (z - m + d / 2) ** 2)); p = np.clip(r.mean(), 0.2, 0.8)
    return abs(d), s


def npc_figures(D, tag, sr_over=10.0):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    S = {k: sel(T, T['ok'] & np.isfinite(T['z'])) for k, T in D.items() if k in ('net', 'cl', 'clp')}
    # pore centres from the network render (denser), used for both methods; centres refined per method
    T = S['net']; px = 10.0
    x0, y0 = np.percentile(T['x'], 0.5), np.percentile(T['y'], 0.5)
    h, xe, ye = np.histogram2d(T['x'], T['y'], bins=[np.arange(x0, np.percentile(T['x'], 99.5), px), np.arange(y0, np.percentile(T['y'], 99.5), px)])
    hs = gaussian_filter(h, 4.0)
    pk = (hs == maximum_filter(hs, size=13)) & (hs > np.percentile(hs[hs > 0], 90))
    ci, cj = np.nonzero(pk); cen = np.c_[xe[ci] + px / 2, ye[cj] + px / 2]
    pores = {k: [] for k in S}
    trees = {k: cKDTree(np.c_[S[k]['x'], S[k]['y']]) for k in S}
    for c in cen:
        rows = {}
        for k in S:
            idx = trees[k].query_ball_point(c, 100.0)
            if len(idx) < 30: break
            xx, yy = S[k]['x'][idx], S[k]['y'][idx]
            cx, cy, r = fit_circle(xx, yy)
            if not (30 < r < 80) or np.hypot(cx - c[0], cy - c[1]) > 40: break
            idx = trees[k].query_ball_point((cx, cy), 95.0)
            rows[k] = dict(cx=cx, cy=cy, r=r, idx=np.array(idx), zc=float(np.median(S[k]['z'][idx])))
        if len(rows) == len(S):
            for k in S: pores[k].append(rows[k])
    out = dict(n_pores=len(pores['net']), methods={})
    fig, axs = plt.subplots(3, 4, figsize=(17, 12.5))
    names = dict(net='network', cl='classical published-style', clp='classical pixel MLE')
    for i, k in enumerate(S):
        P_ = pores[k]
        if not P_: continue
        dx = np.concatenate([S[k]['x'][p['idx']] - p['cx'] for p in P_]); dy = np.concatenate([S[k]['y'][p['idx']] - p['cy'] for p in P_])
        # sample axial drift (not removed by the phase-field drift term, which only takes the interferometer phase): per
        # 500-window block, median of z - pore mean over all pore localizations, smoothed, iterated; pores centred on their mean z
        zz = np.concatenate([S[k]['z'][p['idx']] for p in P_]); ww = np.concatenate([S[k]['w'][p['idx']] for p in P_])
        pid = np.concatenate([np.full(len(p['idx']), i_) for i_, p in enumerate(P_)])
        blk = (ww // 500).astype(int); nb = blk.max() + 1; drift = np.zeros(nb)
        for _ in range(4):
            zc_ = np.bincount(pid, zz - drift[blk]) / np.maximum(np.bincount(pid), 1)
            r = zz - drift[blk] - zc_[pid]
            md = np.array([np.median(r[blk == b]) if (blk == b).sum() > 50 else 0.0 for b in range(nb)])
            drift = drift + gaussian_filter(md, 1.0)
        zc_ = np.bincount(pid, zz - drift[blk]) / np.maximum(np.bincount(pid), 1)
        dz = zz - drift[blk] - zc_[pid]
        zdrift_ptp = float(np.ptp(drift))
        rr = np.hypot(dx, dy)
        hr, er = np.histogram(rr, bins=np.arange(0, 96, 2)); rc = 0.5 * (er[1:] + er[:-1])
        hn = hr / (2 * np.pi * rc); rpk = float(rc[np.argmax(gaussian_filter(hn.astype(float), 1.0) * (rc > 20))])
        ring = (rr > 25) & (rr < 85)
        d, sdz = em2(dz[ring & (np.abs(dz) < 120)])
        hz, ez_ = np.histogram(dz[ring], bins=np.arange(-120, 121, 2)); zc2 = 0.5 * (ez_[1:] + ez_[:-1])
        hzs = gaussian_filter(hz.astype(float), 1.5)
        lm = np.nonzero((hzs[1:-1] > hzs[:-2]) & (hzs[1:-1] >= hzs[2:]))[0] + 1
        lm = lm[np.argsort(-hzs[lm])][:2]
        if len(lm) == 2:
            a_, b_ = sorted(lm); pk_sep = float(zc2[b_] - zc2[a_]); dip = float(hzs[a_:b_ + 1].min() / min(hzs[a_], hzs[b_]))
        else:
            pk_sep, dip = float('nan'), 1.0
        np.savez(f'{OUT}/real/npc_avgpore_{k}{tag}.npz', dx=dx.astype(np.float32), dy=dy.astype(np.float32), dz=dz.astype(np.float32))
        out['methods'][k] = dict(n_locs=int(len(rr)), ring_radius_peak_nm=rpk, ring_radius_median_nm=float(np.median(rr[ring])),
                                 circle_fit_radius_median_nm=float(np.median([p['r'] for p in P_])),
                                 two_ring_z_sep_nm=float(d), ring_z_sd_nm=float(sdz), peak_sep_nm=pk_sep, dip_ratio=dip, z_drift_ptp_nm=zdrift_ptp, n_pores=len(P_))
        ax = axs[0, i]; h2, _, _ = np.histogram2d(dy, dx, bins=np.arange(-100, 101, 2.5))
        ax.imshow(gaussian_filter(h2, 0.7), cmap='magma', extent=[-100, 100, 100, -100]); ax.set_title(f'{names[k]}: averaged pore x-y ({len(P_)} pores)\nring peak r {rpk:.1f} nm', fontsize=8)
        ax = axs[1, i]; h3, _, _ = np.histogram2d(dz, dx, bins=[np.arange(-120, 121, 3), np.arange(-100, 101, 2.5)])
        ax.imshow(gaussian_filter(h3, 0.7)[::-1], cmap='magma', extent=[-100, 100, -120, 120], aspect='auto')
        ax.set_title(f'averaged pore x-z: z peaks {pk_sep:.0f} nm apart, dip {dip:.2f}', fontsize=8)
        ax = axs[2, i]; ax.hist(dz[ring & (np.abs(dz) < 120)], bins=np.arange(-120, 121, 4), color='k'); ax.set_title('z of ring locs (pore-centred)', fontsize=8)
    # overview render (classical) in the 4th column
    zr = tuple(np.percentile(S['net']['z'], [2, 98]))
    bx = (np.percentile(S['net']['x'], 0.5), np.percentile(S['net']['x'], 99.5), np.percentile(S['net']['y'], 0.5), np.percentile(S['net']['y'], 99.5))
    render(axs[0, 3], S['net']['x'], S['net']['y'], S['net']['z'], bx, zr, 20.0, 'network overview')
    render(axs[1, 3], S['cl']['x'], S['cl']['y'], S['cl']['z'], bx, zr, 20.0, 'classical overview')
    axs[2, 3].axis('off')
    plt.tight_layout(); fn = f'{OUT}/fig/NPC_pores{tag}.png'; plt.savefig(fn, dpi=130); plt.close()
    out['fig_avg'] = fn
    # zooms on single pores: 6 pores with the most network localizations, side by side
    if pores['net']:
        top = np.argsort([-len(p['idx']) for p in pores['net']])[:6]
        fig, axs = plt.subplots(3, 6, figsize=(18, 9))
        for j, q in enumerate(top):
            c = (pores['net'][q]['cx'], pores['net'][q]['cy'])
            for i, k in enumerate(S):
                render(axs[i, j], S[k]['x'], S[k]['y'], S[k]['z'], (c[0] - 150, c[0] + 150, c[1] - 150, c[1] + 150), (zr[0], zr[1]), 2.5,
                       f'{names[k]}, pore {q}', vmax=None)
        plt.tight_layout(); fn2 = f'{OUT}/fig/NPC_pore_zooms{tag}.png'; plt.savefig(fn2, dpi=130); plt.close()
        out['fig_zoom'] = fn2
    # big overview side by side
    fig, axs = plt.subplots(1, 2, figsize=(20, 10 * (bx[3] - bx[2]) / (bx[1] - bx[0]) + 1))
    for i, k in enumerate(('net', 'cl')):
        render(axs[i], S[k]['x'], S[k]['y'], S[k]['z'], bx, zr, 10.0, f'{names[k]}: {len(S[k]["x"])} locs, hue z {zr[0]:.0f}..{zr[1]:.0f} nm')
    plt.tight_layout(); fn3 = f'{OUT}/fig/NPC_overview{tag}.png'; plt.savefig(fn3, dpi=110); plt.close()
    out['fig_overview'] = fn3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='MT'); ap.add_argument('--net', required=True); ap.add_argument('--tag', default='')
    ap.add_argument('--files', type=int, nargs='*'); ap.add_argument('--wmax', type=int, default=None)
    ap.add_argument('--work', required=True, help='4Pi work folder')
    a = ap.parse_args()
    global OUT; P4 = a.work; OUT = f'{P4}/classical'
    os.makedirs(f'{OUT}/fig', exist_ok=True)
    D = load(f'{OUT}/real/cl_{a.cell}{a.tag}_all.npz', a.net, a.files, a.wmax, zdrift=(a.cell == 'MT'))
    if a.cell == 'MT':
        import time; t0 = time.time(); lg = lambda s: print(f'[{time.time() - t0:.0f}s] {s}', flush=True)
        fs0 = sorted(__import__('glob').glob(f'{OUT}/real/cl_MT_[0-9][0-9][0-9]{a.tag}_summary.json'))[0]
        D['flag_names'] = json.load(open(fs0))['flag_names']
        zd = D['zdrift']; np.savez(f'{OUT}/real/zdrift_MT{a.tag}.npz', net=zd['net'], cl=zd['cl'], block_windows=500)
        n_ = min(len(zd['net']), len(zd['cl']))
        zinfo = dict(ptp_net_nm=float(np.ptp(zd['net'])), ptp_cl_nm=float(np.ptp(zd['cl'])),
                     rms_diff_net_cl_nm=float(np.sqrt(np.mean((zd['net'][:n_] - zd['cl'][:n_] - np.mean(zd['net'][:n_] - zd['cl'][:n_])) ** 2))))
        lg(f'z drift {zinfo}')
        reg = json.load(open(f'{P4}/patches/regions_MT.json'))
        boxes = [dict(b, id=i) for i, b in enumerate(reg['boxes']) if b.get('save')]
        D['crossings'] = pick_xings(D); lg(f'crossings {D["crossings"]}')
        fo = mt_overview(D, boxes, a.tag); lg(fo)
        fx, xr = mt_crossings(D, a.tag); lg(fx)
        fw = filament_widths(D, a.tag); lg(fw['fig'])
        de = density_emptying(D, a.tag); lg(de['fig'])
        ft = filter_table(D['cl'], D['flag_names'])
        res, L = mt_figures(D, boxes, a.tag); lg('boxes done')
        r0 = numbers(D, 'MT whole field (pixel MLE: boxes 0 and 1 only)'); r0['drift_ref_offset_nm'] = D.get('drift_ref_offset_nm'); lg('whole-field numbers done')
        res.insert(0, r0); L = fmt_numbers(r0) + L
        ns = json.load(open(a.net.replace('_all.npz', '_summary.json')))
        H = [f"figures: {fo}, {fx}, {fw['fig']}, {de['fig']}, " + ', '.join(r['fig'] for r in res if 'fig' in r),
             f"\nNetwork side: {NETNAME}. {PIXNOTE}; whole-field pixel-MLE numbers are boxes 0 and 1 only.",
             f"\n## Sample z drift (structure-based, 500-window blocks, each table on its own)\nptp network {zinfo['ptp_net_nm']:.0f} nm, classical {zinfo['ptp_cl_nm']:.0f} nm; rms difference of the two curves {zinfo['rms_diff_net_cl_nm']:.1f} nm",
             f"\n## Candidates and filters\nclassical (wavelet detector): {ft['n_cand']} candidates, kept {ft['n_keep']} ({ft['kept_frac']:.3f}); network v2: {ns['n_all']} rows, good {ns['n_good']} ({ns['n_good'] / ns['n_all']:.3f})",
             '| classical filter | removed alone | removed sequentially |', '|---|---|---|']
        H += [f"| {k} | {ft['alone'][k]:.3f} | {ft['sequential'][k]:.3f} |" for k in D['flag_names']]
        H.append(f"| other (non-finite etc.) | | {ft['other_removed']:.3f} |")
        H.append('\n## Filament cross-section (isolated straight segments)')
        H.append('| method | segments | lateral FWHM pooled / median per segment [IQR] (nm) | axial FWHM pooled / median per segment [IQR] (nm) |'); H.append('|---|---|---|---|')
        for k, m in fw['methods'].items():
            H.append(f"| {k} | {m['n_segments']} | {m['fwhm_lat_pooled_nm']:.1f} / {m['fwhm_lat_median_nm']:.1f} [{m['fwhm_lat_iqr_nm'][0]:.0f}-{m['fwhm_lat_iqr_nm'][1]:.0f}] | {m['fwhm_ax_pooled_nm']:.1f} / {m['fwhm_ax_median_nm']:.1f} [{m['fwhm_ax_iqr_nm'][0]:.0f}-{m['fwhm_ax_iqr_nm'][1]:.0f}] |")
        H.append('\n## Filter emptying vs local density (100 nm bins by network rows)')
        H.append('| density pct bin | net good / rows | classical kept / candidates | classical kept / network good |'); H.append('|---|---|---|---|')
        pc = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-95', '95-99', '99-100']
        H += [f"| {p} | {r['net_good_frac']:.3f} | {r['cl_kept_frac']:.3f} | {r['cl_over_net']:.3f} |" for p, r in zip(pc, de['deciles'])]
        H.append(f"top 5 % density: net {de['top5']['net_good_frac']:.3f}, classical {de['top5']['cl_kept_frac']:.3f}, ratio {de['top5']['cl_over_net']:.3f}; median (40-60 %): net {de['mid']['net_good_frac']:.3f}, classical {de['mid']['cl_kept_frac']:.3f}, ratio {de['mid']['cl_over_net']:.3f}")
        H.append('| classical filter (alone) | top 5 % density bins | median-density bins |'); H.append('|---|---|---|')
        H += [f"| {k} | {de['filters_alone_top5'][k]:.3f} | {de['filters_alone_mid'][k]:.3f} |" for k in D['flag_names']]
        H.append('\n## Crossings (within 300 nm of the junction)'); H.append('| X | net good | classical kept | pixel MLE | classical kept / cand | net good / rows |'); H.append('|---|---|---|---|---|---|')
        H += [f"| X{r['id']} | {r['n_net']} | {r['n_cl']} | {r['n_clp']} | {r['cl_kept_frac_of_cand']:.3f} | {r['net_good_frac_of_rows']:.3f} |" for r in xr]
        L = H + L
        res.append(dict(zdrift=zinfo, filters=ft, filament=dict((k, v) for k, v in fw.items()), density=de, crossings=xr, figs=[fo, fx, fw['fig'], de['fig']]))
    else:
        res = [numbers(D, 'NPC whole field')]; res[0]['drift_ref_offset_nm'] = D.get('drift_ref_offset_nm'); L = fmt_numbers(res[0])
        po = npc_figures(D, a.tag); res.append(po)
        L.append(f"\n## NPC pores ({po['n_pores']} pores found on the network render, refitted per method)")
        L.append('| method | ring radius (peak / median, nm) | circle fit radius median | z peak separation (nm) | dip (valley / peak) | 2-Gaussian sep / sd (nm) | z drift removed (ptp nm) |'); L.append('|---|---|---|---|---|---|---|')
        for k, m in po['methods'].items():
            L.append(f"| {k} | {m['ring_radius_peak_nm']:.1f} / {m['ring_radius_median_nm']:.1f} | {m['circle_fit_radius_median_nm']:.1f} | {m['peak_sep_nm']:.0f} | {m['dip_ratio']:.2f} | {m['two_ring_z_sep_nm']:.1f} / {m['ring_z_sd_nm']:.1f} | {m['z_drift_ptp_nm']:.0f} |")
    json.dump(res, open(f'{OUT}/compare_{a.cell}{a.tag}.json', 'w'), indent=1, default=float)
    open(f'{OUT}/compare_{a.cell}{a.tag}.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
