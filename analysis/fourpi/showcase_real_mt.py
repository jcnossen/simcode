"""Showcase figure of the REAL 4Pi-SIMFLUX microtubule data (same layout as the simulated showcase_render of
showcase4pi_fig.py, without a ground-truth row).

Rows (methods): 4Pi-SIMCODE network (head v10), classical 4Pi-SIMFLUX published-style, classical pixel MLE (exists only in
the MT boxes 0 and 1 of out/4pi/patches/regions_MT.json), astigmatism-only z (z_ast of the classical table on the
published-style rows and x, y). Columns: x-y overview coloured by z with the zoom boxes, zoom 1 (crossing with a clear z
separation), zoom 2 (filaments overlapping in x-y at different heights) + an x-z / y-z section through it, tube
cross-section (isolated straight segments, pooled, in-plane x axial) and its 1D lateral / axial profiles with FWHM.

Everything that is chosen (box of the overview, zooms, section strip, tube segments) is chosen deterministically from the
classical published-style table only (kept rows inside the drift-corrected cores of boxes 0 and 1), so a rerun with a
different network table keeps the same regions.

Corrections: lateral drift = the classical DME curve (classical table `drift`, applied to the per-file network tables:
xd = x - drift[frame + 3]) or the network table's own xd, yd (--net_all); a remaining constant lateral offset of the network
is measured on matched emitters (same window, < 64 nm on raw positions) and removed. Sample z drift: the structure-based
estimator fig_cl.zdrift_struct on the classical published-style kept rows of the whole movie, the SAME curve subtracted from
every row (network included). Constant z offsets of network, pixel MLE and z_ast to the published-style z are measured on
the same emitters / rows and removed (z of a 4Pi table has an arbitrary reference).

Usage:
  python analysis/fourpi/showcase_real_mt.py --net_locs 'out/4pi/net/locs_MT_*_v10*.npz' --tag v10_first [--samefiles]
  python analysis/fourpi/showcase_real_mt.py --net_all out/4pi/net/net_MT_v10_all.npz --tag v10_full
Writes out/4pi/fig/MT_showcase_real_<tag>.png and .json.
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, convolve, label, distance_transform_edt
from scipy.spatial import cKDTree
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
sys.path.insert(0, f'{REPO}/analysis/fourpi_classical')
sys.path.insert(0, f'{REPO}/analysis/figures')
from simcode.fourpi.setup import P4
import fig_cl as FC                     # read-only: zdrift_struct, em2, fwhm (none of these write files)
from zoompick import pick_crossings
from skel import skeletonize
FC.OUT = f'{P4}/fig/tmp_showcase'       # never write into the classical worker's directory
from simcode.fourpi.setup import PX_NM as PX
FPF = 18000                              # frames per file
OUTD = f'{P4}/fig'
METH = [('net', '4Pi-SIMCODE (network)'), ('cl', 'classical 4Pi-SIMFLUX,\npublished-style'),
        ('clp', 'classical 4Pi-SIMFLUX,\npixel MLE'), ('ast', 'astigmatism-only z')]
ZCMAP = LinearSegmentedColormap.from_list('zdiv', ['#3987e5', '#c3c2b7', '#eb6834'])
C_LAT, C_AX = '#2a78d6', '#eb6834'
VEX = 2.0                                # vertical exaggeration of the section panel
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False})
t0 = time.time()
def lg(s): print(f'[{time.time() - t0:5.0f}s] {s}', flush=True)


# ----------------------------------------------------------------------------------------------------------- loading
CL_EXTRA = ('sx', 'sy', 'sz', 'sxp', 'syp', 'szp', 'N', 'nn', 'chi2px', 'C', 'R')       # quality columns (selection analysis)
NET_EXTRA = ('chi2', 'dof', 'nn', 'pdet', 'cs', 'kin', 'Tsum')


def load_classical(fn, extra=()):
    Z = np.load(fn)
    C = {k: Z[k] for k in ('x', 'y', 'xd', 'yd', 'z', 'z_ast', 'keep', 'xpd', 'ypd', 'zp', 'keep_pix', 'frame', 'file_index') + tuple(extra)}
    C['drift'] = Z['drift']
    m = C['keep'] | C['keep_pix']
    C = {k: (v[m] if k != 'drift' else v) for k, v in C.items()}
    return C


def zdrift_curve(C):
    m = C['keep'] & np.isfinite(C['z']) & np.isfinite(C['xd'])
    d = FC.zdrift_struct(C['xd'][m] * PX, C['yd'][m] * PX, C['z'][m] * 1000, C['frame'][m] // 6)
    return d                                                   # nm per 500-window block (global windows)


def apply_z(z_nm, frame, d):
    return z_nm - d[np.clip(frame // 6 // 500, 0, len(d) - 1)]


def load_net(a, drift, extra=()):
    """network rows -> dict(x, y [nm, lateral drift corrected], z [nm], ux, uy [nm raw], frame (global), f (file))"""
    if a.net_all:
        N = np.load(a.net_all)
        fidx = N['fidx']; w = N['w']
        T = dict(x=N['x'], y=N['y'], z=N['z'], sx=N['sx'], sy=N['sy'], sz=N['sz'], N=N['N'], good=N['good'], fidx=fidx, w=w,
                 xd=N['xd'], yd=N['yd'], **{k: N[k] for k in extra if k in N.files})
        files = sorted(np.unique(fidx).tolist()); src = [a.net_all]
    else:
        fs = sorted(set(sum([glob.glob(g) for g in a.net_locs], [])))
        byf = {}
        for f in fs:                                           # one table per file: whole-field file preferred over the box file
            m = re.search(r'locs_MT_(\d{3})_[^_]+?(_box)?\.npz$', os.path.basename(f))
            if not m: continue
            k = int(m.group(1))
            if k not in byf or (byf[k].endswith('_box.npz') and not f.endswith('_box.npz')): byf[k] = f
        parts = []
        for k, f in sorted(byf.items()):
            L = np.load(f)
            n = len(L['x']); fi = int(L['file_index']) if L['file_index'].ndim == 0 else L['file_index']
            parts.append(dict(x=L['x'], y=L['y'], z=L['z'], sx=L['sx'], sy=L['sy'], sz=L['sz'], N=L['N'], good=L['good'],
                              fidx=np.full(n, fi, np.int64) if np.ndim(fi) == 0 else fi, w=L['w'], **{k: L[k] for k in extra if k in L.files}))
        T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if all(k in p for p in parts)}
        files = sorted(byf); src = [byf[k] for k in files]
    T['frame'] = T['fidx'] * FPF + 6 * T['w']
    if 'xd' not in T:
        fi_ = np.clip(T['frame'] + 3, 0, len(drift) - 1)
        T['xd'] = T['x'] - drift[fi_, 0]; T['yd'] = T['y'] - drift[fi_, 1]
    s = T['good'] & (T['N'] >= 1000) & (T['sz'] < 0.025) & (T['sx'] < 0.12) & (T['sy'] < 0.12) & np.isfinite(T['z'])
    T = {k: v[s] for k, v in T.items()}
    return T, files, src


# ------------------------------------------------------------------------------------------------- region geometry
def box_cores(boxes, drift, inset=1.0):
    """drift-corrected rectangle (nm) in which a structure stays inside the raw box for the whole movie"""
    out = []
    for b in boxes:
        x0 = (b['x0'] - drift[:, 0].min() + inset) * PX; x1 = (b['x1'] - drift[:, 0].max() - inset) * PX
        y0 = (b['y0'] - drift[:, 1].min() + inset) * PX; y1 = (b['y1'] - drift[:, 1].max() - inset) * PX
        out.append((x0, x1, y0, y1))
    return out


def inrect(x, y, r): return (x >= r[0]) & (x < r[1]) & (y >= r[2]) & (y < r[3])


def two_arms(x, y, z, c, r0=150.0, r1=550.0, half=45.0):
    """the two straight lines through c holding most locs of the annulus r0..r1 (3 deg steps, >= 30 deg apart):
    fraction of the annulus on them and the median z of each line"""
    v = np.c_[x - c[0], y - c[1]]; r = np.hypot(v[:, 0], v[:, 1]); a = (r > r0) & (r < r1)
    v, zz = v[a], z[a]
    if len(v) < 100: return None
    th = np.deg2rad(np.arange(0, 180, 3.0)); nrm = np.c_[-np.sin(th), np.cos(th)]
    on = np.abs(v @ nrm.T) < half                                   # (n, n_theta)
    cnt = on.sum(0); i1 = int(np.argmax(cnt))
    dth = np.abs((np.arange(len(th)) - i1 + 30) % 60 - 30) * 3.0  # angular distance (deg, mod 180)
    c2 = np.where(dth >= 30, cnt, -1); i2 = int(np.argmax(c2))
    m1, m2 = on[:, i1] & ~on[:, i2], on[:, i2] & ~on[:, i1]
    if m1.sum() < 30 or m2.sum() < 30: return None
    return dict(clean=float((on[:, i1] | on[:, i2]).mean()), z1=float(np.median(zz[m1])), z2=float(np.median(zz[m2])),
                n1=int(m1.sum()), n2=int(m2.sum()), ang_deg=float(dth[i2]))


def pick_zoom1(S, core, size=1200.0):
    """crossings (zoompick junctions) inside the core: two clean straight arms (>= 50 % of the 150-550 nm annulus on two
    lines), scored by the z difference of the two arms"""
    x, y, z = S['x'], S['y'], S['z']
    m = inrect(x, y, core); x, y, z = x[m], y[m], z[m]
    lim = tuple(v / 1000 for v in core)
    bs = pick_crossings(x / 1000, y / 1000, lim=lim, n=40, size=0.4, px=0.02, margin=size / 2000 + 0.05)
    res = []
    for b in bs:
        c = ((b[0] + b[1]) / 2 * 1000, (b[2] + b[3]) / 2 * 1000)
        r = two_arms(x, y, z, c)
        if r is None: continue
        res.append(dict(c=c, dz=abs(r['z1'] - r['z2']), **r))
    ok = [r for r in res if r['clean'] >= 0.5 and r['ang_deg'] >= 40]
    return ok if ok else res


def bimodal_bins(x, y, z, rect, bin_nm=60.0, nmin=40):
    import pandas as pd
    ix = np.floor((x - rect[0]) / bin_nm).astype(np.int64); iy = np.floor((y - rect[2]) / bin_nm).astype(np.int64)
    nx = int((rect[1] - rect[0]) / bin_nm); ny = int((rect[3] - rect[2]) / bin_nm)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    key = iy[ok] * nx + ix[ok]; g = pd.Series(z[ok]).groupby(key)
    q = g.quantile([0.1, 0.9]).unstack(); cnt = g.size()
    mid = 0.5 * (q[0.1] + q[0.9])
    lo = pd.Series(z[ok] < mid.reindex(key).to_numpy()).groupby(key).mean()
    bim = (cnt >= nmin) & ((q[0.9] - q[0.1]) > 120) & (lo > 0.2) & (lo < 0.8)
    B = np.zeros(ny * nx, bool); B[bim.index[bim.to_numpy()].to_numpy()] = True
    return B.reshape(ny, nx)


def pick_zoom2(S, core, avoid, size=1500.0, bin_nm=60.0, margin=150.0):
    """the size x size window with the most 60 nm bins holding two z populations (> 120 nm apart, 10-90 %)"""
    B = bimodal_bins(S['x'], S['y'], S['z'], core, bin_nm)
    mb = int(margin / bin_nm); k = int(size / bin_nm); I = np.cumsum(np.cumsum(np.pad(B.astype(int), ((1, 0), (1, 0))), 0), 1)
    best = (-1, None)
    for iy in range(mb, B.shape[0] - k + 1 - mb):
        for ix in range(mb, B.shape[1] - k + 1 - mb):
            c = (core[0] + (ix + k / 2) * bin_nm, core[2] + (iy + k / 2) * bin_nm)
            if any(max(abs(c[0] - a[0]), abs(c[1] - a[1])) < (size + a[2]) / 2 + 50 for a in avoid): continue
            s = I[iy + k, ix + k] - I[iy, ix + k] - I[iy + k, ix] + I[iy, ix]
            if s > best[0]: best = (int(s), c)
    return best[1], best[0], B


def pick_section(S, c, size, width=150.0):
    """axis-aligned strip through zoom 2 across its dominant filament direction (x-z section = strip along x at fixed y
    when the filaments run mostly along y, else y-z), at the position with the most 30 nm bins holding two z populations"""
    rect = (c[0] - size / 2, c[0] + size / 2, c[1] - size / 2, c[1] + size / 2)
    m = inrect(S['x'], S['y'], rect); v = np.c_[S['x'][m] - c[0], S['y'][m] - c[1]]
    # dominant direction from the structure tensor of the density gradient
    H = gaussian_filter(np.histogram2d(v[:, 1], v[:, 0], bins=int(size / 20))[0], 1.5)
    gy, gx = np.gradient(H); ang = 0.5 * np.arctan2(2 * (gx * gy).sum(), ((gx ** 2) - (gy ** 2)).sum())   # gradient dir
    along = 'y' if abs(np.cos(ang)) < abs(np.sin(ang)) else 'x'       # gradient mostly along x -> filaments along y -> strip along x
    B = bimodal_bins(S['x'], S['y'], S['z'], rect, 30.0, 25)
    k = int(width / 30.0)
    p = np.convolve(B.sum(1 if along == 'x' else 0), np.ones(k), 'valid').astype(float)
    # a strip lying along a filament (occupied over > 35 % of its length) is not a cross-section: excluded
    nb = int(round(size / 30.0)); e = np.arange(nb + 1) * 30.0
    occ = np.zeros(len(p))
    for i in range(len(p)):
        a0 = (rect[2] if along == 'x' else rect[0]) + i * 30.0
        if along == 'x': mm = m & (S['y'] >= a0) & (S['y'] < a0 + width); q = S['x'][mm] - rect[0]
        else: mm = m & (S['x'] >= a0) & (S['x'] < a0 + width); q = S['y'][mm] - rect[2]
        h = np.histogram(q, e)[0]; occ[i] = np.mean(h >= max(5, 0.1 * h.max())) if h.max() > 0 else 0
    p[occ > 0.35] = -1
    lo = int(0.2 * len(p)); p[:lo] = -1; p[len(p) - lo:] = -1                 # strip in the middle 60 % of the zoom
    i = int(np.argmax(p)); a0 = (rect[2] if along == 'x' else rect[0]) + i * 30.0
    return dict(along=along, a0=float(a0), a1=float(a0 + width), n=int(p.max()), gradient_angle_deg=float(np.rad2deg(ang)))


def find_segments(S, cores, px=20.0, min_len=500.0, jex=250.0, iso=(100.0, 250.0), iso_frac=0.05):
    """isolated straight tube segments on the classical render (fig_cl.filament_widths, relaxed for the dense boxes):
    skeleton minus 250 nm around junctions, >= 0.5 um, perpendicular rms <= 15 nm, < 5 % of the mask 100-250 nm off the line; no second
    z population along the segment (two tubes on top of each other) and fully inside a box core"""
    segs = []
    for core in cores:
        m_ = inrect(S['x'], S['y'], core); x, y, z = S['x'][m_], S['y'][m_], S['z'][m_]
        x0, y0 = core[0], core[2]
        bins = [np.arange(core[2], core[3] + px, px), np.arange(core[0], core[1] + px, px)]
        H = np.histogram2d(y, x, bins=bins)[0]
        Hs = gaussian_filter(H, 2.0); m = Hs > np.percentile(Hs[Hs > 0], 70)
        sk = skeletonize(m); nb = convolve(sk.astype(int), np.ones((3, 3), int), mode='constant') - sk
        J = sk & (nb >= 3); dj = distance_transform_edt(~J)
        lab, nl = label(sk & (dj > jex / px), structure=np.ones((3, 3)))
        my, mx = np.nonzero(m); mxy = np.c_[x0 + (mx + 0.5) * px, y0 + (my + 0.5) * px]; mtree = cKDTree(mxy)
        tr = cKDTree(np.c_[x, y])
        for i in range(1, nl + 1):
            yy, xx = np.nonzero(lab == i)
            if len(xx) < min_len / px: continue
            P = np.c_[x0 + (xx + 0.5) * px, y0 + (yy + 0.5) * px]; c = P.mean(0)
            _, _, vt = np.linalg.svd(P - c, full_matrices=False); d = vt[0]; nrm = np.r_[-d[1], d[0]]
            al = (P - c) @ d; pe = (P - c) @ nrm
            if np.sqrt(np.mean(pe ** 2)) > 15 or np.ptp(al) < min_len: continue
            a0, a1 = al.min() + 100, al.max() - 100
            ends = np.array([c + a0 * d, c + a1 * d])
            if not (inrect(ends[:, 0] - 250, ends[:, 1] - 250, core).all() and inrect(ends[:, 0] + 250, ends[:, 1] + 250, core).all()): continue
            q = mxy[mtree.query_ball_point(c, np.ptp(al) / 2 + 400)]; qa = (q - c) @ d; qp = np.abs((q - c) @ nrm)
            inr = (qa > a0) & (qa < a1)
            if np.mean((qp[inr] > iso[0]) & (qp[inr] < iso[1])) > iso_frac: continue
            idx = np.array(tr.query_ball_point(c, np.ptp(al) / 2 + 300), int)
            v = np.c_[x[idx], y[idx]] - c; sa = v @ d; sp = v @ nrm
            w = (sa > a0) & (sa < a1) & (np.abs(sp) < 60)
            if w.sum() < 150: continue
            p = np.polyfit(sa[w], z[idx][w], 2); r = z[idx][w] - np.polyval(p, sa[w])
            if np.mean(np.abs(r - np.median(r)) > 120) > 0.05: continue          # a second tube above / below
            segs.append(dict(c=c.tolist(), d=d.tolist(), a0=float(a0), a1=float(a1)))
    return segs


def cross_section(T, segs):
    """per method: in-plane offset u and axial offset v about each segment's own centre line (quadratic fits of sp(s) and
    z(s) on this method's locs, iterated on the core), pooled over segments"""
    tr = cKDTree(np.c_[T['x'], T['y']]); U, V, nseg = [], [], 0
    for s in segs:
        c, d = np.array(s['c']), np.array(s['d']); nrm = np.r_[-d[1], d[0]]
        idx = np.array(tr.query_ball_point(c, (s['a1'] - s['a0']) / 2 + 400), int)
        if len(idx) == 0: continue
        v = np.c_[T['x'][idx], T['y'][idx]] - c; sa = v @ d; sp = v @ nrm; z = T['z'][idx]
        w = (sa > s['a0']) & (sa < s['a1']) & (np.abs(sp) < 200)
        if w.sum() < 40: continue
        sa, sp, z = sa[w], sp[w], z[w]
        core = np.abs(sp) < 80
        for _ in range(3):
            if core.sum() < 20: break
            pl = np.polyfit(sa[core], sp[core], 2 if core.sum() > 100 else 1)
            u = sp - np.polyval(pl, sa)
            pz = np.polyfit(sa[core], z[core], 2 if core.sum() > 100 else 1)
            vz = z - np.polyval(pz, sa)
            core = (np.abs(u) < 60) & (np.abs(vz) < 100)
        if core.sum() < 20: continue
        U.append(u); V.append(vz); nseg += 1
    if not U: return np.zeros(0), np.zeros(0), 0
    return np.concatenate(U), np.concatenate(V), nseg


def peaks(v, lo=-60, hi=60, bw=2.0, sm=1.5):
    """two highest local maxima of the smoothed profile within +-40 nm: separation and dip (valley / lower peak)"""
    h, e = np.histogram(v, bins=np.arange(lo, hi + bw, bw)); c = 0.5 * (e[1:] + e[:-1]); hs = gaussian_filter(h.astype(float), sm)
    lm = np.nonzero((hs[1:-1] > hs[:-2]) & (hs[1:-1] >= hs[2:]) & (np.abs(c[1:-1]) < 40))[0] + 1
    if len(lm) < 2: return dict(resolved=False, sep_nm=None, dip=1.0)
    lm = np.sort(lm[np.argsort(-hs[lm])][:2]); a, b = lm
    dip = float(hs[a:b + 1].min() / min(hs[a], hs[b]))
    return dict(resolved=bool(dip < 0.9), sep_nm=float(c[b] - c[a]), dip=dip)


# --------------------------------------------------------------------------------------------------------- render
def zrgb(x, y, z, x0, y0, ext, sr, zr, blur=0.8, ext_y=None):
    n = int(round(ext / sr)); ny = int(round((ext if ext_y is None else ext_y) / sr))
    ix = np.floor((x - x0) / sr).astype(int); iy = np.floor((y - y0) / sr).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < ny)
    c = np.zeros((ny, n)); zs = np.zeros((ny, n))
    np.add.at(c, (iy[ok], ix[ok]), 1); np.add.at(zs, (iy[ok], ix[ok]), z[ok])
    cs = gaussian_filter(c, blur); zz = gaussian_filter(zs, blur) / np.maximum(cs, 1e-9)
    v = np.clip(cs / max(np.percentile(cs[cs > 0], 97), 1e-9), 0, 1) ** 0.7 if (cs > 0).any() else cs
    return ZCMAP(np.clip((zz - zr[0]) / (zr[1] - zr[0]), 0, 1))[..., :3] * v[..., None]


def dens_img(a, b, a0, a1, b0, b1, sr, blur=0.8):
    na, nb = int(round((a1 - a0) / sr)), int(round((b1 - b0) / sr))
    ia = np.floor((a - a0) / sr).astype(int); ib = np.floor((b - b0) / sr).astype(int)
    ok = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    c = np.zeros((nb, na)); np.add.at(c, (ib[ok], ia[ok]), 1); c = gaussian_filter(c, blur)
    return c / max(np.percentile(c[c > 0], 99.7), 1e-9) if (c > 0).any() else c


def scalebar(ax, frac, lab, x=0.05, y=0.06):
    ax.plot([x, x + frac], [y, y], transform=ax.transAxes, color='w', lw=2, solid_capstyle='butt')
    ax.text(x + frac / 2, y + 0.03, lab, transform=ax.transAxes, color='w', ha='center', va='bottom', fontsize=7)


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cl', default=f'{P4}/classical/real/cl_MT_wv_all.npz')
    ap.add_argument('--net_locs', nargs='*', default=None, help='glob(s) of per-file network tables (raw x, y)')
    ap.add_argument('--net_all', default=None, help='merged network table with its own xd, yd (post_w4pi.py)')
    ap.add_argument('--head', default='v10'); ap.add_argument('--tag', required=True)
    ap.add_argument('--samefiles', action='store_true', help='also write <tag>_samefiles: classical rows restricted to the network files')
    ap.add_argument('--regions', default=f'{P4}/patches/regions_MT.json')
    a = ap.parse_args()
    assert a.net_locs or a.net_all
    os.makedirs(OUTD, exist_ok=True)
    R, G, segs, info, files, src, ncl = prepare(a)
    variants = [(a.tag, None)]
    if a.samefiles: variants.append((a.tag + '_samefiles', files))
    for tag, fsel in variants:
        Rv = {k: (v if (k == 'net' or fsel is None) else {q: w[np.isin(v['f'], fsel)] for q, w in v.items()}) for k, v in R.items()}
        nfiles = {k: (len(files) if (k == 'net' or fsel is not None) else ncl) for k in Rv}
        J = figure(Rv, G, segs, tag, a, info, nfiles, files)
        J.update(info=info, geometry=G, network_files=files, network_sources=src, n_files=nfiles)
        json.dump(J, open(f'{OUTD}/MT_showcase_real_{tag}.json', 'w'), indent=1, default=float)
        lg(f'-> {OUTD}/MT_showcase_real_{tag}.png')


def prepare(a, cl_extra=(), net_extra=()):
    """tables of all rows (nm, drift corrected, registered; z drift removed), regions, segments.
    cl_extra / net_extra: extra per-row columns carried along (classical: CL_EXTRA names; the pixel-MLE row gets the same
    columns, its own sigmas are sxp, syp, szp)"""
    C = load_classical(a.cl, cl_extra); drift = C['drift']; lg(f'classical rows (keep | keep_pix) {len(C["x"])}')
    chk = np.clip(C['frame'][:100000] + 3, 0, len(drift) - 1)
    reg_chk = float(np.nanmax(np.abs(C['x'][:100000] - drift[chk, 0] - C['xd'][:100000])))
    zd = zdrift_curve(C); lg(f'z drift curve: {len(zd)} blocks, ptp {np.ptp(zd):.0f} nm; classical xd == x - drift[frame+3] max dev {reg_chk:.2e} px')
    reg = json.load(open(a.regions)); boxes = [dict(b, id=i) for i, b in enumerate(reg['boxes'][:2])]
    cores = box_cores(boxes, drift)
    big = (min(c[0] for c in cores) - 500, max(c[1] for c in cores) + 500, min(c[2] for c in cores) - 500, max(c[3] for c in cores) + 500)

    # classical rows in nm, z drift removed, restricted to the neighbourhood of the two box cores
    xd, yd = C['xd'] * PX, C['yd'] * PX
    near = inrect(xd, yd, big) | inrect(C['xpd'] * PX, C['ypd'] * PX, big)
    C = {k: (v[near] if k != 'drift' else v) for k, v in C.items()}
    fr = C['frame']
    def mk(m, x, y, z): return dict(x=x[m] * PX, y=y[m] * PX, z=apply_z(z[m] * 1000, fr[m], zd), frame=fr[m], f=C['file_index'][m],
                                    ux=C['x'][m] * PX, uy=C['y'][m] * PX, **{k: C[k][m] for k in cl_extra})
    mcl = C['keep'] & np.isfinite(C['z']) & np.isfinite(C['xd'])
    R = dict(cl=mk(mcl, C['xd'], C['yd'], C['z']),
             clp=mk(C['keep_pix'] & np.isfinite(C['zp']) & np.isfinite(C['xpd']), C['xpd'], C['ypd'], C['zp']),
             ast=mk(mcl & np.isfinite(C['z_ast']), C['xd'], C['yd'], C['z_ast']))
    info = dict(head=a.head, tag=a.tag, zdrift_ptp_nm=float(np.ptp(zd)), zdrift_blocks=len(zd), cores_nm=cores,
                classical_xd_equals_x_minus_drift_maxdev_px=reg_chk)
    # constant z offsets to the published-style z, on the same rows
    s = mcl & C['keep_pix'] & np.isfinite(C['zp'])
    off_p = float(np.median((C['zp'][s] - C['z'][s]) * 1000)); R['clp']['z'] -= off_p
    s = mcl & np.isfinite(C['z_ast'])
    off_a = float(np.median((C['z_ast'][s] - C['z'][s]) * 1000)); R['ast']['z'] -= off_a
    info['z_offset_removed_nm'] = dict(clp=off_p, ast=off_a)

    # network
    T, files, src = load_net(a, drift, net_extra); lg(f'network rows (selected) {len(T["x"])}, files {files}')
    net = dict(x=T['xd'] * PX, y=T['yd'] * PX, z=apply_z(T['z'] * 1000, T['frame'], zd), frame=T['frame'], f=T['fidx'],
               ux=T['x'] * PX, uy=T['y'] * PX, sx=T['sx'], sy=T['sy'], sz=T['sz'], N=T['N'], **{k: T[k] for k in net_extra if k in T})
    mz = np.abs(net['z'] - np.median(net['z'])) <= 300
    net = {k: v[mz] for k, v in net.items()}
    net = {k: v[inrect(net['x'], net['y'], big)] for k, v in net.items()}
    # registration on matched emitters (same window, < 64 nm on raw positions)
    A, B = net, R['cl']
    K = 1e7
    trb = cKDTree(np.c_[B['ux'], B['uy'], (B['frame'] // 6) * K])
    dd, kk = trb.query(np.c_[A['ux'], A['uy'], (A['frame'] // 6) * K], distance_upper_bound=64.0)
    ok = np.isfinite(dd); ia, ib = np.nonzero(ok)[0], kk[ok]
    mad = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v))))
    dx = A['x'][ia] - B['x'][ib]; dy = A['y'][ia] - B['y'][ib]; dz = A['z'][ia] - B['z'][ib]
    regi = dict(n_matched=int(len(ia)), frac_of_net=float(len(ia) / max(len(A['x']), 1)),
                dx_median_nm=float(np.median(dx)), dy_median_nm=float(np.median(dy)), dz_median_nm=float(np.median(dz)),
                dx_robust_sd_nm=mad(dx), dy_robust_sd_nm=mad(dy), dz_robust_sd_nm=mad(dz),
                lateral_drift_source='network table xd, yd (own)' if a.net_all else 'classical DME curve drift[frame + 3]')
    net['x'] = net['x'] - regi['dx_median_nm']; net['y'] = net['y'] - regi['dy_median_nm']; net['z'] = net['z'] - regi['dz_median_nm']
    R['net'] = net; info['registration'] = regi
    lg(f'registration {regi}')

    # regions from the classical published-style table (all 100 files, independent of the network table)
    S = R['cl']
    X = []
    for bi, core in enumerate(cores):
        for r in pick_zoom1(S, core): X.append(dict(r, box=bi))
    X.sort(key=lambda r: -r['dz'])
    z1 = X[0]; bi = z1['box']; core = cores[bi]
    c2, sc2, _ = pick_zoom2(S, core, [(z1['c'][0], z1['c'][1], 1200.0)])
    sec = pick_section(S, c2, 1500.0)
    segs = find_segments(S, cores)
    m = inrect(S['x'], S['y'], core)
    zmed = float(np.median(S['z'][m])); zh = float(np.ceil(np.percentile(np.abs(S['z'][m] - zmed), 97) / 50) * 50)
    zr = (zmed - zh, zmed + zh)
    G = dict(box=bi, core=core, zoom1=dict(c=z1['c'], size=1200.0, dz_nm=z1['dz'], arms_clean_frac=z1['clean'], arms_angle_deg=z1['ang_deg']), zoom2=dict(c=c2, size=1500.0, n_bimodal_bins=sc2),
             section=sec, n_segments=len(segs), zr_nm=zr, crossings_scored=X)
    lg(f'regions: box {bi}, zoom1 {z1}, zoom2 {c2} ({sc2} bimodal bins), section {sec}, {len(segs)} segments')
    return R, G, segs, info, files, src, int(len(np.unique(C['file_index'])))


def figure(R, G, segs, tag, a, info, nfiles, files, meth=None, extra_note=''):
    METH_ = METH if meth is None else meth
    core = G['core']; ext, exty = core[1] - core[0], core[3] - core[2]; ox, oy = core[0], core[2]; zr = G['zr_nm']
    zooms = [(G['zoom1']['c'], G['zoom1']['size']), (G['zoom2']['c'], G['zoom2']['size'])]
    sec = G['section']; zc = 0.5 * (zr[0] + zr[1]); zh = 0.5 * (zr[1] - zr[0])
    fig = plt.figure(figsize=(14.2, 2.6 * len(METH_) + 0.9))
    gs = fig.add_gridspec(len(METH_), 6, width_ratios=[1, 1, 1, 1.0, 0.8, 0.95], wspace=0.42, hspace=0.12,
                          left=0.07, right=0.985, top=0.955, bottom=0.135)
    out = dict(rows={})
    for i, (k, lab) in enumerate(METH_):
        T = R[k]
        n_ov = int(inrect(T['x'], T['y'], core).sum())
        ax = fig.add_subplot(gs[i, 0]); ax.imshow(zrgb(T['x'], T['y'], T['z'], ox, oy, ext, 10.0, zr, 0.8, exty)); ax.set_xticks([]); ax.set_yticks([]); [q.set_visible(False) for q in ax.spines.values()]
        ax.set_ylabel(f"{lab}\n{n_ov:,} locs, {nfiles[k]} file{'s' if nfiles[k] != 1 else ''}", fontsize=8)
        for j, (c, s) in enumerate(zooms):
            h = s / 2 / 10.0
            ax.add_patch(plt.Rectangle(((c[0] - ox) / 10 - h, (c[1] - oy) / 10 - h), 2 * h, 2 * h, fill=False, ec='w', lw=0.6))
            if i == 0: ax.text((c[0] - ox) / 10 - h + 3, (c[1] - oy) / 10 - h + 3, str(j + 1), color='w', fontsize=7, va='top')
        if i == 0: scalebar(ax, 1000 / ext, '1 µm')
        zcount = []
        for j, (c, s) in enumerate(zooms):
            ax = fig.add_subplot(gs[i, 1 + j])
            ax.imshow(zrgb(T['x'], T['y'], T['z'], c[0] - s / 2, c[1] - s / 2, s, 4.0, zr, 1.0)); ax.set_xticks([]); ax.set_yticks([]); [q.set_visible(False) for q in ax.spines.values()]
            zcount.append(int(inrect(T['x'], T['y'], (c[0] - s / 2, c[0] + s / 2, c[1] - s / 2, c[1] + s / 2)).sum()))
            if j == 1:                                                    # section strip on zoom 2
                for v in (sec['a0'], sec['a1']):
                    q = (v - (c[1] if sec['along'] == 'x' else c[0]) + s / 2) / 4.0
                    if sec['along'] == 'x': ax.axhline(q, color='w', lw=0.5, ls=':')
                    else: ax.axvline(q, color='w', lw=0.5, ls=':')
            if i == 0: scalebar(ax, 200 / s, '200 nm')
            ax.text(0.97, 0.97, f'{zcount[-1]:,}', transform=ax.transAxes, color='w', fontsize=6.5, ha='right', va='top')
        # section through zoom 2
        c, s = zooms[1]
        if sec['along'] == 'x':
            m = (T['y'] >= sec['a0']) & (T['y'] < sec['a1']); al = T['x'][m] - (c[0] - s / 2)
        else:
            m = (T['x'] >= sec['a0']) & (T['x'] < sec['a1']); al = T['y'][m] - (c[1] - s / 2)
        ax = fig.add_subplot(gs[i, 3])
        im = dens_img(al, T['z'][m] - zc, 0, s, -zh, zh, 6.0, 1.0)
        ax.imshow(im[::-1], cmap='Greys_r', extent=(0, s / 1000, -zh, zh), aspect=VEX / 1000, vmin=0, vmax=1)
        ax.set_yticks([-200, 0, 200] if zh >= 250 else [-100, 0, 100]); ax.set_ylabel('z (nm)', fontsize=7, labelpad=1)
        if i == len(METH_) - 1: ax.set_xlabel(f"{sec['along']} in zoom 2 (µm)")
        else: ax.set_xticklabels([])
        # cross-section
        u, v, nseg = cross_section(T, segs)
        ax = fig.add_subplot(gs[i, 4])
        im = dens_img(u, v, -70, 70, -70, 70, 2.0, 1.0)
        ax.imshow(im[::-1], cmap='Greys_r', extent=(-70, 70, -70, 70), vmin=0, vmax=1)
        ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50]); ax.set_ylabel('axial (nm)', fontsize=7)
        if i == len(METH_) - 1: ax.set_xlabel('in plane, across tube (nm)')
        else: ax.set_xticklabels([])
        ax.text(0.03, 0.97, f'{nseg} seg.\n{len(u):,} locs', transform=ax.transAxes, color='w', fontsize=6.5, va='top')
        # 1D profiles
        ax = fig.add_subplot(gs[i, 5])
        r = dict(n_locs_overview=n_ov, n_locs_zoom=zcount, n_segments=nseg, n_locs_cross_section=int(len(u)), n_files=nfiles[k])
        if len(u) > 100:
            lat = u[np.abs(v) < 60]; axl = v[np.abs(u) < 40]
            b = np.arange(-80, 80.1, 2.0); cc = 0.5 * (b[1:] + b[:-1])
            for vv, col, nm in ((lat, C_LAT, 'lateral'), (axl, C_AX, 'axial')):
                h, _ = np.histogram(vv, b); h = gaussian_filter(h.astype(float), 1.0); ax.plot(cc, h / h.max(), color=col, lw=1.6, label=nm)
            fl, fa = FC.fwhm(lat, -150, 150, 2.0), FC.fwhm(axl, -150, 150, 2.0)
            pl, pa = peaks(lat), peaks(axl)
            r.update(fwhm_lat_nm=fl, fwhm_ax_nm=fa, lat_peaks=pl, ax_peaks=pa)
            t1 = f"FWHM lateral {fl:.0f} nm, axial {fa:.0f} nm\nlateral: " + (f"2 peaks {pl['sep_nm']:.0f} nm apart\n(dip {pl['dip']:.2f})" if pl['resolved'] else 'one peak\n(no dip < 0.9)')
            ax.text(0.02, 1.0, t1, transform=ax.transAxes, fontsize=6.5, va='top', color='#2b2b2b')
        ax.set_ylim(0, 1.5); ax.set_xlim(-80, 80); ax.set_yticks([0, 0.5, 1]); ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylabel('norm. count', fontsize=7)
        if i == 2: ax.legend(loc='lower left', fontsize=6.5, frameon=False, handlelength=1.2)
        if i == len(METH_) - 1: ax.set_xlabel('offset from tube axis (nm)')
        else: ax.set_xticklabels([])
        out['rows'][k] = r
    titles = ['(a) x-y, colour = z', f"(b) zoom 1: crossing, Δz {G['zoom1']['dz_nm']:.0f} nm", '(c) zoom 2: overlap in x-y',
              f"(c) {sec['along']}-z section through zoom 2\n({sec['a1'] - sec['a0']:.0f} nm strip, dotted; z × {VEX:g})", '(d) tube cross-section\n(isolated straight segments)',
              '(d) profiles across the tube']
    for j, t in enumerate(titles): fig.axes[j].set_title(t)
    cax = fig.add_axes([0.07, 0.098, 0.2, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ZCMAP, norm=plt.Normalize(-zh, zh)), cax=cax, orientation='horizontal')
    cb.set_label('z (nm, relative)', fontsize=7)
    rg = info['registration']; fs = ', '.join(str(f) for f in files)
    note = (f"Real 4Pi-SIMFLUX microtubules. Network head {a.head}; files: network {nfiles['net']} ({fs}), classical rows {nfiles['cl']}"
            f"{' (same files as the network)' if nfiles['cl'] == nfiles['net'] else ''}. "
            f"Overview and zooms lie inside MT box {G['box']} (all panels) and the {G['n_segments']} tube segments inside boxes 0 and 1: the pixel-MLE row exists only there. "
            f"Regions and segments chosen automatically on the classical published-style table.\n"
            f"Selection: network good, N ≥ 1000, σz < 25 nm, σx, σy < 15 nm, |z − median| ≤ 300 nm; classical published-style: keep; pixel MLE: keep_pix; "
            f"astigmatism-only: the published-style rows and x, y with z_ast. Lateral drift: classical DME curve (network: {rg['lateral_drift_source']}); "
            f"network offset to classical on {rg['n_matched']:,} matched emitters {rg['dx_median_nm']:.1f} / {rg['dy_median_nm']:.1f} nm "
            f"(robust sd {rg['dx_robust_sd_nm']:.1f} / {rg['dy_robust_sd_nm']:.1f}) removed.\n"
            f"Sample z drift ({info['zdrift_ptp_nm']:.0f} nm ptp) estimated from the structure on the classical published-style table (whole movie) and subtracted from all rows, network included; "
            f"constant z offsets to published-style z removed (network {rg['dz_median_nm']:.0f}, pixel MLE {info['z_offset_removed_nm']['clp']:.0f}, z_ast {info['z_offset_removed_nm']['ast']:.0f} nm). "
            f"Counts: overview (row label), zoom (top right), cross-section." + extra_note)
    fig.text(0.07, 0.008, note, fontsize=6.3, color='#52514e', va='bottom', wrap=True)
    fn = f'{OUTD}/MT_showcase_real_{tag}.png'; fig.savefig(fn, dpi=150); plt.close(fig)
    out['fig'] = fn
    return out


if __name__ == '__main__':
    main()
