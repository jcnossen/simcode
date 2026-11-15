"""Quick-look renders of the microtubule cell field (mt_cells.py) for a few seeds.

Per seed one figure out/fourpi_sim/mt_cells_seed<k>.png:
  (a) overview: the cells around the field of view (outline, nucleus), every microtubule centreline coloured by its
      height above the coverslip, the FOV box;
  (b) the FOV: label positions rendered at 20 nm, colour = mean z of the labels in the pixel, brightness = density;
  (c) zoom (2 x 2 um, 5 nm pixels) at the densest 2 um square of the FOV;
  (d) x-z section through the FOV (labels within +-250 nm of the FOV centre line in y), z exaggerated x4;
  (e) z distribution of the labels and of the centreline length in the FOV;
  (f) label distance from its own microtubule axis (the 25 nm tube + 12 nm binder + 2-5 nm linker shell) and the
      cross-section of one tube (labels within a 200 nm stretch, projected along the axis).
Usage: python analysis/fourpi_sim/render_mt_cells.py [--seeds 1 2 3] [--fov 14.1] [--focus 250] [--zr 750]
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mt_cells as M

OUT = os.path.expanduser('~/simcode2/out/fourpi_sim')
CMAP = plt.get_cmap('turbo')


def render_xy(x, y, z, x0, x1, y0, y1, px, zlim):
    nx, ny = int(round((x1 - x0) / px)), int(round((y1 - y0) / px))
    cnt, _, _ = np.histogram2d(y, x, bins=(ny, nx), range=((y0, y1), (x0, x1)))
    zs, _, _ = np.histogram2d(y, x, bins=(ny, nx), range=((y0, y1), (x0, x1)), weights=z)
    zm = np.where(cnt > 0, zs / np.maximum(cnt, 1), 0)
    rgb = CMAP(Normalize(*zlim, clip=True)(zm))[..., :3]
    ref = np.percentile(cnt[cnt > 0], 99) if (cnt > 0).any() else 1
    v = np.clip(cnt / ref, 0, 1) ** 0.6
    return rgb * v[..., None]


def axis_frame(line, step=2.0):
    """resample a centreline (nm) at `step` nm: points, unit tangents"""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    q = np.arange(0, s[-1], step)
    P = np.c_[[np.interp(q, s, line[:, i]) for i in range(3)]].T
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    return P, T


def radial_and_section(r, rng, nsub=150000):
    """distance of labels from their own axis (subsample) + the cross-section of one tube"""
    from scipy.spatial import cKDTree
    xyz, mt = r['xyz'], r['mt']
    sel = rng.choice(len(xyz), min(nsub, len(xyz)), replace=False)
    dist = []
    for i in np.unique(mt[sel]):
        if i < 0:
            continue
        P, T = axis_frame(r['centrelines'][i])
        q = sel[mt[sel] == i]
        d, _ = cKDTree(P).query(xyz[q])
        dist.append(d)
    dist = np.concatenate(dist) if dist else np.zeros(0)
    # cross-section: the longest centreline in the FOV, a 200 nm stretch in its middle
    W, H = r['fov_um']
    best, bl = None, 0
    for i, m in enumerate(r['centrelines']):
        inn = (m[:, 0] > 0) & (m[:, 0] < W * 1000) & (m[:, 1] > 0) & (m[:, 1] < H * 1000)
        L = inn.sum()
        if L > bl and (mt == i).sum() > 500:
            best, bl = i, L
    sec = None
    if best is not None:
        P, T = axis_frame(r['centrelines'][best])
        lab = xyz[mt == best]
        inn = (P[:, 0] > 0) & (P[:, 0] < W * 1000) & (P[:, 1] > 0) & (P[:, 1] < H * 1000)
        idx = np.flatnonzero(inn)
        c = idx[len(idx) // 2]
        t = T[c]
        a = lab - P[c]
        along = a @ t
        keep = np.abs(along) < 100
        a = a[keep] - along[keep, None] * t
        zh = np.array([0, 0, 1.0])
        e1 = np.cross(t, zh); e1 /= np.linalg.norm(e1)            # horizontal, perpendicular to the tube
        e2 = np.cross(e1, t)                                       # mostly z
        sec = dict(u=a @ e1, v=a @ e2, mt=int(best), z_axis=float(P[c, 2]),
                   tilt_deg=float(np.degrees(np.arcsin(abs(t[2])))))
    return dist, sec


def figure(seed, a, rng):
    t0 = time.time()
    r = M.simulate(seed=seed, fov_um=(a.fov, a.fov), focus_nm=a.focus, z_range_nm=a.zr,
                   label_efficiency=a.eff, offset_um=(a.dx, a.dy))
    t_sim = time.time() - t0
    W, H = r['fov_um']
    xyz = r['xyz']
    cxu, cyu = r['center_um']
    x0w, y0w = cxu - W / 2, cyu - H / 2
    zlim = (0, a.zmax)
    fig = plt.figure(figsize=(20, 11.5), layout='constrained')
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.5])

    # (a) overview
    ax = fig.add_subplot(gs[0, 0:2])
    F = M.get_field(seed, dict(mtDensity=0.9, density=0.33))
    R = a.overview / 2
    cells = F.cells_in_rect(cxu - R, cyu - R, cxu + R, cyu + R)
    segs, cols = [], []
    for c in cells:
        ol = F.cell_outline_world(c); nu = F.nucleus_outline_world(c)
        ax.fill(ol[:, 0], ol[:, 1], color='0.12', lw=0)
        ax.plot(np.r_[ol[:, 0], ol[0, 0]], np.r_[ol[:, 1], ol[0, 1]], color='0.55', lw=0.6)
        ax.fill(nu[:, 0], nu[:, 1], color='0.28', lw=0)
        for m in F.mts_world(c):
            if len(m) < 2:
                continue
            mm = m[::4] if len(m) > 8 else m
            segs.append(np.stack([mm[:-1, :2], mm[1:, :2]], 1)); cols.append(0.5 * (mm[:-1, 2] + mm[1:, 2]) * 1000)
    if segs:
        lc = LineCollection(np.concatenate(segs), array=np.concatenate(cols), cmap=CMAP,
                            norm=Normalize(0, 1500), lw=0.25, alpha=0.8)
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02); cb.set_label('centreline z (nm)')
    ax.add_patch(plt.Rectangle((x0w, y0w), W, H, fill=False, ec='w', lw=1.5))
    ax.set_xlim(cxu - R, cxu + R); ax.set_ylim(cyu + R, cyu - R)
    ax.set_facecolor('k'); ax.set_aspect('equal')
    ax.set_xlabel('world x (um)'); ax.set_ylabel('world y (um)')
    nc = len(r['cells'])
    ax.set_title(f'(a) seed {seed}: cells around the FOV (white box)\n'
                 f'{len(cells)} cells in view, {nc} reach the FOV; centrelines coloured by z', fontsize=10)

    # (b) FOV x-y
    ax = fig.add_subplot(gs[0, 2:4])
    img = render_xy(xyz[:, 0], xyz[:, 1], xyz[:, 2], 0, W * 1000, 0, H * 1000, 20.0, zlim)
    ax.imshow(img, extent=(0, W, H, 0), interpolation='nearest')
    st = r['stats']
    ax.set_title(f'(b) FOV {W:.1f} x {H:.1f} um at ({cxu:.1f}, {cyu:.1f}) um, labels at 20 nm/px, colour = z\n'
                 f'{st["n_labels"]:,} labels ({st["labels_per_um2"]:.0f}/um$^2$, efficiency {a.eff:.0%}), '
                 f'{st["mt_length_per_um2"]:.2f} um tube/um$^2$, z {a.focus:.0f} $\\pm$ {a.zr:.0f} nm', fontsize=10)
    ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)')
    sm = plt.cm.ScalarMappable(norm=Normalize(*zlim), cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02); cb.set_label('label z above coverslip (nm)')
    # densest 2 um square for the zoom
    Z = 2.0
    cnt, ye, xe = np.histogram2d(xyz[:, 1] / 1000, xyz[:, 0] / 1000, bins=(int(H), int(W)), range=((0, int(H)), (0, int(W))))
    iy, ix = np.unravel_index(np.argmax(cnt), cnt.shape)
    zx0 = float(np.clip(xe[ix] + 0.5 - Z / 2, 0, W - Z)); zy0 = float(np.clip(ye[iy] + 0.5 - Z / 2, 0, H - Z))
    ax.add_patch(plt.Rectangle((zx0, zy0), Z, Z, fill=False, ec='w', lw=1))
    ys = zy0 + Z / 2
    ax.axhspan(ys - 0.25, ys + 0.25, color='w', alpha=0.15, lw=0)

    # (c) zoom
    ax = fig.add_subplot(gs[0, 4:6])
    m = ((xyz[:, 0] >= zx0 * 1000) & (xyz[:, 0] < (zx0 + Z) * 1000) & (xyz[:, 1] >= zy0 * 1000) &
         (xyz[:, 1] < (zy0 + Z) * 1000))
    img = render_xy(xyz[m, 0], xyz[m, 1], xyz[m, 2], zx0 * 1000, (zx0 + Z) * 1000, zy0 * 1000, (zy0 + Z) * 1000, 5.0,
                    zlim)
    ax.imshow(img, extent=(zx0, zx0 + Z, zy0 + Z, zy0), interpolation='nearest')
    ax.set_title(f'(c) zoom 2 x 2 um at 5 nm/px (densest square), {m.sum():,} labels', fontsize=10)
    ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)')

    # (d) x-z section
    ax = fig.add_subplot(gs[1, 0:4])
    m = np.abs(xyz[:, 1] - ys * 1000) < 250
    zr0, zr1 = max(0.0, a.focus - a.zr), a.focus + a.zr
    nx, nz = int(W * 1000 / 10), int((zr1 - zr0) / 10)
    cnt, _, _ = np.histogram2d(xyz[m, 2], xyz[m, 0], bins=(nz, nx), range=((zr0, zr1), (0, W * 1000)))
    ref = np.percentile(cnt[cnt > 0], 99) if (cnt > 0).any() else 1
    ax.imshow(np.clip(cnt / ref, 0, 1) ** 0.6, extent=(0, W, zr0, zr1), origin='lower', cmap='magma',
              aspect=0.004, interpolation='nearest')
    ax.axhline(a.focus, color='c', lw=0.6, ls='--')
    ax.set_xlabel('x (um)'); ax.set_ylabel('z above coverslip (nm)')
    ax.set_title(f'(d) x-z section: labels with |y - {ys:.2f} um| < 250 nm (band in b, through the zoom), 10 nm bins, '
                 f'z exaggerated x4; dashed = focus', fontsize=10)

    # (e)+(f) histograms
    ax = fig.add_subplot(gs[1, 4])
    bins = np.arange(zr0, zr1 + 20, 20)
    ax.hist(xyz[:, 2], bins=bins, color='C0', alpha=0.7, density=True, label='labels')
    zl, wl = [], []
    for c in r['centrelines']:
        mid = 0.5 * (c[1:] + c[:-1]); sl = np.linalg.norm(np.diff(c, axis=0), axis=1)
        ok = (mid[:, 0] >= 0) & (mid[:, 0] < W * 1000) & (mid[:, 1] >= 0) & (mid[:, 1] < H * 1000)
        zl.append(mid[ok, 2]); wl.append(sl[ok])
    if zl:
        ax.hist(np.concatenate(zl), bins=bins, weights=np.concatenate(wl), density=True, histtype='step', color='k',
                lw=1.2, label='axis length')
    ax.set_xlabel('z (nm)'); ax.set_ylabel('density'); ax.legend(fontsize=8)
    ax.set_title('(e) z in the FOV', fontsize=10)
    ax = fig.add_subplot(gs[1, 5])
    dist, sec = radial_and_section(r, rng)
    if sec is not None:
        ax.scatter(sec['u'], sec['v'], s=1.5, c='C3', alpha=0.5, lw=0)
        th = np.linspace(0, 2 * np.pi, 200)
        for rr, ls in [(12.5, '-'), (12.5 + a.binder, '--')]:
            ax.plot(rr * np.cos(th), rr * np.sin(th), 'k', lw=0.6, ls=ls)
        ax.set_aspect('equal'); ax.set_xlim(-45, 45); ax.set_ylim(-45, 45)
        ax.set_xlabel('across tube, horizontal (nm)'); ax.set_ylabel('across, ~z (nm)')
        md = np.median(dist) if len(dist) else np.nan
        ax.set_title(f'(f) tube cross-section ({len(sec["u"])} labels, 200 nm)\n'
                     f'label-axis distance median {md:.1f} nm (circles: 12.5, {12.5 + a.binder:.1f})', fontsize=9)
    fig.suptitle(f'webSMLM microtubule cell field (inSiliScope core, Python port) - seed {seed}, start view '
                 f'({r["start"]["x"]}, {r["start"]["y"]}) um + offset ({a.dx}, {a.dy}) um', fontsize=12)
    fn = f'{OUT}/mt_cells_seed{seed}.png'
    fig.savefig(fn, dpi=110, bbox_inches='tight')
    plt.close(fig)
    info = dict(seed=seed, file=fn, t_sim_s=t_sim, center_um=r['center_um'], start=r['start'], stats=r['stats'],
                cells=[{k: v for k, v in c.items() if k not in ('outline_um', 'nucleus_um')} for c in r['cells']],
                label_axis_dist_nm_pct_5_50_95=[float(v) for v in np.percentile(dist, [5, 50, 95])] if len(dist) else None,
                section=None if sec is None else dict(mt=sec['mt'], z_axis_nm=sec['z_axis'], tilt_deg=sec['tilt_deg']))
    print(json.dumps({k: v for k, v in info.items() if k != 'cells'}, default=float), flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--fov', type=float, default=14.1, help='um (webSMLM default structure window 141 px x 100 nm)')
    ap.add_argument('--focus', type=float, default=250.0); ap.add_argument('--zr', type=float, default=750.0)
    ap.add_argument('--eff', type=float, default=0.7); ap.add_argument('--dx', type=float, default=0.0)
    ap.add_argument('--dy', type=float, default=0.0); ap.add_argument('--zmax', type=float, default=1000.0)
    ap.add_argument('--overview', type=float, default=80.0); ap.add_argument('--binder', type=float, default=12.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    infos = [figure(s, a, rng) for s in a.seeds]
    json.dump(dict(args=vars(a), seeds=infos), open(f'{OUT}/mt_cells_render.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
