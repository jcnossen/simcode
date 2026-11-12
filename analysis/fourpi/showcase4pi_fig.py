"""Figures of the 4Pi-SIMFLUX high-density showcase (reads out/4pi/showcase/<tag>/ written by showcase4pi_eval.py).

Per density: render_<tag>.png (rows = methods; x-y coloured by z, two zooms at crossings, averaged tube cross-section),
sections_<tag>.png (x-z and y-z sections through crossings, rows = methods), profiles.png (tube cross-section profiles,
all densities), summary.png (recall, precision, lateral and axial error vs density, network vs classical).
Same z colour scale, zoom boxes and section strips for every method and density.
Usage: python analysis/fourpi/showcase4pi_fig.py
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
REPO = '/home/jc/simcode2'
OUT = f'{REPO}/out/4pi/showcase'
PXN = 128.0
LO, HI = 8.0, 52.0
TAGS = ['sc_d01', 'sc_d03', 'sc_d1', 'sc_d2', 'sc_d4']
METH = [('gt', 'ground truth'), ('net', 'network (in-situ pattern + field)'), ('cl_pub', 'classical, published-style'),
        ('cl_pix', 'classical, pixel MLE'), ('astig', 'astigmatism-only z')]
COL = {'net': '#2a78d6', 'net_bound': '#2a78d6', 'cl_pub': '#eb6834', 'cl_pix': '#1baf7a', 'astig': '#eda100', 'gt': '#52514e'}
ZR = (-0.3, 0.3)
SHORT = {'gt': 'ground truth', 'net': 'network', 'cl_pub': 'classical publ.', 'cl_pix': 'classical pix. MLE', 'astig': 'astig. only'}
ZCMAP = LinearSegmentedColormap.from_list('zdiv', ['#3987e5', '#c3c2b7', '#eb6834'])
ZOOMS = [((18.0, 15.0), 1.2), ((22.0, 38.5), 1.6)]          # centre (px), side (um): crossing A x B; C / E / F / B overlap
# section strips: (label, strip axis, strip from, to (px), crop along the section from, to (um from the region corner))
SECT = [('x-z, through the crossing of zoom 1', 'y', 14.2, 15.8, 0.7, 1.9),
        ('y-z, two tubes 100 nm apart in z', 'x', 29.2, 30.8, 2.4, 3.6),
        ('x-z, through the crossings of zoom 2', 'y', 37.7, 39.3, 0.9, 2.1)]
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False})


def zrgb(x, y, z, x0, y0, ext, sr, blur=0.8):
    n = int(ext / sr)
    ix = ((x - x0) / sr).astype(int); iy = ((y - y0) / sr).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    c = np.zeros((n, n)); zs = np.zeros((n, n))
    np.add.at(c, (iy[ok], ix[ok]), 1); np.add.at(zs, (iy[ok], ix[ok]), z[ok])
    cs = gaussian_filter(c, blur); zz = gaussian_filter(zs, blur) / np.maximum(cs, 1e-9)
    v = np.clip(cs / max(np.percentile(cs[cs > 0], 97), 1e-9), 0, 1) ** 0.7 if (cs > 0).any() else cs
    rgb = ZCMAP(np.clip((zz - ZR[0]) / (ZR[1] - ZR[0]), 0, 1))[..., :3] * v[..., None]
    return rgb


def dens_img(a, b, a0, a1, b0, b1, sr, blur=0.8):
    na, nb = int((a1 - a0) / sr), int((b1 - b0) / sr)
    ia = ((a - a0) / sr).astype(int); ib = ((b - b0) / sr).astype(int)
    ok = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    c = np.zeros((nb, na)); np.add.at(c, (ib[ok], ia[ok]), 1)
    c = gaussian_filter(c, blur)
    return c / max(np.percentile(c[c > 0], 99.7), 1e-9) if (c > 0).any() else c


def load(tag, nm):
    L = np.load(f'{OUT}/{tag}/locs_{nm}.npz')
    return L['x'] * PXN, L['y'] * PXN, L['z']                  # nm, nm, um


def scalebar(ax, n_px, label, x=0.05, y=0.06):
    ax.plot([x, x + n_px], [y, y], transform=ax.transAxes, color='w', lw=2, solid_capstyle='butt')
    ax.text(x + n_px / 2, y + 0.03, label, transform=ax.transAxes, color='w', ha='center', va='bottom', fontsize=7)


def fig_render(tag, E):
    fig, axs = plt.subplots(len(METH), 4, figsize=(10.2, 2.55 * len(METH)), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8]))
    ext = (HI - LO) * PXN
    for i, (nm, lab) in enumerate(METH):
        x, y, z = load(tag, nm)
        ax = axs[i, 0]; ax.imshow(zrgb(x, y, z, LO * PXN, LO * PXN, ext, 8.0)); ax.set_xticks([]); ax.set_yticks([])
        n_ = E['methods'][nm]['n_locs'] if nm != 'gt' else E['gt_units']
        ax.set_ylabel(f'{lab}\n{n_} locs', fontsize=8)
        for (cx, cy), s in ZOOMS:
            h = s * 1000 / 2 / 8.0
            ax.add_patch(plt.Rectangle(((cx - LO) * PXN / 8 - h, (cy - LO) * PXN / 8 - h), 2 * h, 2 * h, fill=False, ec='w', lw=0.6))
        if i == 0:
            scalebar(ax, 1000 / ext, '1 µm')
            for k_, (nm_, ax_, a0, a1, c0, c1) in enumerate(SECT):
                for v in (a0, a1):
                    q = (v - LO) * PXN / 8; r = (c0 * 1000 / 8, c1 * 1000 / 8)
                    if ax_ == 'y': ax.plot(r, (q, q), color='w', lw=0.5, ls=':')
                    else: ax.plot((q, q), r, color='w', lw=0.5, ls=':')
                q = (a1 - LO) * PXN / 8
                ax.text(*(((c1 * 1000 / 8 + 4, q)) if ax_ == 'y' else (q + 4, c0 * 1000 / 8)), f'S{k_ + 1}', color='w', fontsize=6, va='top')
        for j, ((cx, cy), s) in enumerate(ZOOMS):
            e = s * 1000; ax = axs[i, 1 + j]
            ax.imshow(zrgb(x, y, z, cx * PXN - e / 2, cy * PXN - e / 2, e, 4.0, 1.0)); ax.set_xticks([]); ax.set_yticks([])
            if i == 0: scalebar(ax, 200 / e, '200 nm')
        P = np.load(f'{OUT}/{tag}/prof_{nm}.npz')
        ax = axs[i, 3]
        im = dens_img(P['u'], P['v'], -70, 70, -70, 70, 2.0, 1.0)
        ax.imshow(im[::-1], cmap='Greys_r', extent=(-70, 70, -70, 70), vmin=0, vmax=1)
        ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50])
        if i == len(METH) - 1: ax.set_xlabel('across tube, in plane (nm)')
        ax.set_ylabel('across tube, axial (nm)', fontsize=7)
    axs[0, 0].set_title('x-y, colour = z'); axs[0, 1].set_title('zoom 1 (crossing, Δz 270 nm)'); axs[0, 2].set_title('zoom 2 (overlap in x-y)')
    axs[0, 3].set_title('tube cross-section\n(all tubes, crossings excluded)')
    cax = fig.add_axes([0.08, 0.035, 0.25, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ZCMAP, norm=plt.Normalize(ZR[0] * 1000, ZR[1] * 1000)), cax=cax, orientation='horizontal')
    cb.set_label('z (nm)', fontsize=7)
    fig.text(0.99, 0.01, f"{E['dens']:.2f} ON emitters / µm² / frame, {E['T']} frames", ha='right', fontsize=7, color='#52514e')
    plt.tight_layout(rect=(0, 0.045, 1, 1)); fig.savefig(f'{OUT}/render_{tag}.png', dpi=150); plt.close(fig)


def fig_sections(tag, E):
    fig, axs = plt.subplots(len(METH), len(SECT), figsize=(10.2, 1.45 * len(METH) + 0.5), squeeze=False)
    for i, (nm, lab) in enumerate(METH):
        x, y, z = load(tag, nm)
        for j, (snm, ax_, a0, a1, c0, c1) in enumerate(SECT):
            s_ = y if ax_ == 'y' else x
            sel = (s_ >= a0 * PXN) & (s_ < a1 * PXN)
            along = (x if ax_ == 'y' else y)[sel] - LO * PXN
            im = dens_img(along, z[sel] * 1000, c0 * 1000, c1 * 1000, -300, 300, 4.0, 1.0)
            ax = axs[i, j]
            ax.imshow(im[::-1], cmap='Greys_r', extent=(c0, c1, -300, 300), aspect=1 / 1000, vmin=0, vmax=1)
            ax.set_yticks([-200, 0, 200])
            if j == 0: ax.set_ylabel(f'{SHORT[nm]}\nz (nm)', fontsize=7)
            if i == len(METH) - 1: ax.set_xlabel(f"{'x' if ax_ == 'y' else 'y'} (µm)")
            else: ax.set_xticklabels([])
            if i == 0: ax.set_title(f'S{j + 1}: {snm}\n({ax_} strip {(a1 - a0) * PXN:.0f} nm wide)')
    fig.text(0.99, 0.005, f"{E['dens']:.2f} ON emitters / µm² / frame", ha='right', fontsize=7, color='#52514e')
    plt.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f'{OUT}/sections_{tag}.png', dpi=150); plt.close(fig)


def fig_profiles(Es):
    tags = [t for t in TAGS if t in Es]
    tags = tags + [t for t in Es if t not in tags]
    fig, axs = plt.subplots(2, len(tags), figsize=(2.3 * len(tags), 4.4), sharex=True, squeeze=False)
    b = np.arange(-80, 80.1, 3.0); c = 0.5 * (b[1:] + b[:-1])
    for j, t in enumerate(tags):
        for nm, lab in METH:
            P = np.load(f'{OUT}/{t}/prof_{nm}.npz')
            for i, e in enumerate((P['u'], P['v'])):
                if len(e) < 100: continue
                h, _ = np.histogram(e, b); h = h / h.max()
                axs[i, j].plot(c, h, color=COL[nm], lw=1.6 if nm != 'gt' else 1.0, ls='-' if nm != 'gt' else '--', label=lab)
        axs[0, j].set_title(f"{Es[t]['dens']:.2f} /µm²/frame" + (', 25 nm tube' if t.endswith('r125') else ''))
        axs[1, j].set_xlabel('position across the tube (nm)')
    axs[0, 0].set_ylabel('in-plane profile\n(normalised count)'); axs[1, 0].set_ylabel('axial profile\n(normalised count)')
    axs[0, -1].legend(fontsize=6, frameon=False, loc='upper right')
    plt.tight_layout(); fig.savefig(f'{OUT}/profiles.png', dpi=150); plt.close(fig)


def fig_summary(Es):
    tags = [t for t in TAGS if t in Es]
    D = np.array([Es[t]['dens'] for t in tags])
    ms = [('net', 'network, in-situ pattern + field', '-', 'o'), ('net_bound', 'network, true pattern + field', '--', 'o'),
          ('net_nochi2', 'network, in situ, no chi2 selection', ':', 'v'),
          ('cl_pub', 'classical, published-style', '-', 's'), ('cl_pix', 'classical, pixel MLE', '-', 'D'), ('astig', 'astigmatism-only z', '-', '^')]
    fig, axs = plt.subplots(1, 5, figsize=(14.5, 3.8))
    def r24(t, nm): return [r for r in Es[t]['methods'][nm]['rows'] if r['lo'] == 2000][0]
    def err(t, nm, k):        # only with >= 100 matched GT units in the 2-4k e bin
        r = r24(t, nm)
        return np.nan if r.get('n_matched', 0) < 100 or k not in r else r[k]
    for nm, lab, ls, mk in ms:
        rec = [Es[t]['methods'][nm]['recall'] for t in tags]
        pre = [Es[t]['methods'][nm]['precision'] for t in tags]
        lat = [0.5 * (err(t, nm, 'x_rob') + err(t, nm, 'y_rob')) for t in tags]
        ax_ = [err(t, nm, 'z_rob') for t in tags]
        fr = [err(t, nm, 'fringe') * 100 for t in tags]
        kw = dict(color=COL['net' if nm.startswith('net') else nm], ls=ls, marker=mk, ms=5, lw=1.6, mfc='white' if ls != '-' else COL['net' if nm.startswith('net') else nm], label=lab)
        for a, v in zip(axs, (rec, pre, lat, ax_, fr)): a.plot(D, v, **kw)
    cr = [r24(t, 'net').get('crlb', [np.nan] * 3) for t in tags]
    axs[2].plot(D, [0.5 * (c[0] + c[1]) for c in cr], color='#0b0b0b', ls=':', lw=1.2, label='CRLB (isolated emitter)')
    axs[3].plot(D, [c[2] for c in cr], color='#0b0b0b', ls=':', lw=1.2, label='CRLB (isolated emitter)')
    axs[4].set_ylim(0, None)
    for a in axs[2:4]:
        a.set_yscale('log'); a.set_yticks([1, 2, 5, 10, 20, 50, 100]); a.set_yticklabels(['1', '2', '5', '10', '20', '50', '100']); a.set_ylim(1, 150)
    labs = ['recall (GT ≥ 300 e per window)', 'precision', 'lateral error, 2–4k e (nm, robust sd)', 'axial error, 2–4k e (nm, robust sd)', 'fringe errors |Δz| > 100 nm, 2–4k e (%)']
    for a, l in zip(axs, labs):
        a.set_xscale('log'); a.set_xlabel('ON emitters / µm² / frame'); a.set_ylabel(l); a.grid(alpha=0.25, lw=0.5)
        a.set_xticks(D); a.set_xticklabels([f'{v:.2g}' for v in D])
    axs[0].set_ylim(0, 1); axs[1].set_ylim(0, 1.02)
    h, l = axs[3].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=7, frameon=False)
    plt.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f'{OUT}/summary.png', dpi=150); plt.close(fig)


def main():
    Es = {}
    for t in TAGS + ['sc_d03_r125']:
        f = f'{OUT}/{t}/eval.json'
        if os.path.exists(f): Es[t] = json.load(open(f))
    for t, E in Es.items():
        fig_render(t, E); fig_sections(t, E)
    fig_profiles(Es); fig_summary(Es)
    print('figures ->', OUT, list(Es))


if __name__ == '__main__':
    main()
