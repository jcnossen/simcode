"""Figures of the 4Pi-SIMCODE density sweep (reads $P4/sweep4pi/<tag>/ written by sweep4pi_eval.py), written to
~/simcode2/out/fourpi_sim/sweep/.

  render_<tag>.png   rows ground truth (all labelling sites) / network (selection given per density) / classical;
                     x-y coloured by z (same -200..+200 nm scale everywhere), two zooms at microtubule crossings and two
                     x-z sections (200 nm thick y-strips through the zooms); same boxes for every method and density
                     (the structure is the same in every movie).
  ttf.png            time to fidelity: GT-FRC resolution (x-y and x-z sections) vs number of frames, per density, network
                     (best selection) / network (chi2 selection off) / classical / GT units (sampling limit)
  sweep.png          GT-FRC (30000 frames), frames to x-z 35 nm, recall, wrong fringe, fixed-gate ghosts, coverage excess CV vs density
Usage: python analysis/fourpi_sim/sweep4pi_fig.py [--best mtc_d03=net_sig1 ...]
"""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
IN = f'{P4}/sweep4pi'
OUT = os.path.expanduser('~/simcode2/out/fourpi_sim/sweep'); os.makedirs(OUT, exist_ok=True)
NM = 128.0; LO, HI = 8 * NM, 52 * NM
TAGS = ['mtc_d01', 'mtc_d03', 'mtc_d05', 'mtc_d075', 'mtc_d15']
ZR = (-200.0, 200.0)
ZCMAP = LinearSegmentedColormap.from_list('zdiv', ['#3987e5', '#c3c2b7', '#eb6834'])
ZOOMS = [((3945.0, 3785.0), 1200.0, 'crossing, Δz 256 nm'), ((4800.0, 2330.0), 1200.0, '4-5 tubes, Δz 90-190 nm')]   # centre (nm), side (nm)
SEC_W = 200.0
SELCOL = {'net_all': '#9aa7b8', 'net_base': '#2a78d6', 'net_chi16': '#1baf7a', 'net_chi13': '#0b6e4f', 'net_sig0': '#f2c14e', 'net_sig1': '#eda100',
          'net_sig2': '#b25b00', 'net_chi16_sig1': '#8a4fd6', 'cl_pub': '#eb6834', 'cl_pix': '#c43d1a', 'gt_units': '#52514e', 'net_bound_base': '#2a78d6'}
LAB = {'net_all': 'network, all candidates', 'net_base': 'network, chi2 sel. off', 'net_chi16': 'network, chi2 < 1.6',
       'net_chi13': 'network, chi2 < 1.3', 'net_sig0': 'network, σ < 7 / 16 nm', 'net_sig1': 'network, σ < 5 / 12 nm', 'net_sig2': 'network, σ < 3.5 / 8 nm',
       'net_chi16_sig1': 'network, chi2 < 1.6 + σ < 5 / 12 nm', 'cl_pub': 'classical (published-style)', 'cl_pix': 'classical pixel MLE',
       'gt_units': 'GT units ≥ 300 e (sampling limit)', 'net_bound_base': 'network, true pattern + field'}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False})


def zrgb(x, y, z, x0, y0, ext, sr, blur=0.8, ref=None):
    n = int(ext / sr)
    ix = ((x - x0) / sr).astype(int); iy = ((y - y0) / sr).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    c = np.zeros((n, n)); zs = np.zeros((n, n))
    np.add.at(c, (iy[ok], ix[ok]), 1); np.add.at(zs, (iy[ok], ix[ok]), z[ok])
    cs = gaussian_filter(c, blur); zz = gaussian_filter(zs, blur) / np.maximum(cs, 1e-9)
    v = np.clip(cs / max(np.percentile(cs[cs > 0], 97), 1e-9), 0, 1) ** 0.7 if (cs > 0).any() else cs
    return ZCMAP(np.clip((zz - ZR[0]) / (ZR[1] - ZR[0]), 0, 1))[..., :3] * v[..., None]


def dens_img(a, b, a0, a1, b0, b1, sr, blur=0.8):
    na, nb = int((a1 - a0) / sr), int((b1 - b0) / sr)
    ia = ((a - a0) / sr).astype(int); ib = ((b - b0) / sr).astype(int)
    ok = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    c = np.zeros((nb, na)); np.add.at(c, (ib[ok], ia[ok]), 1)
    c = gaussian_filter(c, blur)
    return c / max(np.percentile(c[c > 0], 99.5), 1e-9) if (c > 0).any() else c


def gt_sites(tag):
    S = np.load(f'{P4}/simmovie/{tag}/gt_sites.npz')['sites']
    return S[:, 0] * NM, S[:, 1] * NM, S[:, 2] * 1000


def scalebar(ax, frac, label):
    ax.plot([0.05, 0.05 + frac], [0.06, 0.06], transform=ax.transAxes, color='w', lw=2, solid_capstyle='butt')
    ax.text(0.05 + frac / 2, 0.09, label, transform=ax.transAxes, color='w', ha='center', va='bottom', fontsize=7)


def fig_render(tag, E, best):
    rows = [('gt', 'ground truth (all labelling sites)'), (best, LAB[best]), ('net_base', LAB['net_base']), ('cl_pub', LAB['cl_pub'])]
    rows = [r for i, r in enumerate(rows) if r[0] not in [q[0] for q in rows[:i]]]
    rows = [r for r in rows if r[0] == 'gt' or r[0] in E['methods']]
    fig, axs = plt.subplots(len(rows), 5, figsize=(13, 2.6 * len(rows)), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.0, 1.0]))
    ext = HI - LO
    for i, (nm, lab) in enumerate(rows):
        if nm == 'gt': x, y, z = gt_sites(tag); ntxt = f'{len(x)} sites'
        else:
            L = np.load(f'{IN}/{tag}/locs_{nm}.npz'); x, y, z = L['x'], L['y'], L['z']; r = E['methods'][nm]
            ntxt = f"{len(x)} locs\nGT-FRC xy {r['gtfrc_last_xy']:.0f} / xz {r['gtfrc_last_xz']:.0f} nm"
        ax = axs[i, 0]; ax.imshow(zrgb(x, y, z, LO, LO, ext, 8.0)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f'{lab}\n{ntxt}', fontsize=7)
        for k, ((cx, cy), s, _) in enumerate(ZOOMS):
            ax.add_patch(plt.Rectangle(((cx - s / 2 - LO) / 8, (cy - s / 2 - LO) / 8), s / 8, s / 8, fill=False, ec='w', lw=0.6))
            ax.text((cx + s / 2 - LO) / 8 + 3, (cy - s / 2 - LO) / 8, str(k + 1), color='w', fontsize=7, va='top')
        if i == 0: scalebar(ax, 1000 / ext, '1 µm')
        for j, ((cx, cy), s, t_) in enumerate(ZOOMS):
            ax = axs[i, 1 + j]
            ax.imshow(zrgb(x, y, z, cx - s / 2, cy - s / 2, s, 4.0, 1.0)); ax.set_xticks([]); ax.set_yticks([])
            ax.axhline((s / 2 - SEC_W / 2) / 4, color='w', lw=0.4, ls=':'); ax.axhline((s / 2 + SEC_W / 2) / 4, color='w', lw=0.4, ls=':')
            if i == 0: scalebar(ax, 200 / s, '200 nm'); ax.set_title(f'zoom {j + 1}: {t_}')
            ax = axs[i, 3 + j]
            sel = np.abs(y - cy) < SEC_W / 2
            im = dens_img(x[sel], z[sel], cx - s / 2, cx + s / 2, -300, 300, 4.0, 1.0)
            ax.imshow(im[::-1], cmap='Greys_r', extent=(-s / 2, s / 2, -300, 300), vmin=0, vmax=1, aspect='equal')
            ax.set_yticks([-200, 0, 200]); ax.set_xticks([-500, 0, 500])
            if i == 0: ax.set_title(f'x-z section through zoom {j + 1}\n({SEC_W:.0f} nm y-strip, dotted)')
            if i == len(rows) - 1: ax.set_xlabel('x (nm)')
            if j == 0: ax.set_ylabel('z (nm)')
    axs[0, 0].set_title('x-y, colour = z')
    cax = fig.add_axes([0.06, 0.03, 0.2, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ZCMAP, norm=plt.Normalize(*ZR)), cax=cax, orientation='horizontal'); cb.set_label('z (nm)', fontsize=7)
    fig.text(0.99, 0.01, f"{E['dens_fov']:.2f} ON emitters / µm² / frame (FOV; {E['dens_region']:.2f} in the region), {E['T']} frames",
             ha='right', fontsize=7, color='#52514e')
    plt.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{OUT}/render_{tag}.png', dpi=150); plt.close(fig)


def fig_ttf(Es, best):
    tags = [t for t in TAGS if t in Es]
    fig, axs = plt.subplots(2, len(tags), figsize=(3.2 * len(tags), 6), sharey='row', squeeze=False)
    for j, t in enumerate(tags):
        E = Es[t]; grid = E['ttf_grid']
        for nm in ['gt_units', best[t], 'net_base', 'cl_pub']:
            if nm not in E['methods'] or 'ttf_last_xy' not in E['methods'][nm]: continue
            r = E['methods'][nm]
            lab = LAB[nm] + (' (selected)' if nm == best[t] else '')
            kw = dict(color=SELCOL[nm], lw=1.8 if nm == best[t] else 1.2, ls='--' if nm == 'gt_units' else '-', marker='o', ms=2.5, label=lab)
            axs[0, j].plot(grid, r['ttf_last_xy'], **kw); axs[1, j].plot(grid, r['ttf_last_xz'], **kw)
        axs[0, j].set_title(f"{E['dens_fov']:.2f} /µm²/frame")
        for i in range(2):
            ax = axs[i, j]; ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=0.25, lw=0.5, which='both')
            ax.set_yticks([10, 15, 20, 30, 50, 100, 200]); ax.set_yticklabels(['10', '15', '20', '30', '50', '100', '200']); ax.set_ylim(9, 250)
            ax.set_xlabel('frames')
        axs[0, j].legend(fontsize=6, frameon=False, loc='upper right')
    axs[0, 0].set_ylabel('GT-FRC resolution x-y (nm)'); axs[1, 0].set_ylabel('GT-FRC resolution x-z sections (nm)')
    plt.tight_layout(); fig.savefig(f'{OUT}/ttf.png', dpi=150); plt.close(fig)
    # all densities in one panel (best selection), the headline plot
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4))
    cm = plt.get_cmap('viridis')
    for j, t in enumerate(tags):
        E = Es[t]; grid = E['ttf_grid']; r = E['methods'][best[t]]
        for i, k in enumerate(('ttf_last_xy', 'ttf_last_xz')):
            axs[i].plot(grid, r[k], color=cm(j / max(len(tags) - 1, 1)), lw=1.8, marker='o', ms=2.5, label=f"{E['dens_fov']:.2f} /µm²/frame, {LAB[best[t]].replace('network, ', '')}")
            if 'cl_pub' in E['methods'] and k in E['methods']['cl_pub']:
                axs[i].plot(grid, E['methods']['cl_pub'][k], color=cm(j / max(len(tags) - 1, 1)), lw=1.0, ls=':')
    for i, l in enumerate(('x-y', 'x-z sections')):
        ax = axs[i]; ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=0.25, lw=0.5, which='both')
        ax.set_yticks([10, 15, 20, 30, 50, 100, 200]); ax.set_yticklabels(['10', '15', '20', '30', '50', '100', '200']); ax.set_ylim(9, 250)
        ax.set_xlabel('frames'); ax.set_ylabel(f'GT-FRC resolution {l} (nm)\n(last 1/7 crossing)')
    axs[0].legend(fontsize=6, frameon=False); axs[1].text(0.02, 0.02, 'solid: network (selection per density, see STATUS); dotted: classical', transform=axs[1].transAxes, fontsize=6)
    plt.tight_layout(); fig.savefig(f'{OUT}/ttf_all.png', dpi=150); plt.close(fig)


def fig_sweep(Es):
    tags = [t for t in TAGS if t in Es]; D = np.array([Es[t]['dens_fov'] for t in tags])
    sels = ['net_all', 'net_base', 'net_chi16', 'net_chi13', 'net_sig0', 'net_sig1', 'net_sig2', 'net_chi16_sig1', 'cl_pub']
    keys = [('gtfrc_xy', 'GT-FRC x-y (nm)'), ('gtfrc_xz', 'GT-FRC x-z sections (nm)'), ('ft_xz', 'frames to x-z GT-FRC 35 nm'),
            ('recall', 'recall (GT ≥ 300 e)'), ('fringe', 'wrong fringe (|Δz| > 100 nm)'), ('ghost', 'ghosts (fixed 30/30/75 nm gate)'), ('cov', 'coverage excess CV (250 nm segments)')]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.4))
    for s in sels:
        vals = {k: [] for k, _ in keys}
        for t in tags:
            r = Es[t]['methods'].get(s, {})
            vals['gtfrc_xy'].append(r.get('gtfrc_last_xy', np.nan)); vals['gtfrc_xz'].append(r.get('gtfrc_last_xz', np.nan))
            ft = (r.get('frames_to_last_xz') or {}).get('35'); vals['ft_xz'].append(ft if ft is not None else np.nan)
            vals['recall'].append(r.get('recall', np.nan)); vals['ghost'].append(r.get('ghost_fixed', {}).get('ghost_fraction', np.nan)); vals['fringe'].append(r.get('wrong_fringe', np.nan))
            vals['cov'].append(r.get('coverage', {}).get('excess_cv', np.nan))
        for ax, (k, l) in zip(axs, keys):
            ax.plot(D, vals[k], color=SELCOL[s], marker='o', ms=3, lw=1.3, ls='-' if s.startswith('net') else '--', label=LAB[s])
    for ax, (k, l) in zip(axs, keys):
        ax.set_xscale('log'); ax.set_xlabel('ON emitters / µm² / frame'); ax.set_ylabel(l); ax.grid(alpha=0.25, lw=0.5)
        ax.set_xticks(D); ax.set_xticklabels([f'{v:.2g}' for v in D])
        if k in ('gtfrc_xy', 'gtfrc_xz', 'ft_xz') and any(np.isfinite(l_.get_ydata()).any() for l_ in ax.lines): ax.set_yscale('log')
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=7, frameon=False)
    plt.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f'{OUT}/sweep.png', dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--best', nargs='*', default=[])
    a = ap.parse_args()
    Es = {t: json.load(open(f'{IN}/{t}/eval.json')) for t in TAGS if os.path.exists(f'{IN}/{t}/eval.json')}
    best = {t: 'net_base' for t in Es}
    for b in a.best:
        k, v = b.split('='); best[k] = v
    for t, E in Es.items(): fig_render(t, E, best[t])
    fig_ttf(Es, best); fig_sweep(Es)
    print('figures ->', OUT, list(Es))


if __name__ == '__main__':
    main()
