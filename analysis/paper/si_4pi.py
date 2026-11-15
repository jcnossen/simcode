"""SI figures of the 4Pi-SIMCODE section (paper v2), from EXISTING 4Pi results (no new GPU runs):

  si_4pi_npc   real 4Pi-SIMFLUX Nup96 (NPC) movie, 38 files: 4Pi-SIMCODE (head v10) vs the classical 4Pi-SIMFLUX
               pipeline as published (published-style): overview coloured by z, pore zooms, averaged pore (x-y, x-z side views),
               two-ring z histogram, ring metrics vs number of localizations kept, FRC resolution vs frames (g).
  si_4pi_mt    real 4Pi-SIMFLUX microtubule movie, 100 files: network overview of the cell, box-0 region coloured by z, a
               crossing zoom, y-z section through overlapping tubes, FRC resolution vs frames (f), tube cross-section (14 isolated
               segments) + profiles. The FRC values (titles) and FRC-vs-frames panels of both figures are read from
               cache_si_4pi/frc_vs_frames.json: run si_4pi_frc.py compute first (the one FRC estimator of the 4Pi figures,
               also behind si_4pi_frc_vs_frames).
  si_4pi_sim   simulated 4Pi-SIMFLUX microtubule cell field (webSMLM / inSiliScope structure), density sweep 0.11-1.64 ON
               emitters / um^2 / frame: renders per density (network, one sigma selection; 4Pi-SIMFLUX), x-z sections, GT-FRC vs
               frames (time to fidelity), recall / false positives / wrong-fringe fraction vs density.

Tables (all existing, read-only):
  NPC  out/4pi/net/net_NPC_v10_all.npz (post_w4pi.py), out/4pi/classical/real/cl_NPC_wv_all.npz
  MT   out/4pi/net/net_MT_v10_all.npz, out/4pi/classical/real/cl_MT_wv_all.npz, out/4pi/patches/regions_MT.json
  sim  out/4pi/sweep4pi/<tag>/{eval.json, locs_*.npz}, out/4pi/simmovie/<tag>/gt_sites.npz (sweep4pi_eval.py)
       panels e / f: out/4pi/sweep4pi/<tag>/ttf_fp.json (analysis/fourpi_sim/sweep4pi_ttf_fp.py: interpolated sustained
       GT-FRC crossing vs time, false positives incl. one-fringe-off); time axis 5.575 ms / frame (measured frame interval)
The loading / registration / z-drift / pore and segment finding code is imported from the analysis scripts that made the
4Pi status numbers (analysis/fourpi/select_curve_npc.py, showcase_real_mt.py), so the numbers are the same recipe.

Usage (CPU only; SIMCODE_CONFIG must point at the main checkout's simcode.json so that out/4pi resolves):
  export SIMCODE_CONFIG=~/simcode2/simcode.json
  python analysis/paper/si_4pi.py prep_npc      # ~10 min, ~20 GB RAM -> out/paper_figs_v2/cache_si_4pi/npc.npz
  python analysis/paper/si_4pi.py prep_mt       # ~20 min, ~40 GB RAM -> cache_si_4pi/mt.npz, mt_geom.json
  python analysis/paper/si_4pi.py npc | mt | sim | sim_share   (sim_share: temporary share variant, not for the SI)
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, f'{REPO}/analysis/fourpi', f'{REPO}/analysis/fourpi_classical', f'{REPO}/analysis/figures', f'{REPO}/analysis/fourpi_sim'):
    if p not in sys.path: sys.path.insert(0, p)
from simcode.config import root
P4 = root('fourpi')
OUT = os.path.expanduser('~/simcode2/out/paper_figs_v2')
CACHE = f'{OUT}/cache_si_4pi'
PX = 128.0                                              # 4Pi pixel (nm), out/4pi/PROCESSING_STATUS.md 'Optics facts'
FPF = 18000                                             # frames per file (6-frame windows: 3000 per file)
ZCMAP = LinearSegmentedColormap.from_list('zdiv', ['#3987e5', '#c3c2b7', '#eb6834'])   # as the 4Pi showcase figures
COL = dict(net='#2a78d6', net_m='#0b3f8a', cl='#eb6834', clp='#1baf7a', gt='#52514e')
LAB = dict(net='4Pi-SIMCODE', net_m='4Pi-SIMCODE, matched count', cl='4Pi-SIMFLUX', clp='4Pi-SIMFLUX, pixel MLE')   # figures show net, net_m, cl only (user 2026-11-15)
K3 = ('net', 'net_m', 'cl')
mpl.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'legend.fontsize': 6, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'pdf.fonttype': 42})
t0 = time.time()
def lg(s): print(f'[{time.time() - t0:6.0f}s] {s}', flush=True)


# ======================================================================================== FRC resolution vs frames (NPC g, MT f)
FRC_YMAX = 300.0


def frc_timeline(T, box, n_frames, name):
    """FRC resolution vs frames of the three analyses in box, read from the ONE estimator of si_4pi_frc.py (cache_si_4pi/frc_vs_frames.json,
    region npc_center / mt_box0, first N frames, independent halves = alternating blocks of min(500, N / 12) six-frame windows,
    N = 1000 * 2^(k/4) .. all; x-y FRC of simcode/evaluation/frc.py, 4 nm pixels, 5-ring smoothing, 1/7; 'frc' = interpolated crossing
    by the sustained rule (the value in the titles = the last point), 'frc_first' = first crossing). Checks that the json was made from
    the same rows (counts in the box) as this figure."""
    D = json.load(open(f'{CACHE}/frc_vs_frames.json'))[{'npc': 'npc_center', 'mt': 'mt_box0'}[name]]
    assert np.allclose(D['box'], box), (D['box'], box)
    for k, U in T.items():
        n = int(inrect(U['x'], U['y'], box).sum())
        assert D['n_rows'][k] == n, (name, k, D['n_rows'][k], n)                     # same rows as the figure
        assert D['T'] == n_frames
    out = dict(box=list(map(float, box)), N=None, frc={}, frc_first={}, n={}, source=f'{CACHE}/frc_vs_frames.json')
    for k in T:
        c = D[f'{k}_ind_xy']; out['N'] = c['N']; out['frc'][k] = c['sus']; out['frc_first'][k] = c['first']; out['n'][k] = c['n']
    return out


def frames_to(N, r, t):
    """first N from which the FRC stays <= t (log-interpolated with the point before); None if never"""
    N = np.asarray(N, float); r = np.asarray(r, float); ok = np.isfinite(r) & (r <= t)
    if not ok[-1]: return None
    bad = np.nonzero(~ok)[0]; i = 0 if len(bad) == 0 else bad[-1] + 1
    if i == 0 or not np.isfinite(r[i - 1]): return float(N[i])
    u = (r[i - 1] - t) / (r[i - 1] - r[i])
    return float(np.exp(np.log(N[i - 1]) + u * (np.log(N[i]) - np.log(N[i - 1]))))


def timeline_summary(F, targets=(25, 20, 15)):
    out = {}
    for k, r in F['frc'].items():
        fin = r[-1]
        r1 = F['frc_first'][k]
        out[k] = dict(final_nm=fin, frames_to={str(t): frames_to(F['N'], r, t) for t in targets}, frames_to_within_10pct_of_final=frames_to(F['N'], r, 1.1 * fin),
                      frames_total=F['N'][-1], frc_by_frames={str(N): v for N, v in zip(F['N'], r)},
                      first_crossing=dict(final_nm=r1[-1], frames_to={str(t): frames_to(F['N'], r1, t) for t in targets}, frc_by_frames={str(N): v for N, v in zip(F['N'], r1)}))
    return out


def frc_timeline_panel(ax, F, keys, xmax):
    for k in keys:
        r = np.array(F['frc'][k], float); r[r > FRC_YMAX] = np.nan
        ax.plot(F['N'], r, marker='o', ms=1.4, lw=0.9 if k == 'net' else 1.0, ls='--' if k == 'net' else '-', color=COL[k], label=LAB[k])
    for t in (25, 20, 15): ax.axhline(t, color='0.6', lw=0.4, ls=':', zorder=0)
    u, ticks, lab, f = (1e6, [0, 0.5e6, 1.0e6, 1.5e6], 'frames (10$^6$)', '{:.1f}') if xmax > 1e6 else (1e3, [0, 2e5, 4e5, 6e5], 'frames (10$^3$)', '{:.0f}')
    ax.set_yscale('log'); ax.set_xlim(0, xmax * 1.03); ax.set_ylim(8, FRC_YMAX)
    ax.set_xticks(ticks); ax.set_xticklabels([f.format(t / u) if t else '0' for t in ticks])
    ax.set_yticks([10, 15, 20, 25, 50, 100, 200]); ax.set_yticklabels(['10', '15', '20', '25', '50', '100', '200'])
    ax.yaxis.set_minor_formatter(plt.NullFormatter()); ax.tick_params(length=2, pad=1)
    ax.set_xlabel(lab, labelpad=1); ax.set_ylabel('FRC resolution (nm)', labelpad=1); ax.grid(alpha=0.25, lw=0.4, which='major')


def timeline_caption(S, region, total):
    s = lambda k, t: (lambda v: 'not reached' if v is None else f'{v:,.0f}')(S[k]['frames_to'][str(t)])
    w10 = lambda k: (lambda v: 'not reached' if v is None else f'{v:,.0f}')(S[k]['frames_to_within_10pct_of_final'])
    return (f"FRC resolution versus the number of frames ({region}; the first N frames of the series; two halves = alternating blocks of "
            "min(500, N / 12) six-frame windows, i.e. the two halves of the N frames below 6000 frames and the titles' 500-window blocks from "
            f"6000 frames on; otherwise the FRC of the titles, so the last point (all {total:,} frames) is the titles' value; the sustained crossing "
            "rule keeps the timeline from jumping at the dips of the structure's own spectrum (ring / tube size); dots = the computed N, 1000 x 2^(k/4) "
            "and every 1/20 of the series, linear frames axis; one estimator with si_4pi_frc_vs_frames). Values above 300 nm not drawn; "
            "dotted: 25 / 20 / 15 nm. "
            "Frames until the FRC stays at or below 25 / 20 / 15 nm: "
            + '; '.join(f"{LAB[k]} {s(k, 25)} / {s(k, 20)} / {s(k, 15)} (within 10 % of its all-frames value from {w10(k)} frames)" for k in K3)
            + ". What the timeline measures: the half-split FRC is the agreement of two independent halves of the data. On DNA-PAINT data it is set by "
            "how densely the localizations sample the labelled structure (docking sites revisited in both halves) and by the localization precision; "
            "the halves are kept apart in blocks of up to 500 windows because short blocks put the repeated detections of one binding event into both "
            "halves, and the FRC then measures precision (si_4pi_frc_vs_frames). It is not a ground-truth resolution: it does not see structure that "
            "is missing or misplaced in both halves alike. A curve that still falls at the end of the series means that the later frames still added "
            "sampling.")


def save(fig, name):
    for ext in ('pdf', 'svg'): fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight')
    fig.savefig(f'{OUT}/{name}.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    lg(f'-> {OUT}/{name}.(pdf|svg|png)')


def letter(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, weight='bold', va='bottom', ha='right')


def noaxis(ax):
    ax.set_xticks([]); ax.set_yticks([]); [q.set_visible(False) for q in ax.spines.values()]


def scalebar(ax, frac, lab, x=0.06, y=0.06, c='w'):
    ax.plot([x, x + frac], [y, y], transform=ax.transAxes, color=c, lw=1.5, solid_capstyle='butt')
    ax.text(x + frac / 2, y + 0.025, lab, transform=ax.transAxes, color=c, ha='center', va='bottom', fontsize=6)


def zrgb(x, y, z, x0, y0, ext, sr, zr, blur=0.8, ext_y=None, pct=97, gamma=0.7):
    n = int(round(ext / sr)); ny = int(round((ext if ext_y is None else ext_y) / sr))
    ix = np.floor((x - x0) / sr).astype(np.int64); iy = np.floor((y - y0) / sr).astype(np.int64)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < ny)
    c = np.zeros((ny, n)); zs = np.zeros((ny, n))
    np.add.at(c, (iy[ok], ix[ok]), 1); np.add.at(zs, (iy[ok], ix[ok]), z[ok])
    return zrgb_img(c, zs, zr, blur, pct, gamma)


def zrgb_img(c, zs, zr, blur=0.8, pct=97, gamma=0.7):
    cs = gaussian_filter(c, blur); zz = gaussian_filter(zs, blur) / np.maximum(cs, 1e-9)
    v = np.clip(cs / max(np.percentile(cs[cs > 0], pct), 1e-9), 0, 1) ** gamma if (cs > 0).any() else cs
    return ZCMAP(np.clip((zz - zr[0]) / (zr[1] - zr[0]), 0, 1))[..., :3] * v[..., None]


def dens(a, b, a0, a1, b0, b1, sr, blur=0.8, pct=99.7):
    na, nb = int(round((a1 - a0) / sr)), int(round((b1 - b0) / sr))
    ia = np.floor((a - a0) / sr).astype(np.int64); ib = np.floor((b - b0) / sr).astype(np.int64)
    ok = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    c = np.zeros((nb, na)); np.add.at(c, (ib[ok], ia[ok]), 1); c = gaussian_filter(c, blur)
    return c / max(np.percentile(c[c > 0], pct), 1e-9) if (c > 0).any() else c


def zcbar(fig, rect, zr, label='z (nm)'):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ZCMAP, norm=plt.Normalize(*zr)), cax=cax, orientation='horizontal')
    cb.set_label(label, fontsize=6, labelpad=1); cb.ax.tick_params(labelsize=5.5, length=2, pad=1); cb.outline.set_linewidth(0.4)
    return cb


def inrect(x, y, r): return (x >= r[0]) & (x < r[1]) & (y >= r[2]) & (y < r[3])


# =============================================================================================================== NPC
def prep_npc(a):
    import select_curve_npc as SN
    ns = argparse.Namespace(cl=f'{P4}/classical/real/cl_NPC_wv_all.npz', net=f'{P4}/net/net_NPC_v10_all.npz')
    M, info = SN.load(ns); lg(f"rows { {k: len(v['x']) for k, v in M.items()} }")
    C = M['cl']; fb = (np.percentile(C['x'], 0.5), np.percentile(C['x'], 99.5), np.percentile(C['y'], 0.5), np.percentile(C['y'], 99.5))
    M = {k: SN.sub(T, np.nonzero(inrect(T['x'], T['y'], fb))[0]) for k, T in M.items()}
    P = SN.find_pores(M['cl']); lg(f'{len(P)} pores')
    zptp = {}
    for k, T in M.items():
        d = SN.pore_zdrift(T, P); T['z'] = T['z'] - d[np.clip(T['frame'] // 6 // 500, 0, len(d) - 1)]; zptp[k] = float(np.ptp(d))
    comb = SN.comb_sigma(M['net'], 'net'); o = np.argsort(np.where(np.isfinite(comb), comb, np.inf), kind='stable')
    n_m = len(M['cl']['x']); matched = np.zeros(len(comb), bool); matched[o[:n_m]] = True
    out = dict(P=P, fb=np.array(fb), net_matched=matched, net_sel=M['net']['sel'], net_comb=comb.astype(np.float32),
               thr_matched=np.float64(comb[o[n_m - 1]]))
    for k, T in M.items():
        for q in ('x', 'y', 'z', 'N'): out[f'{k}_{q}'] = T[q].astype(np.float32)
        out[f'{k}_frame'] = T['frame'].astype(np.int64)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(f'{CACHE}/npc.npz', **out)
    info.update(pore_zdrift_ptp_nm=zptp, n_matched=n_m, thr_matched_nm=float(out['thr_matched']), footprint_nm=list(fb), n_pores=int(len(P)))
    json.dump(info, open(f'{CACHE}/npc_info.json', 'w'), indent=1, default=float)
    lg(f'-> {CACHE}/npc.npz; info {info}')


def fig_npc(a):
    import select_curve_npc as SN
    Z = np.load(f'{CACHE}/npc.npz'); info = json.load(open(f'{CACHE}/npc_info.json'))
    T = {k: {q: Z[f'{k}_{q}'].astype(np.float64) if q != 'frame' else Z[f'{k}_{q}'] for q in ('x', 'y', 'z', 'N', 'frame')} for k in ('net', 'cl')}
    T['net_m'] = {q: v[Z['net_matched']] for q, v in T['net'].items()}
    P = Z['P']; fb = Z['fb']
    sel = Z['net_sel']
    S = {q: v[sel] for q, v in T['net'].items()}                                # overview: validated-range rows
    import pickle
    pc = f'{CACHE}/npc_poremetrics.pkl'
    if os.path.exists(pc) and os.path.getmtime(pc) > os.path.getmtime(f'{CACHE}/npc.npz'):
        PM, V = pickle.load(open(pc, 'rb'))
    else:
        PM = {}; V = {}
        for k in K3:
            PM[k], V[k] = SN.pore_metrics(T[k], P); lg(f'{k}: {PM[k]}')
        pickle.dump((PM, V), open(pc, 'wb'))
    ref = json.load(open(f'{P4}/fig/NPC_selection_curves_v10.json'))
    # ---- layout: 7.2 in wide, every axis inside [0, 1] (no bbox growth)
    W, H = 7.2, 7.0; AR = W / H
    fig = plt.figure(figsize=(W, H))
    zr = (-150.0, 150.0); zmed = float(np.median(S['z']))
    # (a) overview
    ext_x, ext_y = fb[1] - fb[0], fb[3] - fb[2]
    wa = 0.36; ha = wa * AR * ext_y / ext_x
    axA = fig.add_axes([0.01, 0.966 - ha, wa, ha])
    axA.imshow(zrgb(S['x'], S['y'], S['z'] - zmed, fb[0], fb[2], ext_x, 25.0, zr, 0.7, ext_y, pct=99), origin='upper'); noaxis(axA)
    scalebar(axA, 5000 / ext_x, '5 µm')
    # zoom: 0.9 um box around a pore near the centre of the pore cloud, with the most pores fully inside
    cen = np.median(P[:, :2], 0); tr = cKDTree(P[:, :2]); s_ = 900.0
    cand = P[tr.query_ball_point(cen, 2500.0)][:, :2]
    best = max(cand, key=lambda c: (np.sum(np.all(np.abs(P[:, :2] - c) < s_ / 2 - 60, 1)), -np.hypot(*(c - cen))))
    zb = (best[0] - s_ / 2, best[0] + s_ / 2, best[1] - s_ / 2, best[1] + s_ / 2)
    axA.add_patch(plt.Rectangle(((zb[0] - fb[0]) / 25, (zb[2] - fb[2]) / 25), s_ / 25, s_ / 25, fill=False, ec='w', lw=0.6))
    letter(axA, 'a', 0.0)
    axA.set_title(f"4Pi-SIMCODE, {len(S['x']):,} localizations (38 files)", fontsize=7, pad=2)
    zcbar(fig, [0.06, 0.966 - ha - 0.03, 0.2, 0.008], zr, 'z (nm), a')
    # (b) zooms, coloured by z, same box
    OPK = {'net': 'network, all kept (good & N >= 1000)', 'net_m': 'network, matched count (combined sigma)', 'cl': 'classical published-style'}
    cx, cy = 0.5 * (fb[0] + fb[1]), 0.5 * (fb[2] + fb[3]); frc_box = (cx - 5000, cx + 5000, cy - 5000, cy + 5000)   # select_curve_npc frc_box
    FT = frc_timeline({k: T[k] for k in K3}, frc_box, 38 * FPF, 'npc')
    FRC1 = {k: ref['operating_points'][OPK[k]]['frc'] for k in K3}                  # first crossing, select_curve_npc.metrics (central 10 x 10 um)
    for k in K3: assert abs(FT['frc_first'][k][-1] - FRC1[k]) < 0.05, (k, FT['frc_first'][k][-1], FRC1[k])    # same rows / halves as the reference
    FRCn = {k: FT['frc'][k][-1] for k in K3}                                         # titles: sustained interpolated crossing, all frames
    zlist = [('net', f"4Pi-SIMCODE\n{int(inrect(T['net']['x'], T['net']['y'], zb).sum()):,} locs"),
             ('net_m', f"4Pi-SIMCODE, matched count\n{int(inrect(T['net_m']['x'], T['net_m']['y'], zb).sum()):,} locs"),
             ('cl', f"4Pi-SIMFLUX\n{int(inrect(T['cl']['x'], T['cl']['y'], zb).sum()):,} locs")]
    zlist = [(k, t + f"\nFRC {FRCn[k]:.1f} nm") for k, t in zlist]
    x0 = 0.395; wz = 0.195; hz = wz * AR
    for i, (k, t) in enumerate(zlist):
        ax = fig.add_axes([x0 + i * (wz + 0.007), 0.966 - hz, wz, hz])
        ax.imshow(zrgb(T[k]['x'], T[k]['y'], T[k]['z'] - zmed, zb[0], zb[2], s_, 3.0, (-60, 60), 1.0, pct=99.5, gamma=0.8)); noaxis(ax)
        ax.set_title(t, fontsize=6.5, pad=2)
        if i == 0: scalebar(ax, 200 / s_, '200 nm'); letter(ax, 'b', 0.0)
    zcbar(fig, [x0 + 0.02, 0.966 - hz - 0.03, 0.2, 0.008], (-60, 60), 'z (nm), b')
    # (c) averaged pore x-y (network matched) + (d) x-z side views of 4 rows
    rows = list(K3)
    ww = 0.168; hh = ww * AR; yb = 0.375; xs = [0.07, 0.07 + ww + 0.07]
    xs += [xs[1] + i * (ww + 0.06) for i in range(1, 3)]
    ax = fig.add_axes([xs[0], yb, ww, hh])
    h = gaussian_filter(np.histogram2d(V['net_m']['dy'], V['net_m']['dx'], bins=np.arange(-100, 101, 2.5))[0], 0.7)
    ax.imshow(h / np.percentile(h, 99.7), cmap='magma', vmin=0, vmax=1, extent=[-100, 100, 100, -100])
    ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50]); ax.tick_params(length=2, pad=1); ax.set_xlabel('x (nm)', labelpad=1); ax.set_ylabel('y (nm)', labelpad=1)
    ax.set_title(f"averaged pore, x-y\n({info['n_pores']:,} pores; matched)", fontsize=6.5, pad=2)
    letter(ax, 'c', -0.2)
    axs = []
    for i, k in enumerate(rows):
        ax = fig.add_axes([xs[1 + i], yb, ww, hh])
        s = np.abs(V[k]['dy']) < 30
        h = gaussian_filter(np.histogram2d(V[k]['dz'][s], V[k]['dx'][s], bins=[np.arange(-100, 101, 2.5), np.arange(-100, 101, 2.5)])[0], 0.7)
        ax.imshow(h[::-1] / np.percentile(h, 99.7), cmap='magma', vmin=0, vmax=1, extent=[-100, 100, -100, 100])
        ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50]); ax.tick_params(length=2, pad=1)
        if i: ax.set_yticklabels([])
        else: ax.set_ylabel('z (nm)', labelpad=1)
        ax.set_xlabel('x (nm), |y| < 30 nm', labelpad=1)
        ax.set_title(LAB[k].replace(', ', ',\n'), fontsize=6.5, pad=2, color=COL[k])
        ax.text(0.04, 0.96, f"{PM[k]['locs_per_pore']:.0f} / pore", transform=ax.transAxes, color='w', fontsize=5.5, va='top')
        axs.append(ax)
    letter(axs[0], 'd', -0.2)
    # (e) ring z histograms, (f) two-ring valley / peak vs number kept, (g) FRC vs frames: one row, same size
    yb2 = 0.06; h2 = 0.215; w2 = 0.255; xr = [0.07, 0.40, 0.73]
    ax = fig.add_axes([xr[0], yb2, w2, h2])
    for k in rows:
        ring = (V[k]['r'] > 25) & (V[k]['r'] < 85)
        hz_, ez = np.histogram(V[k]['dz'][ring], bins=np.arange(-100, 101, 2)); cz = 0.5 * (ez[1:] + ez[:-1])
        hs = gaussian_filter(hz_.astype(float), 1.0); ax.plot(cz, hs / hs.max(), color=COL[k], lw=1.1 if k != 'net' else 0.8, ls='-' if k != 'net' else '--',
                                                            label=f"{LAB[k]}: valley / peak {PM[k]['z_dip']:.2f}")
    ax.set_xlabel('z about the pore mean (nm), ring radius 25-85 nm', labelpad=1); ax.set_ylabel('norm. count', labelpad=1); ax.set_ylim(0, 1.6); ax.set_xlim(-100, 100)
    ax.set_yticks([0, 0.5, 1]); ax.tick_params(length=2, pad=1)
    ax.legend(loc='upper center', frameon=False, fontsize=5.3, handlelength=1.8, borderaxespad=0.1, labelspacing=0.2)
    letter(ax, 'e', -0.1)
    ax = fig.add_axes([xr[1], yb2, w2, h2])
    Cv = ref['curves']
    for k, sn, lab in (('net', 'combined sigma', '4Pi-SIMCODE'), ('cl', 'combined sigma', '4Pi-SIMFLUX')):
        c = Cv[k][sn]; ax.plot([r['n_keep'] / 1e6 for r in c], [r.get('z_dip', np.nan) for r in c], '-o', ms=2, lw=1.0, color=COL[k], label=lab)
        if k == 'net':      # matched count = the point of the 4Pi-SIMCODE curve at the 4Pi-SIMFLUX count
            ax.plot([info['n_matched'] / 1e6], [PM['net_m']['z_dip']], 's', ms=4, color=COL['net_m'], label=LAB['net_m'], zorder=5)
    ax.set_xscale('log'); ax.set_xlabel('localizations kept (millions)', labelpad=1); ax.set_ylabel('two-ring valley / peak', labelpad=1)
    ax.set_xticks([0.3, 1, 3]); ax.set_xticklabels(['0.3', '1', '3']); ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.tick_params(length=2, pad=1)
    ax.axvline(info['n_matched'] / 1e6, color='0.6', lw=0.6, ls=':'); ax.grid(alpha=0.25, lw=0.4)
    ax.set_ylim(0.2, 1.05); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='lower left', frameon=False, fontsize=5.0, handlelength=1.3, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.15)
    letter(ax, 'f', -0.1)
    ax = fig.add_axes([xr[2], yb2, w2, h2]); frc_timeline_panel(ax, FT, K3, FT['N'][-1])
    ax.legend(loc='upper right', frameon=False, fontsize=5.0, handlelength=1.8, borderaxespad=0.1, labelspacing=0.15)
    ax.set_title('central 10 × 10 µm, first N frames', fontsize=6.3, pad=2)
    letter(ax, 'g', -0.1)
    TS = timeline_summary(FT)
    save(fig, 'si_4pi_npc')
    # numbers
    sc = json.load(open(f'{P4}/net/cmp_v10/compare_NPC_v10.json')) if os.path.exists(f'{P4}/net/cmp_v10/compare_NPC_v10.json') else None
    val = dict(lines={k: LAB[k] for k in K3}, pore_metrics={k: PM[k] for k in K3}, n_rows={k: int(len(T[k]['x'])) for k in K3}, n_overview=int(len(S['x'])),
               zoom_box_nm=list(map(float, zb)), info=info, reference_json=f'{P4}/fig/NPC_selection_curves_v10.json',
               frc_titles_nm=FRCn, frc_first_crossing_nm=FRC1,
               panel_g_frc_vs_frames=dict(frc_box_nm=list(map(float, frc_box)), frames=FT['N'], frc_nm=FT['frc'], frc_first_nm=FT['frc_first'], n_locs=FT['n'], summary=TS,
                                          rule='first N frames; halves = alternating blocks of min(500, N/12) six-frame windows; frc(px=4, units=nm) curve; frc_nm = sustained interpolated crossing (drawn, titles), frc_first_nm = first crossing (earlier titles)'),
               panel_f_valley_peak_vs_kept={k: [dict(n_keep=r['n_keep'], valley_peak=r.get('z_dip'), ring_z_sd_nm=r.get('ring_z_sd_nm'))
                                                for r in ref['curves'][k]['combined sigma']] for k in ('net', 'cl')},
               reference_operating_points={k: {q: ref['operating_points'][OPK[k]].get(q) for q in ('n_keep', 'locs_per_pore', 'ring_radius_median_nm', 'radial_width_nm',
                                                                       'two_ring_sep_em2_nm', 'ring_z_sd_nm', 'z_dip', 'dafl', 'dafl_2_4k', 'fringe_hop', 'frc')}
                                           for k in K3})
    json.dump(val, open(f'{OUT}/si_4pi_npc_values.json', 'w'), indent=1, default=float)
    open(f'{OUT}/si_4pi_npc_caption.txt', 'w').write(caption_npc(val))


def caption_npc(v):
    PM = v['pore_metrics']; R = v['reference_operating_points']; I = v['info']
    rn, rm, rc = R['net'], R['net_m'], R['cl']
    f = lambda k: (f"{PM[k]['locs_per_pore']:.0f} localizations per pore, ring radius {PM[k]['ring_radius_median_nm']:.1f} nm (median, 25-85 nm), "
                   f"radial width {PM[k]['radial_width_nm']:.1f} nm, two-ring z separation {PM[k]['two_ring_sep_em2_nm']:.0f} nm, per-ring z sd {PM[k]['ring_z_sd_nm']:.1f} nm, "
                   f"valley / peak {PM[k]['z_dip']:.2f}")
    d = lambda r: f"{r['dafl'][0]:.2f} / {r['dafl'][1]:.2f} / {r['dafl'][2]:.2f} nm"
    L = ['DRAFT CAPTION',
         "4Pi-SIMCODE on the 4Pi-SIMFLUX nuclear pore complex data (Nup96, U-2 OS, DNA-PAINT with a Cy3B imager; data of the original 4Pi-SIMFLUX "
         "publication, 38 files of 18 000 frames = 3000 six-frame windows each; 4 camera channels, 128 nm pixels). Three analyses are shown throughout: "
         "4Pi-SIMCODE (standard filter: chi2/dof < 2.5, sigma_z < 25 nm, sigma_x, sigma_y < 0.12 px, >= 1000 photo-electrons per window); "
         "4Pi-SIMCODE, matched count (its localizations with the smallest reported combined precision sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2), "
         f"<= {I['thr_matched_nm']:.2f} nm, as many as 4Pi-SIMFLUX keeps); and 4Pi-SIMFLUX, the analysis as published (published-style pipeline with its "
         "published rejections). "
         "(a) Overview of the nucleus, 4Pi-SIMCODE localizations in the validated range (|z| <= 0.3 um), colour = z. "
         "(b) Zoom (white box in a), the three analyses, with the localizations in the zoom and the FRC resolution of each analysis (x-y, over "
         "the central 10 x 10 um of the nucleus; 4 nm histogram pixels, two halves = alternating blocks of 500 six-frame windows, ring FRC smoothed over 5 rings, 1/7 threshold, crossing interpolated between rings and taken where the curve stays below 1/7 for 0.01 /nm (the sustained rule of all FRC values in the 4Pi figures, also the simulation's); simcode/evaluation/frc.py). (c) Averaged pore (x-y) of the matched-count 4Pi-SIMCODE localizations; "
         f"{I['n_pores']:,} pores found once on the 4Pi-SIMFLUX table, the same centres for every analysis. (d) Side views (x-z, |y| < 30 nm) of the "
         "averaged pore. (e) z histogram of the ring (radius 25-85 nm) about each pore's mean z: the two Nup96 rings; valley / peak = the "
         "lowest count between the two ring peaks divided by the lower peak (lower = better separated rings). "
         "(f) Two-ring valley / peak (as in e) versus the number of localizations kept: each method's localizations ranked by its own reported "
         "combined precision, the most precise kept first (dotted line: the 4Pi-SIMFLUX count; square: 4Pi-SIMCODE at matched count). "
         "(g) " + timeline_caption(v['panel_g_frc_vs_frames']['summary'], 'central 10 x 10 um, as in b', v['panel_g_frc_vs_frames']['frames'][-1]),
         '',
         'NUMBERS (si_4pi_npc_values.json; same recipe as out/4pi/fig/NPC_selection_curves_v10.json)',
         f"4Pi-SIMCODE: {f('net')}.",
         f"4Pi-SIMCODE matched count: {f('net_m')}.",
         f"4Pi-SIMFLUX: {f('cl')}.",
         f"Localizations in the field (pores' footprint): 4Pi-SIMCODE {v['n_rows']['net']:,}, matched count {v['n_rows']['net_m']:,}, "
         f"4Pi-SIMFLUX {v['n_rows']['cl']:,}; overview rows (validated range) {v['n_overview']:,}.",
         f"Adjacent-window precision (DAFL, x / y / z, all pairs): 4Pi-SIMCODE {d(rn)}, matched count {d(rm)}, 4Pi-SIMFLUX {d(rc)}. "
         f"FRC (central 10 x 10 um): {v['frc_titles_nm']['net']:.1f} / {v['frc_titles_nm']['net_m']:.1f} / {v['frc_titles_nm']['cl']:.1f} nm "
         f"(first-crossing rule: {v['frc_first_crossing_nm']['net']:.1f} / {v['frc_first_crossing_nm']['net_m']:.1f} / {v['frc_first_crossing_nm']['cl']:.1f} nm). Fringe hops (adjacent windows, |dz| > 80 nm): "
         f"{100 * rn['fringe_hop']:.1f} % / {100 * rm['fringe_hop']:.2f} % / {100 * rc['fringe_hop']:.1f} %.",
         f"Registration network -> 4Pi-SIMFLUX on {I['registration']['n_matched']:,} matched emitters (same window, < 64 nm): "
         f"{I['registration']['dx_median_nm']:.1f} / {I['registration']['dy_median_nm']:.1f} / {I['registration']['dz_median_nm']:.1f} nm removed; "
         f"fringe disagreement {100 * I['registration']['dz_fringe_disagree']:.1f} %.",
         '',
         'CONDITIONS',
         "4Pi-SIMCODE: det4pi detector, emitter network v10 (learned read-out windows, 4 intensities per frame and channel), explicit 4Pi solve "
         "(SIM pattern and 4Pi phase field estimated in situ per file, interference contrast scale 1.56), z from the interferometric phase with the fringe "
         "chosen by the network's astigmatic z prior; lateral drift: own DME; sample z drift: own structure-based estimate. "
         "4Pi-SIMFLUX = our implementation of the published pipeline (out/4pi/classical/STATUS.md): wavelet detection on the registered 4-channel "
         "6-frame sum (as the published code), Poisson MLE of the 24-image sum with the incoherent part of the bead-fitted 4Pi PSF (x, y, astigmatic z), "
         "the published two-step SIMFLUX lateral solve, 4Pi z from Gaussian-weighted per-channel moments (phasor) unwrapped against the astigmatic z, "
         "pattern and phase field estimated per file, the published rejections; its own DME and structure-based z drift. Residual z drift from the "
         f"fixed pores removed per table (ptp nm: {', '.join(f'{k} {x:.1f}' for k, x in I['pore_zdrift_ptp_nm'].items() if k != 'clp')}).",
         'CODE: analysis/paper/si_4pi.py (prep_npc, npc); recipe imported from analysis/fourpi/select_curve_npc.py.']
    return '\n'.join(L) + '\n'


# ================================================================================================================ MT
def prep_mt(a):
    import showcase_real_mt as SR
    ns = argparse.Namespace(cl=f'{P4}/classical/real/cl_MT_wv_all.npz', net_all=f'{P4}/net/net_MT_v10_all.npz', net_locs=None,
                            head='v10', tag='si4pi', regions=f'{P4}/patches/regions_MT.json')
    R, G, segs, info, files, src, ncl = SR.prepare(ns)
    out = {}
    for k in ('net', 'cl', 'clp'):
        for q in ('x', 'y', 'z'): out[f'{k}_{q}'] = R[k][q].astype(np.float32)
        out[f'{k}_frame'] = R[k]['frame'].astype(np.int64)
    for q in ('sx', 'sy', 'sz', 'N'): out[f'net_{q}'] = R['net'][q].astype(np.float32)
    # whole-field overview of the network table (its own drift-corrected xd, yd, zd), selection as showcase_real_mt.load_net
    N = np.load(ns.net_all)
    s = N['good'] & (N['N'] >= 1000) & (N['sz'] < 0.025) & (N['sx'] < 0.12) & (N['sy'] < 0.12) & np.isfinite(N['zd'])
    x, y, z = N['xd'][s] * PX, N['yd'][s] * PX, N['zd'][s] * 1000
    z = z - np.median(z); s2 = np.abs(z) <= 300; x, y, z = x[s2], y[s2], z[s2]
    sr = 25.0; ex = [np.percentile(x, 0.05) - 500, np.percentile(x, 99.95) + 500, np.percentile(y, 0.05) - 500, np.percentile(y, 99.95) + 500]
    bx = np.arange(ex[0], ex[1] + sr, sr); by = np.arange(ex[2], ex[3] + sr, sr)
    c = np.histogram2d(y, x, bins=[by, bx])[0]; zs = np.histogram2d(y, x, bins=[by, bx], weights=z)[0]
    out.update(ov_c=c.astype(np.float32), ov_zs=zs.astype(np.float32), ov_ext=np.array(ex), ov_sr=np.float64(sr), ov_n=np.int64(len(x)))
    os.makedirs(CACHE, exist_ok=True); np.savez(f'{CACHE}/mt.npz', **out)
    json.dump(dict(G=G, segs=segs, info=info, files=files, ncl=ncl, net_overview_rows=int(len(x)),
                   net_overview_note='network table xd, yd, zd (own DME, own structure-based z drift), good & N >= 1000 & sigma cuts, |z - median| <= 300 nm'),
              open(f'{CACHE}/mt_geom.json', 'w'), indent=1, default=float)
    lg(f'-> {CACHE}/mt.npz')


def fig_mt(a):
    import showcase_real_mt as SR, fig_cl as FC
    Z = np.load(f'{CACHE}/mt.npz'); J = json.load(open(f'{CACHE}/mt_geom.json')); G, segs, info = J['G'], J['segs'], J['info']
    T = {k: {q: Z[f'{k}_{q}'].astype(np.float64) for q in ('x', 'y', 'z')} for k in ('net', 'cl')}
    for k in T: T[k]['frame'] = Z[f'{k}_frame']
    for q in ('sx', 'sy', 'sz', 'N'): T['net'][q] = Z[f'net_{q}'].astype(np.float64)
    cores = info['cores_nm']; inreg = lambda U: inrect(U['x'], U['y'], cores[0]) | inrect(U['x'], U['y'], cores[1])
    comb = np.sqrt((T['net']['sx'] * PX) ** 2 + (T['net']['sy'] * PX) ** 2 + (T['net']['sz'] * 1000) ** 2)
    n_m = int(inreg(T['cl']).sum()); o = np.sort(comb[inreg(T['net'])]); thr = float(o[n_m - 1])
    T['net_m'] = {q: v[comb <= thr] for q, v in T['net'].items()}
    lg(f'matched count {n_m}, threshold {thr:.3f} nm (status: 5.03)')
    W = 7.2; H = 6.4; AR = W / H
    fig = plt.figure(figsize=(W, H))
    core = G['core']; zr = G['zr_nm']; zc = 0.5 * (zr[0] + zr[1]); zh = 0.5 * (zr[1] - zr[0])
    top = 0.965
    # (a) whole-cell overview
    c, zs, ex, sr = Z['ov_c'], Z['ov_zs'], Z['ov_ext'], float(Z['ov_sr'])
    ny, nx = c.shape
    wa = 0.32; ha = wa * AR * ny / nx
    axA = fig.add_axes([0.01, top - ha, wa, ha])
    axA.imshow(zrgb_img(c.astype(float), zs.astype(float), (-200, 200), 0.8, 98.5, 0.6), origin='lower'); noaxis(axA)
    axA.add_patch(plt.Rectangle(((core[0] - ex[0]) / sr, (core[2] - ex[2]) / sr), (core[1] - core[0]) / sr, (core[3] - core[2]) / sr, fill=False, ec='w', lw=0.6))
    scalebar(axA, 5000 / (nx * sr), '5 µm')
    axA.set_title(f"4Pi-SIMCODE, {int(Z['ov_n']):,} localizations (100 files)", fontsize=6.5, pad=2); letter(axA, 'a', 0.0)
    zcbar(fig, [0.06, top - ha - 0.035, 0.2, 0.008], (-200, 200), 'z (nm), a')
    # (b) box-0 region, network all kept; zoom boxes
    ext, exty = core[1] - core[0], core[3] - core[2]
    wb = 0.29; hb = wb * AR * exty / ext
    axB = fig.add_axes([0.355, top - hb, wb, hb])
    axB.imshow(zrgb(T['net']['x'], T['net']['y'], T['net']['z'] - zc, core[0], core[2], ext, 10.0, (-zh, zh), 0.8, exty)); noaxis(axB)
    zooms = [(G['zoom1']['c'], G['zoom1']['size']), (G['zoom2']['c'], G['zoom2']['size'])]
    for j, (cc, s) in enumerate(zooms):
        axB.add_patch(plt.Rectangle(((cc[0] - s / 2 - core[0]) / 10, (cc[1] - s / 2 - core[2]) / 10), s / 10, s / 10, fill=False, ec='w', lw=0.6))
        axB.text((cc[0] - s / 2 - core[0]) / 10 + 3, (cc[1] - s / 2 - core[2]) / 10 + 3, str(j + 1), color='w', fontsize=6, va='top')
    scalebar(axB, 1000 / ext, '1 µm'); axB.set_title('4Pi-SIMCODE (box in a)', fontsize=6.5, pad=2); letter(axB, 'b', 0.0)
    zcbar(fig, [0.40, top - hb - 0.035, 0.2, 0.008], (-zh, zh), 'z (nm), b-d')
    # (c) zoom 1 (crossing), (d) zoom 2: the three analyses
    wc = 0.108; hc = wc * AR; xc = [0.665 + i * (wc + 0.006) for i in range(3)]
    ZT = dict(net='4Pi-SIMCODE', net_m='matched count', cl='4Pi-SIMFLUX')
    zoom_matched = []
    refm = json.load(open(f'{P4}/fig/MT_selection_curves_v10_full.json'))['operating_points']
    FRC1 = {k: refm[o]['frc'] for k, o in (('net', 'all'), ('net_m', 'matched count'), ('cl', 'classical published (same files)'))}   # first crossing, box 0
    FT = frc_timeline({k: T[k] for k in K3}, cores[0], 100 * FPF, 'mt')
    for k in K3: assert abs(FT['frc_first'][k][-1] - FRC1[k]) < 0.05, (k, FT['frc_first'][k][-1], FRC1[k])    # same rows / halves as the reference
    FRCm = {k: FT['frc'][k][-1] for k in K3}                                         # titles: sustained interpolated crossing, all frames, box 0 = region b
    for r, (cc, s) in enumerate(zooms):
        y0 = top - hc - r * (hc + 0.05)
        zr_ = (cc[0] - s / 2, cc[0] + s / 2, cc[1] - s / 2, cc[1] + s / 2)
        inz = inrect(T['net']['x'], T['net']['y'], zr_); n_cl = int(inrect(T['cl']['x'], T['cl']['y'], zr_).sum())
        iz = np.nonzero(inz)[0]; iz = iz[np.argsort(comb[iz], kind='stable')[:n_cl]]            # matched WITHIN the zoom
        Uz = dict(net=T['net'], net_m={q: v[iz] for q, v in T['net'].items()}, cl=T['cl'])
        zoom_matched.append(dict(zoom=r + 1, n=n_cl, threshold_nm=float(comb[iz].max()) if len(iz) else float('nan')))
        for i, k in enumerate(K3):
            ax = fig.add_axes([xc[i], y0, wc, hc]); U = Uz[k]
            ax.imshow(zrgb(U['x'], U['y'], U['z'] - zc, cc[0] - s / 2, cc[1] - s / 2, s, 4.0, (-zh, zh), 1.0)); noaxis(ax)
            n = int(inrect(U['x'], U['y'], zr_).sum())
            ax.set_title(f"{ZT[k]}" + (f"\nFRC {FRCm[k]:.1f} nm" if r == 0 else '') + f"\nzoom {r + 1}: {n:,} locs", fontsize=5.6, pad=2, color=COL[k])
            if r == 1:
                sec = G['section']
                for v in (sec['a0'], sec['a1']):
                    q = (v - (cc[1] if sec['along'] == 'x' else cc[0]) + s / 2) / 4.0
                    (ax.axhline if sec['along'] == 'x' else ax.axvline)(q, color='w', lw=0.5, ls=':')
            if i == 0: scalebar(ax, 200 / s, '200 nm', x=0.2); letter(ax, 'cd'[r], -0.08, 1.3)
    # (e) section through zoom 2: three methods
    sec = G['section']; cc, s = zooms[1]
    ye = 0.40; he = 0.105; we = 0.19; xe = [0.075, 0.28, 0.485]
    for i, k in enumerate(K3):
        U = T[k]
        if sec['along'] == 'x': m = (U['y'] >= sec['a0']) & (U['y'] < sec['a1']); al = U['x'][m] - (cc[0] - s / 2)
        else: m = (U['x'] >= sec['a0']) & (U['x'] < sec['a1']); al = U['y'][m] - (cc[1] - s / 2)
        ax = fig.add_axes([xe[i], ye, we, he])
        im = dens(al, U['z'][m] - zc, 0, s, -zh, zh, 6.0, 1.0)
        ax.imshow(im[::-1], cmap='Greys_r', extent=(0, s / 1000, -zh, zh), aspect='auto', vmin=0, vmax=1)
        ax.set_yticks([-100, 0, 100]); ax.tick_params(length=2, pad=1); ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel(f"{sec['along']} along zoom 2 (µm)", labelpad=1)
        if i == 0: ax.set_ylabel('z (nm)', labelpad=1); letter(ax, 'e', -0.04)
        else: ax.set_yticklabels([])
        ax.set_title(f"{LAB[k]}\n({int(m.sum()):,} locs)", fontsize=6.3, pad=2, color=COL[k])
    # (f) FRC resolution vs frames (box 0 = the region of the zoom-1 FRC values)
    ax = fig.add_axes([0.755, ye, 0.23, 0.205]); frc_timeline_panel(ax, FT, K3, FT['N'][-1])
    ax.legend(loc='upper right', frameon=False, fontsize=5.0, handlelength=1.8, borderaxespad=0.1, labelspacing=0.15)
    ax.set_title('box 0, first N frames', fontsize=6.3, pad=2); letter(ax, 'f', -0.12)
    TS = timeline_summary(FT)
    # (g) tube cross-sections + (h) profiles
    yf = 0.07; wf = 0.16; hf = wf * AR
    prof = {}
    for i, k in enumerate(K3):
        u, v, nseg = SR.cross_section(T[k], segs)
        ax = fig.add_axes([0.075 + i * (wf + 0.012), yf, wf, hf])
        ax.imshow(dens(u, v, -70, 70, -70, 70, 2.0, 1.0)[::-1], cmap='Greys_r', extent=(-70, 70, -70, 70), vmin=0, vmax=1)
        ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50]); ax.tick_params(length=2, pad=1)
        ax.set_xlabel('across tube (nm)', labelpad=1)
        if i == 0: ax.set_ylabel('axial (nm)', labelpad=1); letter(ax, 'g', -0.12)
        else: ax.set_yticklabels([])
        ax.set_title(LAB[k].replace(', ', ',\n'), fontsize=6.3, pad=2, color=COL[k])
        lat = u[np.abs(v) < 60]; axl = v[np.abs(u) < 40]
        prof[k] = dict(lat=lat, ax=axl, n=int(len(u)), nseg=nseg, fwhm_lat=FC.fwhm(lat, -150, 150, 2.0), fwhm_ax=FC.fwhm(axl, -150, 150, 2.0),
                       lat_peaks=SR.peaks(lat), ax_peaks=SR.peaks(axl))
        ax.text(0.04, 0.96, f"{len(u):,}", transform=ax.transAxes, color='w', fontsize=5.5, va='top')
    b = np.arange(-80, 80.1, 2.0); cb_ = 0.5 * (b[1:] + b[:-1])
    xg = 0.075 + 3 * (wf + 0.012) + 0.05
    for j, (key, ttl) in enumerate((('lat', 'lateral profile'), ('ax', 'axial profile'))):
        ax = fig.add_axes([xg + j * 0.155, yf, 0.14, hf])
        for k in K3:
            hh = gaussian_filter(np.histogram(prof[k][key], b)[0].astype(float), 1.0)
            fw = prof[k]['fwhm_lat' if key == 'lat' else 'fwhm_ax']
            ax.plot(cb_, hh / hh.max(), color=COL[k], lw=1.0 if k != 'net' else 0.8, ls='--' if k == 'net' else '-', label=f'{fw:.0f} nm')
        ax.set_xlim(-80, 80); ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.5, 1]); ax.set_xticks([-50, 0, 50])
        ax.set_xlabel('offset (nm)', labelpad=1); ax.tick_params(length=2, pad=1)
        ax.set_title(ttl, fontsize=6.3, pad=2); ax.grid(alpha=0.25, lw=0.4)
        ax.legend(title='FWHM', title_fontsize=5.0, loc='upper right', frameon=False, fontsize=5.0, handlelength=1.3, borderaxespad=0.0, labelspacing=0.1)
        if j == 0: ax.set_ylabel('norm. count', labelpad=1); letter(ax, 'h', -0.2)
        else: ax.set_yticklabels([])
    save(fig, 'si_4pi_mt')
    ref = json.load(open(f'{P4}/fig/MT_selection_curves_v10_full.json'))
    val = dict(threshold_matched_nm=thr, n_matched=n_m, zoom_matched=zoom_matched, G={k: G[k] for k in ('box', 'core', 'zoom1', 'zoom2', 'section', 'n_segments', 'zr_nm')},
               profiles={k: {q: v for q, v in p.items() if q not in ('lat', 'ax')} for k, p in prof.items()},
               n_overview=int(Z['ov_n']), n_rows_region={k: int(inreg(T[k]).sum()) for k in K3}, info=info, lines={k: LAB[k] for k in K3},
               reference_json=f'{P4}/fig/MT_selection_curves_v10_full.json',
               frc_titles_nm=FRCm, frc_first_crossing_nm=FRC1,
               panel_f_frc_vs_frames=dict(frc_box_nm=list(map(float, cores[0])), frames=FT['N'], frc_nm=FT['frc'], frc_first_nm=FT['frc_first'], n_locs=FT['n'], summary=TS,
                                          rule='first N frames; halves = alternating blocks of min(500, N/12) six-frame windows; frc(px=4, units=nm) curve; frc_nm = sustained interpolated crossing (drawn, titles), frc_first_nm = first crossing (earlier titles)'),
               reference_operating_points={k: {q: ref['operating_points'][o].get(q) for q in ('n_keep', 'fwhm_lat', 'fwhm_ax', 'dip', 'dafl', 'dafl_2_4k', 'fringe_hop', 'frc')}
                                           for k, o in (('net', 'all'), ('net_m', 'matched count'), ('cl', 'classical published (same files)'))})
    json.dump(val, open(f'{OUT}/si_4pi_mt_values.json', 'w'), indent=1, default=float)
    open(f'{OUT}/si_4pi_mt_caption.txt', 'w').write(caption_mt(val))


def caption_mt(v):
    P = v['profiles']; R = v['reference_operating_points']; G = v['G']; I = v['info']
    p = lambda k: (f"{P[k]['fwhm_lat']:.0f} / {P[k]['fwhm_ax']:.0f} nm, lateral valley / peak {P[k]['lat_peaks']['dip']:.2f}"
                   + (f" (two peaks {P[k]['lat_peaks']['sep_nm']:.0f} nm apart)" if P[k]['lat_peaks']['resolved'] else ''))
    d = lambda r: f"{r['dafl'][0]:.2f} / {r['dafl'][1]:.2f} / {r['dafl'][2]:.2f}"
    ra, rm, rc = R['net'], R['net_m'], R['cl']
    L = ['DRAFT CAPTION',
         "4Pi-SIMCODE on the 4Pi-SIMFLUX microtubule data (COS-7, DNA-PAINT with a Cy3B imager; data of the original 4Pi-SIMFLUX publication, 100 files "
         "of 18 000 frames). Three analyses are shown throughout: 4Pi-SIMCODE (standard filter: chi2/dof < 2.5, sigma_z < 25 nm, sigma_x, sigma_y "
         "< 0.12 px, >= 1000 photo-electrons per window, |z - median| <= 300 nm); 4Pi-SIMCODE, matched count (its localizations with the smallest "
         f"reported combined precision sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2) <= {v['threshold_matched_nm']:.2f} nm, as many in boxes 0 and 1 as "
         f"4Pi-SIMFLUX keeps, {v['n_matched']:,}); and 4Pi-SIMFLUX, the analysis as published (published-style pipeline with its published rejections). "
         "(a) The whole field, 4Pi-SIMCODE localizations coloured by z. (b) The 6 x 6 um region of box 0 (white box in a), 4Pi-SIMCODE. (c) Zoom 1, "
         f"a crossing of two microtubules {G['zoom1']['dz_nm']:.0f} nm apart in z, the three analyses, each with its FRC resolution "
         "(x-y, over the whole region of b = box 0; 4 nm histogram pixels, two halves = alternating blocks of 500 six-frame windows, ring FRC smoothed over 5 rings, 1/7 threshold, crossing interpolated between rings and taken where the curve stays below 1/7 for 0.01 /nm (the sustained rule of all FRC values in the 4Pi figures, also the simulation's); simcode/evaluation/frc.py); in the zooms (c, d) the matched count is taken "
         "WITHIN each zoom (the most precise 4Pi-SIMCODE localizations of that zoom, as many as 4Pi-SIMFLUX has there: "
         + ' / '.join(f"{q['n']:,} (combined precision <= {q['threshold_nm']:.2f} nm)" for q in v['zoom_matched'])
         + " for zooms 1 / 2); everywhere else (e-h, numbers) it is the global match over boxes 0 and 1. "
         f"(d) Zoom 2, microtubules overlapping in x-y at different heights; dotted: the {int(G['section']['a1'] - G['section']['a0'])} nm strip of e. "
         f"(e) {G['section']['along']}-z sections along that strip. (f) " + timeline_caption(v['panel_f_frc_vs_frames']['summary'], 'box 0 = the region of b, as the FRC in c', v['panel_f_frc_vs_frames']['frames'][-1]) + " "
         f"(g) Tube cross-section pooled over {P['net']['nseg']} isolated straight segments in "
         "boxes 0 and 1 (in-plane offset across the tube x axial offset, each segment's own centre line). (h) Lateral and axial profiles of g with their "
         "FWHM (dashed: 4Pi-SIMCODE); valley / peak = the lowest count between the two walls of the hollow tube divided by the lower wall peak "
         "(lower = better separated walls). Regions, zooms, section strip and segments are chosen automatically on the 4Pi-SIMFLUX table, so they do "
         "not favour 4Pi-SIMCODE.",
         '',
         'NUMBERS (si_4pi_mt_values.json; reference out/4pi/fig/MT_selection_curves_v10_full.json)',
         f"Tube FWHM lateral / axial: 4Pi-SIMCODE {p('net')}; matched count {p('net_m')}; 4Pi-SIMFLUX {p('cl')}.",
         f"Localizations in boxes 0 + 1: 4Pi-SIMCODE {v['n_rows_region']['net']:,}, matched count {v['n_rows_region']['net_m']:,}, "
         f"4Pi-SIMFLUX {v['n_rows_region']['cl']:,}; whole field (a) {v['n_overview']:,}.",
         f"Adjacent-window precision (DAFL x / y / z nm, boxes 0 + 1): 4Pi-SIMCODE {d(ra)}, matched count {d(rm)}, 4Pi-SIMFLUX {d(rc)}. "
         f"Fringe hops: {100 * ra['fringe_hop']:.1f} / {100 * rm['fringe_hop']:.1f} / {100 * rc['fringe_hop']:.1f} %. "
         f"FRC (box 0): {v['frc_titles_nm']['net']:.1f} / {v['frc_titles_nm']['net_m']:.1f} / {v['frc_titles_nm']['cl']:.1f} nm "
         f"(first-crossing rule: {v['frc_first_crossing_nm']['net']:.1f} / {v['frc_first_crossing_nm']['net_m']:.1f} / {v['frc_first_crossing_nm']['cl']:.1f} nm).",
         f"Registration network -> 4Pi-SIMFLUX on {I['registration']['n_matched']:,} matched emitters: {I['registration']['dx_median_nm']:.1f} / "
         f"{I['registration']['dy_median_nm']:.1f} / {I['registration']['dz_median_nm']:.1f} nm removed. Sample z drift {I['zdrift_ptp_nm']:.0f} nm ptp "
         "(structure-based, 4Pi-SIMFLUX table), the same curve subtracted from every row.",
         '',
         'CONDITIONS',
         "4Pi-SIMCODE: det4pi detector, emitter network v10, explicit 4Pi solve with the in-situ SIM pattern and 4Pi phase field per file; lateral drift: "
         "own DME. 4Pi-SIMFLUX: our implementation of the published pipeline, published-style analysis with its rejections (out/4pi/classical/STATUS.md). "
         "Panels b-h: 4Pi-SIMFLUX rows and 4Pi-SIMCODE share the 4Pi-SIMFLUX z-drift curve and are registered on matched emitters; panel a uses the "
         "4Pi-SIMCODE table's own drift and z drift.",
         'CODE: analysis/paper/si_4pi.py (prep_mt, mt); recipe imported from analysis/fourpi/showcase_real_mt.py.']
    return '\n'.join(L) + '\n'


# =============================================================================================================== sim
SW = f'{P4}/sweep4pi'
NM = 128.0; LO, HI = 8 * NM, 52 * NM
DENS = ['mtc_d01', 'mtc_d03', 'mtc_d05', 'mtc_d075', 'mtc_d15']
SHOW = ['mtc_d01', 'mtc_d03', 'mtc_d075', 'mtc_d15']
BEST = {t: 'net_sig1' for t in ('mtc_d01', 'mtc_d03', 'mtc_d05', 'mtc_d075', 'mtc_d15')}   # ONE selection at every density (user 2026-11-15)
DREG = os.path.expanduser('~/simcode2/out/fourpi_sim/sweep/densregion.json')   # analysis/fourpi_sim/sweep4pi_densregion.py


def dense_ref():
    """real MT ON density (per um^2 per frame): densest 7.68 um window (= simulated field), densest 20 um window, field average"""
    d = json.load(open(DREG)); f = d['fixed']
    return dict(d768=f['sim_field_7.68um']['on_per_um2_frame_mean'], d768_sd=f['sim_field_7.68um']['on_per_um2_frame_sd'],
                d20=f['20um']['on_per_um2_frame_mean'], d20_sd=f['20um']['on_per_um2_frame_sd'], field=d['field_on_per_um2_frame_mean'],
                t768=f['sim_field_7.68um']['on_per_frame_per_um_tube_mean'], nfiles=len(d['files']))


FRAME_S = 5.575e-3                                       # measured frame interval of the real movies (dcimg time stamps); not used in the sim figure (frames only, user)
NMIN_CL = 50                                             # fewer kept 4Pi-SIMFLUX localizations: rates not measurable
SELTXT = dict(net_sig0='max(σx, σy) < 7, σz < 16 nm', net_sig1='max(σx, σy) < 5, σz < 12 nm')
ZOOM1 = ((3945.0, 3785.0), 1200.0)                       # sweep4pi_fig.ZOOMS[0]: crossing, dz 256 nm
ZOOM2 = ((4800.0, 2330.0), 1200.0)                       # 4-5 tubes


def fig_sim(a):
    """a.share: the temporary share variant si_4pi_sim_share (not for the SI): no x-z row, only the zoom box (labelled 'zoom') in a,
    no strip lines in b, panels re-lettered a-e"""
    share = bool(getattr(a, 'share', False))
    DR = dense_ref()
    Es = {t: json.load(open(f'{SW}/{t}/eval.json')) for t in DENS}
    gt = np.load(f'{P4}/simmovie/{SHOW[0]}/gt_sites.npz')['sites']; gx, gy, gz = gt[:, 0] * NM, gt[:, 1] * NM, gt[:, 2] * 1000
    W, H = 7.2, 7.6
    fig = plt.figure(figsize=(W, H))
    ncol = 1 + len(SHOW); gap = 0.006; x0 = 0.075; wcol = (1 - x0 - 0.005 - gap * (ncol - 1)) / ncol; hcol = wcol * W / H
    ext = HI - LO; zr = (-200, 200)
    rows = [('ov', 'x-y, colour = z'), ('z2net', 'zoom'), ('z2cl', 'zoom'), ('sec', 'x-z')]
    ytop = 0.975
    yrow = [ytop - hcol, ytop - 2 * hcol - 0.012, ytop - 3 * hcol - 0.024]
    hsec = hcol * 0.62; yrow.append(yrow[2] - 0.022 - hsec)
    drop = (0.022 + hsec + 0.017) if share else 0.0                     # share: the x-z row is removed, upper rows move down
    Hs = H * (1 - drop)
    if share: fig.set_size_inches(W, Hs)
    up = lambda y: (y - drop) * H / Hs                                  # upper block (rows a-c, colour bar) -> figure fraction
    lo_ = lambda y: y * H / Hs                                          # lower block (summary panels)
    hh_ = lambda h: h * H / Hs
    def L(t, nm): return np.load(f'{SW}/{t}/locs_{nm}.npz')
    for j in range(ncol):
        xl = x0 + j * (wcol + gap)
        if j == 0:
            data = {'net': (gx, gy, gz), 'cl': None}; ttl = f'ground truth\n{len(gx):,} labelling sites'
        else:
            t = SHOW[j - 1]; E = Es[t]; Ln = L(t, BEST[t]); Lc = L(t, 'cl_pub')
            data = {'net': (Ln['x'], Ln['y'], Ln['z']), 'cl': (Lc['x'], Lc['y'], Lc['z'])}
            ttl = f"{E['dens_fov'] / DR['d768']:.2g}x densest real region\n({E['crowd_frame_mean']:.1f} others per 15 px patch)" if 'crowd_frame_mean' in E and np.isscalar(E['crowd_frame_mean']) \
                else f"{E['dens_fov'] / DR['d768']:.2g}x densest real region"
        x, y, z = data['net']
        ax = fig.add_axes([xl, up(yrow[0]), wcol, hh_(hcol)]); ax.imshow(zrgb(x, y, z, LO, LO, ext, 8.0, zr, 0.8)); noaxis(ax); ax.set_title(ttl, fontsize=6.5, pad=2)
        for (cc, s), lab in ((ZOOM2, 'zoom'),):                       # only the zoom-2 box, labelled 'zoom' (user 2026-11-15)
            ax.add_patch(plt.Rectangle(((cc[0] - s / 2 - LO) / 8, (cc[1] - s / 2 - LO) / 8), s / 8, s / 8, fill=False, ec='w', lw=0.5))
            ax.text((cc[0] + s / 2 - LO) / 8 + 2, (cc[1] - s / 2 - LO) / 8, lab, color='w', fontsize=5.5, va='top')
        if j == 0: scalebar(ax, 1000 / ext, '1 µm'); letter(ax, 'a', -0.02)
        if j > 0:
            r = Es[t]['methods'][BEST[t]]
            ax.text(0.97, 0.03, f"{len(x):,} locs\n{SELTXT[BEST[t]]}", transform=ax.transAxes, color='w', fontsize=5.3, ha='right', va='bottom')
        cc, s = ZOOM2
        ax = fig.add_axes([xl, up(yrow[1]), wcol, hh_(hcol)]); ax.imshow(zrgb(x, y, z, cc[0] - s / 2, cc[1] - s / 2, s, 4.0, zr, 1.0)); noaxis(ax)
        if not share: ax.axhline((s / 2 - 100) / 4, color='w', lw=0.4, ls=':'); ax.axhline((s / 2 + 100) / 4, color='w', lw=0.4, ls=':')
        if not share and j == 0:                                       # marker: the strip that panel d sections
            ax.text(0.03, 0.5, 'x-z section in d', transform=ax.transAxes, color='w', fontsize=5.5, ha='left', va='center',
                    bbox=dict(fc='k', ec='none', alpha=0.6, pad=0.6))
        if j == 0: scalebar(ax, 200 / s, '200 nm'); letter(ax, 'b', -0.02)
        ax = fig.add_axes([xl, up(yrow[2]), wcol, hh_(hcol)])
        if data['cl'] is not None:
            xc, yc, zc_ = data['cl']; ax.imshow(zrgb(xc, yc, zc_, cc[0] - s / 2, cc[1] - s / 2, s, 4.0, zr, 1.0))
            ax.text(0.97, 0.03, f"{int(inrect(xc, yc, (cc[0] - s / 2, cc[0] + s / 2, cc[1] - s / 2, cc[1] + s / 2)).sum()):,}", transform=ax.transAxes, color='w', fontsize=5.3, ha='right', va='bottom')
        else:
            ax.set_visible(False)
        noaxis(ax)
        if j == 0: fig.text(x0 - 0.02 * wcol, up(yrow[2] + hcol * 1.02), 'c', fontsize=9, weight='bold', va='bottom', ha='right')
        if j > 0:
            n_ = int(inrect(x, y, (cc[0] - s / 2, cc[0] + s / 2, cc[1] - s / 2, cc[1] + s / 2)).sum())
            fig.axes[-2].text(0.97, 0.03, f"{n_:,}", transform=fig.axes[-2].transAxes, color='w', fontsize=5.3, ha='right', va='bottom')
        # x-z section along the 200 nm y-strip through the centre of the zoom (ZOOM2; dotted in b), network
        if share: continue
        ax = fig.add_axes([xl, yrow[3], wcol, hsec])
        cc, s = ZOOM2
        m = np.abs(y - cc[1]) < 100
        im = dens(x[m], z[m], cc[0] - s / 2, cc[0] + s / 2, -350, 350, 5.0, 1.0, 99.5)
        ax.imshow(im[::-1], cmap='Greys_r', extent=(-s / 2, s / 2, -350, 350), vmin=0, vmax=1, aspect='auto')
        ax.set_yticks([-200, 0, 200]); ax.set_xticks([-500, 0, 500]); ax.tick_params(length=2, pad=1)
        ax.set_xlabel('x (nm)', labelpad=1)
        if j == 0: ax.set_ylabel('z (nm)', labelpad=0); letter(ax, 'd', -0.02)
        else: ax.set_yticklabels([])
    # row labels
    zl = 'zoom'
    rl = ((yrow[0], hcol, '4Pi-SIMCODE'), (yrow[1], hcol, f'4Pi-SIMCODE\n{zl}'), (yrow[2], hcol, f'4Pi-SIMFLUX\n{zl}'))
    for yy, hh, t in (rl if share else rl + ((yrow[3], hsec, '4Pi-SIMCODE\nx-z'),)):
        fig.text(0.008 if hh != hsec else 0.0, up(yy + hh / 2), t, rotation=90, ha='center', va='center', fontsize=6.5)
    zcbar(fig, [x0 + 0.01, up(yrow[2] - 0.045) if share else yrow[3] - 0.062, 0.17, hh_(0.007)], zr, 'z (nm)')
    # summary panels
    ys = 0.05; hs = 0.15; ws = 0.19
    cm = plt.get_cmap('viridis')
    cols = {t: cm(i / (len(DENS) - 1)) for i, t in enumerate(DENS)}
    TF = {t: json.load(open(f'{SW}/{t}/ttf_fp.json')) for t in DENS}       # analysis/fourpi_sim/sweep4pi_ttf_fp.py
    from matplotlib.lines import Line2D
    for i, (key, ttl) in enumerate((('ttf_xy', 'GT-FRC x-y (nm)'), ('ttf_xz', 'GT-FRC x-z sections (nm)'))):
        ax = fig.add_axes([0.075 + i * 0.255, lo_(ys), ws, hh_(hs)])
        for t in DENS:
            F = TF[t]; g = np.array(F['grid'], float)                     # frames
            ax.plot(g, F['methods'][BEST[t]][key], color=cols[t], lw=1.2, label=f"{F['dens_fov'] / DR['d768']:.2g}x densest real region")
            c = np.array(F['methods']['cl_pub'][key], float)
            ax.plot(g, c, color=cols[t], lw=0.9, ls=':')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(15, 250); ax.set_yticks([20, 30, 50, 100, 200]); ax.set_yticklabels(['20', '30', '50', '100', '200'])
        ax.yaxis.set_minor_formatter(plt.NullFormatter()); ax.grid(alpha=0.25, lw=0.4, which='both')
        ax.axhline(25 if 'xy' in key else 35, color='0.4', lw=0.6, ls='--')
        ax.set_xticks([1000, 3000, 10000, 30000]); ax.set_xticklabels(['1,000', '3,000', '10,000', '30,000']); ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel('frames', labelpad=1); ax.set_ylabel(ttl, labelpad=1); ax.tick_params(length=2, pad=1)
        if i == 0:
            h_, l_ = ax.get_legend_handles_labels()
            h_ += [Line2D([], [], color='k', lw=1.2), Line2D([], [], color='k', lw=0.9, ls=':')]
            l_ += ['4Pi-SIMCODE (solid)', '4Pi-SIMFLUX (dotted)']
            fig.legend(h_, l_, frameon=False, fontsize=5.5, loc='lower left', bbox_to_anchor=(0.27, lo_(ys + hs + 0.025)),
                       ncol=4, handlelength=1.8, columnspacing=1.0, labelspacing=0.2)
            letter(ax, 'd' if share else 'e', -0.15)
    ax = fig.add_axes([0.075 + 2 * 0.255 + 0.03, lo_(ys), 0.215, hh_(hs)])
    D = np.array([Es[t]['dens_fov'] for t in DENS])
    def g(t, nm, k):
        if k == 'recall': return Es[t]['methods'][nm]['recall']
        return TF[t]['methods'][nm]['fp_fraction']
    for k, ls in (('recall', '-'), ('fp', '--')):
        ax.plot(D, [g(t, BEST[t], k) for t in DENS], ls, color=COL['net'], marker='o', ms=2, lw=1.1)
        y = np.array([g(t, 'cl_pub', k) for t in DENS], float)
        nm_ = np.array([TF[t]['methods']['cl_pub']['n_locs'] < NMIN_CL for t in DENS])     # too few kept: not measurable
        ax.plot(D[~nm_], y[~nm_], ls, color=COL['cl'], marker='s', ms=2, lw=1.1)
        for xx in D[nm_]: ax.text(xx, 0.02 if k == 'recall' else 0.06, 'n.m.', color=COL['cl'], fontsize=5, ha='center')
    ax.set_xscale('log'); ax.set_xticks(D); ax.set_xticklabels([f'{v / DR["d768"]:.2g}x' for v in D]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('density (x densest real region)', labelpad=1); ax.set_ylabel('fraction', labelpad=1); ax.set_ylim(0, 0.75); ax.grid(alpha=0.25, lw=0.4); ax.tick_params(length=2, pad=1)
    hl = [Line2D([], [], color=COL['net'], lw=1.1, label='4Pi-SIMCODE'), Line2D([], [], color=COL['cl'], lw=1.1, label='4Pi-SIMFLUX'),
          Line2D([], [], color='k', ls='-', lw=0.9, label='recall'), Line2D([], [], color='k', ls='--', lw=0.9, label='false positives')]
    ax.legend(handles=hl, frameon=False, fontsize=5.3, loc='upper right', handlelength=1.8, labelspacing=0.15)
    letter(ax, 'e' if share else 'f', -0.15)
    if share:
        for ext_ in ('pdf', 'png'): fig.savefig(f'{OUT}/si_4pi_sim_share.{ext_}', bbox_inches='tight', **({'dpi': 220} if ext_ == 'png' else {}))
        plt.close(fig); lg(f'-> {OUT}/si_4pi_sim_share.(pdf|png)'); return
    save(fig, 'si_4pi_sim')
    val = {}
    for t in DENS:
        E = Es[t]; val[t] = dict(dens_fov=E['dens_fov'], dens_region=E['dens_region'], T=E['T'], crowd=E.get('crowd_frame_mean'))
        for nm in (BEST[t], 'net_base', 'net_chi16', 'cl_pub', 'net_bound_base'):
            if nm not in E['methods']: continue
            r = E['methods'][nm]
            val[t][nm] = dict(n_locs=r['n_locs'], recall=r['recall'], precision=r['precision'], false_pos_fixed_gate=r['ghost_fixed']['ghost_fraction'],
                              wrong_fringe=r['wrong_fringe'], lat_rob=r['lat_rob'], ax_rob=r['ax_rob'], gtfrc_last_xy=r['gtfrc_last_xy'], gtfrc_last_xz=r['gtfrc_last_xz'],
                              frames_to_xy25=(r.get('frames_to_last_xy') or {}).get('25'), frames_to_xz35=(r.get('frames_to_last_xz') or {}).get('35'),
                              coverage_excess_cv=r.get('coverage', {}).get('excess_cv'), dip_lat=r.get('tube', {}).get('dip_lat'), dip_ax=r.get('tube', {}).get('dip_ax'))
            if nm in TF[t]['methods']:
                q = TF[t]['methods'][nm]
                val[t][nm].update(false_pos_3d=q['fp_fraction'], false_pos_wrong_fringe_only=q['fp_wrong_fringe_only'],
                                  frames_to_xy25=q['frames_to_xy'].get('25'), frames_to_xz35=q['frames_to_xz'].get('35'),
                                  gtfrc_end_xy=q['ttf_xy'][-1], gtfrc_end_xz=q['ttf_xz'][-1])
    json.dump(dict(selection=BEST, per_density=val), open(f'{OUT}/si_4pi_sim_values.json', 'w'), indent=1, default=float)
    open(f'{OUT}/si_4pi_sim_caption.txt', 'w').write(caption_sim(val))


def dens_limit(V, nm, thr=40.0):
    """highest density with GT-FRC x-y <= thr and x-z <= thr (30 000 frames), log-interpolated between sweep points; (value, bound)"""
    D = np.array([V[t]['dens_fov'] for t in DENS])
    q = np.array([max(V[t][nm if nm != 'net' else BEST[t]]['gtfrc_end_xy'], V[t][nm if nm != 'net' else BEST[t]]['gtfrc_end_xz']) for t in DENS])
    if q[-1] <= thr: return float(D[-1]), '>='
    if q[0] > thr: return float(D[0]), '<'
    i = int(np.nonzero(q > thr)[0][0]); u = (thr - q[i - 1]) / (q[i] - q[i - 1])
    return float(np.exp(np.log(D[i - 1]) + u * (np.log(D[i]) - np.log(D[i - 1])))), '='


def dens_limit_text(V, DR, thr=40.0):
    dn, bn = dens_limit(V, 'net', thr); dc, bc = dens_limit(V, 'cl_pub', thr)
    rel = '>=' if bn == '>=' else '='
    return (f"DENSITY LIMIT (one criterion for both: the highest density at which a method still gives GT-FRC <= {thr:.0f} nm in x-y AND in x-z "
            f"sections within 30 000 frames; 4Pi-SIMFLUX reaches 34 / 36 nm at the lowest density): 4Pi-SIMFLUX up to {dc / DR['d768']:.2f}x the densest real "
            f"region (interpolated; {dc:.2f} ON /um^2/frame), 4Pi-SIMCODE {bn} {dn / DR['d768']:.1f}x ({dn:.2f} ON /um^2/frame; the highest simulated "
            f"density), i.e. 4Pi-SIMCODE tolerates {rel} {dn / dc:.1f}x the emitter density of 4Pi-SIMFLUX.")


def caption_sim(V):
    DR = dense_ref()
    fmt = lambda x, n=0: '-' if x is None else (f'{x:.{n}f}' if np.isfinite(x) else '-')
    sec = lambda f: 'not reached' if f is None else f'{f:,.0f} frames'
    L = ['DRAFT CAPTION',
         "4Pi-SIMCODE versus 4Pi-SIMFLUX (the analysis as published: our implementation of the published pipeline, sim_cl.py published-style, with "
         "the published rejections) on simulated 4Pi-SIMFLUX movies of a microtubule cell field at increasing labelling "
         "density. Structure: the 'Microtubules (cells)' world model of webSMLM / inSiliScope (flat lamella below the nucleus, 59 microtubules, labels "
         "24.6 nm from the tube axis, 70 % labelling, heights 0-300 nm, focus at 150 nm). Movies: 30 000 frames (5000 six-frame windows), "
         "4 channels of 60 x 60 px (128 nm), the bead-fitted 4Pi PSF of the real data, measured camera noise and background, the real SIM pattern geometry, "
         "a drifting 4Pi phase field and DNA-PAINT kinetics (mean on-time 10 frames, median 450 photo-electrons per frame). Both pipelines estimate the "
         "SIM pattern and the 4Pi phase field in situ. "
         "Densities are averages over the simulated 7.7 x 7.7 um field, which is as crowded as the densest regions of the real cell; the multiples "
         "refer to the densest real region of the same size: the densest 7.7 x 7.7 um region of the real 4Pi-SIMFLUX microtubule movie has "
         f"{DR['d768']:.2f} +- {DR['d768_sd']:.2f} active (ON) emitters per um^2 per frame ({DR['nfiles']} files over the series) = 1x. "
         "In absolute units the five simulated movies have "
         + ' / '.join(f"{V[t]['dens_fov']:.2f}" for t in DENS) + " ON emitters /um^2/frame. "
         "(a) x-y renders coloured by z (ground truth = all labelling sites; 4Pi-SIMCODE with one selection at every density: reported max(sigma_x, sigma_y) < 5 nm and sigma_z < 12 nm, on top of the base filters). "
         "(b, c) The zoom (white box in a; four to five tubes at 90-190 nm height differences): 4Pi-SIMCODE (b) and 4Pi-SIMFLUX (c); numbers = "
         "localizations in the zoom. (d) 4Pi-SIMCODE x-z sections taken along the 200 nm wide y-strip through the centre of the zoom (between the "
         "dotted lines in b), over the zoom's full x range. (e) Ground-truth FRC resolution in x-y and in x-z sections "
         "versus the number of frames acquired for every density (solid: 4Pi-SIMCODE, dotted: 4Pi-SIMFLUX; dashed: the 25 / 35 nm fidelity targets): 1/7 crossing "
         "of the 5-shell smoothed curve, interpolated between shells, taken where the curve stays below 1/7 for 0.01 /nm, best value up to that time. "
         "The step from ~65 to ~35 nm is the hollow-tube spectrum (Bessel zero at 1/64 nm for 24.6 nm radius): the curve crosses 1/7 before that null "
         "until the next lobe rises above 1/7. (f) Recall of the ground-truth emitter-windows (>= 300 e) and the false-positive fraction versus density: "
         "a kept localization is a false positive when no emitter that is on in its own window lies within 30 / 30 / 75 nm in x / y / z; "
         "a localization placed one 4Pi fringe off in z (|dz| > 100 nm from its laterally matched emitter) is counted as a false positive too "
         "(false positives / all kept localizations).",
         '',
         'NUMBERS per density (si_4pi_sim_values.json; full table out/fourpi_sim/sweep/tables.md; e / f from out/4pi/sweep4pi/<tag>/ttf_fp.json)']
    for t, v in V.items():
        b = BEST[t]; n, c = v[b], v['cl_pub']
        L.append(f"{v['dens_fov'] / DR['d768']:.2g}x densest real region ({v['dens_fov']:.2f} ON /um^2/frame; {v['dens_region']:.2f} in the evaluation region): 4Pi-SIMCODE ({SELTXT[b]}) {n['n_locs']:,} locs, recall {n['recall']:.2f}, "
                 f"false positives {100 * n['false_pos_3d']:.1f} % (of which one-fringe-off only {100 * n['false_pos_wrong_fringe_only']:.1f} %), robust error lateral / axial "
                 f"{n['lat_rob']:.1f} / {n['ax_rob']:.1f} nm, GT-FRC x-y / x-z {n['gtfrc_end_xy']:.0f} / {n['gtfrc_end_xz']:.0f} nm, frames to x-y 25 nm "
                 f"{fmt(n['frames_to_xy25'])}, to x-z 35 nm {fmt(n['frames_to_xz35'])}, "
                 f"coverage excess CV {n['coverage_excess_cv']:.2f}; "
                 f"4Pi-SIMFLUX {c['n_locs']:,} locs, recall {c['recall']:.2f}, false positives {100 * c['false_pos_3d']:.1f} %, "
                 f"{c['lat_rob']:.1f} / {c['ax_rob']:.1f} nm, GT-FRC {c['gtfrc_end_xy']:.0f} / {c['gtfrc_end_xz']:.0f} nm, frames to 25 / 35 nm "
                 f"{fmt(c['frames_to_xy25'])} / {fmt(c['frames_to_xz35'])}.")
        if 'net_bound_base' in v:
            q = v['net_bound_base']
            L.append(f"   bound at this density (4Pi-SIMCODE with the TRUE pattern + 4Pi phase field, no selection): GT-FRC x-y / x-z (last crossing) {q['gtfrc_last_xy']:.0f} / "
                     f"{q['gtfrc_last_xz']:.0f} nm, wrong fringe {100 * q['wrong_fringe']:.1f} %.")
    both = lambda m: None if (m['frames_to_xy25'] is None or m['frames_to_xz35'] is None) else max(m['frames_to_xy25'], m['frames_to_xz35'])
    fb = {t: both(V[t][BEST[t]]) for t in V}
    ok = {t: f for t, f in fb.items() if f is not None}
    tb = min(ok, key=ok.get) if ok else None
    cl_any = any(both(V[t]['cl_pub']) is not None for t in V)
    rd = f"{V[tb]['dens_fov'] / DR['d768']:.2g}x the densest real region" if tb else '-'
    MX = ' / '.join(f"{V[t]['dens_fov'] / DR['d768']:.2g}x" for t in DENS)
    t01 = 'mtc_d01'
    ratio = (fb[t01] / fb[tb]) if (tb and fb.get(t01)) else None
    L += ['',
          f'READING (densities as multiples of the densest real region, {MX}): the 4Pi-SIMFLUX pipeline keeps 22 % / 7 % / 3 % / 1 % / 0 % of the ground-truth units and '
          + ('reaches 25 nm (x-y) and 35 nm (x-z) within 30 000 frames at some density; ' if cl_any else 'never reaches both 25 nm (x-y) and 35 nm (x-z) within 30 000 frames; ')
          + (f"4Pi-SIMCODE reaches both soonest at {rd}, after {ok[tb]:,.0f} frames" if tb else '4Pi-SIMCODE does not reach both')
          + (f", {ratio:.1f}x faster than at {V[t01]['dens_fov'] / DR['d768']:.2g}x ({fb[t01]:,.0f} frames). " if ratio else '. ')
          + f'Time to both targets per density ({MX}): '
          + ' / '.join(sec(fb[t]) for t in DENS)
          + '; coverage excess CV ' + ' / '.join(f"{V[t][BEST[t]]['coverage_excess_cv']:.2f}" for t in DENS)
          + ' (holes along the tubes grow with density); false positives '
          + ' / '.join(f"{100 * V[t][BEST[t]]['false_pos_3d']:.1f}" for t in DENS) + ' % (4Pi-SIMFLUX '
          + ' / '.join(f"{100 * V[t]['cl_pub']['false_pos_3d']:.1f}" for t in DENS) + ' %); at the highest density the network is out of distribution '
          '(neighbours along the tube bias the astigmatic prior that picks the 4Pi fringe; out/fourpi_sim/sweep/STATUS.md). '
          f"The x-z plateau (~31 nm) is set by the in-situ 4Pi phase field (21 nm with the true field at {V['mtc_d03']['dens_fov'] / DR['d768']:.2g}x). "
          + dens_limit_text(V, DR),
          'CONDITIONS: detector det4pi, emitter network v10 (trained with <= 3 neighbours within 7 px: out of distribution at >= 2.1x), explicit solve with '
          'in-situ pattern + phase field (simlin4pi.py --pattern iter --phi field); one selection at every density: base filters (sigma_z < 25 nm, '
          f'sigma_x, sigma_y < 0.12 px, N > 300) + reported max(sigma_x, sigma_y) < 5 nm and sigma_z < 12 nm (net_sig1 of the sweep). 4Pi-SIMFLUX: the analysis as published (sim_cl.py, published-style, with its published rejections; rates marked n.m. where it keeps < {NMIN_CL} '
          'localizations: none here). One structure realisation, one movie per density.',
          'CODE: analysis/paper/si_4pi.py (sim); data analysis/fourpi_sim/sweep4pi_{movie,run,eval}.py, panels e / f analysis/fourpi_sim/sweep4pi_ttf_fp.py.']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('what', choices=['prep_npc', 'prep_mt', 'npc', 'mt', 'sim', 'sim_share'])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == 'sim_share': a.share = True
    dict(prep_npc=prep_npc, prep_mt=prep_mt, npc=fig_npc, mt=fig_mt, sim=fig_sim, sim_share=fig_sim)[a.what](a)
