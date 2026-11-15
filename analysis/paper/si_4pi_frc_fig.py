"""Figure, caption and values of si_4pi_frc_vs_frames (from cache_si_4pi/frc_vs_frames.json, made by si_4pi_frc.py compute)."""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.expanduser('~/simcode2/out/paper_figs_v2')
NAME = 'si_4pi_frc_vs_frames'
COL = dict(net='#2a78d6', cl='#eb6834')
LAB = dict(net='4Pi-SIMCODE', cl='4Pi-SIMFLUX')
TARGETS = (25, 20, 15)
mpl.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'legend.fontsize': 5.5, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'pdf.fonttype': 42})


def frames_to(N, r, t):
    """first N from which the curve stays <= t (log-interpolated with the point before); None if never"""
    N = np.asarray(N, float); r = np.asarray(r, float)
    ok = np.isfinite(r) & (r <= t)
    if not ok[-1]: return None
    bad = np.nonzero(~ok)[0]
    i = 0 if len(bad) == 0 else bad[-1] + 1
    if i == 0 or not np.isfinite(r[i - 1]): return float(N[i])
    u = (r[i - 1] - t) / (r[i - 1] - r[i])
    return float(np.exp(np.log(N[i - 1]) + u * (np.log(N[i]) - np.log(N[i - 1]))))


def saturation(N, r, rel=0.10):
    """first N from which the curve stays within rel of its final (all-frames) value"""
    r = np.asarray(r, float); f = r[-1]
    return frames_to(N, r, f * (1 + rel)), float(f)


def cur(D, key, meth, mode, metric):
    c = D[key][f'{meth}_{mode}_{metric}']; return np.array(c['N'], float), np.array(c['sus'], float)


def fmt(v): return '-' if v is None else f'{v:,.0f}'


def setx(ax, xmax=2e6):
    ax.set_xscale('log'); ax.set_xlim(900, xmax); ax.set_xlabel('frames')


def targets(ax, ts=TARGETS):
    for t in ts: ax.axhline(t, color='0.6', lw=0.4, ls=(0, (2, 2)), zorder=0)


def letter(ax, s): ax.text(-0.2, 1.06, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')


def main(res):
    D = json.load(open(res)); S = D['structure']
    V = dict(regions={}, sim={}, structure={})
    # ------------------------------------------------------------------ values
    for key in ('mt_dense', 'mt_box0', 'mt_field', 'npc_center'):
        q = dict(box_nm=D[key]['box'], frames_total=D[key]['T'], n_rows=D[key]['n_rows'])
        for meth in ('net', 'cl'):
            for mode in ('ind', 'spread', 'blk'):
                for metric in ('xy', 'xz'):
                    k = f'{meth}_{mode}_{metric}'
                    if k not in D[key]: continue
                    N, r = cur(D, key, meth, mode, metric); sat, fin = saturation(N, r)
                    tt = (25, 20, 15) if metric == 'xy' else (35, 30, 25)
                    q[k] = dict(final_nm=fin, frames_to={str(t): frames_to(N, r, t) for t in tt}, saturation_frames_10pct=sat,
                                at_frames={str(int(n)): float(v) for n, v in zip(N, r) if int(n) in (1000, 2000, 4000, 8000, 11314, 16000, 32000, 64000, 128000, 256000, 512000) or n == N[-1]})
        V['regions'][key] = q
    for key in [k for k in D if k.startswith('sim_')]:
        q = dict(x_dense=D[key]['x_dense'])
        for meth in ('net', 'cl'):
            g = D[key]['gt'][meth]
            q[f'{meth}_gt_xy_runmin'] = dict(frames_to={str(t): frames_to(g['N'], g['xy_runmin'], t) for t in (25, 20)}, final_nm=g['xy_runmin'][-1])
            for mode in ('ind', 'blk', 'spread'):
                N, r = cur(D, key, meth, mode, 'xy')
                q[f'{meth}_{mode}_xy'] = dict(final_nm=float(r[-1]), frames_to={str(t): frames_to(N, r, t) for t in (25, 20)})
        V['sim'][key] = q
    for key in ('mt_dense', 'mt_box0'):
        s = S[key]; q = dict(ref=s['ref'])
        for meth in ('net', 'cl'):
            for mode in ('ind', 'spread'):
                for k in ('1', '3', '10'):
                    c = np.array(s[f'{meth}_{mode}'][k])
                    q[f'{meth}_{mode}_ge{k}'] = dict(final=float(c[-1]), **{f'frames_to_{p}': frames_to(s['grid'], -c, -p) for p in (0.9, 0.95, 0.99)})
        V['structure'][key] = q
    s = S['npc']; q = dict(n_pores=s['n_pores'])
    for meth in ('net', 'cl'):
        for mode in ('ind', 'spread'):
            c = np.array(s[f'{meth}_{mode}']['corners_ge1']); f = c[-1]
            q[f'{meth}_{mode}_corners_ge1'] = dict(final=float(f), frames_to_90pct_of_final=frames_to(s['grid'], -c, -0.9 * f), frames_to_95pct_of_final=frames_to(s['grid'], -c, -0.95 * f),
                                                   frames_to_99pct_of_final=frames_to(s['grid'], -c, -0.99 * f))
        q[f'{meth}_dip'] = dict(grid=s['grid_dip'], **s[f'{meth}_dip'])
    V['structure']['npc'] = q
    V['lag'] = {key: D[key]['lag'] for key in D if isinstance(D[key], dict) and 'lag' in D[key]}

    # ------------------------------------------------------------------ figure
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.0))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.075, wspace=0.42, hspace=0.62)
    LS = dict(ind='-', spread=(0, (1, 1.2)), blk=(0, (4, 1.5)))
    LW = dict(ind=1.1, spread=0.9, blk=0.6)

    def frc_panel(ax, key, metric, modes=('ind', 'spread', 'blk'), title='', ylim=(8, 200)):
        for meth in ('net', 'cl'):
            for mode in modes:
                k = f'{meth}_{mode}_{metric}'
                if k not in D[key]: continue
                N, r = cur(D, key, meth, mode, metric)
                ax.plot(N, r, color=COL[meth], ls=LS[mode], lw=LW[mode])
        ax.set_yscale('log'); ax.set_ylim(*ylim); setx(ax)
        ax.set_yticks([10, 15, 20, 25, 35, 50, 100, 200]); ax.set_yticklabels(['10', '15', '20', '25', '35', '50', '100', '200'])
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.axvline(30000, color='0.75', lw=0.5, zorder=0)
        ax.set_title(title, fontsize=6.5, pad=3)

    ax = axs[0, 0]; frc_panel(ax, 'mt_dense', 'xy', title='MT, densest 7.7 × 7.7 µm region'); targets(ax); ax.set_ylabel('FRC x-y (nm)'); letter(ax, 'a')
    ax.text(30000 * 1.1, 8.6, 'length of the\nsimulated movies', fontsize=5, color='0.5', va='bottom')
    ax = axs[0, 1]; frc_panel(ax, 'mt_dense', 'xz', title='MT, densest region, x-z sections'); targets(ax, (35, 30, 25)); ax.set_ylabel('FRC x-z sections (nm)'); letter(ax, 'b')
    ax = axs[0, 2]
    s = S['mt_dense']
    for meth in ('net', 'cl'):
        for mode in ('ind', 'spread'):
            ax.plot(s['grid'], s[f'{meth}_{mode}']['3'], color=COL[meth], ls=LS[mode], lw=LW[mode])
    setx(ax); ax.set_ylim(0, 1.02); ax.set_ylabel('skeleton covered\n(≥ 3 locs within ±30 nm)'); ax.set_title('MT, densest region, coverage', fontsize=6.5, pad=3); letter(ax, 'c')
    ax.axvline(30000, color='0.75', lw=0.5, zorder=0)

    ax = axs[1, 0]
    for meth in ('net', 'cl'):
        for metric, ls in (('xy', '-'), ('xz', (0, (4, 1.5)))):
            N, r = cur(D, 'mt_box0', meth, 'ind', metric); ax.plot(N, r, color=COL[meth], ls=ls, lw=1.0)
        N, r = cur(D, 'mt_field', meth, 'ind', 'xy'); ax.plot(N, r, color=COL[meth], ls=(0, (1, 1.2)), lw=0.8)
    ax.set_yscale('log'); ax.set_ylim(8, 200); setx(ax); targets(ax)
    ax.set_yticks([10, 15, 20, 25, 35, 50, 100, 200]); ax.set_yticklabels(['10', '15', '20', '25', '35', '50', '100', '200']); ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.axvline(30000, color='0.75', lw=0.5, zorder=0)
    ax.set_ylabel('FRC (nm)'); ax.set_title('MT, box 0 and 21 × 11 µm field', fontsize=6.5, pad=3); letter(ax, 'd')
    ax.legend(handles=[Line2D([], [], color='0.3', lw=1.0, ls='-', label='box 0, x-y'), Line2D([], [], color='0.3', lw=1.0, ls=(0, (4, 1.5)), label='box 0, x-z sections'),
                       Line2D([], [], color='0.3', lw=0.8, ls=(0, (1, 1.2)), label='field, x-y')], loc='lower left', frameon=False, handlelength=2.2)

    ax = axs[1, 1]
    for meth in ('net', 'cl'):
        for metric, ls in (('xy', '-'), ('xz', (0, (4, 1.5)))):
            N, r = cur(D, 'npc_center', meth, 'ind', metric); ax.plot(N, r, color=COL[meth], ls=ls, lw=1.0)
        N, r = cur(D, 'npc_center', meth, 'spread', 'xy'); ax.plot(N, r, color=COL[meth], ls=(0, (1, 1.2)), lw=0.8)
    ax.set_yscale('log'); ax.set_ylim(8, 200); setx(ax, 8e5); targets(ax)
    ax.set_yticks([10, 15, 20, 25, 35, 50, 100, 200]); ax.set_yticklabels(['10', '15', '20', '25', '35', '50', '100', '200']); ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.axvline(30000, color='0.75', lw=0.5, zorder=0)
    ax.set_ylabel('FRC (nm)'); ax.set_title('NPC, central 10 × 10 µm', fontsize=6.5, pad=3); letter(ax, 'e')
    ax.legend(handles=[Line2D([], [], color='0.3', lw=1.0, ls='-', label='x-y'), Line2D([], [], color='0.3', lw=1.0, ls=(0, (4, 1.5)), label='x-z sections'),
                       Line2D([], [], color='0.3', lw=0.8, ls=(0, (1, 1.2)), label='x-y, spread')], loc='lower left', frameon=False, handlelength=2.2)

    ax = axs[1, 2]; s = S['npc']
    for meth in ('net', 'cl'):
        for mode in ('ind', 'spread'):
            ax.plot(s['grid'], s[f'{meth}_{mode}']['corners_ge1'], color=COL[meth], ls=LS[mode], lw=LW[mode])
    setx(ax, 8e5); ax.set_ylim(0, 8.2); ax.set_yticks(range(0, 9, 2)); ax.set_ylabel('corners seen per pore (of 8)')
    ax.set_title(f"NPC ring completeness ({s['n_pores']:,} pores)", fontsize=6.5, pad=3); letter(ax, 'f'); ax.axvline(30000, color='0.75', lw=0.5, zorder=0)

    ax = axs[2, 0]; key = 'sim_mtc_d03'
    for meth in ('net', 'cl'):
        g = D[key]['gt'][meth]; ax.plot(g['N'], g['xy_runmin'], color=COL[meth], lw=2.2, alpha=0.35, solid_capstyle='butt')
        for mode in ('ind', 'blk'):
            N, r = cur(D, key, meth, mode, 'xy'); ax.plot(N, r, color=COL[meth], ls=LS[mode], lw=LW[mode])
    ax.set_yscale('log'); ax.set_ylim(8, 200); ax.set_xscale('log'); ax.set_xlim(900, 32000); ax.set_xlabel('frames'); targets(ax)
    ax.set_yticks([10, 15, 20, 25, 35, 50, 100, 200]); ax.set_yticklabels(['10', '15', '20', '25', '35', '50', '100', '200']); ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylabel('FRC x-y (nm)'); ax.set_title('simulation, 0.84× densest real region', fontsize=6.5, pad=3); letter(ax, 'g')
    ax.legend(handles=[Line2D([], [], color='0.3', lw=2.2, alpha=0.35, label='ground-truth FRC'), Line2D([], [], color='0.3', lw=LW['ind'], ls='-', label='half-split, independent halves'),
                       Line2D([], [], color='0.3', lw=LW['blk'], ls=LS['blk'], label='half-split, short blocks')], loc='lower left', frameon=False, handlelength=2.2)

    ax = axs[2, 1]
    for key, lab, ls in (('mt_dense', 'MT', '-'), ('npc_center', 'NPC', (0, (4, 1.5))), ('sim_mtc_d03', 'simulation', (0, (1, 1.2)))):
        for meth in ('net', 'cl'):
            L = D[key]['lag'][meth]; e = np.array(L['lag_edges_windows'], float); c = np.sqrt(e[:-1] * e[1:]) * 6
            r = np.array(L['ratio']); ok = np.array(L['pairs_r8']) >= 20
            ax.plot(c[ok], r[ok], color=COL[meth], ls=ls, lw=0.9, marker='o', ms=1.5)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(5, 2e6); ax.set_ylim(0.5, 60); ax.set_xlabel('time lag (frames)'); ax.set_ylabel('pairs < 8 nm / pairs at 8-24 nm\n(per area)')
    ax.axvline(3000, color='0.6', lw=0.5, ls=(0, (2, 2))); ax.text(2800, 0.55, '500\nwindows', fontsize=5, color='0.4', va='bottom', ha='right')
    ax.set_title('same-molecule correlation vs lag', fontsize=6.5, pad=3); letter(ax, 'h')
    ax.legend(handles=[Line2D([], [], color='0.3', lw=0.9, ls='-', label='MT'), Line2D([], [], color='0.3', lw=0.9, ls=(0, (4, 1.5)), label='NPC'),
                       Line2D([], [], color='0.3', lw=0.9, ls=(0, (1, 1.2)), label='simulation 0.84×')], loc='upper right', frameon=False, handlelength=2.2)

    # (i) frames to 25 / 20 / 15 nm (x-y, independent halves)
    ax = axs[2, 2]
    rows = [('mt_dense', 'MT dense'), ('mt_box0', 'MT box 0'), ('npc_center', 'NPC')]
    mk = {25: 'o', 20: 's', 15: 'D'}
    for i, (key, lab) in enumerate(rows):
        for j, meth in enumerate(('net', 'cl')):
            y = i + (0.15 if meth == 'net' else -0.15)
            for t in TARGETS:
                v = V['regions'][key][f'{meth}_ind_xy']['frames_to'][str(t)]
                if v is not None: ax.plot(v, y, marker=mk[t], color=COL[meth], ms=3.2, ls='none', mfc=COL[meth] if t == 15 else 'w', mew=0.8)
            sat = V['regions'][key][f'{meth}_ind_xy']['saturation_frames_10pct']
            if sat: ax.plot([sat, sat], [y - 0.1, y + 0.1], color=COL[meth], lw=1.4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows]); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale('log'); ax.set_xlim(900, 2e6); ax.set_xlabel('frames'); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.set_title('frames to reach x-y FRC', fontsize=6.5, pad=3); letter(ax, 'i')
    ax.legend(handles=[Line2D([], [], marker=mk[t], color='0.3', ls='none', ms=3.2, mfc='0.3' if t == 15 else 'w', label=f'{t} nm') for t in TARGETS]
              + [Line2D([], [], color='0.3', lw=1.4, label='within 10 % of final')], loc='upper left', frameon=False, ncol=1, handlelength=1.2)

    fig.legend(handles=[Line2D([], [], color=COL['net'], lw=1.2, label=LAB['net']), Line2D([], [], color=COL['cl'], lw=1.2, label=LAB['cl']),
                        Line2D([], [], color='0.3', lw=LW['ind'], ls=LS['ind'], label='first N frames, independent halves'),
                        Line2D([], [], color='0.3', lw=LW['spread'], ls=LS['spread'], label='N frames spread over the series'),
                        Line2D([], [], color='0.3', lw=LW['blk'], ls=LS['blk'], label='first N frames, short blocks (a, b)')],
               loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6, handlelength=2.4, columnspacing=1.2)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(f'{OUT}/{NAME}.{ext}', dpi=300 if ext == 'png' else None)
    json.dump(V, open(f'{OUT}/{NAME}_values.json', 'w'), indent=1)
    open(f'{OUT}/{NAME}_caption.txt', 'w').write(caption(V, D))
    print(f'-> {OUT}/{NAME}.pdf / .svg / .png, values')
    return V


def caption(V, D):
    R = V['regions']; St = V['structure']; Sm = V['sim']
    def ft(key, k, ts=('25', '20', '15')):
        q = R[key][k]; return ' / '.join(fmt(q['frames_to'][t]) for t in ts) + f" (all frames {q['final_nm']:.1f} nm; within 10 % of that from {fmt(q['saturation_frames_10pct'])})"
    def at(key, k, n): return R[key][k]['at_frames'].get(str(n))
    lag = lambda key, m, i: D[key]['lag'][m]['ratio'][i]
    cov = lambda key, k, p: fmt(St[key][k][f'frames_to_{p}'])
    L = ['DRAFT CAPTION',
         "How many frames do the real 4Pi movies need? Half-split FRC resolution versus the number of frames N, real data and simulation, "
         "4Pi-SIMCODE (standard filter, as si_4pi_npc / si_4pi_mt) and 4Pi-SIMFLUX (the analysis as published). FRC: localizations of the N frames "
         "split into two halves, x-y histograms with 4 nm pixels over a fixed region (x-z: (x, z) histograms with 5 nm pixels in 250 nm y-slabs, "
         "+-300 nm about the median z, correlation pooled over the slabs, as the simulation's ground-truth x-z FRC), 5-ring smoothing, 1/7 threshold, "
         "crossing interpolated between rings where the curve stays below 1/7 for 0.01 /nm (the rule of si_4pi_sim e); raw values per N (no running "
         "best). Three ways of taking N frames and splitting them: solid, the first N frames, halves = alternating blocks of min(500, N / 12) "
         "six-frame windows (two halves below 6000 frames, 500-window blocks = the figures' FRC from 6000 frames on; independent halves); "
         "dotted, the same blocks spread evenly over the whole series (frames from the whole series instead of its start); thin dashed (a, b, g), "
         "the first N frames with short alternating blocks (>= 10 blocks, min(500, N / 60) windows), which put the repeated detections of one "
         "binding event into both halves. "
         "(a, b) Microtubules, the densest 7.7 x 7.7 um region of the cell (the density reference of si_4pi_sim; 100 files, 1.8 M frames), x-y and "
         "x-z sections; grey vertical line: 30 000 frames, the length of the simulated movies. (c) Microtubule skeleton coverage in that region: "
         "fraction of the skeleton (20 nm pixels, skeleton of both full tables together, "
         f"{St['mt_dense']['ref']['skeleton_um']:.0f} um of tube) with >= 3 localizations within +-30 nm. "
         "(d) Box 0 (the region of si_4pi_mt b; x-y solid, x-z sections dashed) and the 21 x 11 um field around boxes 0 and 1 (x-y, dotted). "
         "(e) Nuclear pores, central 10 x 10 um (as si_4pi_npc; 38 files, 684 000 frames): x-y, x-z sections and x-y spread over the series. "
         f"(f) Ring completeness: corners seen per pore (of 8; >= 1 localization in the 45 deg sector, ring radius 25-85 nm, per-pore rotation from both full tables), all {St['npc']['n_pores']:,} pores. "
         "(g) The simulation at 0.84x the densest real region (si_4pi_sim, 30 000 frames): ground-truth FRC (thick, running best as in si_4pi_sim e) "
         "and the half-split FRC of the same localizations (solid: independent halves; thin dashed: short blocks). "
         "(h) Same-molecule correlation versus time lag: pairs of localizations closer than 8 nm in x-y relative to pairs 8-24 nm apart "
         "(per area; ~1 = only the structure; MT: densest region, NPC: central region, simulation 0.84x). "
         "(i) Frames until the x-y FRC stays at or below 25 / 20 / 15 nm (open circle / open square / filled diamond; first N frames, independent "
         "halves) and from which it stays within 10 % of its all-frames value (bar).",
         '',
         'NUMBERS (si_4pi_frc_vs_frames_values.json)',
         "Frames until the x-y FRC stays <= 25 / 20 / 15 nm (first N frames, independent halves):",
         f"  MT densest region: 4Pi-SIMCODE {ft('mt_dense', 'net_ind_xy')}; 4Pi-SIMFLUX {ft('mt_dense', 'cl_ind_xy')}.",
         f"  MT box 0: 4Pi-SIMCODE {ft('mt_box0', 'net_ind_xy')}; 4Pi-SIMFLUX {ft('mt_box0', 'cl_ind_xy')}.",
         f"  MT 21 x 11 um field: 4Pi-SIMCODE {ft('mt_field', 'net_ind_xy')}; 4Pi-SIMFLUX {ft('mt_field', 'cl_ind_xy')}.",
         f"  NPC central 10 um: 4Pi-SIMCODE {ft('npc_center', 'net_ind_xy')}; 4Pi-SIMFLUX {ft('npc_center', 'cl_ind_xy')}.",
         "Frames until the x-z section FRC stays <= 35 / 30 / 25 nm:",
         f"  MT densest region: 4Pi-SIMCODE {ft('mt_dense', 'net_ind_xz', ('35', '30', '25'))}; 4Pi-SIMFLUX {ft('mt_dense', 'cl_ind_xz', ('35', '30', '25'))}.",
         f"  MT box 0: 4Pi-SIMCODE {ft('mt_box0', 'net_ind_xz', ('35', '30', '25'))}; 4Pi-SIMFLUX {ft('mt_box0', 'cl_ind_xz', ('35', '30', '25'))}.",
         f"  NPC: 4Pi-SIMCODE {ft('npc_center', 'net_ind_xz', ('35', '30', '25'))}; 4Pi-SIMFLUX {ft('npc_center', 'cl_ind_xz', ('35', '30', '25'))}.",
         f"Spread over the series (x-y, all-frames value equal): MT densest region 4Pi-SIMCODE {ft('mt_dense', 'net_spread_xy')}; NPC 4Pi-SIMCODE {ft('npc_center', 'net_spread_xy')}.",
         f"Short blocks (precision-dominated): MT densest region 4Pi-SIMCODE x-y at 1000 / 2000 / 4000 frames {at('mt_dense', 'net_blk_xy', 1000):.0f} / "
         f"{at('mt_dense', 'net_blk_xy', 2000):.0f} / {at('mt_dense', 'net_blk_xy', 4000):.0f} nm, independent halves "
         f"{at('mt_dense', 'net_ind_xy', 1000):.0f} / {at('mt_dense', 'net_ind_xy', 2000):.0f} / {at('mt_dense', 'net_ind_xy', 4000):.0f} nm.",
         f"Skeleton coverage (>= 3 localizations within +-30 nm) 90 / 95 / 99 %: MT densest region 4Pi-SIMCODE {cov('mt_dense', 'net_ind_ge3', 0.9)} / "
         f"{cov('mt_dense', 'net_ind_ge3', 0.95)} / {cov('mt_dense', 'net_ind_ge3', 0.99)} frames, 4Pi-SIMFLUX {cov('mt_dense', 'cl_ind_ge3', 0.9)} / "
         f"{cov('mt_dense', 'cl_ind_ge3', 0.95)} / {cov('mt_dense', 'cl_ind_ge3', 0.99)} (final {St['mt_dense']['cl_ind_ge3']['final']:.3f}).",
         f"NPC corners seen per pore, all frames: 4Pi-SIMCODE {St['npc']['net_ind_corners_ge1']['final']:.2f}, 4Pi-SIMFLUX {St['npc']['cl_ind_corners_ge1']['final']:.2f}; "
         f"90 / 95 / 99 % of that after 4Pi-SIMCODE {fmt(St['npc']['net_ind_corners_ge1']['frames_to_90pct_of_final'])} / {fmt(St['npc']['net_ind_corners_ge1']['frames_to_95pct_of_final'])} / "
         f"{fmt(St['npc']['net_ind_corners_ge1']['frames_to_99pct_of_final'])} frames, 4Pi-SIMFLUX {fmt(St['npc']['cl_ind_corners_ge1']['frames_to_90pct_of_final'])} / "
         f"{fmt(St['npc']['cl_ind_corners_ge1']['frames_to_95pct_of_final'])} / {fmt(St['npc']['cl_ind_corners_ge1']['frames_to_99pct_of_final'])}.",
         "Simulation (30 000 frames; GT = ground-truth FRC, running best; half = half-split, independent halves), x-y at 30 000 frames and frames to 25 nm:",
         *[f"  {Sm[k]['x_dense']}x: 4Pi-SIMCODE GT {Sm[k]['net_gt_xy_runmin']['final_nm']:.0f} nm ({fmt(Sm[k]['net_gt_xy_runmin']['frames_to']['25'])}), half "
           f"{Sm[k]['net_ind_xy']['final_nm']:.0f} nm ({fmt(Sm[k]['net_ind_xy']['frames_to']['25'])}), short blocks {Sm[k]['net_blk_xy']['final_nm']:.0f} nm; "
           f"4Pi-SIMFLUX GT {Sm[k]['cl_gt_xy_runmin']['final_nm']:.0f}, half {Sm[k]['cl_ind_xy']['final_nm']:.0f} nm." for k in sorted(Sm)],
         f"Same-molecule correlation (h), pairs < 8 nm relative to 8-24 nm: MT 4Pi-SIMCODE {lag('mt_dense', 'net', 0):.0f} at a lag of 1 window, "
         f"{lag('mt_dense', 'net', 4):.1f} at 30-100 windows, {lag('mt_dense', 'net', 6):.1f} at 300-1000, {lag('mt_dense', 'net', 9):.1f} at 10 000-30 000; "
         f"NPC {lag('npc_center', 'net', 0):.0f} / {lag('npc_center', 'net', 4):.1f} / {lag('npc_center', 'net', 6):.1f} / {lag('npc_center', 'net', 9):.1f}.",
         f"NPC two-ring valley / peak of all 4Pi-SIMCODE rows (pore_metrics, first N frames) at " + ', '.join(f"{g:,}: {d:.2f}" for g, d in zip(St['npc']['net_dip']['grid'], St['npc']['net_dip']['dip']) if g >= 32000)
         + "; 4Pi-SIMFLUX " + ', '.join(f"{g:,}: {d:.2f}" for g, d in zip(St['npc']['cl_dip']['grid'], St['npc']['cl_dip']['dip']) if g >= 32000) + ".",
         '',
         'READING',
         "1. Short blocks measure precision, not the image: with blocks shorter than the same-molecule correlation time (h: repeated detections "
         "of one binding event and rebinding stay correlated for ~100-300 windows on these data, much longer than in the simulation) the half-split "
         "FRC of the first 1000 frames is 13 nm on MT, better than with all 1.8 M frames. Only independent halves (>= 500-window blocks) give a "
         "usable timeline. 2. With independent halves the MT curves fall over the whole series (no plateau: within 10 % of the all-frames value "
         "only after ~1.1-1.4 M of 1.8 M frames), with a step where the curve passes the hollow-tube spectral dip; the NPC curves flatten after "
         "~0.2 M of 0.68 M frames. 3. Frames from across the series agree less than consecutive frames (dotted vs solid; MT 25 nm after 0.38 M "
         "instead of 0.13 M frames). Two causes, not separable here: adjacent 500-window blocks still share some correlated detections across their "
         "boundaries (h: at 300-1000 windows the MT ratio is 1.7 against 1.3 at long lags), and files far apart are less consistent with each other "
         "(per-file 4Pi registration and in-situ calibration, residual drift); there is no sign that the late frames themselves are worse (the "
         "spread curves end at the same value). The first-N curves are therefore somewhat optimistic for a shorter acquisition. 4. The half-split FRC is a much stricter and different quantity than the simulation's "
         "ground-truth FRC: in the simulation at 0.84x the densest real region (g) the ground-truth FRC reaches 25 nm after 11 000 frames and "
         "18 nm at 30 000 frames, while the half-split FRC of the same localizations is still 82 nm (both halves must sample the same sites; "
         "the ground truth is complete). The simulation's 'faithful image after ~11 000 frames' therefore cannot be checked with the real data's "
         "half-split FRC, and the real timelines are not a test of it. 5. Structure completeness saturates much earlier than the FRC: the MT "
         "skeleton is 99 % covered (>= 3 4Pi-SIMCODE localizations within +-30 nm) after ~0.2 M frames (4Pi-SIMFLUX: 90 % after 0.63 M), and "
         "4Pi-SIMCODE sees 95 % of the NPC corners after ~27 000 frames (4Pi-SIMFLUX ~96 000). 6. The NPC two-ring valley / peak of the "
         "4Pi-SIMCODE rows is best at ~0.13-0.26 M frames and worsens with the later files (z consistency between files), so the NPC 3D result "
         "would not suffer from stopping at about a third of the series.",
         '',
         'CODE: analysis/paper/si_4pi_frc.py (compute: cache_si_4pi/frc_vs_frames.json from the si_4pi caches), analysis/paper/si_4pi_frc_fig.py.']
    return '\n'.join(L) + '\n'
