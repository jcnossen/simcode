"""
Figures of the per-step pattern analysis (plain matplotlib PNGs in out/nanoruler/perstep/).
usage: python analysis/nanoruler/pst_fig.py <per-step config A (coarse)> <per-step config B (fine / chosen)> <shared A> <shared B>
"""
import sys, json, math
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L

# categorical slots 1-3 (blue, orange, aqua) = the three phase steps of an orientation; neutral ink for the shared model
STEP = ['#2a78d6', '#eb6834', '#1baf7a']; INK = '#0b0b0b'; SEC = '#52514e'; GRID = '#e4e3df'; SURF = '#fcfcfb'; MUTED = '#8a8983'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': GRID,
                     'axes.labelcolor': SEC, 'xtick.color': SEC, 'ytick.color': SEC, 'text.color': INK, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False, 'lines.linewidth': 1.6})
NM = [219.9 / (2 * math.pi), 219.5 / (2 * math.pi)]
T_MIN = 14.01e-3 / 60                                   # minutes per frame


def main():
    cA, cB, sA, sB = sys.argv[1:5]
    Z = np.load(L.OUT + 'curves.npz'); S = json.load(open(L.OUT + 'fit_summary.json')); fr = Z['frames']; t = fr * T_MIN
    rel = lambda Dm: Dm - np.repeat(np.stack([Dm[:3].mean(0), Dm[3:].mean(0)]), 3, 0)

    # ---- figure 1: drift curves, shared vs per step
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, sharey='row')
    for j, (cp, cs) in enumerate(((cA, sA), (cB, sB))):
        for a in (0, 1):
            x = ax[a, j]
            for i in range(3):
                x.plot(t, Z[f'{cp}__D'][3 * a + i] * NM[a], color=STEP[i], label=f'step {3 * a + i} (per-step fit)')
            x.plot(t, Z[f'{cs}__D'][3 * a] * NM[a], color=INK, lw=1.2, ls='--', label='shared drift (one per orientation)')
            x.set_title(f'orientation {a}: {cp} vs {cs}')
            if j == 0: x.set_ylabel(f'pattern phase drift (nm; 1 nm = {1 / NM[a]:.4f} rad)')
            if a == 1: x.set_xlabel('time (min)')
    ax[0, 0].legend(fontsize=8, ncol=2, loc='upper right')
    fig.suptitle('Pattern phase drift of the six steps (mean per curve removed)', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(L.OUT + 'fig_drift_curves.png', dpi=130); plt.close(fig)

    # ---- figure 2: step-relative drift with the two half fits
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.2), sharex=True, sharey='col')
    for j, cp in enumerate((cA, cB)):
        r = rel(Z[f'{cp}__D']); r = r - r.mean(1, keepdims=True)
        h = [rel(Z[f'{cp}__D_split0_half{k}']) for k in (0, 1)]; h = [q - q.mean(1, keepdims=True) for q in h]
        st = S['configs'][cp]['rel']
        for a in (0, 1):
            x = ax[a, j]
            for i in range(3):
                s = 3 * a + i
                x.fill_between(t, h[0][s] * NM[a], h[1][s] * NM[a], color=STEP[i], alpha=0.25, lw=0)
                x.plot(t, r[s] * NM[a], color=STEP[i],
                       label=f'step {s}: rms {st["rms_nm_full_fit"][s]:.1f} nm, fit noise {st["noise_rms_nm_full_fit"][s]:.1f} nm, half-to-half corr. {st["corr_between_halves"][s]:+.2f}')
            x.axhline(0, color=MUTED, lw=0.8)
            x.set_title(f'orientation {a}: {cp}'); x.legend(fontsize=7, loc='lower left', ncol=1)
            lim = 3.2 * max(np.percentile(np.abs(r), 99) * NM[0], 3); x.set_ylim(-1.6 * lim, lim)
            if j == 0: x.set_ylabel('phase relative to the orientation mean (nm)')
            if a == 1: x.set_xlabel('time (min)')
    fig.suptitle('Step-relative drift: what one shared drift per orientation cannot represent (band = fits on two disjoint halves of the sites)',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(L.OUT + 'fig_relative_drift.png', dpi=130); plt.close(fig)

    # ---- figure 3: phase steps over time
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True)
    for a in (0, 1):
        for j, (cp, ls) in enumerate(((cA, '-'), (cB, ':'))):
            st = Z[f'{cp}__steps']
            for i in range(2):
                v = st[2 * a + i]; v = (v + math.pi) % (2 * math.pi) - math.pi
                ax[a].plot(t, np.rad2deg(v), color=STEP[i], ls=ls, lw=1.6 if j == 0 else 1.0,
                           label=f'step {3 * a + i + 1} - step {3 * a + i} ({cp})')
        ax[a].axhline(-120, color=MUTED, lw=0.8); ax[a].text(t[-1], -120, ' -120', va='center', ha='left', color=SEC, fontsize=8)
        ax[a].set_title(f'orientation {a}: phase step'); ax[a].set_xlabel('time (min)'); ax[a].legend(fontsize=8, ncol=2)
    ax[0].set_ylabel('phase step (deg)')
    fig.tight_layout(); fig.savefig(L.OUT + 'fig_phase_steps.png', dpi=130); plt.close(fig)

    # ---- figure 4: held-out robust loss vs knot spacing
    fig, x = plt.subplots(figsize=(7.2, 4.2))
    H = S['heldout']
    for model, col, mk in (('shared', INK, 'o'), ('perstep', STEP[0], 's')):
        rows = sorted([(h['knot_frames'], h['held_rel_change_vs_ref'], h['held_rel_change_se']) for h in H.values() if h['model'] == model])
        kf, v, se = map(np.array, zip(*rows))
        x.errorbar(kf, 100 * v, 100 * se, color=col, marker=mk, ms=6, capsize=2, label={'shared': 'shared drift per orientation', 'perstep': 'six independent drifts'}[model])
    hy = [(n_, h) for n_, h in H.items() if h['model'] == 'hybrid']
    for i, (n_, h) in enumerate(hy):
        x.errorbar(h['knot_frames'] * (1.06 + 0.0 * i), 100 * h['held_rel_change_vs_ref'], 100 * h['held_rel_change_se'], color=STEP[1], marker='D', ms=6, capsize=2,
                   label='shared (fine) + per-step (coarse knots)' if i == 0 else None)
        x.annotate(f'+{h["knot_frames2"]}', (h['knot_frames'] * 1.06, 100 * h['held_rel_change_vs_ref']), textcoords='offset points', xytext=(6, -3), fontsize=7, color=SEC)
    x.set_xscale('log'); x.set_xlabel('knot spacing (frames)'); x.set_ylabel(f'held-out robust loss vs {S["shared_ref"]} (%)')
    x.set_title('Held-out emitters: pattern from one half of the sites, loss on the other'); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(L.OUT + 'fig_heldout.png', dpi=130); plt.close(fig)

    # ---- figure 5: jitter estimates with the nulls + temporal correlation
    J = json.load(open(L.OUT + 'jitter.json'))
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for r_, kind in enumerate(('phase', 'amp')):
        x = ax[r_, 0]
        for j, (nm, col) in enumerate((('shared', INK), ('perstep', STEP[0]))):
            e = J['jitter'][nm][kind]; keys = [str(s) for s in range(6)] + ['o0', 'o1', 'all']
            xs = np.arange(len(keys)) + (j - 0.5) * 0.3
            v = np.array([e[k]['var'] for k in keys]); se = np.array([e[k]['var_se'] for k in keys])
            x.errorbar(xs, v, 1.96 * se, color=col, marker='o', ms=5, ls='none', capsize=2, label=f'{J[nm + "_cfg"]}: pairwise estimate (95 % CI)')
            nl = np.array([e[k]['null_local']['mean'] for k in keys]); ng = np.array([e[k]['null_global']['mean'] for k in keys])
            x.plot(xs, nl, ls='none', marker='_', ms=12, mew=2, color=STEP[1], label='null: cycle labels shuffled within +-25 cycles' if j == 0 else None)
            x.plot(xs, ng, ls='none', marker='x', ms=6, color=STEP[2], label='null: shuffled over the whole movie' if j == 0 else None)
        x.axhline(0, color=MUTED, lw=0.8); x.set_xticks(np.arange(len(keys))); x.set_xticklabels([f'step {k}' if len(k) == 1 else {'o0': 'orient. 0', 'o1': 'orient. 1', 'all': 'all'}[k] for k in keys])
        x.set_ylabel('variance of the frame-common term ' + ('(rad$^2$)' if kind == 'phase' else '(relative$^2$)'))
        x.set_title(('Phase' if kind == 'phase' else 'Amplitude') + ' error common to all emitters of a frame'); x.legend(fontsize=7)
        x = ax[r_, 1]
        for j, (nm, col) in enumerate((('shared', INK), ('perstep', STEP[0]))):
            e = J['jitter'][nm][kind]['all']
            lg = [0] + [int(k) for k in e['lag_cov']]; v = [e['var']] + [q[0] for q in e['lag_cov'].values()]; se = [e['var_se']] + [q[1] for q in e['lag_cov'].values()]
            x.errorbar(np.arange(len(lg)) + (j - 0.5) * 0.15, v, 1.96 * np.array(se), color=col, marker='o', ms=5, capsize=2, label=J[nm + '_cfg'])
        x.axhline(0, color=MUTED, lw=0.8); x.set_xticks(np.arange(len(lg))); x.set_xticklabels(lg)
        x.set_xlabel('lag (cycles of 6 frames = 84 ms; same step)'); x.set_ylabel('covariance, all steps pooled')
        x.set_title('Temporal correlation (different emitters at the two frames)'); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(L.OUT + 'fig_jitter.png', dpi=130); plt.close(fig)

    # ---- figure 6: excess relative variance vs predicted photons
    fig, x = plt.subplots(figsize=(8.6, 5.2))
    for nm, col in (('shared', INK), ('perstep', STEP[0])):
        N = J['noise'][nm]; b = np.array(N['binned']); p = N['ml_trim4']
        x.errorbar(b[:, 0], b[:, 1], 1.96 * b[:, 2], color=col, marker='o', ms=5, ls='none', capsize=2,
                   label=f'{J[nm + "_cfg"]}: c = {p["c"]:.3f} +- {N["ml_trim4_boot"]["c_se"]:.3f}')
        mu = np.geomspace(b[0, 0], b[-1, 0], 100)
        x.plot(mu, p['a0'] / mu ** 2 + p['a1'] / mu + p['c2'], color=col, lw=1.2)
        x.axhline(p['c2'], color=col, lw=0.8, ls=':')
    try:
        CF = json.load(open(L.OUT + 'cfree.json'))['shared']['phase_free']; b = np.array(CF['binned'])
        x.errorbar(b[:, 0], b[:, 1], 1.96 * b[:, 2], color=STEP[1], marker='s', ms=5, ls='none', capsize=2,
                   label=f'shared, sample phase free: c = {CF["c"]:.3f} +- {CF["c_se"]:.3f}')
        mu = np.geomspace(b[0, 0], b[-1, 0], 100); x.plot(mu, CF['a0'] / mu ** 2 + CF['a1'] / mu + CF['c2'], color=STEP[1], lw=1.2)
    except FileNotFoundError:
        pass
    try:
        PR = json.load(open(L.OUT + 'pairs.json')); b = np.array(PR['binned_mu_var_se_n']); q = PR['a1eq1']
        x.errorbar(b[:, 0], b[:, 1] / b[:, 0] ** 2, 1.96 * b[:, 2] / b[:, 0] ** 2, color=STEP[2], marker='^', ms=6, ls='none', capsize=2,
                   label=f'model-free: same step, adjacent cycles (c = {q["c"]:.3f} +- {q["c_se"]:.3f} with a1 = 1)')
        mu = np.geomspace(b[0, 0], b[-1, 0], 100); x.plot(mu, q['a0'] / mu ** 2 + 1 / mu + q['c2'], color=STEP[2], lw=1.2)
    except FileNotFoundError:
        pass
    x.set_xscale('log'); x.set_yscale('log'); x.set_xlabel('predicted photons per frame (A x)'); x.set_ylabel('relative variance of the intensity, amplitude-fit noise removed')
    x.set_title('Fully-on frames: relative intensity variance; lines = a0 + a1 mu + c$^2$ mu$^2$ fits'); x.legend(fontsize=7, loc='lower left')
    fig.tight_layout(); fig.savefig(L.OUT + 'fig_excess_variance.png', dpi=130); plt.close(fig)

    # ---- figure 7: step-relative drift, first 5000 frames: per-step fit vs the per-pattern offsets of the 2023 cache
    try:
        C = np.load(L.OUT + 'compare_cache_drift.npz'); tt = C['frames'] * T_MIN
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
        for a in (0, 1):
            for i in range(3):
                s = 3 * a + i
                ax[a].plot(tt, rel(C['D_perstep'])[s] * NM[a], color=STEP[i], label=f'step {s}: this fit')
                ax[a].plot(tt, rel(C['D_cache'])[s] * NM[a], color=STEP[i], ls=':', lw=1.2, label=f'step {s}: earlier per-pattern offsets')
            ax[a].set_title(f'orientation {a}: step-relative drift, first 5000 frames'); ax[a].set_xlabel('time (min)'); ax[a].legend(fontsize=7, ncol=3)
        ax[0].set_ylabel('phase relative to the orientation mean (nm)')
        fig.tight_layout(); fig.savefig(L.OUT + 'fig_compare_cache.png', dpi=130); plt.close(fig)
    except FileNotFoundError:
        pass

    # ---- figure 8: cross-site lag covariance of the frame-common phase error (shared_3000 pattern) + simulation
    try:
        LCV = json.load(open(L.OUT + 'lagcov.json')); SIM = json.load(open(L.OUT + 'simcheck.json'))
        e = LCV['shared_3000']['phase']; keys = list(e.keys()); xs = np.arange(len(keys))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for j, (k, lab) in enumerate((('same', 'same step'), ('cross', 'different steps, same orientation (lags 0-2 leak-prone)'), ('xori', 'different orientations'))):
            ax[0].errorbar(xs + (j - 1) * 0.12, [e[q][k][0] for q in keys], [1.96 * e[q][k][1] for q in keys], color=STEP[j], marker='o', ms=5, capsize=2, label=lab)
        sm = SIM['common_ar1']
        ax[0].plot(xs, [sm[q]['same'][0] for q in keys], color=INK, ls='--', lw=1.2, label='simulation: common AR(1), var 0.003 rad$^2$, 5 cycles (same step)')
        ax[0].axhline(0, color=MUTED, lw=0.8); ax[0].set_xticks(xs); ax[0].set_xticklabels(keys, rotation=30)
        ax[0].set_xlabel('lag (cycles of 84 ms)'); ax[0].set_ylabel('covariance of the phase error (rad$^2$)')
        ax[0].set_title('Phase error common to different emitters (95 % CI)'); ax[0].legend(fontsize=7)
        for a in (0, 1):
            ax[1].errorbar(xs + (a - 0.5) * 0.12, [e[q][f'var_rel{a}'][0] for q in keys], [1.96 * e[q][f'var_rel{a}'][1] for q in keys], color=STEP[a], marker='o', ms=5,
                           capsize=2, label=f'orientation {a}')
        sr = SIM['common_ar1_plus_rel']
        ax[1].plot(xs, [sr[q]['var_rel'][0] for q in keys], color=INK, ls='--', lw=1.2,
                   label=f'simulation with a slow step-relative drift of {math.sqrt(sr["truth"]["rel_var_actual"]):.3f} rad rms')
        ax[1].axhline(0, color=MUTED, lw=0.8); ax[1].set_xticks(xs); ax[1].set_xticklabels(keys, rotation=30)
        ax[1].set_xlabel('lag (cycles of 84 ms)'); ax[1].set_ylabel('(2/3) (same - cross) (rad$^2$), uncalibrated')
        ax[1].set_title('Step-relative part (what a shared drift cannot represent)'); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(L.OUT + 'fig_lagcov.png', dpi=130); plt.close(fig)
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    main()
