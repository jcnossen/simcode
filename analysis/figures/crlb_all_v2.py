"""Combined CRLB figure v2 (user 2026-11-09) -> <out>/plots/crlb_all_v2.png + <out>/sc7/crlb_all_v2.txt.
Rows: x/y position RMSD (gate npz, seeds 301-303, detector-matched within 1 px on x_det at score >= 0.5, resolvable
targets), per-frame I_t RMSD and per-frame bg RMSD (icrlb npz, seeds 301-302, det = matched detector candidates, forced =
at the GT positions). Columns: iso / rs05q / rs05qm. Networks: sc5nb, sc6n (+ PM pattern solve) [+ sc7bg if present].
critic 34 (2026-11-10): + joint SIMFLUX MLE (lmref.SFRef, sf_gate.py -> sc7/gate_lm/*_sf11.npz; all columns) and repaired SIMCODE (ni6_3, paper
filters; rs columns, on the targets matched by both SIMCODE and sc5nb, with sc5nb + PM on the same set); bg row reference relabelled.
Reference: basic single-emitter LM (analysis/figures/lmref.py; lm_gate.py / lm_icrlb.py, run_lm.sh; 11 x 11 ROI, z fixed 50 nm, x, y, 6 I_t, 6 bg_t; sCMOS Poisson):
pattern-free LM x/y and LM I_t + the same explicit pattern solve (PM) -> position; LM I_t / bg_t. The LM runs on the SAME
detector candidates as the networks (position row: the same matched set), or at the GT positions (forced).
Inputs (<out> = simcode2.config out root): gate/<scene>__s30?_<tag>.npz (simcode2.hd.gate, simcode2.eff.simcode_gate),
sc7/gate_lm/*_lm11*.npz / *_sf11.npz (lm_gate.py, sf_gate.py), sc7/icrlb_<net>_<scene>.npz (icrlb.py, lm_icrlb.py).
usage: python crlb_all_v2.py [nets sc5nb,sc6n[,sc7bg]] [LM variant 11z] [--out png] [--txt txt]"""
import sys, os, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
_ap = argparse.ArgumentParser(description='combined CRLB figure v2')
_ap.add_argument('nets', nargs='?', default='sc5nb,sc6n')
_ap.add_argument('lmv', nargs='?', default='11z')
_ap.add_argument('--out', default='', help='png (default <out>/plots/crlb_all_v2.png)')
_ap.add_argument('--txt', default='', help='text table (default <out>/sc7/crlb_all_v2.txt)')
A = _ap.parse_args()
sys.argv = sys.argv[:1] + [A.nets]                   # icrlb_report reads sys.argv at import (as in the development run)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icrlb_report import imet, bmet
from simcode2 import config
from simcode2.hd import gate_report as GR
from plot_crlb_sc3 import curves
import plot_crlb_sc3 as P3

OUTR = config.root('out')
D7 = os.path.join(OUTR, 'sc7')
NETS = A.nets.split(',')
LMV = A.lmv      # plotted LM variant: '11z' = 11 x 11, z free; '11' = z fixed 50 nm
LMX = [v for v in ('11', '11z', '15') if v != LMV]   # other variants: text only
SCS = ['iso', 'rs05q', 'rs05qm']
GD = os.path.join(OUTR, 'gate')
GDIR = {'sc5nb': (GD, '_hd_sc5nbb15pm'), 'sc6n': (GD, '_hd_sc6nb15pm'),
        'sc7bg': (GD, '_hd_sc7bgb15pm'), 'lm': (f'{D7}/gate_lm', '_lm' + LMV), 'lm11': (f'{D7}/gate_lm', '_lm11'), 'lm11z': (f'{D7}/gate_lm', '_lm11z'),
        'sf': (f'{D7}/gate_lm', '_sf11'), 'scode': (GD, '_scode_ni6_3')}
COL = {'sc5nb': '#eb6834', 'sc6n': '#1baf7a', 'sc7bg': '#2a78d6', 'lm': '#7a4fb5', 'sf': '#111111', 'scode': '#8c6d1f'}
SFLAB = 'joint SIMFLUX MLE (pattern known; x, y, z, I, bg/angle)'
SCLAB = 'SIMCODE (paper net ni6_3, paper filters)'
TITLE = {'iso': 'iso: isolated emitters, exact model',
         'rs05q': 'rs05q: 0.5 nM-like crowded scene at paper\nbrightness, exact model',
         'rs05qm': 'rs05qm: as rs05q + realistic mismatch (estimated pattern,\nPSF / halo jitter, per-frame / per-step intensity\nvariation, fluctuating bg, camera error)'}
LMLAB = 'LM ' + ('11x11, z free' if LMV == '11z' else ('11x11, z 50 nm' if LMV == '11' else LMV))
PBINS = np.array([150, 300, 500, 800, 1300, 2000, 3200, 5000, 8000, 13000, 20000])
P3.BINS = PBINS
cen = np.sqrt(PBINS[:-1] * PBINS[1:])


def load_icrlb(nm, sc):
    f = f'{D7}/icrlb_{nm}_{sc}.npz'
    if not os.path.exists(f): return None
    return dict(np.load(f))


def matched(R, f='sc', m='sc'):
    """(ti, oi) as plot_crlb_sc3.curves: score >= 0.5, 1 px one-to-one on x_det, targets, resolvable"""
    sel = np.isfinite(R[f'{f}_x_{m}']) & (R[f'{f}_score'] >= 0.5)
    idx = np.nonzero(sel)[0]
    og = GR.match(R['gt_fov'], R['gt_x'], R['gt_y'], R[f'{f}_b'][idx], R[f'{f}_x_det'][idx], R[f'{f}_y_det'][idx], 1.0)
    ti = og[og >= 0]; oi = idx[og >= 0]
    res = np.hypot(R['gt_sf_m'][:, 0], R['gt_sf_m'][:, 1]) / np.sqrt(2) * 108.33 < 25
    k = (R['gt_target'][ti] > 0) & res[ti]
    return ti[k], oi[k]


def curve_on(R, ti, oi, f='sc', m='sc'):
    """rows [rmsd nm, CRLB_SF_m, CRLB_SF_i, CRLB_SM_m, n, ratio to CRLB_SF_m] per PBINS bin on the given (ti, oi)"""
    e = np.stack([R[f'{f}_x_{m}'][oi] - R['gt_x'][ti], R[f'{f}_y_{m}'][oi] - R['gt_y'][ti]], 1); ph = R['gt_phot'][ti]
    rows = []
    for lo, hi in zip(PBINS[:-1], PBINS[1:]):
        s = (ph >= lo) & (ph < hi)
        if s.sum() < 30: rows.append([np.nan] * 6); continue
        rms = lambda c: np.sqrt(np.nanmean(c[ti[s]] ** 2)) * 108.33
        rm = np.sqrt(np.mean(e[s] ** 2)) * 108.33
        rows.append([rm, rms(R['gt_sf_m']), rms(R['gt_sf_i']), rms(R['gt_sm_m']), s.sum(), rm / rms(R['gt_sf_m'])])
    return np.array(rows)


def lm_as_icrlb(sc, P=LMV):
    """LM I / bg arrays merged into the sc5nb icrlb record (same targets, checked by lm_icrlb.py)"""
    ref = load_icrlb('sc5nb', sc); q = load_icrlb(f'lm{P}', sc)
    if ref is None or q is None: return None
    d = dict(ref)
    d.update(I_det=q['I_det'], I_f=q['I_f'], bg_det=q['bg_det'], bg_f=q['bg_f'])
    return d


def resolvable(d):
    res = np.hypot(d['sf_m'][:, 0], d['sf_m'][:, 1]) / np.sqrt(2) * 108.33 < 25
    return {k: v[res] for k, v in d.items()}


def irow(d, kind):
    ys, yc, yf, ns = [], [], [], []
    for lo, hi in zip(PBINS[:-1], PBINS[1:]):
        m = (d['ph'] >= lo) & (d['ph'] < hi)
        if kind == 'I':
            dd_ = dict(d); cm = np.nanmedian(np.where(d['I_gt'][m] > 0, d['fr_m'][m], np.nan)) if m.any() else np.nan
            dd_['fr_m'] = np.where(d['fr_m'] <= 3 * cm, d['fr_m'], np.nan)
            r = imet(dd_, m & np.isfinite(d['I_det']).all(1), 'I_det'); rf = imet(dd_, m, 'I_f')
        else:
            r = bmet(d, m & np.isfinite(d['bg_det']).all(1), 'bg_det'); rf = bmet(d, m, 'bg_f')
        ys.append(r['rmsd'] if r else np.nan); yc.append(rf['crlb'] if rf else np.nan); yf.append(rf['rmsd'] if rf else np.nan)
        ns.append(r['n'] if r else 0)
    return np.array(ys), np.array(yc), np.array(yf), ns


def main():
    L = []; pr = lambda s: (L.append(s), print(s, flush=True))
    pr('crlb_all_v2: RMSD per true-window-photon bin (label units). Position: gate seeds 301-303, detector-matched (1 px, x_det, score >= 0.5), resolvable;')
    pr(f'  I_t / bg: icrlb seeds 301-302, det = matched detector candidates, forced = at GT. LM = basic single-emitter LM reference (plotted: {LMLAB}), evaluation only.')
    fig, axs = plt.subplots(4, 3, figsize=(15, 18), gridspec_kw=dict(height_ratios=[1.25, 0.8, 1, 1]))
    for j, sc in enumerate(SCS):
        # ---------------- position
        ax = axs[0, j]; axr = axs[1, j]; axr.axhline(1.0, color='k', lw=2); pr(f'===== {sc}: position x/y RMSD (nm) [ratio to CRLB_SF multi on the same set] (recall 1 px)')
        C0 = None; rows = {}
        for nm in NETS + ['lm'] + ['lm' + v for v in LMX if v != '15']:
            dd, tag = GDIR[nm]
            try:
                R, nf = GR.load(dd, sc, tag)
                if nf == 0: raise FileNotFoundError
            except Exception:
                continue
            if nm.startswith('lm') and nm != 'lm':
                rows[f'(LM {nm[2:]} pattern-free)'] = curves(R, 'sc', 'lm'); rows[f'(LM {nm[2:]} + PM)'] = curves(R, 'sc', 'sc')
            elif nm == 'lm':
                rows['LM pattern-free'] = curves(R, 'sc', 'lm'); rows['LM + pattern solve (PM)'] = curves(R, 'sc', 'sc')
            else:
                rows[nm + ' + PM'] = curves(R, 'sc', 'sc')
            if C0 is None:
                C0 = rows[list(rows)[0]]
                if sc == 'iso':          # isolated-only scene: the multi-emitter CRLB is not meaningful here (user 2026-11-10)
                    ax.loglog(cen, C0[:, 2], '-', color='k', lw=2, label='CRLB SIMFLUX (isolated)')
                    ax.loglog(cen, C0[:, 3], '--', color='0.5', lw=1, label='CRLB SMLM (isolated)')
                else:
                    ax.loglog(cen, C0[:, 1], '-', color='k', lw=2, label='CRLB SIMFLUX multi-emitter')
                    ax.loglog(cen, C0[:, 2], ':', color='k', lw=1.5, label='CRLB SIMFLUX isolated')
                    ax.loglog(cen, C0[:, 3], '--', color='0.5', lw=1, label='CRLB SMLM multi-emitter')
        for lab, C in rows.items():
            if lab.startswith('('):
                pr(f'  {lab:26s} ' + ' '.join(f'{lo:.0f}-{hi:.0f}: {C[k, 0]:.2f} [{C[k, 5]:.2f}]' for k, (lo, hi) in enumerate(zip(PBINS[:-1], PBINS[1:])) if np.isfinite(C[k, 0])) + '  (not plotted)')
                continue
            if lab.startswith('LM'):
                st = dict(color=COL['lm'], lw=1.6, ms=4, marker='s', ls='--' if 'free' in lab else '-')
            else:
                st = dict(color=COL[lab.split()[0]], lw=2, ms=4, marker='o', ls='-')
            ax.loglog(cen, C[:, 0], label=lab, **st); axr.semilogx(cen, C[:, 5], **st)
            pr(f'  {lab:26s} ' + ' '.join(f'{lo:.0f}-{hi:.0f}: {C[k, 0]:.2f} [{C[k, 5]:.2f}] ({C[k, 4]:.2f})' for k, (lo, hi) in enumerate(zip(PBINS[:-1], PBINS[1:])) if np.isfinite(C[k, 0])))
        # joint SIMFLUX MLE reference (critic 34 #4; evaluation only): same detector candidates / matched set as the LM
        try:
            Rsf, _ = GR.load(GDIR['sf'][0], sc, GDIR['sf'][1])
            C = curves(Rsf, 'sc', 'sc')
            st_ = dict(color=COL['sf'], lw=1.6, ms=4.5, marker='D', ls='-.'); ax.loglog(cen, C[:, 0], label=SFLAB, **st_); axr.semilogx(cen, C[:, 5], **st_)
            pr(f'  {"joint SIMFLUX MLE":26s} ' + ' '.join(f'{lo:.0f}-{hi:.0f}: {C[k, 0]:.2f} [{C[k, 5]:.2f}] ({C[k, 4]:.2f})' for k, (lo, hi) in enumerate(zip(PBINS[:-1], PBINS[1:])) if np.isfinite(C[k, 0])))
        except FileNotFoundError:
            pass
        # repaired SIMCODE (critic 34 #4): rs columns, on the targets matched by BOTH SIMCODE and the network detector (sc5nb set)
        if sc != 'iso':
            try:
                Rs, _ = GR.load(GDIR['scode'][0], sc, GDIR['scode'][1]); Rn, _ = GR.load(GDIR['sc5nb'][0], sc, GDIR['sc5nb'][1])
                assert np.allclose(Rs['gt_x'], Rn['gt_x']) and np.array_equal(Rs['gt_fov'], Rn['gt_fov'])
                tS, oS = matched(Rs); tN, oN = matched(Rn)
                com, iS, iN = np.intersect1d(tS, tN, return_indices=True)
                CS = curve_on(Rs, com, oS[iS]); CN = curve_on(Rn, com, oN[iN])
                stS = dict(color=COL['scode'], lw=2, ms=5, marker='v', ls='-'); stN = dict(color=COL['sc5nb'], lw=1.2, ms=3, marker='o', ls=':')
                ax.loglog(cen, CS[:, 0], label=SCLAB + ', common set', **stS); ax.loglog(cen, CN[:, 0], label='sc5nb + PM on the same common set', **stN)
                axr.semilogx(cen, CS[:, 5], **stS); axr.semilogx(cen, CN[:, 5], **stN)
                pr(f'  SIMCODE common set: {len(com)} targets (SIMCODE matched {len(tS)}, sc5nb matched {len(tN)})')
                for lab, C in (('SIMCODE (common set)', CS), ('sc5nb + PM (common set)', CN)):
                    pr(f'  {lab:26s} ' + ' '.join(f'{lo:.0f}-{hi:.0f}: {C[k, 0]:.2f} [{C[k, 5]:.2f}] (n {C[k, 4]:.0f})' for k, (lo, hi) in enumerate(zip(PBINS[:-1], PBINS[1:])) if np.isfinite(C[k, 0])))
            except FileNotFoundError:
                pass
        ax.set_title(TITLE[sc], fontsize=9.5); ax.set_ylabel('x/y RMSD (nm)'); ax.grid(True, which='both', alpha=0.3)
        if sc == 'iso':
            ax.legend(fontsize=6.5, loc='lower left')
        else:      # rs: room above the contaminated single-emitter references
            ax.set_ylim(top=4000); ax.legend(fontsize=6.3, loc='upper right', ncol=2)
        axr.set_ylim(0.8, 2.6); axr.grid(True, which='both', alpha=0.3)
        axr.set_ylabel('RMSD / CRLB SIMFLUX ' + ('(isolated)' if sc == 'iso' else '(multi-emitter)'))
        if sc != 'iso': axr.text(0.02, 0.95, 'single-emitter LM / joint MLE: 7-23x, off scale', transform=axr.transAxes, fontsize=7, va='top')
        # ---------------- I_t and bg
        D = {nm: load_icrlb(nm, sc) for nm in NETS}; D['lm'] = lm_as_icrlb(sc)
        DX = {v: lm_as_icrlb(sc, v) for v in LMX}
        for row, (kind, ylab) in enumerate((('I', 'per-frame I_t RMSD (photons)'), ('bg', 'per-frame bg RMSD (ph/px)')), start=2):
            ax = axs[row, j]; first = True
            pr(f'===== {sc}: per-frame {kind} RMSD, det | forced' + (' (ph/px)' if kind == 'bg' else ' (photons; frames with CRLB > 3x the bin median dropped)'))
            for nm in NETS + ['lm']:
                d = D.get(nm)
                if d is None: continue
                d = resolvable(d)
                ys, yc, yf, ns = irow(d, kind)
                if first:
                    ax.loglog(cen, yc, '-', color='k', lw=2, label=('CRLB per-frame I' + ('' if sc == 'iso' else ' (multi-emitter)')) if kind == 'I' else 'flat-bg reference, 21x21 ROI')
                    pr(('  CRLB                       ' if kind == 'I' else '  flat-bg reference 21x21   ') + ' '.join(f'{lo:.0f}-{hi:.0f}: {c:.3g}' for (lo, hi), c in zip(zip(PBINS[:-1], PBINS[1:]), yc) if np.isfinite(c)))
                    first = False
                lab = LMLAB if nm == 'lm' else nm
                mk = 's' if nm == 'lm' else 'o'
                ax.loglog(cen, ys, marker=mk, ls='-', color=COL[nm], ms=4, lw=2 if nm != 'lm' else 1.6, label=f'{lab} detected')
                ax.loglog(cen, yf, ls='--', color=COL[nm], lw=1, label=f'{lab} forced at GT')
                pr(f'  {lab:26s} ' + ' '.join(f'{lo:.0f}-{hi:.0f}: {a:.3g} | {b:.3g}' for (lo, hi), a, b in zip(zip(PBINS[:-1], PBINS[1:]), ys, yf) if np.isfinite(a) or np.isfinite(b)))
            for v, DV in DX.items():
                if DV is None: continue
                ys, yc, yf, ns = irow(resolvable(DV), kind)
                pr(f'  {"LM " + v + " (not plotted)":26s} ' + ' '.join(f'{lo:.0f}-{hi:.0f}: {a:.3g} | {b:.3g}' for (lo, hi), a, b in zip(zip(PBINS[:-1], PBINS[1:]), ys, yf) if np.isfinite(a) or np.isfinite(b)))
            ax.set_ylabel(ylab); ax.grid(True, which='both', alpha=0.3)
            if row == 3: ax.set_xlabel('true photons per window (label units)')
            ax.legend(fontsize=6, loc='upper left', ncol=2)
    for ax in axs.ravel():
        ax.xaxis.set_major_locator(LogLocator(base=10)); ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle(f'{" / ".join(NETS)} (network I_t -> explicit pattern solve, PM) vs evaluation-only references: single-emitter LM ({LMLAB}; '
                 'two-stage: LM I_t -> PM), joint SIMFLUX MLE (one-stage, pattern known)\nand repaired SIMCODE (rs columns, on the targets it shares '
                 'with the networks). Resolvable targets; position: same detector-matched set for networks / LM / MLE; I_t row drops frames with CRLB > 3x '
                 'the bin median', fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.965)); out = A.out or os.path.join(config.out_dir('plots'), 'crlb_all_v2.png')
    plt.savefig(out, dpi=90); print('saved', out)
    open(A.txt or f'{D7}/crlb_all_v2.txt', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
