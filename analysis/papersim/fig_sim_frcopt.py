"""Fig 2 variant at each method's FRC-optimal selection (frcopt.py sweep), webSMLM hollow-tube sets.

Layout as the paper Fig 2 (fig_sim_fig2.py), with:
  A GT | B GT zoom | C SMLM | D GT-FRC vs fraction kept / E localization classes | F SIMFLUX | G DECODE | H SIMCODE /
  I GT-FRC curves (optimal solid, current dashed) | J half-split FRC vs fraction kept
Zoom renders (C, F, G, H): the same box / renderer as Fig 2, rows = the method's selection at the threshold that minimizes its
GT-FRC (--crit, default gt_first = the Fig 2 GT-FRC, first 1/7 crossing from shell 3; gt_interp: interpolated between shells;
ties -> more rows); tile titles: GT-FRC current -> optimal, rows kept (% of the method's Fig 2 base set) and the threshold.
SIMCODE threshold family: --v2fam (sig_p = production p >= 0.63 + calibrated sigma threshold).
E: per method two stacked bars (current | optimal): precise (2D error to the nearest ON emitter <= --r_prec nm), imprecise
(error > r_prec, an emitter ON within --r_fp nm), false positive (no emitter ON in the window within r_fp nm).
usage: python analysis/papersim/fig_sim_frcopt.py [--set Dexp1_v2psf_wsm] [--crit gt_interp] [--v2fam sig_p]
output: out/paper_figs_v2/fig_sim_<set without the density>_frcopt.{png,pdf}, _caption.txt"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
from fig_density import render, crop_of, COL
from fig_sim_fig2 import letter, scalebar, FS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

METH = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2', 'SIMCODE')]
COLM = dict(smlm=COL['smlm'], sf=COL['sf'], decode=COL['decode'], v2=COL['v2sig'])
FRCD = os.path.join(C.OUT, 'frcopt')


def classes(B, m, r_prec, r_fp):
    """(precise, imprecise, false positive) counts of the rows m (2D error to the nearest ON emitter of the row's window)"""
    e = np.hypot(B['ex'][m], B['ey'][m]) * C.PX_NM; e = np.where(np.isfinite(e), e, np.inf)
    return int((e <= r_prec).sum()), int(((e > r_prec) & (e <= r_fp)).sum()), int((e > r_fp).sum())


def load(name, v2fam):
    R = json.load(open(os.path.join(FRCD, f'{name}.json')))
    Z = np.load(os.path.join(FRCD, f'{name}_rows.npz'))
    B = {}
    for k, _ in METH:
        ck = 'v2_' + v2fam if k == 'v2' else k
        B[k] = {c.split('__', 1)[1]: Z[c] for c in Z.files if c.split('__', 1)[0] == ck}
    return R, B


def selection(R, B, k, crit, v2fam):
    """(current mask, optimal mask, current row, optimal row, family rows) of method k"""
    M = R['methods'][k]; fam = v2fam if k == 'v2' else 'sig'; rows = M['fam'][fam]
    j = M['opt'][fam][crit]['idx']; ro = rows[j]
    key = B[k]['key']
    cur_thr = 12.0 / C.PX_NM if k == 'v2' else np.inf
    cur = key <= cur_thr; opt = key <= ro['thr']
    rc = M['variants'][fam]['at12'] if k == 'v2' else M['current']
    return cur, opt, rc, ro, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='Dexp1_v2psf_wsm'); ap.add_argument('--crit', default='gt_first')
    ap.add_argument('--hs', default='hs_first', help='half-split criterion of panel J')
    ap.add_argument('--v2fam', default='sig_p'); ap.add_argument('--r_prec', type=float, default=20.0); ap.add_argument('--r_fp', type=float, default=50.0)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(C.OUT), 'paper_figs_v2')); a = ap.parse_args()
    md = os.path.join(C.OUT, a.set); R, B = load(a.set, a.v2fam)
    gt = np.load(os.path.join(md, 'gt.npz')); W = R['W']; (cx, cy), w = crop_of(md)
    lab = dict(METH)
    fig = plt.figure(figsize=(13.0, 10.9))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2, 1.0], hspace=0.30, left=0.055, right=0.985, top=0.93, bottom=0.065)
    gs = outer[0].subgridspec(2, 4, hspace=0.36, wspace=0.14)
    A = fig.add_subplot(gs[0, 0]); A.set_facecolor('k')
    p = gt['pts']; A.plot(p[:, 0], p[:, 1], ',', color='#ffb070', alpha=0.9, ms=0.5, rasterized=True)
    A.add_patch(Rectangle((cx, cy), w, w, fill=False, ec='w', lw=2.5))
    A.set_xlim(-0.5, W - 0.5); A.set_ylim(W - 0.5, -0.5); A.set_xticks([]); A.set_yticks([])
    L = 2000 / C.PX_NM; A.plot([0.05 * W, 0.05 * W + L], [0.93 * W] * 2, 'w-', lw=4, solid_capstyle='butt')
    A.text(0.05 * W, 0.90 * W, '2000 nm', color='w', fontsize=10, va='bottom')
    A.set_title('Ground truth\n(all labelling sites)', fontsize=FS); letter(A, 'A')
    A = fig.add_subplot(gs[0, 1]); A.set_xticks([]); A.set_yticks([])
    A.imshow(render(p[:, 0], p[:, 1], (cx, cy), w), cmap='inferno'); A.set_title('Ground truth (zoom)', fontsize=FS); letter(A, 'B'); scalebar(A, w, 500)
    slots = {'smlm': (0, 2, 'C'), 'sf': (1, 1, 'F'), 'decode': (1, 2, 'G'), 'v2': (1, 3, 'H')}
    S = {}
    for k, lk in METH:
        cur, opt, rc, ro, rows = selection(R, B, k, a.crit, a.v2fam); S[k] = (cur, opt, rc, ro, rows)
        r, c, Lt = slots[k]; A = fig.add_subplot(gs[r, c]); A.set_xticks([]); A.set_yticks([])
        A.imshow(render(B[k]['x'][opt], B[k]['y'][opt], (cx, cy), w), cmap='inferno')
        nb = R['methods'][k]['n_base']
        A.set_title(f"{lk}  GT-FRC {rc[a.crit]:.1f} → {ro[a.crit]:.1f} nm\n{ro['n'] / nb * 100:.0f} % kept, σ ≤ {ro['thr'] * C.PX_NM:.1f} nm",
                    fontsize=FS - 2.5); letter(A, Lt); scalebar(A, w, 500)
    # D: GT-FRC vs fraction kept
    A = fig.add_subplot(gs[0, 3])
    for k, lk in METH:
        cur, opt, rc, ro, rows = S[k]; nb = R['methods'][k]['n_base']
        fr = np.array([r_['n'] for r_ in rows]) / nb; v = np.array([r_.get(a.crit, np.nan) for r_ in rows])
        A.plot(fr * 100, v, '-', color=COLM[k], lw=1.6, label=lk)
        A.plot(rc['n'] / nb * 100, rc[a.crit], 'o', mfc='none', mec=COLM[k], ms=9, mew=1.8)
        A.plot(ro['n'] / nb * 100, ro[a.crit], '*', color=COLM[k], ms=13, mec='k', mew=0.6)
    A.plot([], [], 'o', mfc='none', mec='0.3', label='current'); A.plot([], [], '*', color='0.5', mec='k', ms=11, label='optimum')
    A.set_yscale('log'); A.set_yticks([15, 20, 30, 50, 100]); A.set_yticklabels(['15', '20', '30', '50', '100']); A.minorticks_off()
    A.set_ylim(14, 260); A.set_xlim(0, 103); A.set_xlabel('kept (% of the base set)', fontsize=10); A.set_ylabel('GT-FRC (nm)', fontsize=10)
    A.legend(fontsize=7.5, frameon=False, loc='upper center', ncol=3); A.tick_params(labelsize=9)
    A.set_title('GT-FRC vs selection strictness', fontsize=FS); letter(A, 'D', dx=-0.2)
    # E: classes, current | optimal per method
    A = fig.add_subplot(gs[1, 0]); xs = []; cls = {}
    for i, (k, lk) in enumerate(METH):
        cur, opt, *_ = S[k]
        for j, (m, dx) in enumerate(((cur, -0.2), (opt, 0.2))):
            pc, ic, fc = classes(B[k], m, a.r_prec, a.r_fp); cls[(k, j)] = (pc, ic, fc); x = i + dx
            A.bar(x, pc / 1e5, 0.38, color=COLM[k]); A.bar(x, ic / 1e5, 0.38, bottom=pc / 1e5, color=COLM[k], alpha=0.4)
            A.bar(x, fc / 1e5, 0.38, bottom=(pc + ic) / 1e5, color='w', ec=COLM[k], hatch='////', lw=0.8)
            A.text(x, (pc + ic + fc) / 1e5 + 0.02, 'cur' if j == 0 else 'opt', ha='center', va='bottom', fontsize=6.5, rotation=90)
    A.bar([0], [0], color='0.35', label=f'precise (≤ {a.r_prec:g} nm)'); A.bar([0], [0], color='0.35', alpha=0.4, label=f'imprecise ({a.r_prec:g}-{a.r_fp:g} nm)')
    A.bar([0], [0], color='w', ec='0.35', hatch='////', label=f'false positive (> {a.r_fp:g} nm)')
    A.set_xticks(range(len(METH))); A.set_xticklabels([lk for _, lk in METH], rotation=35, ha='right', fontsize=9, rotation_mode='anchor')
    top = max(sum(v) for v in cls.values()) / 1e5
    A.set_ylim(0, top * 1.55); A.set_ylabel('Counts (x 10$^5$)', fontsize=11); A.tick_params(axis='y', labelsize=10)
    A.legend(fontsize=7, frameon=False, loc='upper left'); A.set_title('Localizations: current | optimal', fontsize=FS); letter(A, 'E', dx=-0.17)
    # I: GT-FRC curves; J: half-split FRC vs kept
    gb = outer[1].subgridspec(1, 2, wspace=0.22); A = fig.add_subplot(gb[0]); Bx = fig.add_subplot(gb[1])
    M_ = R['methods']
    for k, lk in METH:
        cur, opt, rc, ro, rows = S[k]
        import gtfrc as G; import frcopt as FO
        Sc = FO.Scorer(W, gt['pts'][:, :2]); g = Sc(B[k], opt, curves=True)['gt_curve']; gc = rc.get('gt_curve') or Sc(B[k], cur, curves=True)['gt_curve']
        A.plot(gc['freq'][2:], gc['curve'][2:], '--', color=COLM[k], lw=1.0)
        A.plot(g['freq'][2:], g['curve'][2:], '-', color=COLM[k], lw=2.2 if k == 'v2' else 1.5, label=f"{lk}  {ro[a.crit]:.1f} nm")
        nb = M_[k]['n_base']; fr = np.array([r_['n'] for r_ in rows]) / nb; v = np.array([r_.get(a.hs, np.nan) for r_ in rows])
        Bx.plot(fr * 100, v, '-', color=COLM[k], lw=1.6, label=lk)
        jh = M_[k]['opt'][a.v2fam if k == 'v2' else 'sig'][a.hs]['idx']
        Bx.plot(rows[jh]['n'] / nb * 100, rows[jh][a.hs], 'D', color=COLM[k], mec='k', ms=7)
        Bx.axvline(ro['n'] / nb * 100, color=COLM[k], lw=0.8, ls=':')
    A.axhline(1 / 7, color='grey', ls='--', lw=1); A.set_xlim(0, 0.09); A.set_ylim(-0.03, 1.0)
    A.set_xlabel('Spatial freq. [nm$^{-1}$]', fontsize=11); A.set_ylabel('GT-FRC', fontsize=11); A.tick_params(labelsize=10)
    A.set_title('FRC against the ground truth (solid: optimal, dashed: current)', fontsize=FS - 1); A.legend(fontsize=9, loc='upper right'); letter(A, 'I', dx=-0.09)
    Bx.set_yscale('log'); Bx.set_yticks([20, 30, 50, 100]); Bx.set_yticklabels(['20', '30', '50', '100']); Bx.minorticks_off()
    Bx.set_xlim(0, 103); Bx.set_xlabel('kept (% of the base set)', fontsize=11); Bx.set_ylabel('half-split FRC (nm)', fontsize=11)
    Bx.plot([], [], 'D', color='0.5', mec='k', label='half-split optimum'); Bx.plot([], [], ':', color='0.3', label='GT-FRC optimum')
    Bx.legend(fontsize=8, frameon=False, ncol=2, loc='upper right'); Bx.tick_params(labelsize=10)
    Bx.set_title('Half-split FRC (no ground truth) vs strictness', fontsize=FS); letter(Bx, 'J', dx=-0.09)
    base = a.set.replace('Dexp1_', 'Dexp_')
    os.makedirs(a.out, exist_ok=True); out = os.path.join(a.out, f'fig_sim_{base}_frcopt')
    cap = [f"Fig 2 variant at each method's FRC-optimal selection ({a.set}, 60 000 frames). Every method's Fig 2 base set (disjoint 6-frame "
           f"windows, own estimated precision max(σx, σy) < 16.25 nm; SIMCODE: p ≥ 0.63) is thresholded on the method's own reported "
           f"precision (SMLM / SIMFLUX: CRLB; DECODE: its predicted σ; SIMCODE: calibrated σ) from all rows down to 10 %; the rendered "
           f"selection minimizes the GT-FRC (as in Fig 2: 1/7 crossing, 5 nm pixels; ties -> more rows). Tiles: GT-FRC at the current "
           f"Fig 2 selection -> at the optimum, rows kept (% of the base set) and the threshold. (D) GT-FRC vs rows kept; circle = current "
           f"selection, star = optimum. (E) Localizations of the current | optimal selection: precise = 2D error to the nearest emitter ON "
           f"in the window ≤ {a.r_prec:g} nm, imprecise = {a.r_prec:g}-{a.r_fp:g} nm, false positive = no emitter ON within {a.r_fp:g} nm. "
           + '; '.join(f"{lk} {cls[(k, 0)][0]}/{cls[(k, 0)][1]}/{cls[(k, 0)][2]} -> {cls[(k, 1)][0]}/{cls[(k, 1)][1]}/{cls[(k, 1)][2]}" for k, lk in METH)
           + ". (I) GT-FRC curves, optimal (solid) and current (dashed). (J) Half-split FRC (alternating 300-frame halves, no ground "
           f"truth; {a.hs}) vs rows kept; diamond = its optimum, dotted = the GT-FRC optimum."]
    open(out + '_caption.txt', 'w').write('\n'.join(cap) + '\n')
    for ext, kw in (('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        plt.savefig(f'{out}.{ext}', **kw); print(f'{out}.{ext}')


if __name__ == '__main__':
    main()
