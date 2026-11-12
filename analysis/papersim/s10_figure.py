"""SI S10 figure (panels A, B, C as the preprint's SI figure) from s10.py's json.
  A  constant bg 1 photon / px / frame: x RMSD (2 % trimmed) of SMLM, SIMFLUX (one bg, the paper's model), SIMFLUX (bg per frame),
     network output before the pattern solve (pass-1 pattern-free head), SIMCODE v2 (pass 2 + solve); CRLB SMLM / SIMFLUX
  B  per-frame bg ~ U(1, 6): network output (pass 1; pass 2 dotted) RMSD and its predicted sigma vs the SMLM CRLB
  C  per-frame bg ~ U(1, 6): SIMCODE v2 RMSD and its predicted (calibrated) sigma vs the SIMFLUX CRLB with a bg per frame, + both
     SIMFLUX fits
Network rows are on the detected emitters (detector p >= 0.5 within 1 px; recall printed below 99 %).
usage: python analysis/papersim/s10_figure.py [--netset m1809] -> out/papersim/s10/s10_fig_<netset>.png / .svg / _table.md"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--netset', default='m1809'); a = ap.parse_args()
    D = os.path.join(C.OUT, 's10'); R = json.load(open(os.path.join(D, f's10_{a.netset}.json')))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    def col(cond, key, f='rmsd_trim2'):
        B = sorted(R['cond'][cond], key=int); r = [R['cond'][cond][b] for b in B]
        ph = np.array([q['photons'] for q in r])
        if key.startswith('crlb'): return ph, np.array([q[key][0] for q in r]) * C.PX_NM
        return ph, np.array([q[key][f][0] for q in r]) * C.PX_NM
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    A = ax[0]
    for k, ls, c, lab in (('crlb_smlm', '--', 'tab:blue', 'CRLB SMLM'), ('crlb_sf', '--', 'g', 'CRLB SIMFLUX')):
        A.loglog(*col('const1', k), ls, color=c, alpha=0.7, label=lab)
    for k, st, c, lab in (('smlm', 'o-', 'k', 'SMLM MLE'), ('sf', 's-', 'g', 'SIMFLUX MLE (one bg, paper model)'), ('sf_pfbg', 'D:', 'olive', 'SIMFLUX MLE (bg per frame)'),
                          ('nn_p1', '^-', 'tab:blue', 'network output (pass 1, before pattern solve)'), ('v2_p2', 'v-', 'tab:red', 'SIMCODE v2')):
        A.loglog(*col('const1', k), st, color=c, label=lab, ms=4)
    A.set_title('A: constant bg 1 photon/px/frame', fontsize=10)
    B = ax[1]
    B.loglog(*col('unif16', 'crlb_smlm'), '--', color='tab:blue', alpha=0.7, label='CRLB SMLM')
    B.loglog(*col('unif16', 'nn_p1'), 'o-', color='tab:blue', label='network output (pass 1) RMSD')
    B.loglog(*col('unif16', 'nn_p1', 'pred_mean'), 'x:', color='tab:blue', label='network output (pass 1) predicted')
    B.loglog(*col('unif16', 'nn_p2'), 'o-', color='c', lw=1, ms=3, label='pass-2 head prior RMSD')
    B.set_title('B: per-frame bg ~ U(1, 6): network output', fontsize=10)
    Cc = ax[2]
    Cc.loglog(*col('unif16', 'crlb_sf_pfbg'), '--', color='tab:red', alpha=0.7, label='CRLB SIMFLUX (bg per frame)')
    Cc.loglog(*col('unif16', 'v2_p2'), 'v-', color='tab:red', label='SIMCODE v2 RMSD')
    Cc.loglog(*col('unif16', 'v2_p2', 'pred_mean'), 'x:', color='tab:red', label='SIMCODE v2 predicted')
    Cc.loglog(*col('unif16', 'sf'), 's-', color='g', ms=3, lw=1, label='SIMFLUX MLE (one bg, paper model)')
    Cc.loglog(*col('unif16', 'sf_pfbg'), 'D:', color='olive', ms=3, lw=1, label='SIMFLUX MLE (bg per frame)')
    Cc.set_title('C: per-frame bg ~ U(1, 6): SIMCODE v2', fontsize=10)
    for x in ax:
        x.set_xlabel('total signal photons (6 frames)'); x.set_ylabel('x precision (nm)'); x.grid(True, which='both', alpha=0.3); x.legend(fontsize=7)
    fig.suptitle(f"precision vs CRLB, averaged 1809/2 PSF, 16 x 16 px ROIs (in 80 x 80 px fields), {R['protocol']['n_per_bin']} emitters per bin, "
                 f"RMSD with the 2 % largest errors trimmed; SIMCODE v2 net set {a.netset}", fontsize=9)
    plt.tight_layout()
    for ext in ('png', 'svg'): plt.savefig(os.path.join(D, f's10_fig_{a.netset}.{ext}'), dpi=110)
    L = ['| cond | photons | CRLB SMLM / SF / SF(bg per frame) | SMLM | SF (1 bg) | SF (bg per frame) | NN p1 (pred) | v2 (pred) | v2 RMSD / CRLB SF | v2 pred / RMSD | recall v2 |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for cond in R['cond']:
        for b in sorted(R['cond'][cond], key=int):
            q = R['cond'][cond][b]; f = lambda k, g='rmsd_trim2': q[k][g][0] * C.PX_NM
            cr = q['crlb_sf' if cond == 'const1' else 'crlb_sf_pfbg'][0] * C.PX_NM
            L.append(f"| {cond} | {q['photons']:.0f} | {q['crlb_smlm'][0] * C.PX_NM:.2f} / {q['crlb_sf'][0] * C.PX_NM:.2f} / {q['crlb_sf_pfbg'][0] * C.PX_NM:.2f} | "
                     f"{f('smlm'):.2f} | {f('sf'):.2f} | {f('sf_pfbg'):.2f} | {f('nn_p1'):.2f} ({f('nn_p1', 'pred_mean'):.2f}) | {f('v2_p2'):.2f} ({f('v2_p2', 'pred_mean'):.2f}) | "
                     f"{f('v2_p2') / cr:.2f} | {f('v2_p2', 'pred_mean') / f('v2_p2'):.2f} | {q['v2_p2']['recall']:.3f} |")
    open(os.path.join(D, f's10_table_{a.netset}.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
