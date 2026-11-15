"""Paper-ready SI figure 'si_crlb_precision' (label fig:crlb_vs_precision): precision vs CRLB from s10.py's json.
  A  constant background 1 photon / px / frame: SMLM, SIMFLUX, NN output, SIMCODE (v2) RMSD; CRLB SMLM / SIMFLUX dashed
  B  background per frame ~ U(1, 6): NN output RMSD and predicted precision vs the SMLM CRLB
  C  background per frame ~ U(1, 6): SIMCODE RMSD and predicted precision vs the SIMFLUX CRLB (bg per frame); SIMFLUX fits
Colours = the paper's method colours (analysis/paper/fig_exp_new.py: SMLM blue, SIMFLUX green, DECODE orange, SIMCODE red); the
NN output (no own colour in the paper) is purple. CRLBs dashed in the colour of the method they bound; predicted precision dotted;
open symbols = bins where the detector found < 50 % of the emitters (network curves only; the RMSD there is over the found ones).
Options (the user decides; defaults = current choice): --sf both|paper|pfbg (SIMFLUX under fluctuating bg), --p2prior 0|1
(panel B: also the pass-2 head prior), --cut 0|<photons> (drop bins below that photon count instead of marking them).
x axis: 'signal photons' (the intensity parameter, old scenes) or 'detected signal photons' for the PSF-v2 scenes (s10.py --psf v2;
read from the json's protocol.photons_are).
Also writes the per-bin table s10_table_<tag>.md (RMSD trim 2 %, CRLBs, predicted precision, ratios, recall; x precision, nm).
usage: python analysis/papersim/si_crlb_figure.py [--tag m1809] -> out/papersim/s10/si_crlb_precision_<tag>.{pdf,svg,png}, s10_table_<tag>.md"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

COL = dict(smlm='blue', sf='green', nn='purple', v2='red')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='m1809'); ap.add_argument('--sf', default='both')
    ap.add_argument('--p2prior', type=int, default=0); ap.add_argument('--cut', type=float, default=0)
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, NullFormatter
    matplotlib.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 3.2, 'svg.fonttype': 'none',
                                'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans', 'legend.fontsize': 6, 'legend.frameon': False,
                                'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4})
    D = os.path.join(C.OUT, 's10'); R = json.load(open(os.path.join(D, f's10_{a.tag}.json')))
    def series(cond, key, f='rmsd_trim2'):
        B = [b for b in sorted(R['cond'][cond], key=int) if R['cond'][cond][b]['photons'] >= a.cut]
        q = [R['cond'][cond][b] for b in B]; ph = np.array([x['photons'] for x in q])
        if key.startswith('crlb'): return ph, np.array([x[key][0] for x in q]) * C.PX_NM, None
        rec = np.array([x[key].get('recall', 1.0) if x[key].get('recall') == x[key].get('recall') else 1.0 for x in q])
        return ph, np.array([x[key][f][0] for x in q]) * C.PX_NM, rec
    def line(ax, cond, key, color, marker, ls='-', label=None, f='rmsd_trim2', open_low=False):
        ph, y, rec = series(cond, key, f)
        ax.plot(ph, y, ls=ls, color=color, label=label, lw=1.0 if f == 'rmsd_trim2' else 0.9)
        if marker:
            low = (rec < 0.5) if (open_low and rec is not None) else np.zeros(len(ph), bool)
            ax.plot(ph[~low], y[~low], marker, color=color, ls='none', ms=3.2)
            ax.plot(ph[low], y[low], marker, color=color, ls='none', ms=3.2, mfc='white')
    fig, axs = plt.subplots(1, 3, figsize=(17 / 2.54, 6.2 / 2.54))
    A, B, Cc = axs
    line(A, 'const1', 'crlb_smlm', COL['smlm'], None, '--', 'CRLB SMLM'); line(A, 'const1', 'crlb_sf', COL['sf'], None, '--', 'CRLB SIMFLUX')
    line(A, 'const1', 'smlm', COL['smlm'], 'o', '-', 'SMLM'); line(A, 'const1', 'sf', COL['sf'], 's', '-', 'SIMFLUX')
    line(A, 'const1', 'nn_p1', COL['nn'], '^', '-', 'NN output', open_low=True); line(A, 'const1', 'v2_p2', COL['v2'], 'v', '-', 'SIMCODE', open_low=True)
    line(B, 'unif16', 'crlb_smlm', COL['smlm'], None, '--', 'CRLB SMLM')
    line(B, 'unif16', 'nn_p1', COL['nn'], '^', '-', 'NN output', open_low=True)
    line(B, 'unif16', 'nn_p1', COL['nn'], None, ':', 'NN output, predicted', f='pred_mean')
    if a.p2prior: line(B, 'unif16', 'nn_p2', 'orchid', '^', '-', 'NN output (pattern-aware head)', open_low=True)
    line(Cc, 'unif16', 'crlb_sf_pfbg', COL['sf'], None, '--', 'CRLB SIMFLUX')
    line(Cc, 'unif16', 'v2_p2', COL['v2'], 'v', '-', 'SIMCODE', open_low=True)
    line(Cc, 'unif16', 'v2_p2', COL['v2'], None, ':', 'SIMCODE, predicted', f='pred_mean')
    if a.sf in ('both', 'paper'): line(Cc, 'unif16', 'sf', COL['sf'], 's', '-', 'SIMFLUX (one background)')
    if a.sf in ('both', 'pfbg'): line(Cc, 'unif16', 'sf_pfbg', COL['sf'], 'D', '-.', 'SIMFLUX (background per frame)')
    Cc.set_yscale('log'); Cc.set_ylim(top=Cc.get_ylim()[1] * 30)
    A.set_yscale('log'); A.set_ylim(top=A.get_ylim()[1] * 4)            # room for the 2-column legend          # room for the legend above the curves
    titles = ('constant background', 'fluctuating background: NN output', 'fluctuating background: SIMCODE')
    for ax, letter, t in zip(axs, 'ABC', titles):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.xaxis.set_major_locator(FixedLocator([100, 300, 1000, 3000])); ax.xaxis.set_major_formatter(FixedFormatter(['100', '300', '1000', '3000']))
        ax.xaxis.set_minor_locator(NullLocator())
        yt = [1, 2, 5, 10, 20, 50, 100, 200]
        lo, hi = ax.get_ylim(); yt = [v for v in yt if lo <= v <= hi * 1.01]
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in yt])); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(('detected ' if str(R['protocol'].get('photons_are', '')).startswith('detected') else '') + 'signal photons (6 frames)'); ax.set_ylabel('x precision (nm)')
        ax.set_title(t, fontsize=7); ax.text(-0.2, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', handlelength=2.2, borderaxespad=0.3, ncol=2 if letter == 'A' else 1, columnspacing=0.8)
        ax.grid(True, which='major', lw=0.3, alpha=0.4)
    fig.tight_layout(w_pad=0.8)
    base = os.path.join(D, f'si_crlb_precision_{a.tag}')
    for ext in ('pdf', 'svg', 'png'): fig.savefig(f'{base}.{ext}', dpi=300)
    print(base)
    table(R, os.path.join(D, f's10_table_{a.tag}.md'))


def table(R, fn):
    """per-bin table (x precision in nm; v2 RMSD / CRLB: SIMFLUX one-bg CRLB at const1, per-frame-bg CRLB at unif16)"""
    nm = lambda v: v * C.PX_NM
    L = ['| cond | photons | CRLB SMLM / SF / SF(bg per frame) | SMLM | SF (1 bg) | SF (bg per frame) | NN Output (pred) | SIMCODE (pred) | SIMCODE RMSD / CRLB SF | SIMCODE pred / RMSD | recall SIMCODE |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for cond in R['cond']:
        for b in sorted(R['cond'][cond], key=int):
            q = R['cond'][cond][b]; r = lambda k: nm(q[k]['rmsd_trim2'][0]); pr = lambda k: nm(q[k]['pred_mean'][0])
            cr = nm(q['crlb_sf'][0] if cond == 'const1' else q['crlb_sf_pfbg'][0])
            L.append(f"| {cond} | {q['photons']:.0f} | {nm(q['crlb_smlm'][0]):.2f} / {nm(q['crlb_sf'][0]):.2f} / {nm(q['crlb_sf_pfbg'][0]):.2f} | {r('smlm'):.2f} | {r('sf'):.2f} | "
                     f"{r('sf_pfbg'):.2f} | {r('nn_p1'):.2f} ({pr('nn_p1'):.2f}) | {r('v2_p2'):.2f} ({pr('v2_p2'):.2f}) | {r('v2_p2') / cr:.2f} | {pr('v2_p2') / r('v2_p2'):.2f} | {q['v2_p2']['recall']:.3f} |")
    open(fn, 'w').write('\n'.join(L) + '\n'); print(fn)


if __name__ == '__main__':
    main()
