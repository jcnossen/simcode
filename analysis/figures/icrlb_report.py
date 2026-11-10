"""Tables + plot for icrlb.py outputs (per-frame intensity and bg vs CRLB) and the combined CRLB figure
plots/crlb_sc6_all.png (rows: x/y position from the gate npz, per-frame intensity I_t, per-frame bg; columns: scenes).
I_t metrics over (target, frame) with I_gt > 0, per GT window-photon bin (label units):
  ratio = RMS(I_net - I_gt) / RMS(CRLB fr_m); med|z| = 1.4826 med |e / fr_m|; total bias = sum I_net / sum I_gt - 1;
  slope-corrected ratio: I_net / a with a = the through-origin LS slope of I_net on I_gt in the bin.
bg metrics over (target, frame): ratio = RMS(bg_net - bg_true_px) / RMS(CRLB flat bg); bias = mean error; floor = RMS of
(pixel bg - patch mean), and the ratio vs the patch-mean target (what a flat-bg estimator can reach).
usage: python icrlb_report.py [nets sc5n,sc5nb,sc6n] [scenes iso,rs05q,rs05qm] [dest dir]
       -> <out>/sc7/icrlb_report.txt, <out>/plots/crlb_sc6_all.png (or both into dest dir)"""
import sys, os
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcode2 import config
D7 = os.path.join(config.root('out'), 'sc7')
NETS = (sys.argv[1] if len(sys.argv) > 1 else 'sc5n,sc5nb,sc6n').split(',')
SCS = (sys.argv[2] if len(sys.argv) > 2 else 'iso,rs05q,rs05qm').split(',')
DEST = sys.argv[3] if len(sys.argv) > 3 else ''
GTAG = {'sc5n': '_hd_sc5nb15pm', 'sc5nb': '_hd_sc5nbb15pm', 'sc6n': '_hd_sc6nb15pm'}
COL = {'sc5n': '#2a78d6', 'sc5nb': '#eb6834', 'sc6n': '#1baf7a'}
TB = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9)]
NB = [(0, 1.5), (1.5, 3), (3, 6), (6, 1e9)]
PBINS = np.array([150, 300, 500, 800, 1300, 2000, 3200, 5000, 8000, 13000, 20000])


def imet(d, m, key):
    I = d[key][m]; G = d['I_gt'][m]; C = d['fr_m'][m]
    ok = (G > 0) & np.isfinite(I) & np.isfinite(C)
    if ok.sum() < 30: return None
    e = I[ok] - G[ok]; c = C[ok]
    a = np.sum(I[ok] * G[ok]) / np.sum(G[ok] ** 2)
    tot = np.nansum(np.where(G > 0, I, 0), 1); tg = np.sum(G, 1); okr = np.isfinite(tot) & np.isfinite(I).all(1)
    return dict(n=int(m.sum()), ratio=np.sqrt(np.mean(e ** 2) / np.mean(c ** 2)), zrms=np.sqrt(np.mean((e / c) ** 2)), medz=1.4826 * np.median(np.abs(e / c)),
                bias=np.sum(tot[okr]) / np.sum(tg[okr]) - 1, slope=a, ratio_sc=np.sqrt(np.mean((I[ok] / a - G[ok]) ** 2) / np.mean(c ** 2)), zrms_sc=np.sqrt(np.mean(((I[ok] / a - G[ok]) / c) ** 2)),
                rmsd=np.sqrt(np.mean(e ** 2)), crlb=np.sqrt(np.mean(c ** 2)))


def bmet(d, m, key):
    B = d[key][m]; t = d['bg_t'][m]; c = d['bg_crlb'][m]; pmn = d['bg_pmean'][m]
    ok = np.isfinite(B) & np.isfinite(c)
    if ok.sum() < 30: return None
    e = B[ok] - t[ok]; e2 = B[ok] - pmn[ok]
    return dict(n=int(m.sum()), ratio=np.sqrt(np.mean(e ** 2) / np.mean(c[ok] ** 2)), bias=float(np.mean(e)), rel_bias=float(np.mean(e) / np.mean(t[ok])),
                floor=float(np.sqrt(np.mean(d['bg_floor'][m][ok] ** 2))), ratio_pm=np.sqrt(np.mean(e2 ** 2) / np.mean(c[ok] ** 2)),
                rmsd=np.sqrt(np.mean(e ** 2)), crlb=np.sqrt(np.mean(c[ok] ** 2)), bg=float(np.mean(t[ok])))


def main():
    L = []; pr = lambda s: (L.append(s), print(s, flush=True))
    pr('Per-frame intensity I_t vs the multi-emitter per-frame CRLB (fr_m) and per-frame bg vs the flat-bg patch CRLB; bins = GT window photons (label units)')
    pr('  resolvable targets only (multi-emitter position CRLB < 25 nm); RMS/CRLB = RMS(e) / RMS(CRLB), z-rms = RMS(e / CRLB)')
    pr('  det = detector candidates matched 1:1 within 1 px; forced = head at all GT positions (no selection)')
    Dall = {}
    for sc in SCS:
        for nm in NETS:
            f = f'{D7}/icrlb_{nm}_{sc}.npz'
            if not os.path.exists(f): continue
            d = dict(np.load(f))
            res = np.hypot(d['sf_m'][:, 0], d['sf_m'][:, 1]) / np.sqrt(2) * 108.33 < 25     # resolvable targets (as the gate report)
            d = {k: v[res] for k, v in d.items()}; Dall[nm, sc] = d
            pr(f'===== {sc} {nm}: {len(d["ph"])} targets, det-matched {d["matched"].mean():.3f}')
            for lo, hi in TB:
                m = (d['ph'] >= lo) & (d['ph'] < hi)
                s = f'  {lo:5.0f}-{hi:<6.0f}'
                for key, lab in (('I_det', 'det'), ('I_f', 'forced')):
                    mm = m & (np.isfinite(d[key]).all(1))
                    r = imet(d, mm, key)
                    if r: s += (f' | I {lab} n {r["n"]:5d} RMS/CRLB {r["ratio"]:.2f} z-rms {r["zrms"]:.2f} med|z| {r["medz"]:.2f} bias {100 * r["bias"]:+5.1f}% '
                                f'slope {r["slope"]:.3f} slope-corr RMS/CRLB {r["ratio_sc"]:.2f} z-rms {r["zrms_sc"]:.2f}')
                pr(s)
            for lo, hi in TB:
                m = (d['ph'] >= lo) & (d['ph'] < hi)
                s = f'  {lo:5.0f}-{hi:<6.0f}'
                for key, lab in (('bg_det', 'det'), ('bg_f', 'forced')):
                    r = bmet(d, m & np.isfinite(d[key]).all(1), key)
                    if r: s += (f' | bg {lab} n {r["n"]:5d} rmsd {r["rmsd"]:.3f} crlb {r["crlb"]:.3f} ratio {r["ratio"]:.2f} bias {r["bias"]:+.3f} '
                                f'({100 * r["rel_bias"]:+.1f}%) floor {r["floor"]:.3f} ratio_vs_patchmean {r["ratio_pm"]:.2f}')
                pr(s)
            pr('  by nn (all photons >= 200, forced): ' + ' | '.join(
                f'nn {lo}-{hi}: I z-rms {imet(d, (d["ph"] >= 200) & (d["nn"] >= lo) & (d["nn"] < hi), "I_f")["zrms"]:.2f} '
                f'bg ratio {bmet(d, (d["ph"] >= 200) & (d["nn"] >= lo) & (d["nn"] < hi), "bg_f")["ratio"]:.2f} bias {bmet(d, (d["ph"] >= 200) & (d["nn"] >= lo) & (d["nn"] < hi), "bg_f")["bias"]:+.3f}'
                for lo, hi in NB if ((d['ph'] >= 200) & (d['nn'] >= lo) & (d['nn'] < hi)).sum() >= 30))
    open(os.path.join(DEST or D7, 'icrlb_report.txt'), 'w').write('\n'.join(L) + '\n')
    # ---------------- combined figure
    from simcode2.hd import gate_report as GR
    from plot_crlb_sc3 import curves
    import plot_crlb_sc3 as P3
    P3.BINS = PBINS
    cen = np.sqrt(PBINS[:-1] * PBINS[1:])
    fig, axs = plt.subplots(3, len(SCS), figsize=(4.2 * len(SCS), 11), squeeze=False)
    for j, sc in enumerate(SCS):
        ax = axs[0, j]; first = True
        for nm in NETS:
            try:
                R, _ = GR.load(os.path.join(config.root('out'), 'gate'), sc, GTAG[nm])
            except Exception:
                continue
            C = curves(R, 'sc', 'sc')
            if first:
                ax.loglog(cen, C[:, 1], '-', color='k', lw=2, label='CRLB SIMFLUX multi-emitter')
                ax.loglog(cen, C[:, 2], ':', color='k', lw=1.5, label='CRLB SIMFLUX isolated'); first = False
            ax.loglog(cen, C[:, 0], 'o-', color=COL[nm], ms=4, lw=2, label=nm)
        ax.set_title(sc, fontsize=10); ax.set_ylabel('x/y RMSD (nm)'); ax.grid(True, which='both', alpha=0.3)
        if j == 0: ax.legend(fontsize=7, loc='lower left')
        for row, (kind, ylab) in enumerate((('I', 'per-frame I_t RMSD (photons)'), ('bg', 'per-frame bg RMSD (ph/px)')), start=1):
            ax = axs[row, j]; first = True
            for nm in NETS:
                d = Dall.get((nm, sc))
                if d is None: continue
                ys = []; yc = []; yf = []
                for lo, hi in zip(PBINS[:-1], PBINS[1:]):
                    m = (d['ph'] >= lo) & (d['ph'] < hi)
                    if kind == 'I':
                        dd_ = dict(d); cm = np.nanmedian(np.where(d['I_gt'][m] > 0, d['fr_m'][m], np.nan)) if m.any() else np.nan
                        dd_['fr_m'] = np.where(d['fr_m'] <= 3 * cm, d['fr_m'], np.nan)     # drop unidentifiable frames (CRLB > 3x bin median)
                        r = imet(dd_, m & np.isfinite(d['I_det']).all(1), 'I_det'); rf = imet(dd_, m, 'I_f')
                    else:
                        r = bmet(d, m & np.isfinite(d['bg_det']).all(1), 'bg_det'); rf = bmet(d, m, 'bg_f')
                    ys.append(r['rmsd'] if r else np.nan); yc.append(rf['crlb'] if rf else np.nan); yf.append(rf['rmsd'] if rf else np.nan)
                if first:
                    ax.loglog(cen, yc, '-', color='k', lw=2, label='CRLB ' + ('per-frame I' if kind == 'I' else 'flat bg, 21x21 patch'))
                    first = False
                ax.loglog(cen, ys, 'o-', color=COL[nm], ms=4, lw=2, label=f'{nm} detected')
                ax.loglog(cen, yf, '--', color=COL[nm], lw=1, label=f'{nm} forced at GT')
            ax.set_ylabel(ylab); ax.grid(True, which='both', alpha=0.3)
            if row == 2: ax.set_xlabel('true photons per window (label units)')
            if j == 0: ax.legend(fontsize=6, loc='upper left', ncol=2)
    from matplotlib.ticker import LogLocator, NullFormatter
    for ax in axs.ravel():
        ax.xaxis.set_major_locator(LogLocator(base=10)); ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle('sc5n / sc5nb / sc6n (+PM): position, per-frame intensity and per-frame bg vs CRLB (resolvable targets; I_t row drops frames with CRLB > 3x the bin median)', fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.98)); out = os.path.join(DEST or config.out_dir('plots'), 'crlb_sc6_all.png')
    plt.savefig(out, dpi=90); print('saved', out)


if __name__ == '__main__':
    main()
