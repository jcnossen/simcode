"""sigma layer (2026-11-17): the kap = 1 precision check. Production solve weights (kap_bins of the model) vs kap_p = kap_I2 = 1
(baseline_runs/runs_k1.sh) on the held-out sim movies, per model, pass and network-photon bin. Same rows (pass 1 and the pass-2 network outputs
do not depend on kap); each run's rows with p >= 0.63 inside the crop are matched to the ground truth (fit_sigma_layer: 1 px,
per window, minus the per-movie median error); compared on the rows matched in both. Precision = robust error sd (1.4826 MAD,
both axes, nm), RMS of the rows with |e| < 50 nm, gross % (|e| >= 50 nm); ratio = kap 1 / production.
usage: python k1_compare.py <out json> <model>:<sim> [...]"""
import json, os, sys
import numpy as np
from simcode import table as SP
from simcode.evaluation import metrics as E
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalsim import gt_of, gt_windows, load_gt, SL as O

PX = 108.33
PH = [0, 300, 500, 800, 1200, 2000, 3000, 5000, 1e9]


def errs(run, tag, GW):
    R = SP.load_rows(run, tag)
    crop = SP.run_crop(run)
    ii = np.nonzero((R['p'] >= 0.63) & SP.in_crop(R['x_hy0'], R['y_hy0'], crop))[0]
    gi = E.match_fast(GW, R['wstart'][ii].astype(np.int64), R['x_hy0'][ii].astype(np.float64), R['y_hy0'][ii].astype(np.float64), r=1.0)
    e = np.full((len(R['x_hy0']), 2), np.nan)
    m = gi >= 0
    e[ii[m]] = np.c_[R['x_hy0'][ii][m] - GW[gi[m], 2], R['y_hy0'][ii][m] - GW[gi[m], 3]]
    e -= np.nanmedian(e, 0)
    return e, R['phot'], len(R['x_hy0'])


def prec(e):
    r = np.hypot(*e.T) * PX
    v = e.ravel() * PX
    c = e[r < 50] * PX
    return dict(sd_rob=float(1.4826 * np.median(np.abs(v))), rms50=float(np.sqrt(np.mean(c ** 2))), gross_pct=float(100 * np.mean(r >= 50)))


def main():
    outj, specs = sys.argv[1], sys.argv[2:]
    res = {}
    for sp in specs:
        m, s = sp.split(':', 1)
        GW = gt_windows(*load_gt(gt_of(s)))
        for ps, tag in (('pass1', 'est'), ('pass2', 'est2')):
            ea, ph, na = errs(f'{O}/runs/{m}_id/{s}', tag, GW)
            eb, phb, nb = errs(f'{O}/runs/{m}_k1/{s}', tag, GW)
            assert na == nb and np.array_equal(ph, phb), f'{sp} {ps}: rows differ'
            ok = np.isfinite(ea).all(1) & np.isfinite(eb).all(1)
            res.setdefault(m, {}).setdefault(ps, {})[s] = dict(e_prod=ea[ok], e_k1=eb[ok], ph=ph[ok])
    out = {}
    for m, P in res.items():
        for ps, S in P.items():
            ea = np.concatenate([v['e_prod'] for v in S.values()]); eb = np.concatenate([v['e_k1'] for v in S.values()])
            ph = np.concatenate([v['ph'] for v in S.values()])
            rows = []
            for lo, hi in zip(PH[:-1], PH[1:]):
                k = (ph >= lo) & (ph < hi)
                if k.sum() < 200: continue
                a, b = prec(ea[k]), prec(eb[k])
                # paired bootstrap of the robust sd ratio (rows resampled)
                rng = np.random.default_rng(0); idx = np.nonzero(k)[0]; rr = []
                for _ in range(200):
                    j = rng.choice(idx, len(idx))
                    rr.append(prec(eb[j])['sd_rob'] / prec(ea[j])['sd_rob'])
                rows.append(dict(lo=lo, hi=hi, n=int(k.sum()), prod=a, k1=b, ratio_sd_rob=b['sd_rob'] / a['sd_rob'],
                                 ratio_rms50=b['rms50'] / a['rms50'], ci_sd_rob=[float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]))
                print(f'{m} {ps} {lo:>5g}-{hi:<6g} n {k.sum():7d} sd_rob {a["sd_rob"]:6.2f} -> {b["sd_rob"]:6.2f} nm (x{rows[-1]["ratio_sd_rob"]:.3f} '
                      f'[{rows[-1]["ci_sd_rob"][0]:.3f}, {rows[-1]["ci_sd_rob"][1]:.3f}])  rms<50 {a["rms50"]:6.2f} -> {b["rms50"]:6.2f} '
                      f'(x{rows[-1]["ratio_rms50"]:.3f})  gross {a["gross_pct"]:.2f} -> {b["gross_pct"]:.2f} %', flush=True)
            out.setdefault(m, {})[ps] = dict(sims=list(S), bins=rows)
    json.dump(out, open(outj, 'w'), indent=1)


if __name__ == '__main__':
    main()
