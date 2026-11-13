"""Opt-in step: add the column p_valid (pvalid.py model) to a SIMCODE v2 localization table -> a NEW file <table>_pvalid.npz
(the input table and simcode/strict.py are not changed). Hook for later: strict.build_table could call pvalid.features + the
model when a --pvalid <model.joblib> option is given and write the column into locs_p2.npz.
Also the GT-free sanity check used for real data: revisit consistency. For every disjoint-window row, the nearest row of the
window 6 frames later within 1 px is taken as the same emitter (DNA-PAINT events last several windows); per p_valid bin the
pair separation d is reported as median d / sqrt(sx1^2 + sy1^2 + sx2^2 + sy2^2)/sqrt(2)... (z, ~0.83 for calibrated Gaussian
errors on both rows) and the fraction d > 60 nm (a gross or wrong-emitter row in the pair).
usage: python analysis/papersim/add_pvalid.py <locs.npz> [--model out/papersim/selection/pvalid_m1809.joblib] [--out ...]"""
import os, sys, json, argparse
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import pvalid as PV


def apply(locs, model):
    import joblib
    M = joblib.load(model); z = np.load(locs)
    X = PV.features(z, PV.pattern_k(locs))
    return z, PV.predict(M['valid'], X), M


def revisit(z, pv, bins=(0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0001)):
    d = z['frame'] % 6 == 0
    i = np.nonzero(d)[0]
    x, y, f = z['x'][i].astype(np.float64), z['y'][i].astype(np.float64), z['frame'][i].astype(np.float64)
    tree = cKDTree(np.c_[x, y, f * 1000.0])
    dd, j = tree.query(np.c_[x, y, (f + 6) * 1000.0], distance_upper_bound=1.0)
    ok = np.isfinite(dd); a, b = i[ok], i[j[ok]]
    sep = dd[ok] * C.PX_NM
    sc = np.sqrt((z['sx'][a] ** 2 + z['sy'][a] ** 2 + z['sx'][b] ** 2 + z['sy'][b] ** 2) / 2) * C.PX_NM
    pmin = np.minimum(pv[a], pv[b])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pmin >= lo) & (pmin < hi)
        if m.sum() < 50: continue
        out.append(dict(p_valid_min=f'{lo:.2f}-{min(hi, 1):.2f}', n_pairs=int(m.sum()), median_sep_nm=float(np.median(sep[m])),
                        median_z=float(np.median(sep[m] / sc[m])), frac_sep_gt60=float((sep[m] > 60).mean())))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('locs'); ap.add_argument('--model', default=os.path.join(PV.OUTD, 'pvalid_m1809.joblib'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    z, pv, M = apply(a.locs, a.model)
    out = a.out or a.locs.replace('.npz', '_pvalid.npz')
    np.savez(out, **{k: z[k] for k in z.files}, p_valid=pv.astype(np.float32))
    print('written', out, 'p_valid quantiles', np.percentile(pv, [5, 25, 50, 75, 95]).round(3).tolist())


if __name__ == '__main__':
    main()
