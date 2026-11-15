"""(2026-11-11 in-situ 3D PSF) Compare psf_heldout outputs (archive-2026-11-15:simcode/calibration/psf_heldout.py) on the SAME held-out samples.
Per model: paired held-out NLL difference vs the first (reference) file (total, per sample, z = total / SE), and the residual
second moments ddM = dM(data) - dM(model) (dM = Mxx - Myy, px^2; ratio of sums over samples, bootstrap SE over samples)
overall / per pattern angle / per aperture, the angle contrast (angle 1 - angle 0), the radial width ratio W/S data / model, and a
per-tile ddM chi^2 (4 x 4 tiles of 30 px).
usage: python -m simcode.calibration.psf_compare ref.npz other.npz ... [--json out.json]"""
import sys, json
import numpy as np


def ratio(M, di, an, ai, comp, idx=None):
    A = M if idx is None else M[idx]
    num = A[:, di, an, ai, comp].sum() if an is not None else A[:, di, :, ai, comp].sum()
    den = A[:, di, an, ai, 0].sum() if an is not None else A[:, di, :, ai, 0].sum()
    return num / den


def ddm(M, an, ai, idx=None):
    return ratio(M, 0, an, ai, 1, idx) - ratio(M, 1, an, ai, 1, idx)


def boot(f, n, nb=200, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.std([f(rng.integers(0, n, n)) for _ in range(nb)]))


def summarize(fn, ref=None):
    H = dict(np.load(fn)); M = H['M']; n = len(H['nll']); ap = list(H['apertures'])
    r = dict(file=fn, n=n, nll_per_px=float(H['nll'].sum() / H['npix'].sum()), chi2_per_px=float(H['chi2'].sum() / H['npix'].sum()),
             z_median=float(np.median(H['z'])))
    if ref is not None:
        assert np.array_equal(ref['sel'], H['sel']), 'different held-out samples'
        d = H['nll'] - ref['nll']
        r.update(dnll_total=float(d.sum()), dnll_per_sample=float(d.mean()), dnll_se_per_sample=float(d.std() / np.sqrt(n)),
                 dnll_z=float(-d.mean() / (d.std() / np.sqrt(n))), frac_improved=float((d < 0).mean()), dnll_median=float(np.median(d)))
        k = np.abs(d - np.median(d)) < 10       # trimmed (|d - median| < 10 nats): robust to residual z local minima
        r.update(dnll_trim_per_sample=float(d[k].mean()), dnll_trim_z=float(-d[k].mean() / (d[k].std() / np.sqrt(k.sum()))), n_trimmed=int((~k).sum()))
    for ai, rad in enumerate(ap):
        key = f'r{rad:g}'
        r[f'ddM_{key}'] = float(ddm(M, None, ai)); r[f'ddM_{key}_se'] = boot(lambda i: ddm(M, None, ai, i), n)
        for an in range(2):
            r[f'ddM_{key}_a{an}'] = float(ddm(M, an, ai)); r[f'ddM_{key}_a{an}_se'] = boot(lambda i: ddm(M, an, ai, i), n)
        con = lambda i=None: (ratio(M, 0, 1, ai, 1, i) - ratio(M, 0, 0, ai, 1, i)) - (ratio(M, 1, 1, ai, 1, i) - ratio(M, 1, 0, ai, 1, i))
        r[f'angcontrast_{key}_data'] = float(ratio(M, 0, 1, ai, 1) - ratio(M, 0, 0, ai, 1))
        r[f'angcontrast_{key}_model'] = float(ratio(M, 1, 1, ai, 1) - ratio(M, 1, 0, ai, 1))
        r[f'angcontrast_{key}_resid'] = float(con()); r[f'angcontrast_{key}_resid_se'] = boot(con, n)
        r[f'Wratio_{key}'] = float(ratio(M, 0, None, ai, 2) / ratio(M, 1, None, ai, 2))
    # per tile (r < 3)
    ai = ap.index(3.0)
    t = (np.clip(H['x'] // 30, 0, 3) * 4 + np.clip(H['y'] // 30, 0, 3)).astype(int)
    vals, ses = [], []
    for k in range(16):
        ii = np.flatnonzero(t == k)
        if len(ii) < 100: continue
        vals.append(ddm(M, None, ai, ii)); ses.append(boot(lambda i: ddm(M, None, ai, ii[i]), len(ii), nb=100))
    vals, ses = np.array(vals), np.array(ses)
    r['tile_ddM_r3'] = vals.round(4).tolist(); r['tile_chi2_dof'] = float((((vals - vals.mean()) / ses) ** 2).sum() / max(len(vals) - 1, 1))
    return r, H


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    js = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    if js: args = [a for a in args if a != js]
    out = []; ref = None
    for fn in args:
        r, H = summarize(fn, ref)
        if ref is None: ref = H
        out.append(r)
        print(f"{fn.split('/')[-1]:28s} n {r['n']} nll/px {r['nll_per_px']:.5f} "
              + (f"dNLL {r['dnll_total']:+9.1f} ({r['dnll_per_sample']:+.3f}±{r['dnll_se_per_sample']:.3f}/smp, z {r['dnll_z']:+.1f}, trim z {r['dnll_trim_z']:+.1f} n{r['n_trimmed']}, impr {r['frac_improved']:.2f}) " if 'dnll_total' in r else ' ' * 76)
              + f"ddM r3 {r['ddM_r3']:+.4f}±{r['ddM_r3_se']:.4f} [a0 {r['ddM_r3_a0']:+.4f} a1 {r['ddM_r3_a1']:+.4f}] r2.5 {r['ddM_r2.5']:+.4f} r4 {r['ddM_r4']:+.4f} "
              + f"| ang1-0 data {r['angcontrast_r3_data']:+.4f} model {r['angcontrast_r3_model']:+.4f} resid {r['angcontrast_r3_resid']:+.4f}±{r['angcontrast_r3_resid_se']:.4f} "
              + f"| W d/m {r['Wratio_r3']:.4f} | tiles chi2/dof {r['tile_chi2_dof']:.2f} | z {r['z_median']:.3f}")
    if js: json.dump(out, open(js, 'w'), indent=1)


if __name__ == '__main__':
    main()
