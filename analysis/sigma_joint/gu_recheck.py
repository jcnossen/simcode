"""sigma study (2026-11-17): the g(u) re-check with the joint-covariance runs (held-out simulated movies). u = per-10-px-tile rank of the
mean sigma among the rows with p >= 0.63 in the crop (simcode.table.tile_rank). For each pass: rms(error / sigma) (|z| < 4) per u bin
of the run's sigma; g_rank(u) = that / the overall value (the rank dependence only, as refitted with a calibrated sigma); the
count at 16.25 nm with sigma, sigma x g_rank(u), and sigma x the model's old g(u) (fitted for the old sigma), and the per-u z after
each.
usage: python gu_recheck.py <out json> <model> <label> <sim> [...]"""
import json, sys
import numpy as np
from evalsim import gt_windows, load_gt, gt_of, rundir, old_files, PX, UE, TRIM

UG = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]


def core(z):
    v = z.ravel(); c = v[np.abs(v) < TRIM]
    return float(np.sqrt(np.mean(c ** 2)) / 0.99893) if len(c) > 20 else float('nan')


def rows(run, tag, GW):
    from simcode import table as SP
    from simcode.evaluation import metrics as E
    R = SP.load_rows(run, tag); crop = SP.run_crop(run)
    ii = np.nonzero((R['p'] >= 0.63) & SP.in_crop(R['x_hy0'], R['y_hy0'], crop))[0]
    s = np.c_[R['sx_cal'], R['sy_cal']][ii].astype(np.float64)
    x, y = R['x_hy0'][ii], R['y_hy0'][ii]
    gi = E.match_fast(GW, R['wstart'][ii].astype(np.int64), x.astype(np.float64), y.astype(np.float64), r=1.0)
    m = gi >= 0
    e = np.full((len(ii), 2), np.nan); e[m] = np.c_[x[m] - GW[gi[m], 2], y[m] - GW[gi[m], 3]]
    e -= np.nanmedian(e[m], 0)
    return dict(x=x, y=y, s=s, e=e, m=m, crop=crop)


def main():
    from simcode import table as SP
    outj, model, label, sims = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    res = {}
    for ps in ('est', 'est2'):
        _, Gold = old_files(model, ps)
        D = [rows(rundir(label).format(m=model) + '/' + s, ps, gt_windows(*load_gt(gt_of(s)))) for s in sims]
        U = [SP.tile_rank(d['x'], d['y'], d['s'].mean(1), d['crop']) for d in D]
        Z = np.concatenate([(d['e'] / d['s'])[d['m']] for d in D]); Um = np.concatenate([u[d['m']] for u, d in zip(U, D)])
        call = core(Z)
        um, g = [], []
        for lo, hi in zip(UG[:-1], UG[1:]):
            k = (Um >= lo) & (Um < hi); um.append(float(np.median(Um[k]))); g.append(core(Z[k]) / call)
        G = dict(u_mid=um, g=g)
        out = dict(g_rank=G, z_all=call)
        for nm, GG in (('none', None), ('g_rank', G), ('g_old', Gold)):
            n16 = 0; Zs, Us = [], []
            for u, d in zip(U, D):
                s = d['s'] * (np.interp(u, GG['u_mid'], GG['g'])[:, None] if GG else 1.0)
                n16 += int((s.max(1) * PX <= 16.25).sum())
                u2 = SP.tile_rank(d['x'], d['y'], s.mean(1), d['crop'])
                Zs.append((d['e'] / s)[d['m']]); Us.append(u2[d['m']])
            Zs, Us = np.concatenate(Zs), np.concatenate(Us)
            out[nm] = dict(count16=n16, z_all=core(Zs), z_u=[core(Zs[(Us >= lo) & (Us < hi)]) for lo, hi in zip(UE[:-1], UE[1:])])
            print(f"{model} {ps} {nm:7s} count <= 16.25 nm {n16} | z all {out[nm]['z_all']:.3f} | z per u {' '.join(f'{v:.2f}' for v in out[nm]['z_u'])}", flush=True)
        print(f"{model} {ps} g_rank(u): " + ' '.join(f'{u:.2f}:{v:.3f}' for u, v in zip(um, g)), flush=True)
        res[ps] = out
    json.dump(res, open(outj, 'w'), indent=1)


if __name__ == '__main__':
    main()
