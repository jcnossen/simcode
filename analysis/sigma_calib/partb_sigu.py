"""sigma study Part B, item 4 (2026-11-17): what the selection-aware sigma rescale g(u) (the model's sig_u; simcode.table.build_table)
does at the paper's cut max(sx, sy) < 0.15 px (16.25 nm). Per run: the table built from the chunk files without and with g(u)
(the model's sigma_rescale json), the count of the selection (p >= 0.63, inside the crop, precision cut) and its median sigma;
on simulated movies with ground truth (run:sim_dir) also the robust zcal (1.4826 median |e / sigma|, both axes, |e| < 50 nm, the
matched set's median error removed, as analysis/psf3d_nets/sigcal.py) and the GT RMS of the selected rows.
usage: python partb_sigu.py <out json> <tag est|est2> <sig_u json> <run>[:<sim dir>[:<f_lo>:<f_hi>]] [...]"""
import json, sys
import numpy as np
from simcode import table as SP
from simcode.evaluation import metrics as E

PX = 108.33
CUT = 0.15 * PX
rob = lambda v: float(1.4826 * np.median(np.abs(v))) if len(v) else float('nan')


def one(spec, tag, sig_u):
    p = spec.split(':'); run = p[0]; sim = p[1] if len(p) > 1 else ''
    flo, fhi = (int(p[2]), int(p[3])) if len(p) > 3 else (0, 1 << 40)
    T0 = SP.build_table(run, tag, sig_u=None)[0]
    T1 = SP.build_table(run, tag, sig_u=sig_u)[0]
    crop = SP.run_crop(run)
    fr = (T0['frame'] >= flo) & (T0['frame'] < fhi)
    out = dict(run=run, sim=sim, frames=[flo, min(fhi, int(T0['frame'].max()) + 1)])
    base = SP.select(T0, None, crop=crop) & fr
    for nm, T in (('no_g', T0), ('g', T1)):
        k = SP.select(T, CUT, crop=crop) & fr
        out[nm] = dict(n_sel=int(k.sum()), n_base=int(base.sum()), frac=float(k.sum() / max(base.sum(), 1)),
                       sigma_med_nm=float(np.median(SP.precision_nm(T)[k])) if k.any() else float('nan'))
    both = SP.select(T0, CUT, crop=crop) & SP.select(T1, CUT, crop=crop) & fr
    out['n_both'] = int(both.sum())
    if sim:
        G = dict(np.load(f'{sim}/sim_gt.npz', allow_pickle=True)); G['nf'] = int(G['nf'])
        GW = E.gt_windows_fast(G, grid=3)
        ib = np.nonzero(base)[0]
        gi = E.match_fast(GW, T0['frame'][ib].astype(np.int64), T0['x'][ib], T0['y'][ib], r=1.0)
        m = gi >= 0
        e = np.full((len(ib), 2), np.nan); e[m] = np.c_[T0['x'][ib][m] - GW[gi[m], 2], T0['y'][ib][m] - GW[gi[m], 3]]
        e -= np.nanmedian(e[m], 0)
        ok = m & (np.hypot(*e.T) * PX < 50)
        for nm, T in (('no_g', T0), ('g', T1)):
            s = np.c_[T['sx'][ib], T['sy'][ib]]
            k = SP.select(T, CUT, crop=crop)[ib] & ok
            out[nm].update(zcal_sel=rob((e[k] / s[k]).ravel()), zcal_all=rob((e[ok] / s[ok]).ravel()), rms_sel_nm=float(np.sqrt(np.nanmean(e[k] ** 2)) * PX),
                           n_matched_sel=int(k.sum()))
            u = SP.tile_rank(T0['x'][ib], T0['y'][ib], 0.5 * (T0['sx'][ib] + T0['sy'][ib]).astype(np.float64), crop)
            out[nm]['zcal_by_u'] = {f'{lo:.2f}-{hi:.2f}': rob((e[ok & (u >= lo) & (u < hi)] / s[ok & (u >= lo) & (u < hi)]).ravel())
                                    for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0))}
    return out


def main():
    outf, tag, sig_u = sys.argv[1:4]
    res = dict(tag=tag, sig_u=sig_u, cut_nm=CUT, runs=[one(s, tag, sig_u) for s in sys.argv[4:]])
    for r in res['runs']:
        a, b = r['no_g'], r['g']
        print(f"{r['run']} frames {r['frames']}: base (p>=0.63, crop) {a['n_base']}; at {CUT:.2f} nm: no g {a['n_sel']} ({a['frac']:.3f}), with g {b['n_sel']} "
              f"({b['frac']:.3f}), change {100 * (b['n_sel'] / a['n_sel'] - 1):+.1f} %; in both {r['n_both']}; median sigma nm {a['sigma_med_nm']:.2f} -> {b['sigma_med_nm']:.2f}")
        if r['sim']:
            print(f"   GT: zcal(sel) {a['zcal_sel']:.3f} -> {b['zcal_sel']:.3f}; zcal(all p>=0.63) {a['zcal_all']:.3f} -> {b['zcal_all']:.3f}; RMS(sel) nm {a['rms_sel_nm']:.2f} -> {b['rms_sel_nm']:.2f}")
            print('   zcal by u (no g):', ' '.join(f'{k} {v:.3f}' for k, v in a['zcal_by_u'].items()))
            print('   zcal by u (g)   :', ' '.join(f'{k} {v:.3f}' for k, v in b['zcal_by_u'].items()))
    json.dump(res, open(outf, 'w'), indent=1)


if __name__ == '__main__':
    main()
