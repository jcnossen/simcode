"""SIMCODE v2 pixel-bias numbers for the SI pixel-bias figure, in the format of out/decode_checks/pixlock.json (pixlock.py):
  R[movie][method] = dict(n, n_used, n_matched, x, y (pblib.ustats of the rows' sub-pixel coordinates: c1_abs, c1_re, noise, A_nm, ks_D ...),
                          x_matched, y_matched, gt_matched_x, gt_matched_y (ustats), bias_x, bias_y (sim.bias_curve: A_nm, A_se_nm, mu_nm ...),
                          bias_x_ph, bias_y_ph ({T1, T2, T3}: photon terciles of the method's own matched rows), ph_terciles)
Rows inside [0.5, W - 0.5), window start + 6 <= nframes, as pixlock.py. Methods:
  v2sig        SIMCODE v2 production (locs_p2.hdf5, the paper's selection; = pixlock.json 'v2sig')
  v2sig_pagg   the same rows' table but selected on the aggregated detection probability p_agg >= 0.63 (stage_solves.PAggRecorder)
               instead of p >= 0.63 (sigma <= 12 nm with the production sig_u, crop) -- the selection fix
  v2_nop       the paper's selection without the p condition (sigma <= 12 nm, crop)
  v2_stage:<stage>   the stages of the production rows (stage_bias.STAGES; same rows as v2sig, positions of that stage)
  v2long_<ckpt>, v2long_<ckpt>_pagg   longer-trained pass-2 emitter network (ckpt_eval re-runs; selection without sig_u, see ckpt_eval)
+ R[movie]['_density'] = the movie's emitter density label. usage: python analysis/pixel_bias_simcode/export_json.py [--long name=dir ...]
-> <out>/pixel_bias_simcode.json"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stage_bias as SB
B, PS, M = SB.B, SB.PS, SB.M
from simcode import table as SP
from simcode.dataset import Dataset

DENS = {'iso': 'isolated (4 emitters / 40 x 40 px window)', 'D0.5': '0.5 um^-2', 'D1': '1 um^-2', 'D2': '2 um^-2', 'D4': '4 um^-2', 'Dexp1': 'experimental-photon movie, 1 um^-2'}


def entry(x, y, f, ph, gt, nf, W, iso=False):
    m = np.isfinite(x) & np.isfinite(y) & (x >= 0.5) & (x < W - 0.5) & (y >= 0.5) & (y < W - 0.5) & (f + 6 <= nf)
    idx = np.nonzero(m)[0]
    x, y, f, ph = x[idx], y[idx], f[idx], ph[idx]
    tx, ty, _ = PS.gt_rows(gt, nf, f, x, y)
    g = np.isfinite(tx)
    if iso: g &= (f % 12 == 0)
    r = dict(n=int(m.sum()), n_used=int(len(idx)), n_matched=int(g.sum()), x=B.ustats(B.sub(x)), y=B.ustats(B.sub(y)),
             x_matched=B.ustats(B.sub(x[g])), y_matched=B.ustats(B.sub(y[g])), gt_matched_x=B.ustats(B.sub(tx[g])), gt_matched_y=B.ustats(B.sub(ty[g])))
    q = np.quantile(ph[g], [1 / 3, 2 / 3])
    for ax, v, t in (('x', x, tx), ('y', y, ty)):
        e = v[g] - t[g]; fg = B.sub(t[g]); pg = ph[g]
        r['bias_' + ax] = PS.bias_curve(e, fg)
        r['bias_' + ax + '_ph'] = {f'T{i + 1}': PS.bias_curve(e[b], fg[b]) for i, b in enumerate([pg < q[0], (pg >= q[0]) & (pg < q[1]), pg >= q[1]])}
    r['ph_terciles'] = [float(v) for v in q]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movies', default=','.join(SB.MOV))
    ap.add_argument('--long', default='', help='name=stages_dir,... of longer-trained networks (ckpt re-runs)')
    a = ap.parse_args()
    R = {}
    for mv in a.movies.split(','):
        md = os.path.join(SB.REPO, SB.MOV[mv]); d = os.path.join(md, 'full', 'v2_m1809v2')
        gt = dict(np.load(os.path.join(md, 'gt.npz'))); nf = int(gt['nframes']); W = int(gt['W']); iso = mv == 'iso'
        Z = np.load(os.path.join(SB.OUT, 'stages', mv + '.npz'))
        S = SB.load_run(d, Z)
        T = Dataset.from_run(d, pass_=2, info=False).columns()
        f = S['w2']; ph = S['ph']
        R[mv] = {'_density': DENS[mv]}
        sels = dict(v2sig=S['sel2'], v2_nop=SP.select(T, min_score=None, crop=S['crop']))
        if 'p2_p_agg' in Z.files: sels['v2sig_pagg'] = sels['v2_nop'] & (np.nan_to_num(Z['p2_p_agg']) >= SP.P_GOOD)
        for k, m in sels.items():
            R[mv][k] = entry(T['x'][m], T['y'][m], f[m], ph[m], gt, nf, W, iso)
        for st in SB.STAGES:
            x, y = S['st'][st]; m = S['sel2']
            R[mv]['v2_stage:' + st] = entry(x[m], y[m], f[m], ph[m], gt, nf, W, iso)
        for nv in [v for v in a.long.split(',') if v]:
            name, dd = nv.split('=', 1); zf = os.path.join(dd, mv + '.npz')
            if not os.path.exists(zf): continue
            Zl = np.load(zf)
            x, y = Zl['p2_r'][:, 0].astype(np.float64), Zl['p2_r'][:, 1].astype(np.float64)
            sig = np.maximum(Zl['p2_sx_cal'], Zl['p2_sy_cal']) * SP.PX_NM_DEFAULT
            base = SP.in_crop(x, y, S['crop']) & (sig <= SP.SIGMA_MAX_NM)
            fl = Zl['p2_wstart'].astype(np.int64); pl = Zl['p2_phot'].astype(np.float64)
            for sfx, pv in (('', Zl['p2_p']), ('_pagg', np.nan_to_num(Zl['p2_p_agg']))):
                m = base & (pv >= SP.P_GOOD)
                R[mv][name + sfx] = entry(x[m], y[m], fl[m], pl[m], gt, nf, W, iso)
        for k, r in R[mv].items():
            if k.startswith('_'): continue
            print(f"{mv:6s} {k:22s} n {r['n']:7d} |c1| {r['x']['c1_abs']:.4f}/{r['y']['c1_abs']:.4f} A {r['bias_x']['A_nm']:+.2f}+-{r['bias_x']['A_se_nm']:.2f}/"
                  f"{r['bias_y']['A_nm']:+.2f}+-{r['bias_y']['A_se_nm']:.2f}", flush=True)
    R['_meta'] = dict(format='out/decode_checks/pixlock.json (pixlock.py)', script='analysis/pixel_bias_simcode/export_json.py',
                      runs='<movie>/full/v2_m1809v2 (model m1809, PSF v2); iso = out/decode_checks/iso_v2psf (true pattern)',
                      A_sign='A > 0 = pull to the pixel centre', methods=__doc__)
    json.dump(R, open(os.path.join(SB.OUT, 'pixel_bias_simcode.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
