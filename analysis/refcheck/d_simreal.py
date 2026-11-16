"""Referee check (d): why the simulated half-split FRC (Fig 2 sets: SIMFLUX 77.6 / SIMCODE 31.8 nm) is so much worse than the
real 1809_2 one (Fig 3: 19.1 / 17.5 nm). Numbers on the same footing for both:
  * field area, frames, localizations (FRC set), filament length in the field (sim: the ground truth's 74.9 um; real and, as a
    check of the estimator, sim: skeleton length of the SIMCODE render, 20 nm pixels, Gaussian 1 px, threshold at 3x the median
    of the non-zero pixels, skimage skeletonize x 1.12 for the diagonal steps) -> localizations per um of filament per FRC half;
  * revisits: simcode.evaluation.metrics.revisit P(another localization of the same set within 27 nm at window f + lag) for
    lags 6 ... 30000 frames; sim ground truth: events per labelling site and the fraction of sites seen in both FRC halves;
  * the sim half-split FRC with the TRUE positions of the same rows (sampling limit), first and sustained crossing, and the
    measured one under both rules; the real curves' sustained crossing.
usage: CUDA_VISIBLE_DEVICES=0 python analysis/refcheck/d_simreal.py
output: out/refcheck/d_simreal.json"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'analysis', 'papersim'), os.path.join(REPO, 'analysis', 'paper')]
import frc_gpu as FG
import metrics as M
from simcode.evaluation.metrics import revisit
from simcode.evaluation.frc import frc_halves

CACHE = os.path.join(REPO, 'out', 'refcheck', 'cache'); PX = 108.33
LAGS = (6, 12, 30, 60, 300, 3000, 30000)


def skel_len_um(x, y, lo, hi, pix_nm=20.0):
    from skimage.morphology import skeletonize
    from scipy.ndimage import gaussian_filter
    n = int((hi - lo) * PX / pix_nm)
    H = np.histogram2d(y, x, bins=n, range=[[lo, hi], [lo, hi]])[0]
    G = gaussian_filter(H, 1.0); thr = 3 * np.median(G[G > 0])
    S = skeletonize(G > thr)
    return float(S.sum() * pix_nm * 1.12 / 1000)



def main():
    res = dict(sim={}, real={})
    # ---- simulation (density 1, Fig 2 selections)
    name = 'Dexp1_v2psf_wsm'; md = os.path.join(REPO, 'out', 'papersim', name)
    g0 = json.load(open(os.path.join(md, 'gen.json'))); gt = np.load(os.path.join(md, 'gt.npz'))
    R = json.load(open(os.path.join(REPO, 'out', 'papersim', 'frcopt', f'{name}.json')))
    Z = np.load(os.path.join(REPO, 'out', 'papersim', 'frcopt', f'{name}_rows.npz'))
    W = g0['W']; nf = g0['nframes']; area = (W * PX / 1000) ** 2
    ev_id = gt['ev_id'].astype(int); t0 = gt['ev_t0']; npts = len(gt['pts'])
    ev_per_site = np.bincount(ev_id, minlength=npts)
    half = ((t0 // 300) % 2).astype(int)
    both = (np.bincount(ev_id[half == 0], minlength=npts) > 0) & (np.bincount(ev_id[half == 1], minlength=npts) > 0)
    L_gt = g0['wsm']['mt_length_um']
    res['sim'] = dict(field_um2=area, frames=nf, filament_um=L_gt, labels=int(npts), labels_per_nm=g0['wsm']['labels_per_nm'],
                      events=int(len(t0)), events_per_site_mean=float(ev_per_site.mean()),
                      frac_sites_never_on=float(np.mean(ev_per_site == 0)), frac_sites_in_both_halves=float(both.mean()), methods={})
    for k, ck, fam in (('sf', 'sf', 'sig'), ('v2', 'v2_sig_p', 'sig_p'), ('smlm', 'smlm', 'sig'), ('decode', 'decode', 'sig')):
        B = {c.split('__', 1)[1]: Z[c] for c in Z.files if c.split('__', 1)[0] == ck}
        o = R['methods'][k]['opt'][fam]['gt_sust']; m = B['key'] <= o['thr']
        x, y, f = B['x'][m], B['y'][m], B['f'][m]
        h, fq, cv = frc_halves(x, y, f, return_curve=True)
        t = np.isfinite(B['ex'][m]); xt, yt = x[t] - B['ex'][m][t], y[t] - B['ey'][m][t]
        h0, fq0, cv0 = frc_halves(xt, yt, f[t], return_curve=True)
        r = dict(n=int(m.sum()), hs_first_nm=float(h), hs_sust_nm=M.frc_sustained(fq, cv), true_pos_hs_first_nm=float(h0),
                 true_pos_hs_sust_nm=M.frc_sustained(fq0, cv0), locs_per_um=float(m.sum() / L_gt),
                 locs_per_um_per_half=float(m.sum() / L_gt / 2), locs_per_um2=float(m.sum() / area),
                 revisit=revisit(x, y, f, lags=[l for l in LAGS if l < nf], r=0.25))
        if k == 'v2': r['skeleton_len_um_check'] = skel_len_um(x, y, 0.0, float(W))
        res['sim']['methods'][k] = r
        print('sim', k, json.dumps(r), flush=True)
    # ---- real 1809_2 (Fig 3 sets)
    area_r = (110 * PX / 1000) ** 2; nf_r = 120000
    Lr = None
    for k in ('sf', 'v2', 'smlm', 'decode'):
        Zr = np.load(os.path.join(CACHE, f'1809_2_{k}.npz')); m = Zr['filt']
        x, y, f = Zr['xdc'][m], Zr['ydc'][m], Zr['f'][m]
        fr = FG.FRC(x, y); h, fq, cv = fr.halves(x, y, f, curve=True)
        if k == 'v2': Lr = skel_len_um(x, y, 5.0, 115.0)
        r = dict(n=int(m.sum()), hs_first_nm=float(h), hs_sust_nm=M.frc_sustained(fq, cv), locs_per_um2=float(m.sum() / area_r),
                 revisit=revisit(x, y, f, lags=LAGS, r=0.25))
        res['real']['methods'] = res['real'].get('methods', {}); res['real']['methods'][k] = r
        print('real', k, json.dumps(r), flush=True)
    res['real'].update(field_um2=area_r, frames=nf_r, filament_um_skeleton=Lr)
    for k, r in res['real']['methods'].items():
        r['locs_per_um'] = r['n'] / Lr; r['locs_per_um_per_half'] = r['n'] / Lr / 2
    json.dump(res, open(os.path.join(REPO, 'out', 'refcheck', 'd_simreal.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'methods'} for k, v in res.items()}))


if __name__ == '__main__' and not os.environ.get('CROPS'):
    main()


def crops_real(out_fn=os.path.join(REPO, 'out', 'refcheck', 'd_simreal.json')):
    """real 1809_2 cut to the simulation's acquisition: first 60 000 frames and 40 x 40 px tiles (3 x 3 tiles at offsets 5, 40,
    75 px; the tiles with structure) -> half-split FRC (300-frame halves) and localizations per tile, vs the full field / movie"""
    from simcode.evaluation.frc import frc_halves as fh
    res = json.load(open(out_fn)); res['real_cut'] = {}
    for k in ('sf', 'v2', 'smlm', 'decode'):
        Zr = np.load(os.path.join(CACHE, f'1809_2_{k}.npz')); m = Zr['filt']
        x, y, f = Zr['xdc'][m], Zr['ydc'][m], Zr['f'][m]
        o = dict(full=float(fh(x, y, f)))
        t = f < 60000; o['first60k'] = float(fh(x[t], y[t], f[t]))
        tiles = []
        for x0 in (5, 40, 75):
            for y0 in (5, 40, 75):
                c = t & (x >= x0) & (x < x0 + 40) & (y >= y0) & (y < y0 + 40)
                if c.sum() < 2000: continue
                h, fq, cv = fh(x[c], y[c], f[c], return_curve=True)
                tiles.append(dict(x0=x0, y0=y0, n=int(c.sum()), first=float(h), sust=M.frc_sustained(fq, cv)))
        v = np.array([q['first'] for q in tiles])
        o['tiles_60k'] = tiles; o['tiles_60k_median_first'] = float(np.nanmedian(v)); o['tiles_60k_range'] = [float(np.nanmin(v)), float(np.nanmax(v))]
        o['tiles_60k_median_n'] = float(np.median([q['n'] for q in tiles]))
        res['real_cut'][k] = o
        print('real cut', k, json.dumps({kk: vv for kk, vv in o.items() if kk != 'tiles_60k'}), flush=True)
    json.dump(res, open(out_fn, 'w'), indent=1, default=float)


if __name__ == '__main__' and os.environ.get('CROPS'):
    crops_real()
