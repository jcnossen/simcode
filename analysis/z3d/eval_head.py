"""z3d: pass-1 heads (2D and 3D checkpoints) on the same fixed simulated batches of the training generator (m1809 recipe:
paper photons + bright tail, pairs, dens / med of the paper models), detector candidates matched to the GT (< 1 px, p >= 0.5).
Two z distributions: 'real' (the presets' lognormal, median 50-55 nm) and 'train' (+ uniform 0-0.3 um share, hd/data.Z_UNIF).
Metrics: x, y prior RMSE (nm) and z-score RMS, intensity NLL, modulation slope; z (3D heads): RMSE (nm) of the prior z and
of z | true x, y vs the GT spread, z-score RMS, by photon bin and z bin.
usage: python analysis/z3d/eval_head.py --heads a.pt,b.pt --det det_m1809.pt --calib <psf json> [--nb 40] [--out x.json]"""
import argparse, json
import numpy as np, torch
from simcode import config
from simcode.hd import data as D
from simcode.hd import sc as SCM
from simcode.hd.sc4 import match_gt, targets, nbr_map, nbr_patches, nll_full, NBR_PMIN, BRIGHT_TAIL
from simcode.hd.sc import PSFCol, patches, mod_slope
from simcode.det.model import select_subpixel

ap = argparse.ArgumentParser()
ap.add_argument('--heads', required=True); ap.add_argument('--det', required=True); ap.add_argument('--calib', required=True)
ap.add_argument('--nb', type=int, default=40); ap.add_argument('--seed', type=int, default=777); ap.add_argument('--out', default='')
a = ap.parse_args()
dev = 'cuda'
det, _ = SCM.load_det_front(a.det)
heads = {h: SCM.load(h) for h in a.heads.split(',')}
gen = D.Gen(64, 64, 6, seed=a.seed, calib=[config.resolve(a.calib)])
sim = gen.rsm[0]['0.5nM'].sim; col = PSFCol(sim); PX = sim.c['pixelsize_um'] * 1000
RES = {}
for zmode in ('real', 'train'):
    for rs in gen.all_rs(): rs.p['z_unif'] = D.Z_UNIF if zmode == 'train' else 0.0
    torch.manual_seed(a.seed)
    acc = {h: {k: [] for k in ('ex', 'ey', 'sx', 'sy', 'nll', 'I', 'Ig', 'z', 'zc', 'sz', 'szm', 'tz', 'ph', 'nn')} for h in heads}
    for it in range(a.nb):
        with torch.no_grad():
            o = gen.batch(16, dens=(0.0005, 0.04), med_photons=(40, 2500), pair_frac=0.25, pair_curr=True, paper=True, tau_r=(15, 80),
                          bright_tail=BRIGHT_TAIL)
            fr = o['frames']
            with torch.autocast('cuda', dtype=torch.bfloat16):
                dd = det(fr)
            lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
            c = select_subpixel(lgm, dd['dxy'].float(), 640, p_min=NBR_PMIN, radius=0.7)
            kp = torch.isfinite(c['logit'])
            nbr = nbr_map(lgm, *(c[k][kp] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit')))
            c = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))
            m = match_gt(o, c)
            if m is None: continue
            t = targets(o, m, col)
            off = torch.stack([m['x'] - m['ix'].float(), m['y'] - m['iy'].float()], 1)
            pt = patches(fr, m['b'], m['iy'], m['ix']); psf = col(off)
            nbp = nbr_patches(nbr, m['b'], m['iy'], m['ix'], m['x'], m['y'], torch.sigmoid(lgm[m['b'], m['iy'], m['ix']]))
            tz = D.z_target_px(o['gt']['z'][m['g']], sim)
            for hn, h in heads.items():
                out = h(pt, None, off, psf, nbp)
                x = m['x'] + out['dxy'][:, 0]; y = m['y'] + out['dxy'][:, 1]
                nll, _ = nll_full(t['Ig'], out['I'].float(), out['L'].float())
                bz = out['bz']; sx, sy = out['lsxy'][:, 0].exp(), out['lsxy'][:, 1].exp(); sz = out['lsz'].exp()
                A = acc[hn]
                for k, v in (('ex', x - t['tx']), ('ey', y - t['ty']), ('sx', sx), ('sy', sy), ('nll', nll), ('I', out['I'].float()), ('Ig', t['Ig']),
                             ('z', out['z']), ('zc', out['z'] + bz[:, 0] * (t['tx'] - x) + bz[:, 1] * (t['ty'] - y)), ('sz', sz),
                             ('szm', (sz ** 2 + (bz[:, 0] * sx) ** 2 + (bz[:, 1] * sy) ** 2).sqrt()), ('tz', tz), ('ph', t['Ig'].sum(1)), ('nn', m['nn'])):
                    A[k].append(v.float())
    for hn in heads:
        A = {k: torch.cat(v) for k, v in acc[hn].items()}
        R = {}
        bins = {'all': A['ph'] >= 0, 'ph<500': A['ph'] < 500, 'ph500-2000': (A['ph'] >= 500) & (A['ph'] < 2000), 'ph>=2000': A['ph'] >= 2000,
                'nn<2.5': A['nn'] < 2.5}
        zum = A['tz'] * sim.c['pixelsize_um'] + sim.focus
        for lo, hi in ((0, .05), (.05, .1), (.1, .15), (.15, .2), (.2, .3)):
            bins[f'z{lo:.2f}-{hi:.2f}'] = (zum >= lo) & (zum < hi)
        for bn, mk in bins.items():
            if mk.sum() < 30: continue
            r = dict(n=int(mk.sum()),
                     xy_rmse_nm=float(torch.cat([A['ex'][mk], A['ey'][mk]]).pow(2).mean().sqrt() * PX),
                     xy_zrms=float(torch.cat([A['ex'][mk] / A['sx'][mk], A['ey'][mk] / A['sy'][mk]]).pow(2).mean().sqrt()),
                     nll_I=float(A['nll'][mk].mean()), slope=mod_slope(A['I'][mk], A['Ig'][mk]),
                     z_gt_sd_nm=float(A['tz'][mk].std() * PX), z_rmse_nm=float((A['z'][mk] - A['tz'][mk]).pow(2).mean().sqrt() * PX),
                     zc_rmse_nm=float((A['zc'][mk] - A['tz'][mk]).pow(2).mean().sqrt() * PX),
                     z_bias_nm=float((A['z'][mk] - A['tz'][mk]).mean() * PX),
                     z_zrms=float(((A['z'][mk] - A['tz'][mk]) / A['szm'][mk]).pow(2).mean().sqrt()),
                     z_sd_med_nm=float(A['szm'][mk].median() * PX))
            R[bn] = r
        RES[f'{zmode}|{hn}'] = R
        print(zmode, hn)
        for bn, r in R.items():
            print(f"  {bn:12s} n {r['n']:6d}  xy {r['xy_rmse_nm']:6.2f} nm z-rms {r['xy_zrms']:.3f}  nll_I {r['nll_I']:.3f} slope {r['slope']:.3f}"
                  f" | z gt sd {r['z_gt_sd_nm']:5.1f} rmse {r['z_rmse_nm']:5.1f} (|xy {r['zc_rmse_nm']:5.1f}) bias {r['z_bias_nm']:+5.1f}"
                  f" z-rms {r['z_zrms']:.2f} sd_med {r['z_sd_med_nm']:5.1f}")
if a.out: json.dump(RES, open(a.out, 'w'), indent=1)
