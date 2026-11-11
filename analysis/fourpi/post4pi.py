"""Post-processing of the 4Pi-SIMFLUX network localizations (analysis/fourpi/solve_real4pi.py): merge all files of a
movie, filter, lateral drift by DME v2 (analysis/baselines/dme_drift.py, subprocess), renders (xy coloured by z, xz of a
strip), summary json.
Global frame = (file - first file) * 18000 + 6 w (the inter-file gaps are ignored, as in the classical pass).
Axial drift: sample z drift is common to the astigmatic prior and the 4Pi phase; the per-block phase offset d(t) of
the field absorbs the interferometer drift only (it is referenced to the prior z), so z needs no separate correction.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/post4pi.py --cell NPC --model v2
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np
REPO = '/home/jc/simcode2'
OUT = f'{REPO}/out/4pi/net'
PX = 128.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='NPC'); p.add_argument('--model', default='v2')
    p.add_argument('--sz_max', type=float, default=0.025); p.add_argument('--sxy_max', type=float, default=0.12)
    p.add_argument('--files', type=int, nargs='*')
    a = p.parse_args()
    fs = sorted(glob.glob(f'{OUT}/locs_{a.cell}_[0-9][0-9][0-9]_{a.model}.npz'))
    if a.files: fs = [f for f in fs if int(f.split('_')[-2]) in a.files]
    first = min(int(f.split('_')[-2]) for f in fs)
    keys = ('x', 'y', 'z', 'sx', 'sy', 'sz', 'N', 'on', 'chi2_on', 'chi2_free', 'w', 'nn', 'Tsum', 'phi', 'Phi0', 'zprior')
    R = {k: [] for k in keys + ('fidx', 'frame')}
    for f in fs:
        D = np.load(f); fi = int(f.split('_')[-2])
        for k in keys: R[k].append(D[k])
        R['fidx'].append(np.full(len(D['x']), fi)); R['frame'].append((fi - first) * 18000 + 6 * D['w'])
    R = {k: np.concatenate(v) for k, v in R.items()}
    n_all = len(R['x'])
    good = (R['sz'] < a.sz_max) & (R['sx'] < a.sxy_max) & (R['sy'] < a.sxy_max) & np.isfinite(R['z']) & (R['N'] > 300)
    F = int(R['frame'].max()) + 6
    tag = f'net_{a.cell}_{a.model}'
    fn = f'{OUT}/{tag}_dmein.npz'
    sel = good & R['on']
    np.savez(fn, frame=R['frame'][sel], x=R['x'][sel], y=R['y'][sel], sx=R['sx'][sel], sy=R['sy'][sel])
    r = subprocess.run([sys.executable, f'{REPO}/analysis/baselines/dme_drift.py', fn, f'{OUT}/{tag}_drift.npy', str(F)],
                       capture_output=True, text=True, env=dict(os.environ))
    print(r.stdout[-1500:], r.stderr[-1500:])
    d = np.load(f'{OUT}/{tag}_drift.npy')
    fi_ = np.clip(R['frame'] + 3, 0, F - 1)
    R['xd'] = R['x'] - d[fi_, 0]; R['yd'] = R['y'] - d[fi_, 1]
    R['good'] = good
    np.savez(f'{OUT}/{tag}_all.npz', **R, drift=d)
    # renders
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    g = good
    x, y, z = R['xd'][g] * PX, R['yd'][g] * PX, R['z'][g] * 1000
    sr = 8.0
    x0, y0 = np.percentile(x, 0.5), np.percentile(y, 0.5); x1, y1 = np.percentile(x, 99.5), np.percentile(y, 99.5)
    W, H = int((x1 - x0) / sr) + 1, int((y1 - y0) / sr) + 1
    ix = np.clip(((x - x0) / sr).astype(int), 0, W - 1); iy = np.clip(((y - y0) / sr).astype(int), 0, H - 1)
    zl, zh = np.percentile(z, [2, 98])
    cnt = np.zeros((H, W)); zs = np.zeros((H, W))
    np.add.at(cnt, (iy, ix), 1); np.add.at(zs, (iy, ix), z)
    from scipy.ndimage import gaussian_filter
    c = gaussian_filter(cnt, 0.7); zz = gaussian_filter(zs, 0.7) / np.maximum(c, 1e-9)
    v = np.clip(c / np.percentile(c[c > 0], 99.5), 0, 1)
    hsv = np.stack([np.clip((zz - zl) / (zh - zl), 0, 1) * 0.75, np.ones_like(v), v], -1)
    plt.figure(figsize=(12, 12 * H / W)); plt.imshow(hsv_to_rgb(hsv)); plt.axis('off')
    plt.title(f'{a.cell} 4Pi-SIMFLUX (network), {g.sum()} locs, z {zl:.0f}..{zh:.0f} nm (hue), {sr:.0f} nm px')
    plt.savefig(f'{OUT}/{tag}_render.png', dpi=150, bbox_inches='tight'); plt.close()
    summ = dict(cell=a.cell, model=a.model, files=len(fs), n_all=int(n_all), n_good=int(g.sum()), kin_frac=float(R['on'].mean()),
                med_sx_nm=float(np.median(R['sx'][g & R['on']]) * PX), med_sy_nm=float(np.median(R['sy'][g & R['on']]) * PX),
                med_sz_nm=float(np.median(R['sz'][g]) * 1000), med_N=float(np.median(R['N'][g])),
                z_p2_p98_nm=[float(zl), float(zh)], drift_ptp_nm=(np.ptp(d, 0) * PX).tolist())
    json.dump(summ, open(f'{OUT}/{tag}_summary.json', 'w'), indent=1)
    print(summ)


if __name__ == '__main__':
    main()
