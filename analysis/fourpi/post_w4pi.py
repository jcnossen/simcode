"""Post-processing of the learned-window 4Pi pipeline tables (real_w4pi.py): merge the files of a movie, selection,
lateral drift by DME v2 (analysis/baselines/dme_drift.py, subprocess), sample z drift (structure-based estimator of the
classical worker, analysis/fourpi_classical/fig_cl.zdrift_struct, imported read-only; per 500-window block), sigma
multipliers from adjacent-window pairs (DAFL; precision only), render, summary json.
Global frame = (file - first file) * 18000 + 6 w (as post4pi.py and the classical tables).
Columns of net_<cell>_<head>_all.npz: x, y (px, raw), xd, yd (drift corrected), z (um, raw), zd (um, z drift removed),
sx, sy, sz (reported), sx_cal, sy_cal, sz_cal, N, good, sel (>= 1000 e and |z| <= 0.3 um), kin, chi2, dof, frame, w, fidx, on (all True), drift [F,2], zdrift.
Usage: python analysis/fourpi/post_w4pi.py --cell NPC --head v10
"""
import argparse, glob, json, os, re, subprocess, sys
import numpy as np
REPO = '/home/jc/simcode2'
OUT = f'{REPO}/out/4pi/net'
PX = 128.0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='NPC'); p.add_argument('--head', default='v10')
    p.add_argument('--sz_max', type=float, default=0.025); p.add_argument('--sxy_max', type=float, default=0.12)
    p.add_argument('--files', type=int, nargs='*')
    p.add_argument('--ksum', type=int, default=1, help='K-summed tables: the global frame of a summed window is the centre of its K source windows (stride --kstride)')
    p.add_argument('--kstride', type=int, default=25)
    a = p.parse_args()
    if REPO not in sys.path: sys.path.insert(0, REPO)
    from simcode2.util.proctitle import auto; auto('4pi-post-w')
    fs = sorted(glob.glob(f'{OUT}/locs_{a.cell}_[0-9][0-9][0-9]_{a.head}.npz'))
    fidx = lambda f: int(re.search(rf'locs_{a.cell}_(\d\d\d)_', os.path.basename(f)).group(1))     # the head tag may contain underscores
    if a.files: fs = [f for f in fs if fidx(f) in a.files]
    first = min(fidx(f) for f in glob.glob(f'{OUT}/locs_{a.cell}_[0-9][0-9][0-9]_{a.head}.npz'))
    keys = ('x', 'y', 'z', 'sx', 'sy', 'sz', 'N', 'good', 'kin', 'chi2', 'dof', 'w', 'nn', 'Tsum', 'Phi0', 'zprior', 'pdet', 'cs')
    R = {k: [] for k in keys + ('fidx', 'frame')}
    for f in fs:
        D = np.load(f); fi = fidx(f)
        for k in keys: R[k].append(D[k])
        wr = D['w'] if a.ksum == 1 else (D['w'] // a.kstride) * (a.ksum * a.kstride) + D['w'] % a.kstride + (a.ksum - 1) * a.kstride // 2
        R['fidx'].append(np.full(len(D['x']), fi)); R['frame'].append((fi - first) * 18000 + 6 * wr)
    R = {k: np.concatenate(v) for k, v in R.items()}
    n_all = len(R['x'])
    R['chi2_good'] = R['good'].copy()
    good = R['good'] & (R['sz'] < a.sz_max) & (R['sx'] < a.sxy_max) & (R['sy'] < a.sxy_max) & np.isfinite(R['z']) & (R['N'] > 300)
    R['good'] = good; R['on'] = np.ones(n_all, bool)
    F = int(R['frame'].max()) + 6
    tag = f'net_{a.cell}_{a.head}' + (f'_files{min(a.files)}-{max(a.files)}' if a.files else '')
    fn = f'{OUT}/{tag}_dmein.npz'
    np.savez(fn, frame=R['frame'][good], x=R['x'][good], y=R['y'][good], sx=R['sx'][good], sy=R['sy'][good])
    r = subprocess.run([sys.executable, f'{REPO}/analysis/baselines/dme_drift.py', fn, f'{OUT}/{tag}_drift.npy', str(F)],
                       capture_output=True, text=True, env=dict(os.environ))
    print(r.stdout[-800:], r.stderr[-800:])
    d = np.load(f'{OUT}/{tag}_drift.npy'); os.remove(fn)
    fi_ = np.clip(R['frame'] + 3, 0, F - 1)
    R['xd'] = R['x'] - d[fi_, 0]; R['yd'] = R['y'] - d[fi_, 1]
    # sample z drift (structure based), from the good rows; global window index
    sys.path.insert(0, f'{REPO}/analysis/fourpi_classical')
    from fig_cl import zdrift_struct
    wg = R['frame'] // 6
    zd = zdrift_struct(R['xd'][good] * PX, R['yd'][good] * PX, R['z'][good] * 1000, wg[good])
    R['zd'] = R['z'] - zd[np.clip(wg // 500, 0, len(zd) - 1)] / 1000.0
    # sigma multipliers from adjacent-window pairs (per axis and photon bin)
    from cmp_real4pi import pairs
    idx = np.nonzero(good)[0]
    I, J = pairs(wg[idx], R['xd'][idx], R['yd'][idx]); I, J = idx[I], idx[J]
    edges = (300, 1000, 2000, 4000, 8000, 1e9); cal = []
    for c in ('sx', 'sy', 'sz'): R[c + '_cal'] = R[c].copy()
    for lo, hi in zip(edges[:-1], edges[1:]):
        Np = 0.5 * (R['N'][I] + R['N'][J]); s = (Np >= lo) & (Np < hi); row = dict(lo=lo, hi=hi, n_pairs=int(s.sum()))
        for col, v, sc in (('sx', 'xd', PX), ('sy', 'yd', PX), ('sz', 'zd', 1000.0)):
            if s.sum() < 200: row[col] = None; continue
            dv = (R[v][I[s]] - R[v][J[s]]) * sc
            emp = 1.4826 * np.median(np.abs(dv - np.median(dv))) / np.sqrt(2)
            rep = np.sqrt(np.median(0.5 * (R[col][I[s]] ** 2 + R[col][J[s]] ** 2))) * sc
            row[col] = float(emp / rep); row[col + '_dafl_nm'] = float(emp)
            rows = (R['N'] >= lo) & (R['N'] < hi); R[col + '_cal'][rows] = R[col][rows] * row[col]
        cal.append(row)
    # 'sel': the rows inside the validated range (>= 1000 e, |z| <= 0.3 um after the z-drift correction): below 1000 e the z has 24 %
    # adjacent-window fringe hops, and beyond 0.3 um the rows are defocused emitters outside the range where PSF and head are validated
    R['sel'] = good & (R['N'] >= 1000) & (np.abs(R['zd'] - np.median(R['zd'][good & (R['N'] >= 2000)])) <= 0.3)
    if __import__('shutil').disk_usage(OUT).free < 10e9:
        print('WARNING: < 10 GB free; table not written'); return
    np.savez(f'{OUT}/{tag}_all.npz', **R, drift=d, zdrift=zd)
    # render
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from scipy.ndimage import gaussian_filter
    x, y, z = R['xd'][good] * PX, R['yd'][good] * PX, R['zd'][good] * 1000
    sr = 8.0
    x0, y0 = np.percentile(x, 0.5), np.percentile(y, 0.5); x1, y1 = np.percentile(x, 99.5), np.percentile(y, 99.5)
    W, H = int((x1 - x0) / sr) + 1, int((y1 - y0) / sr) + 1
    ix = np.clip(((x - x0) / sr).astype(int), 0, W - 1); iy = np.clip(((y - y0) / sr).astype(int), 0, H - 1)
    zl, zh = np.percentile(z, [2, 98])
    cnt = np.zeros((H, W)); zs = np.zeros((H, W)); np.add.at(cnt, (iy, ix), 1); np.add.at(zs, (iy, ix), z)
    c = gaussian_filter(cnt, 0.7); zz = gaussian_filter(zs, 0.7) / np.maximum(c, 1e-9)
    v = np.clip(c / np.percentile(c[c > 0], 99.5), 0, 1)
    hsv = np.stack([np.clip((zz - zl) / (zh - zl), 0, 1) * 0.75, np.ones_like(v), v], -1)
    plt.figure(figsize=(12, 12 * H / W)); plt.imshow(hsv_to_rgb(hsv)); plt.axis('off')
    plt.title(f'{a.cell} 4Pi-SIMFLUX network pipeline (head {a.head}), {good.sum()} localizations, z {zl:.0f} to {zh:.0f} nm (hue), {sr:.0f} nm pixels', fontsize=9)
    plt.savefig(f'{OUT}/{tag}_render.png', dpi=150, bbox_inches='tight'); plt.close()
    summ = dict(cell=a.cell, head=a.head, files=len(fs), n_all=int(n_all), n_good=int(good.sum()), n_good_1000=int((good & (R['N'] >= 1000)).sum()), n_sel=int(R['sel'].sum()),
                kin_frac=float(R['kin'].mean()), med_sx_nm=float(np.median(R['sx'][good]) * PX), med_sy_nm=float(np.median(R['sy'][good]) * PX),
                med_sz_nm=float(np.median(R['sz'][good]) * 1000), med_N=float(np.median(R['N'][good])), z_p2_p98_nm=[float(zl), float(zh)],
                drift_ptp_nm=(np.ptp(d, 0) * PX).tolist(), zdrift_ptp_nm=float(np.ptp(zd)), sigma_cal=cal)
    json.dump(summ, open(f'{OUT}/{tag}_summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'sigma_cal'}))
    for row in cal: print(row)


if __name__ == '__main__':
    main()
