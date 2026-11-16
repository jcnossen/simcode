"""Real-data equivalent of the sweep densities: the ON-emitter rate per um of microtubule in the real 4Pi-SIMFLUX MT data
(network v10 tables, <work>/net/locs_MT_<k>_v10.npz) vs the simulated movies, for the imager-concentration translation.

Per data set: localizations of the network with the base filters (sigma_z < 25 nm, sigma_xy < 0.12 px, N > 300) per
6-frame window per um of tube. Tube length: x-y render at 40 nm, Gaussian blur 1 px, threshold at `--thr` x the 99th
percentile, skeleton (skimage), length = sum of the 8-connected skeleton links (1 or sqrt 2 px) / 2. The same estimator runs
on the simulated net_base localizations of every sweep movie and is calibrated against the true tube length there.
Then: ON emitters per frame per um of tube in the real data = real rate / (sim locs per ON emitter-frame, interpolated at
the matching rate), and the equivalent FOV density of the sweep's structure.
Output: out/fourpi_sim/sweep/realdens.json
Local tube density: skeleton length per 2 x 2 um block (blocks with > 0.25 um/um2), calibrated.
Usage: python analysis/fourpi_sim/sweep4pi_realdens.py --work <4Pi work folder> [--files 0 1 2]
"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from skimage.morphology import skeletonize
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path: sys.path.insert(0, REPO)
NM = 128.0
OUT = os.path.expanduser('~/simcode2/out/fourpi_sim/sweep')
TAGS = ['mtc_d01', 'mtc_d03', 'mtc_d05', 'mtc_d075', 'mtc_d15']


def tube_length(x_nm, y_nm, nwin, px=40.0, thr=0.15):
    x0, y0 = x_nm.min(), y_nm.min()
    nx, ny = int((x_nm.max() - x0) / px) + 1, int((y_nm.max() - y0) / px) + 1
    img = np.zeros((ny, nx)); np.add.at(img, (((y_nm - y0) / px).astype(int), ((x_nm - x0) / px).astype(int)), 1)
    b = gaussian_filter(img / nwin, 1.0)
    m = b > thr * np.percentile(b[b > 0], 99)
    sk = skeletonize(m)
    s = sk.astype(float)
    hv = convolve(s, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float), mode='constant') * s
    dg = convolve(s, np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], float), mode='constant') * s
    seg = (hv + np.sqrt(2) * dg) / 2 * px / 1000                              # tube length (um) per skeleton pixel
    L = seg.sum()
    # local tube density: length per 2 x 2 um block, blocks that contain tube
    nb = int(2000 / px); hb, wb = seg.shape[0] // nb, seg.shape[1] // nb
    blk = seg[:hb * nb, :wb * nb].reshape(hb, nb, wb, nb).sum((1, 3)) / 4.0
    blk = blk[blk > 0.25]
    return float(L), blk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--files', type=int, nargs='*', default=[0, 30, 60, 90]); ap.add_argument('--thr', type=float, default=0.15)
    ap.add_argument('--work', required=True, help='4Pi work folder')
    a = ap.parse_args(); P4 = a.work
    R = dict(thr=a.thr, sim={}, real={})
    cal = []
    for t in TAGS:
        f = f'{P4}/sweep4pi/{t}/locs_net_base.npz'
        if not os.path.exists(f): continue
        E = json.load(open(f'{P4}/sweep4pi/{t}/eval.json')); L = np.load(f)
        nwin = E['T'] // 6
        Lest, blk = tube_length(L['x'], L['y'], nwin, thr=a.thr)
        ntrue = E['tube_um_region']
        on_per_um = E['dens_region'] * E['region_um'] ** 2 / ntrue          # ON emitters per frame per um of tube
        rate = len(L['x']) / nwin / ntrue                                      # base locs per window per um of tube
        R['sim'][t] = dict(dens_fov=E['dens_fov'], dens_region=E['dens_region'], tube_true_um=ntrue, tube_est_um=Lest, block_tube_density_p50_p90=np.percentile(blk / (Lest / ntrue), [50, 90]).tolist(),
                           on_per_frame_per_um=on_per_um, locs_per_window_per_um=rate, locs_per_on=rate / on_per_um)
        cal.append(Lest / ntrue)
        print(t, R['sim'][t], flush=True)
    k = float(np.median(cal)); R['length_calibration'] = k
    sims = sorted(R['sim'].values(), key=lambda v: v['on_per_frame_per_um'])
    rr = np.array([v['locs_per_window_per_um'] for v in sims]); oo = np.array([v['on_per_frame_per_um'] for v in sims]); dd = np.array([v['dens_fov'] for v in sims])
    for i in a.files:
        f = f'{P4}/net/locs_MT_{i:03d}_v10.npz'
        if not os.path.exists(f): continue
        L = np.load(f)
        ok = (L['sz'] < 0.025) & (L['sx'] < 0.12) & (L['sy'] < 0.12) & (L['N'] > 300)
        nwin = int(L['w'].max() - L['w'].min() + 1)
        x, y = L['x'][ok] * NM, L['y'][ok] * NM
        Lest, blk = tube_length(x, y, nwin, thr=a.thr)
        Lc = Lest / k
        rate = ok.sum() / nwin / Lc
        on = float(np.interp(rate, rr, oo))                 # sim ON per frame per um at the same base-loc rate
        dens_eq = float(np.interp(on, oo, dd))
        R['real'][i] = dict(n_base=int(ok.sum()), windows=nwin, tube_est_um=Lest, tube_um=Lc, block_tube_density_p25_p50_p75_p90=np.percentile(blk / k, [25, 50, 75, 90]).tolist(),
                            locs_per_window_per_um=rate, on_per_frame_per_um=on, sweep_equivalent_dens_fov=dens_eq,
                            on_per_um2_local_p50_p90=(on * np.percentile(blk / k, [50, 90])).tolist())
        print('real', i, R['real'][i], flush=True)
    v = list(R['real'].values())
    R['real_mean'] = {k_: float(np.mean([q[k_] for q in v])) for k_ in ('on_per_frame_per_um', 'sweep_equivalent_dens_fov', 'locs_per_window_per_um')}
    R['real_mean']['block_tube_density_p25_p50_p75_p90'] = np.mean([q['block_tube_density_p25_p50_p75_p90'] for q in v], 0).tolist()
    R['real_mean']['on_per_um2_local_p50_p90'] = np.mean([q['on_per_um2_local_p50_p90'] for q in v], 0).tolist()
    print('real mean', R['real_mean'])
    json.dump(R, open(f'{OUT}/realdens.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
