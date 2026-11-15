"""Densest-region emitter density of the real 4Pi-SIMFLUX MT movie, for comparison with the density sweep (si_4pi_sim).

Real data: network v10 per-file tables out/4pi/net/locs_MT_<k>_v10.npz (local disk; no raw frames), base filters as the sweep's
net_base (sigma_z < 25 nm, sigma_x, sigma_y < 0.12 px, N > 300). Per file (3000 six-frame windows):
  * base localizations per window, histogrammed on 2 px (256 nm) bins over the lit field (x 55..305 px, y 0..200 px);
  * sliding square windows (all positions on the 2 px grid, fully inside the field): 20 x 20 um (156 px) and the simulated
    field size 7.68 x 7.68 um (60 px); the densest window by base-loc rate;
  * ON emitters per um^2 per frame = base locs per window per um^2 / r, with r = the sim's base localizations per ON emitter-frame
    at the same areal rate (sweep4pi eval: net_base locs per window per um^2 of the evaluation region / dens_region; interpolated;
    r = 1.03-1.14). The sweep densities (dens_fov) are ON emitters per um^2 per frame over the simulated 7.68 um field.
  * ON emitters per frame per um of tube: skeleton tube length in the window (sweep4pi_realdens.tube_length recipe, 42.7 nm render,
    calibrated by realdens.json length_calibration), same r.
The window positions: per file its own densest window, and one fixed window (densest on the mean map over all files) evaluated in
every file (time trend / spread).
Output: out/fourpi_sim/sweep/densregion.json
Usage: python analysis/fourpi_sim/sweep4pi_densregion.py [--files 0 10 ... 99]
"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from skimage.morphology import skeletonize
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
NM = 128.0
OUT = os.path.expanduser('~/simcode2/out/fourpi_sim/sweep')
TAGS = ['mtc_d01', 'mtc_d03', 'mtc_d05', 'mtc_d075', 'mtc_d15']
X0, X1, Y0, Y1, B = 55, 305, 0, 200, 2                     # lit field (px) and bin (px)
WIN = {'20um': 156, 'sim_field_7.68um': 60}                 # window sizes (px): 20 um = 156.25 px


def seg_map(x_px, y_px, nwin, px_nm=B * NM / 6, thr=0.15):
    """skeleton tube length (um) per 42.7 nm pixel (6 per bin) on the lit field (sweep4pi_realdens.tube_length recipe, there 40 nm)"""
    nx, ny = int((X1 - X0) * NM / px_nm), int((Y1 - Y0) * NM / px_nm)
    ix = ((x_px - X0) * NM / px_nm).astype(int); iy = ((y_px - Y0) * NM / px_nm).astype(int)
    k = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    img = np.zeros((ny, nx)); np.add.at(img, (iy[k], ix[k]), 1)
    b = gaussian_filter(img / nwin, 1.0)
    m = b > thr * np.percentile(b[b > 0], 99)
    s = skeletonize(m).astype(float)
    hv = convolve(s, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float), mode='constant') * s
    dg = convolve(s, np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], float), mode='constant') * s
    return (hv + np.sqrt(2) * dg) / 2 * px_nm / 1000


def box(a, n):
    """sums over all n x n windows of a 2D array (valid positions)"""
    c = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--files', type=int, nargs='*', default=list(range(0, 100, 10)) + [99]); a = ap.parse_args()
    RD = json.load(open(f'{OUT}/realdens.json')); kcal = RD['length_calibration']
    # sim: base locs per window per um^2 (region) vs ON per um^2 per frame (region)
    sim = []
    for t in TAGS:
        E = json.load(open(f'{P4}/sweep4pi/{t}/eval.json')); s = RD['sim'][t]
        rate_area = s['locs_per_window_per_um'] * E['tube_um_region'] / E['region_um'] ** 2
        sim.append((rate_area, E['dens_region'], E['dens_fov'], s['on_per_frame_per_um']))
    sim = np.array(sorted(sim)); r_of = lambda rate: float(np.interp(rate, sim[:, 0], sim[:, 0] / sim[:, 1]))
    nbx, nby = (X1 - X0) // B, (Y1 - Y0) // B
    maps, segs, per = {}, {}, {}
    for f in a.files:
        L = np.load(f'{P4}/net/locs_MT_{f:03d}_v10.npz')
        ok = (L['sz'] < 0.025) & (L['sx'] < 0.12) & (L['sy'] < 0.12) & (L['N'] > 300)
        nwin = int(L['w'].max() - L['w'].min() + 1)
        x, y = L['x'][ok], L['y'][ok]
        H = np.histogram2d(y, x, bins=(nby, nbx), range=((Y0, Y0 + nby * B), (X0, X0 + nbx * B)))[0] / nwin     # base locs per window per bin
        S = seg_map(x, y, nwin) / kcal
        sb = 6
        S2 = S[:nby * sb, :nbx * sb].reshape(nby, sb, nbx, sb).sum((1, 3))
        maps[f], segs[f] = H, S2
        q = dict(windows=nwin, n_base=int(ok.sum()), field_rate_per_um2=float(H.sum() / (nbx * nby * (B * NM / 1000) ** 2)))
        for nm, n in WIN.items():
            nb = n // B; A = (nb * B * NM / 1000) ** 2
            R = box(H, nb) / A; T = box(S2, nb)
            i = np.unravel_index(np.argmax(R), R.shape)
            rate = float(R[i]); r = r_of(rate); on = rate / r
            q[nm] = dict(side_um=nb * B * NM / 1000, origin_px=[int(X0 + i[1] * B), int(Y0 + i[0] * B)], base_rate_per_um2=rate, locs_per_on=r,
                         on_per_um2_frame=on, tube_um=float(T[i]), tube_um_per_um2=float(T[i] / A), on_per_frame_per_um_tube=float(on * A / T[i]))
        q['field'] = dict(on_per_um2_frame=q['field_rate_per_um2'] / r_of(q['field_rate_per_um2']),
                          tube_um_per_um2=float(S2.sum() / (nbx * nby * (B * NM / 1000) ** 2)))
        per[f] = q
        print(f, json.dumps(default=float, obj={k: (v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in q.items()}), flush=True)
    # fixed windows on the mean map
    Hm = np.mean([maps[f] for f in per], 0); Sm = np.mean([segs[f] for f in per], 0)
    fixed = {}
    for nm, n in WIN.items():
        nb = n // B; A = (nb * B * NM / 1000) ** 2
        R = box(Hm, nb) / A; i = np.unravel_index(np.argmax(R), R.shape)
        vals = []
        for f in per:
            rate = float(box(maps[f], nb)[i] / A); on = rate / r_of(rate); T = float(box(segs[f], nb)[i])
            vals.append(dict(file=f, on_per_um2_frame=on, on_per_frame_per_um_tube=on * A / T, tube_um_per_um2=T / A))
        o = np.array([v['on_per_um2_frame'] for v in vals]); ot = np.array([v['on_per_frame_per_um_tube'] for v in vals])
        fx = np.array([v['file'] for v in vals]); slope = np.polyfit(fx, o, 1)
        fixed[nm] = dict(origin_px=[int(X0 + i[1] * B), int(Y0 + i[0] * B)], side_um=nb * B * NM / 1000, per_file=vals,
                         on_per_um2_frame_mean=float(o.mean()), on_per_um2_frame_sd=float(o.std(ddof=1)), on_per_um2_frame_min_max=[float(o.min()), float(o.max())],
                         on_per_frame_per_um_tube_mean=float(ot.mean()), on_per_frame_per_um_tube_sd=float(ot.std(ddof=1)),
                         trend_rel_change_over_series=float(slope[0] * 99 / np.polyval(slope, 0)))
        print('fixed', nm, {k: v for k, v in fixed[nm].items() if k != 'per_file'}, flush=True)
    sweep = {t: dict(dens_fov=float(d[2]), dens_region=float(d[1]), on_per_frame_per_um_tube=float(d[3])) for t, d in zip(TAGS, sorted(sim, key=lambda q: q[2]))}
    ref = fixed['sim_field_7.68um']['on_per_um2_frame_mean']; ref20 = fixed['20um']['on_per_um2_frame_mean']; reft = fixed['sim_field_7.68um']['on_per_frame_per_um_tube_mean']
    for t, v in sweep.items():
        v['x_dense_7.68um'] = v['dens_fov'] / ref; v['x_dense_20um'] = v['dens_fov'] / ref20; v['x_dense_7.68um_per_tube'] = v['on_per_frame_per_um_tube'] / reft
    out = dict(selection='net_base (sigma_z < 25 nm, sigma_xy < 0.12 px, N > 300)', files=a.files, sim_rate_on=sim.tolist(), per_file=per, fixed=fixed, sweep=sweep,
               field_on_per_um2_frame_mean=float(np.mean([q['field']['on_per_um2_frame'] for q in per.values()])))
    json.dump(out, open(f'{OUT}/densregion.json', 'w'), indent=1, default=float)
    print(json.dumps(sweep, indent=1))


if __name__ == '__main__':
    main()
