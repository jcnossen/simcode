"""SIMFLUX - SMLM offset over the movie (CPU): for the re-implemented baseline tables and for the published tables (both NOT
drift corrected, so both methods share the raw frame), and baseline - published per method. Mutual nearest neighbours in the
same cycle within 0.6 px; medians per tenth of the movie in nm. A time-dependent SIMFLUX - SMLM offset is a pattern-phase
tracking error (it adds to the apparent drift of the SIMFLUX table).

  python analysis/nanoruler/r24b_sfcheck.py nr40|nr20      -> out/nanoruler/<movie>/ref/sfcheck.txt / .json"""
import sys, json, numpy as np
from scipy.spatial import cKDTree
REPO = '/home/jc/simcode2'
sys.path.insert(0, f'{REPO}/analysis/nanoruler')
import nra_lib as L
NCYC = dict(nr40=6666, nr20=16666)


def mm(a, b, r=0.6 * 65):
    A = np.c_[a['x_nm'], a['y_nm'], a['frame'] * 1e6]; P = np.c_[b['x_nm'], b['y_nm'], b['frame'] * 1e6]
    da, ja = cKDTree(P).query(A, distance_upper_bound=r); dp, jp = cKDTree(A).query(P, distance_upper_bound=r)
    i = np.nonzero(np.isfinite(da))[0]; i = i[jp[ja[i]] == i]
    return i, ja[i]


def main():
    mv = sys.argv[1]; d = f'{REPO}/out/nanoruler/{mv}/ref/'; n = NCYC[mv]
    T = {k: dict(np.load(d + f)) for k, f in (('bl_smlm', 'tab_bl_smlm.npz'), ('bl_sf', 'tab_bl_simflux.npz'), ('pub_smlm', 'tab_pub_smlm_nodc.npz'), ('pub_sf', 'tab_pub_simflux_nodc.npz'))}
    R = {}; o = [f'{mv}: A - B, mutual nearest neighbours per cycle within 0.6 px; nm; medians per tenth of the movie']
    for a, b in (('bl_sf', 'bl_smlm'), ('pub_sf', 'pub_smlm'), ('bl_smlm', 'pub_smlm'), ('bl_sf', 'pub_sf'), ('bl_sf', 'pub_smlm')):
        i, j = mm(T[a], T[b]); dx = T[a]['x_nm'][i] - T[b]['x_nm'][j]; dy = T[a]['y_nm'][i] - T[b]['y_nm'][j]
        tb = np.minimum(T[a]['frame'][i].astype(int) * 10 // n, 9)
        r = dict(n=int(len(i)), median=[float(np.median(dx)), float(np.median(dy))], rsd=[L.rsd(dx), L.rsd(dy)],
                 per_tenth_dx=[float(np.median(dx[tb == k])) for k in range(10)], per_tenth_dy=[float(np.median(dy[tb == k])) for k in range(10)])
        R[f'{a} - {b}'] = r
        o.append(f'{a:8s} - {b:8s}: n {len(i):6d}, median {r["median"][0]:+.2f} {r["median"][1]:+.2f}, robust sd {r["rsd"][0]:.2f} {r["rsd"][1]:.2f}')
        o.append(f'      dx {np.round(r["per_tenth_dx"], 2).tolist()}'); o.append(f'      dy {np.round(r["per_tenth_dy"], 2).tolist()}')
    json.dump(R, open(d + 'sfcheck.json', 'w'), indent=1); open(d + 'sfcheck.txt', 'w').write('\n'.join(o) + '\n'); print('\n'.join(o))


if __name__ == '__main__':
    main()
