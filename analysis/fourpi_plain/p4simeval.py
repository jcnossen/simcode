"""Accuracy of simulated-movie runs (analysis/fourpi_plain/sim_bench.py outputs) per z bin: bias and robust sd of x, y, z
on matched isolated interior emitters (> 2000 photons), and the photon ratio; several runs side by side.

  python analysis/fourpi_plain/p4simeval.py name:<bench dir> [...] --work <plain-4Pi work folder> --data <extracted Zenodo folder> [--json out.json]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sim_bench import evaluate, isolated_interior

ZB = np.array([-0.9, -0.7, -0.55, -0.4, -0.25, -0.1, 0.1])
PX = 128.0


def main():
    p = argparse.ArgumentParser(); p.add_argument('runs', nargs='+'); p.add_argument('--json', default='')
    p.add_argument('--nmin', type=float, default=2000)
    p.add_argument('--work', required=True, help='plain-4Pi work folder (configs(work, data) of fourpi_plain.py)')
    p.add_argument('--data', required=True, help='folder of the extracted Zenodo data')
    a = p.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fourpi'))
    from train_det4pi import acquisition
    c = acquisition(os.path.join(os.path.dirname(__file__), '..', '..', 'simcode', 'pipeline', 'configs', 'fourpi_plain.py'), a.work, a.data)
    R = {}
    for spec in a.runs:
        nm, d = spec.split(':', 1)
        z = np.load(os.path.join(d, 'sim_movie.npz')); G = {k[3:]: z[k] for k in z.files if k.startswith('gt_')}
        L = dict(np.load(os.path.join(d, 'locs_sim.npz')))
        iso = isolated_interior(G, out_fr=c)
        Gi = {k: v[iso] for k, v in G.items()}
        _, gi, li = evaluate(L, Gi, PX)
        s = Gi['N'][gi] >= a.nmin
        gi, li = gi[s], li[s]
        ex, ey, ez = (L['x'][li] - Gi['x'][gi]) * PX, (L['y'][li] - Gi['y'][gi]) * PX, (L['z'][li] - Gi['z'][gi]) * 1000
        ok = np.abs(ez) < 100                                                   # period slips reported separately
        rs = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 10 else None
        rows = []
        k = np.digitize(Gi['z'][gi], ZB) - 1
        for b in range(len(ZB) - 1):
            q = (k == b) & ok
            if q.sum() < 20: continue
            rows.append(dict(z=[float(ZB[b]), float(ZB[b + 1])], n=int(q.sum()), bias=[float(np.median(v[q])) for v in (ex, ey, ez)],
                             sd=[rs(v[q]) for v in (ex, ey, ez)], N_ratio=float(np.median(L['N'][li][q] / Gi['N'][gi][q]))))
        R[nm] = dict(n=int(len(gi)), slips=float(np.mean(~ok)), all_sd=[rs(v[ok]) for v in (ex, ey, ez)],
                     all_bias=[float(np.median(v[ok])) for v in (ex, ey, ez)], zbins=rows,
                     chi2_dof_median=float(np.median(L['chi2'][li] / L['dof'][li])))
        print(f'{nm}: n {len(gi)}, slips {R[nm]["slips"]:.3f}, sd x/y/z {np.round(R[nm]["all_sd"], 2).tolist()} nm, chi2/dof {R[nm]["chi2_dof_median"]:.2f}')
        for r in rows:
            print(f'   z {r["z"][0]:+.2f}..{r["z"][1]:+.2f} n {r["n"]:5d} bias x/y/z {np.round(r["bias"], 1).tolist()} sd {np.round(r["sd"], 1).tolist()} N ratio {r["N_ratio"]:.3f}')
    if a.json: json.dump(R, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
