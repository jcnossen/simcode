"""Compare two simcode2.run output dirs that differ only in --cand_pmin (critic 40 low item): pattern parameters, automatic
c (theta), and the final positions / sigma of the identical p >= 0.63 rows (matched on window start + prior position).

  CUDA_VISIBLE_DEVICES=1 python -m simcode2.run --movie <0.5 nM movie> --calib insitu/calib_0.5nM.json --out p03 --chunk_list 12000 --cand_pmin 0.3
  CUDA_VISIBLE_DEVICES=2 python -m simcode2.run --movie <0.5 nM movie> --calib insitu/calib_0.5nM.json --out p05 --chunk_list 12000 --cand_pmin 0.5
  python analysis/checks/cand_pmin_check.py p03 p05 [--px_nm 108.33]
"""
import glob, json, sys
import numpy as np

a, b = sys.argv[1], sys.argv[2]
px = float(sys.argv[sys.argv.index('--px_nm') + 1]) if '--px_nm' in sys.argv else 108.33
P = {}
for nm, d in (('A', a), ('B', b)):
    pe = json.load(open(f'{d}/pattern_est.json')); h = pe['hist'][-1]
    th = json.load(open(f'{d}/autoc_est.json'))['theta'] if glob.glob(f'{d}/autoc_est.json') else None
    P[nm] = h
    print(f'{nm} {d}: pattern rows {pe["k"]["0"]["n"]}; pitch px {pe["k"]["0"]["pitch_px"]:.5f} / {pe["k"]["1"]["pitch_px"]:.5f}; '
          f'angle deg {pe["k"]["0"]["angle_deg"]:.4f} / {pe["k"]["1"]["angle_deg"]:.4f}')
    print(f'   phase rad {np.round(h["psi"], 4).tolist()}\n   depth {np.round(h["amp"], 4).tolist()}  m {np.round(h["m"], 4).tolist()}'
          f'\n   rel. intensity {np.round(h["rel"], 4).tolist()}\n   autoc theta (s, c, c_a) {np.round(th, 4).tolist() if th else None}')
dpsi = np.angle(np.exp(1j * (np.array(P['A']['psi']) - np.array(P['B']['psi']))))
pitch = 2.071
print(f'A - B: phase {np.round(dpsi, 4).tolist()} rad = {np.round(dpsi / (2 * np.pi) * pitch * px, 2).tolist()} nm; '
      f'depth {np.round(np.array(P["A"]["amp"]) - np.array(P["B"]["amp"]), 4).tolist()}')
def rows(d):
    Q = {}
    for f in sorted(glob.glob(f'{d}/pass2_est_*.npz')):
        z = np.load(f)
        for k in ('wstart', 'x_prior', 'y_prior', 'x_hy0', 'y_hy0', 'sx_cal', 'sy_cal', 'p'): Q.setdefault(k, []).append(z[k])
    Q = {k: np.concatenate(v) for k, v in Q.items()}; m = Q['p'] >= 0.63
    return {k: v[m] for k, v in Q.items()}
A, B = rows(a), rows(b)
ka = {(int(w), round(float(x), 4), round(float(y), 4)): i for i, (w, x, y) in enumerate(zip(A['wstart'], A['x_prior'], A['y_prior']))}
ij = []
for j, (w, x, y) in enumerate(zip(B['wstart'], B['x_prior'], B['y_prior'])):
    i = ka.get((int(w), round(float(x), 4), round(float(y), 4)))
    if i is not None: ij.append((i, j))
ia, ib = np.array(ij).T
dx = (A['x_hy0'][ia] - B['x_hy0'][ib]) * px; dy = (A['y_hy0'][ia] - B['y_hy0'][ib]) * px
rsd = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
sa = 0.5 * (A['sx_cal'][ia] + A['sy_cal'][ia]); sb = 0.5 * (B['sx_cal'][ib] + B['sy_cal'][ib])
print(f'p >= 0.63 rows A {len(A["p"])} B {len(B["p"])}, identical {len(ia)}; final A - B (nm): median {np.median(dx):.2f} / {np.median(dy):.2f}, '
      f'robust sd {rsd(dx):.2f} / {rsd(dy):.2f}, rms {np.sqrt(np.mean(dx ** 2)):.2f} / {np.sqrt(np.mean(dy ** 2)):.2f}, '
      f'|d| > 5 nm {100 * np.mean(np.hypot(dx, dy) > 5):.2f} %; median sigma {np.median(sa) * px:.2f} vs {np.median(sb) * px:.2f} nm '
      f'(ratio {np.median(sa / sb):.3f})')
