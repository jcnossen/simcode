"""in-situ registration: robust order-2 polynomial from bright isolated emitters in frames 0-3600 of the reference
file (3 iterations: detection / ROI centring with the refined map), then the residual of that map on other files
(registration drift). usage: python run_reg_insitu.py <cell> <ref_fi>,<fi2>,...
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register import *
from dc import movie_files, file_index
out = '/home/jc/simcode2/out/4pi/smlm/reg'
cell = sys.argv[1]; fis = [int(v) for v in sys.argv[2].split(',')]
off = np.load(f'/home/jc/simcode2/out/4pi/smlm/camera/camera_{cell}.npz')['off']
pm = PolyMap.load(f'{out}/reg_beads_{cell}.json')
fn = lambda fi: [f for f in movie_files(cell) if file_index(f) == fi][0]
S6 = load_s6(cell, fn(fis[0]), 0, 3600, off)
for it in range(3):
    P = clean(insitu_positions(S6, pm))
    pm, rep = insitu_register(P, pm, out_json=f'{out}/reg_insitu_{cell}.json', source=f'in situ {cell} file {fis[0]} frames 0-3600, iteration {it}')
    print(cell, fis[0], 'iter', it, json.dumps(rep), flush=True)
np.save(f'{out}/insitu_pos_{cell}_{fis[0]}.npy', P)
for fi in fis[1:]:
    S6 = load_s6(cell, fn(fi), 0, 3600, off)
    P = clean(insitu_positions(S6, pm))
    _, rep = insitu_register(P, pm, out_json=f'{out}/reg_insitu_{cell}_{fi}.json', source=f'in situ {cell} file {fi} frames 0-3600')
    print(cell, fi, 'residual of the reference map:', json.dumps({c: (rep[c]['start_map_median'], rep[c]['start_map_robust_sd']) for c in (1, 2, 3)}),
          'own map sd', json.dumps({c: rep[c]['insitu_robust_sd'] for c in (1, 2, 3)}), flush=True)
    np.save(f'{out}/insitu_pos_{cell}_{fi}.npy', P)
