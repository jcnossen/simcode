"""Pattern estimation with the spatial-hash backend (GPU and/or CPU) vs a pattern_est.pt computed by the original
serial-cKDTree code on the same run dir (read-only), same row loading as run_uncalibrated's pattern stage.
usage: python analysis/speed/pattern_hash_vs_file.py <run dir> <movie> <sc_cal> <tag> [devices cuda,cpu,kdtree (= the original serial cKDTree)]"""
import sys, os, glob, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, gpu_state, dump
from simcode2 import config
from simcode2.pipeline import pattern_est as PE
import tifffile
shape_of = lambda mv: tuple(tifffile.TiffFile(mv).pages[0].shape)      # (Movie() would index all pages of the TIFF)
d, mv, sc_cal, tag = sys.argv[1:5]; devs = (sys.argv[5] if len(sys.argv) > 5 else 'cuda,cpu').split(',')
fs = sorted(glob.glob(f'{d}/pass1_*.npz'), key=lambda f: int(f.split('_')[-1][:-4])); chunks = [int(f.split('_')[-1][:-4]) for f in fs]
cols = ('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc')
Ls = [np.load(f) for f in fs]
keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (f0 == chunks[-1])) for L, f0 in zip(Ls, chunks)])
R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}; del Ls
po = json.load(open(config.resolve(sc_cal))).get('prior_off')
if po is not None: R['x'] = R['x'] - po[0]; R['y'] = R['y'] - po[1]
R['out'] = (R['tc'] >= 1.0) & (R['tc'] < 4.0) & (R['x'] >= 5) & (R['x'] < 115) & (R['y'] >= 5) & (R['y'] < 115)
shape = shape_of(mv)
P0 = torch.load(f'{d}/pattern_est.pt', weights_only=False)
REP = dict(run_dir=d, n_rows=len(R['x']), gpu_state=gpu_state())
for dev in devs:
    PE.NB_BACKEND, PE.NB_DEVICE, PE.KD_WORKERS = ('kdtree', None, 1) if dev == 'kdtree' else ('hash', dev, 1); PE.NB_STATS.clear(); t = time.time()
    P1, _ = PE.estimate_pattern(R, shape, verbose=lambda *a: None)
    r = {k: float((torch.as_tensor(P0[k]).double() - torch.as_tensor(P1[k]).double()).abs().max()) for k in ('k', 'phi', 'm', 'logrel', 'dphi')}
    r.update(identical=all(torch.equal(torch.as_tensor(P0[k]), torch.as_tensor(P1[k])) for k in ('k', 'phi', 'm', 'logrel', 'dphi')),
             s=round(time.time() - t, 1), ties=PE.NB_STATS.get('ties'))
    REP[dev] = r; print(dev, r, flush=True)
dump(REP, f'{OUT}/pattern_hash_vs_file_{tag}.json')
