"""DME (exact, GPU port of the original algorithm; $DME_PATH/gpu/dme_pipeline.py, default ~/dme, imported read-only) on the full
0.5 nM movie: ours (sc5nb, disjoint windows p>=0.63, time = window centre) vs SIMCODE sfhd-ndi, at several frames-per-bin down
to 1. Input of recon_full_dme_cross.py.
usage: CUDA_VISIBLE_DEVICES=1 python dme_full05.py [--run full_0.5nM_sc5nb_bright_real] [--simcode <sfhd-ndi.hdf5>] [--out_dir drift]
-> <out>/<out_dir>/drift_{ours,simcode}_fpb{200,50,10,1}.npy"""
import os, sys, argparse, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_metrics as SM
from simcode2 import config
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run', default='full_0.5nM_sc5nb_bright_real', help='relative: under the out root')
ap.add_argument('--simcode', default=SM.SIMCODE_05, help='relative: under the data root')
ap.add_argument('--out_dir', default='drift', help='relative: under the out root')
a = ap.parse_args()
_D = os.path.expanduser(os.environ.get('DME_PATH', '~/dme'))
sys.path[:0] = [f'{_D}/gpu']
from dme_pipeline import dme_estimate_gpu
SIMH5 = config.resolve(a.simcode, 'data'); OD = config.out_dir(a.out_dir)
A = SM.load_ours(os.path.join(config.root('out'), a.run), 'est')
dj = (A['wstart'] % 6 == 0) & (A['p'] >= 0.63)
pa = np.c_[A['x_hy0'][dj], A['y_hy0'][dj]]; fa = (A['wstart'][dj] + 3).astype(np.int32)
ca = np.c_[A['sx_cal'][dj], A['sy_cal'][dj]]          # px
S = SM.load_simcode(SIMH5, 0, 90000)
ps = np.c_[S['x'], S['y']]; fs = (S['frame'] + 3).astype(np.int32)
import h5py; L = h5py.File(SIMH5, 'r')['locs'][:]
cs = np.c_[L['lpx'], L['lpy']] if 'lpx' in L.dtype.names else np.full((len(ps), 2), np.median(ca, 0))
print('ours', len(pa), 'median sigma px', np.median(ca, 0), '| SIMCODE', len(ps), 'median sigma px', np.median(cs, 0), flush=True)
out = {}
for fpb in (200, 50, 10, 1):
    for nm, p, f, c in (('ours', pa, fa, ca), ('simcode', ps, fs, cs)):
        d, (d1, d2) = dme_estimate_gpu(p, f, c, framesperbin=fpb, numframes=90000, verbose=False)
        sh = np.sqrt(np.mean((d1 - d2) ** 2, 0)) / np.sqrt(2) * 108.33          # split-half precision per axis (nm)
        out[f'{nm}_{fpb}'] = d; np.save(f'{OD}/drift_{nm}_fpb{fpb}.npy', d)
        print(f'fpb {fpb:4d} {nm:8s}: drift range {np.ptp(d, 0) * 108.33} nm, split-half precision {sh} nm', flush=True)
    dd = (out[f'ours_{fpb}'] - out[f'simcode_{fpb}']); dd -= dd.mean(0)
    print(f'fpb {fpb:4d} ours vs SIMCODE drift difference rms {np.sqrt(np.mean(dd ** 2, 0)) * 108.33} nm', flush=True)
