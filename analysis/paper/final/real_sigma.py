"""Median reported sigma (nm, max(sx, sy)) and count of the fair-FRC rows (frc_fair.masks: disjoint 6-frame windows, crop, max(sigma)
< 0.15 px; SIMCODE p >= 0.63 at 16.25 nm) of the SIMCODE and DECODE tables of Figs 3 / 4, old vs new (2026-11-17).
usage: python real_sigma.py <out json>"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import frc_fair as FF

D = '/mnt/data/dmdsf/bn_dnapaint'; P2 = os.path.expanduser('~/simcode2/out/paper2')
SETS = {'1809_2': dict(v2_old=f'{D}/1809/cover1/2/runs/simcode_v2_psfv2_20261115', v2_new=f'{D}/1809/cover1/2/runs/simcode_final_20261117',
                       dec_old=f'{P2}/1809_2/decode/locs.npz', dec_new=f'{P2}/1809_2/decode/locs_v3.npz'),
        '2.5nM_2': dict(v2_old=f'{D}/dnapaint/2.5nM/2/runs/simcode_v2_psfv2_final', v2_new=f'{D}/dnapaint/2.5nM/2/runs/simcode_final_20261117',
                        dec_old=f'{P2}/2.5nM_2/decode/locs.npz', dec_new=f'{P2}/2.5nM_2/decode/locs_v3.npz')}


def stats(kind, path):
    if kind == 'v2':
        from simcode.dataset import Dataset
        tab = 'locs_p2_net' if os.path.exists(f'{path}/locs_p2_net.hdf5') else 'locs_p2'
        L = Dataset.from_run(path, table=tab).select(16.25)
        F = L.numpy(); T = dict(x=F['pos'][:, 0].astype(np.float64), y=F['pos'][:, 1].astype(np.float64), f=F['frame'].astype(np.int64),
                                sx=F['crlb'][:, 0].astype(np.float64), sy=F['crlb'][:, 1].astype(np.float64))
    else:
        T = FF.load('decode', path)
    _, m = FF.masks(T)
    s = np.maximum(T['sx'], T['sy'])[m] * FF.PX
    return dict(path=path, n=int(m.sum()), median_sigma_nm=float(np.median(s)))


if __name__ == '__main__':
    out = {}
    for ds, P in SETS.items():
        if not os.path.exists(f"{P['v2_new']}/locs_p2_net.hdf5"): continue
        out[ds] = {k: stats('v2' if k.startswith('v2') else 'decode', p) for k, p in P.items()}
        print(ds, json.dumps(out[ds]), flush=True)
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
