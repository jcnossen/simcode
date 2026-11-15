"""Merge the per-file classical tables of a movie, lateral drift by DME v2 (analysis/baselines/dme_drift.py, the same
call as the network's post_w4pi.py) on the kept published-style rows, applied to all classical positions.
Global frame = (file - first file) * 18000 + 6 w, first file MT 000, NPC 002 (as the network tables).
Axial: as the network: sample z drift moves z_ast and the 4Pi phase together; the per-block term of the phase field
takes the interferometer drift; no separate z correction.
Output: out/4pi/classical/real/cl_<cell>_all<tag>.npz (+ _summary.json)
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi_classical/post_cl.py --cell NPC [--tag _short]
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
OUT = f'{P4}/classical/real'
FIRST = {'MT': 0, 'NPC': 2}
KEYS = ('w', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'N', 'x0', 'y0', 'z_ast', 'sz_ast', 'xp', 'yp', 'zp', 'sxp', 'syp', 'szp', 'keep', 'keep_pix',
        'cx', 'cy', 'C', 'R', 'chi2px', 'nn', 'file_index', 'row', 'box')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cell', default='NPC'); ap.add_argument('--tag', default='')
    ap.add_argument('--files', type=int, nargs='*')
    a = ap.parse_args()
    if REPO not in sys.path: sys.path.insert(0, REPO)
    from simcode.util.proctitle import set_title; set_title('4pi-classic')
    fs = sorted(glob.glob(f'{OUT}/cl_{a.cell}_[0-9][0-9][0-9]{a.tag}.npz'))
    if a.files: fs = [f for f in fs if int(os.path.basename(f).split('_')[2][:3]) in a.files]
    R = {}
    flags = []
    for f in fs:
        D = np.load(f); fi = int(D['file_index'][0])
        for k in KEYS:
            if k in D.files: R.setdefault(k, []).append(D[k])
        flags.append(D['flags'])
        R.setdefault('frame', []).append((fi - FIRST[a.cell]) * 18000 + 6 * D['w'])
    R = {k: np.concatenate(v) for k, v in R.items()}
    R['flags'] = np.concatenate(flags)
    F = int(R['frame'].max()) + 6
    tag = f'cl_{a.cell}{a.tag}'
    k = R['keep'] & np.isfinite(R['x'])
    from simcode.util.drift import drift_of_table
    from simcode.io.work import atomic_save
    d, dinfo = drift_of_table(dict(frame=R['frame'][k], x=R['x'][k], y=R['y'][k], sx=R['sx'][k], sy=R['sy'][k]), F, return_info=True)
    print(f'DME v2 {dinfo["version"]} on {dinfo["n"]} rows, chosen {dinfo["chosen"]}; drift ptp {np.ptp(d, 0) * 128.0} nm')
    atomic_save(f'{OUT}/{tag}_drift.npy', d)
    fi_ = np.clip(R['frame'] + 3, 0, F - 1)
    for a_, b_ in (('x', 'xd'), ('y', 'yd'), ('xp', 'xpd'), ('yp', 'ypd'), ('x0', 'x0d'), ('y0', 'y0d')):
        R[b_] = R[a_] - d[fi_, 0 if a_[0] == 'x' else 1]
    np.savez(f'{OUT}/{tag}_all.npz', **R, drift=d)
    g, gp = R['keep'], R['keep_pix']
    summ = dict(cell=a.cell, files=len(fs), n_cand=int(len(g)), n_keep=int(g.sum()), n_keep_pix=int(gp.sum()),
                kept_frac=float(g.mean()), filter_removed_alone=dict(zip(json.load(open(fs[0].replace('.npz', '_summary.json')))['flag_names'],
                                                                         R['flags'].mean(0).round(4).tolist())),
                med_sigma_published_nm=[float(np.median(R['sx'][g]) * 128), float(np.median(R['sy'][g]) * 128), float(np.median(R['sz'][g]) * 1000)],
                med_sigma_pixel_nm=[float(np.nanmedian(R['sxp'][gp]) * 128), float(np.nanmedian(R['syp'][gp]) * 128), float(np.nanmedian(R['szp'][gp]) * 1000)],
                med_N=float(np.median(R['N'][g])), drift_ptp_nm=(np.ptp(d, 0) * 128).tolist())
    json.dump(summ, open(f'{OUT}/{tag}_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
