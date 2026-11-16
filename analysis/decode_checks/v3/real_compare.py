"""Old vs new DECODE on a paper movie: fair FRC (frc_fair protocol: disjoint windows, max(sigma) < 0.15 px, crop, own DME v2
drift, 300-frame halves) and RSP (fig_exp_new: paper_common.compute_pc of the fair-FRC set with the widefield mean image), as
fig_exp_new.compute does for each method. Old = decode_table(locs.npz) (frame-label fixed), new = locs_v3.npz.
usage: python analysis/decode_checks/v3/real_compare.py <ds> -> out/decode_checks/v3/real_compare_<ds>.json"""
import os, sys, json
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [REPO, os.path.join(REPO, 'analysis', 'paper')]
import frc_fair as FF
import fig_exp_new as FE
from paper_common import compute_pc
from simcode.dataset import decode_table
OUT = os.path.join(REPO, 'out', 'decode_checks', 'v3')
NF = {'1809_2': 120000, '2.5nM_2': 180000}


def main():
    ds = sys.argv[1]; F = NF[ds]
    d = os.path.join(REPO, 'out', 'paper2', ds, 'decode'); tifp = FE.SETS[ds]['tif']
    tabs = dict(old=decode_table(os.path.join(d, 'locs.npz')), new=os.path.join(d, 'locs_v3.npz'))
    res, dsets = {}, []
    for k, p in tabs.items():
        T = FF.load('decode', p); keep = T['f'] + 6 <= F; T = {kk: v[keep] for kk, v in T.items()}
        R = FF.evaluate(T, F, drift_cache=os.path.join(OUT, f'real_compare_{ds}_{k}_drift.npy'))
        m = R['mask']
        dsets.append(FE.to_dataset(R['x_dc'][m], R['y_dc'][m], T['f'][m], T['sx'][m], T['sy'][m], T['photons'][m]))
        res[k] = dict(path=p, **{kk: v for kk, v in FF.summary(R).items()})
        print(k, json.dumps(res[k], default=float), flush=True)
    pc, rmsd, _ = compute_pc(tifp, dsets)
    for k, v in zip(tabs, pc): res[k]['rsp'] = float(v)
    json.dump(res, open(os.path.join(OUT, f'real_compare_{ds}.json'), 'w'), indent=1, default=float)
    for k in tabs: print(k, 'fair FRC %.2f nm  n_frc %d  n_all %d  RSP %.3f' % (res[k]['frc_nm'], res[k]['n_frc'], res[k]['n_all'], res[k]['rsp']))


if __name__ == '__main__':
    main()
