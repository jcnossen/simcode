"""Helpers of the pass-2 gate scripts (merged_d1.py, vs_simcode_p2.py), from the development p2/eval_gate.py (cals, miss_anatomy;
the benchmark runner / calibration fit of that file are not needed by these scripts and are not ported).
  cals()            solve calibrations by name: 'p1' = calib/calib_sc_hd_sc5nbb15_pm_bright.json (pass 1, bright, sim sigma),
                    'p2L' = calib/calib_p2L.json (pass-2 gate calibration), + any development calibrations
                    <out>/p2/calib_<d2c|p2s|p2a|p2b|ctl>.json that exist (used by merged_d1.py for the p2s columns)
  miss_anatomy(R)   per target: matched / merged / undetected on the detector point, and the nearest visible GT neighbour
The dumps are read by analysis/calib/plib.py (<out>/p2/fe, simcode/p2/dump.py)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib'))
import plib as P          # analysis/calib/plib.py
from simcode import config

FIT = ('d2c', 'p2s', 'p2a', 'p2b', 'ctl')


def cals():
    C = {'p1': json.load(open(config.model(config.CAL_P1_SIM))), 'p2L': json.load(open(config.model(config.CAL_P2_GATE)))}
    d = os.path.join(config.root('out'), 'p2')
    for n in FIT:
        if os.path.exists(f'{d}/calib_{n}.json'): C[n] = json.load(open(f'{d}/calib_{n}.json'))
    return C


def miss_anatomy(R, thr=0.5):
    """per target: matched (x_det, 1 px) / merged (unmatched, its nearest GT neighbour matched by a candidate within 1 px of
    the target too) / undetected"""
    idx = np.nonzero(R['sc_score'] >= thr)[0]
    og = P.match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], R['sc_x_det'][idx], R['sc_y_det'][idx], 1.0)
    cand_of = np.full(len(R['gt_x']), -1); cand_of[og[og >= 0]] = idx[og >= 0]
    from scipy.spatial import cKDTree
    st = np.full(len(R['gt_x']), 0)      # 0 undetected, 1 matched, 2 merged
    st[cand_of >= 0] = 1
    nbr = np.full(len(R['gt_x']), -1)
    for f in np.unique(R['gt_fov']):
        gi = np.nonzero(R['gt_fov'] == f)[0]
        vis = gi[R['gt_phot'][gi] >= 100]
        if len(vis) < 2: continue
        tr = cKDTree(np.c_[R['gt_x'][vis], R['gt_y'][vis]])
        d, j = tr.query(np.c_[R['gt_x'][gi], R['gt_y'][gi]], 2)
        jj = np.where(vis[j[:, 0]] == gi, vis[j[:, 1]], vis[j[:, 0]])
        nbr[gi] = jj
    for g in np.nonzero(st == 0)[0]:
        q = nbr[g]
        if q >= 0 and cand_of[q] >= 0:
            c = cand_of[q]
            if np.hypot(R['sc_x_det'][c] - R['gt_x'][g], R['sc_y_det'][c] - R['gt_y'][g]) < 1.0: st[g] = 2
    return st, nbr
