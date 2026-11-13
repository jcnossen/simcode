"""Nanoruler (resume 2026-11-13 evening): the sub-set of matched events (abl_match.py output <prefix>_pipe.npz / _ref.npz [/ _3way_*]) whose
pipeline row is inside the paper's main selection 'm12' (strict_u finite and max(sx, sy) * 65 nm <= 12 nm; abl_lib.selection), i.e. the
same kind of sub-set as abl_match's <prefix>_u45_*. CPU only.
usage: python analysis/nanoruler/nr12_sub.py <locs.npz> <match prefix>   -> <prefix>_m12_pipe.npz, <prefix>_m12_ref.npz"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
L = A.L

T = dict(np.load(sys.argv[1])); pre = sys.argv[2]; sel = A.selection(T, 'm12')
P = dict(np.load(pre + '_pipe.npz')); R = dict(np.load(pre + '_ref.npz'))
k = sel[P['row'].astype(np.int64)]
L.save_table(pre + '_m12_pipe.npz', **{q: v[k] for q, v in P.items()}); L.save_table(pre + '_m12_ref.npz', **{q: v[k] for q, v in R.items()})
print(f'{os.path.basename(pre)}: {k.sum()} of {len(k)} matched events inside m12 ({100 * k.mean():.1f} %)')
