"""R24: cut tables for the merged-row test: rows of an nra table (r24_tab.py; column 'row' = index into the pipeline table) whose merged-score
(r24_merged.py <prefix>_scores.npz) is below the quantile that keeps the given yield of the scored rows. Rows without a score are dropped.
Also 'oracle' = all rows except the GT-merged ones (f2 >= 0.05), the upper bound of what any cut can reach.
usage: python r24_cut.py <table.npz> <scores.npz> <out prefix> [--stat combo,photons] [--yield 0.9,0.8,0.7]  -> <out>_<stat><yield>.npz, <out>_oracle.npz"""
import sys, numpy as np
a = sys.argv[1:]; g = lambda k, d: a[a.index(k) + 1] if k in a else d
T = dict(np.load(a[0])); S = np.load(a[1]); out = a[2]; row = T['row'].astype(np.int64); pos = {r: i for i, r in enumerate(S['row'])}
ix = np.array([pos.get(r, -1) for r in row]); has = ix >= 0
def save(name, keep):
    np.savez_compressed(f'{out}_{name}.npz', **{k: v[keep] for k, v in T.items() if np.ndim(v) == 1 and len(v) == len(row)}); print(name, int(keep.sum()), 'of', len(row))
for st in g('--stat', 'combo,photons').split(','):
    s = np.nan_to_num(S[st])
    for y in g('--yield', '0.9,0.8,0.7').split(','):
        thr = np.quantile(s, float(y)); k = has.copy(); k[has] = s[ix[has]] <= thr; save(f'{st}{int(round(100 * float(y)))}', k)
k = has.copy(); k[has] = S['f2'][ix[has]] < 0.05; save('oracle', k)
