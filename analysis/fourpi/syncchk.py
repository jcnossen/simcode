"""Within-file pattern-step sync check on the processed subset files, every frame: background polarization signature
r_t = (ch0 + ch2) / (ch1 + ch3) (per-frame channel means over the illuminated area, normalised by a 6-frame running mean).
For each block of 50 windows (300 frames) the 6-slot profile is matched (cyclic shift) to the file's first block.
A step slip shows as a block with best shift != 0. usage: python syncchk.py <cell> <fi> ..."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
import sys, json
import numpy as np
OUT = f'{P4}/smlm'
cell = sys.argv[1]; res = {}
for fi in [int(v) for v in sys.argv[2:]]:
    f = np.load(f'{OUT}/p1/p1_{cell}_{fi:03d}.npz')['fmean']
    r = (f[:, 0] + f[:, 2]) / (f[:, 1] + f[:, 3])
    r = r / np.convolve(r, np.ones(6) / 6, 'same')
    W = r[:len(r) // 6 * 6].reshape(-1, 6) - 1
    ref = W[1:51].mean(0)
    blocks = []
    for b in range(0, len(W), 50):
        p = W[b:b + 50].mean(0)
        cc = [float(np.dot(p, np.roll(ref, -k))) for k in range(6)]
        blocks.append(int(np.argmax(cc)))
    # per-window: fraction of windows whose own best shift is 0 (noise level)
    cw = np.stack([W @ np.roll(ref, -k) for k in range(6)], 1).argmax(1)
    res[fi] = dict(ref_profile=ref.round(4).tolist(), block_shifts_nonzero=[i for i, s in enumerate(blocks) if s != 0],
                   n_blocks=len(blocks), frac_windows_shift0=float(np.mean(cw == 0)))
    print(cell, fi, json.dumps(res[fi]))
json.dump(res, open(f'{OUT}/frameorder/syncchk_{cell}.json', 'w'))
