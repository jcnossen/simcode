"""Frame-order (pattern-step sync) check for EVERY movie file: reads only frames [0, NH) and [n - NT, n) of each file
(plus the footer counters / timestamps of the first and last frame), localizes emitters in 6-frame windows with quick
photometry, and compares the per-slot pattern coefficients Z[j, a] (j = frame % 6 relative to the file start, a = angle)
with those of a reference file (the first subset file). Free per-angle phase rotation (pattern phase drift);
the step offset s in 0..5 is the cyclic shift that best matches. Also reports the per-angle phase rotation psi_a
(the pattern phase drift relative to the reference file) at the start and at the end of each file.
usage: python frameorder.py <cell> <out.npz> [nproc]
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
import os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc import Dcimg, movie_files, crops, file_index
from quickloc import affine_warp, window_photometry, rows_to_arrays, slot_coeffs, refine_k, GAIN

NH, NT = 240, 120
K0 = {'MT': (np.array([3.84, -0.026]), np.array([0.0, 3.852])), 'NPC': (np.array([3.84, -0.026]), np.array([0.0, 3.852]))}
REF = {'MT': 5, 'NPC': 10}
CAM = f'{P4}/smlm/camera'


def load_off(cell):
    return np.load(f'{CAM}/camera_{cell}.npz')['off']


def _reader(cell, fs, q, nworkers):
    """single sequential reader (the data disk is one HDD: concurrent readers are seek-bound at ~30 MB/s)"""
    off = load_off(cell)
    for fn in fs:
        d = Dcimg(fn)
        f0, f1 = d.footer(0), d.footer(d.nframes - 1)
        meta = dict(fi=file_index(fn), n=d.nframes, fc0=int(f0[0]), fc1=int(f1[0]), t0=f0[1] + 1e-6 * f0[2], t1=f1[1] + 1e-6 * f1[2])
        parts = {}
        for tag, a0, n in (('head', 0, NH), ('tail', d.nframes - NT, NT)):
            parts[tag] = ((crops(d.raw(a0, n)) - off) * GAIN).astype(np.float32)
        q.put((meta, parts))
    for _ in range(nworkers): q.put(None)


def _worker(cell, q, r):
    warp = affine_warp(cell)
    while True:
        it = q.get()
        if it is None: break
        meta, parts = it
        res = dict(meta)
        for tag, E in parts.items():
            rows = window_photometry(E, warp)
            w, x, y, I, bg = rows_to_arrays(rows)
            res[tag] = dict(x=x, y=y, I=I, w=w, bgm=E.mean((0, 2, 3)), fmean=E[:, :, :, 60:300].mean((2, 3)))
        r.put(res)


if __name__ == '__main__':
    import multiprocessing as mp
    cell, out = sys.argv[1], sys.argv[2]; npr = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    fs = movie_files(cell)
    q, r = mp.Queue(maxsize=4), mp.Queue()
    rd = mp.Process(target=_reader, args=(cell, fs, q, npr)); rd.start()
    ws = [mp.Process(target=_worker, args=(cell, q, r)) for _ in range(npr)]
    for w in ws: w.start()
    t = time.time(); R = []
    for i in range(len(fs)):
        R.append(r.get())
        if i % 10 == 0: print(cell, i, R[-1]['fi'], f'{time.time() - t:.0f}s', flush=True)
    for w in ws: w.join()
    rd.join()
    np.save(out, np.array(R, dtype=object), allow_pickle=True)
    print('saved', out, f'{time.time() - t:.0f}s', flush=True)
