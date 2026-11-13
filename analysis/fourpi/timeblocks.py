"""Pattern and 4Pi phase versus time over a WHOLE movie series from sampled blocks (no whole-file processing):
for every file, the first NB and the last NB frames (one sequential reader, workers do quick photometry with the in-situ
registration map). Per block and angle: SIM period (k refined per block), per-step phases psi_j, step differences,
median per-emitter modulation depth, plane-wave coherence; plus the background polarization step signature (per-frame
angle assignment) and the population 4Pi phase (mean exp(i Phi) of the channel aperture fractions).
usage: python timeblocks.py <cell> <out.npy> [NB] [nworkers]"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc import Dcimg, movie_files, crops, file_index
from quickloc import Warp, window_photometry, rows_to_arrays
from register import PolyMap

from simcode.fourpi.setup import GAIN
OUT = f'{P4}/smlm'


def _reader(cell, fs, NB, q, nw):
    off = np.load(f'{OUT}/camera/camera_{cell}.npz')['off']
    for fn in fs:
        d = Dcimg(fn)
        f0, f1 = d.footer(0), d.footer(d.nframes - 1)
        meta = dict(fi=file_index(fn), n=d.nframes, fc0=int(f0[0]), fc1=int(f1[0]), t0=f0[1] + 1e-6 * f0[2], t1=f1[1] + 1e-6 * f1[2])
        for tag, a0 in (('head', 0), ('tail', d.nframes - NB)):
            for c0 in range(a0, a0 + NB, 600):
                E = ((crops(d.raw(c0, 600)) - off) * GAIN).astype(np.float32)
                q.put((meta, tag, c0, E))
    for _ in range(nw): q.put(None)


def _worker(cell, q, r):
    pm = PolyMap.load(f'{OUT}/reg/reg_insitu_{cell}.json')
    warp = Warp(lambda c, xx, yy: pm.inv(c, xx, yy))
    while True:
        it = q.get()
        if it is None: break
        meta, tag, c0, E = it
        rows = window_photometry(E, warp)
        w, x, y, I, bg = rows_to_arrays(rows) if rows else (np.zeros(0),) * 5
        r.put((meta, tag, c0, dict(w=w + c0 // 6, x=x, y=y, I=I, fmean=E[:, :, :, 60:300].mean((2, 3)))))


if __name__ == '__main__':
    import multiprocessing as mp
    cell, out = sys.argv[1], sys.argv[2]
    NB = int(sys.argv[3]) if len(sys.argv) > 3 else 1200; nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    fs = movie_files(cell)
    q, r = mp.Queue(maxsize=4), mp.Queue()
    rd = mp.Process(target=_reader, args=(cell, fs, NB, q, nw)); rd.start()
    ws = [mp.Process(target=_worker, args=(cell, q, r)) for _ in range(nw)]
    for w in ws: w.start()
    nexp = len(fs) * 2 * (NB // 600); got = []; t = time.time()
    for i in range(nexp):
        got.append(r.get())
        if i % 40 == 0: print(cell, i, '/', nexp, f'{time.time() - t:.0f}s', flush=True)
    for w in ws: w.join()
    rd.join()
    np.save(out, np.array(got, dtype=object), allow_pickle=True)
    print('saved', out, f'{time.time() - t:.0f}s', flush=True)
