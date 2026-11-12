"""Classical 4Pi-SIMFLUX pipeline on K-summed real data (real dense data by summing K source windows), for the comparison
with the network pipeline on the same summed data.

The classical worker's code (analysis/fourpi_classical: wavelet detection, summed fit, pattern, SIMFLUX two-step, 4Pi
moment phase, phase field, published rejections, 24-image pixel MLE) is imported READ-ONLY and run unchanged, with:
  * the summed movie from the same virtual reader as real_w4pi.py --ksum (stride S windows), e = (sum ADU - K offset) gain;
  * its read variance per pixel and image set to K x (read noise) + (K x offset error)^2 (module global cl4pi.V_PX);
  * its output directory redirected to out/4pi/net/cl_ksum/ (nothing is written into out/4pi/classical).
Patches are kept in host RAM (float16 e), never on disk.
Output: out/4pi/net/cl_ksum/cl_<cell>_<file>_k<K>.npz (the classical table format; w = summed window index).
Usage: CUDA_VISIBLE_DEVICES=1 python analysis/fourpi/cl_ksum4pi.py --cell NPC --files 10 --ksum 8
"""
import argparse, json, math, os, sys, time, threading, queue
import numpy as np, torch
REPO = '/home/jc/simcode2'
for p_ in (REPO, f'{REPO}/analysis/fourpi', f'{REPO}/analysis/fourpi_classical'):
    if p_ not in sys.path: sys.path.insert(0, p_)
from dc import Dcimg, movie_files, file_index, crops
from front4pi import Front
from simcode2.fourpi.lut4pi import LUT4Pi
import cl4pi
from cl4pi import Classic4Pi
import mov4pi_cl as M
import real_cl as RC
import full_cl as FCL

OUT = f'{REPO}/out/4pi/net/cl_ksum'
GAIN = 0.23


class RamSource:
    """e patches in host RAM -> chunks (the classical MemSource interface)"""
    def __init__(self, img, doff, w, cx, cy, dev):
        self.img, self.doff, self.w, self.cx, self.cy, self.dev = img, doff, w, cx, cy, dev
        self.n = len(w)

    def info(self):
        return self.w, self.cx, self.cy, self.doff

    def chunks(self, size, rows=None):
        idx = np.arange(self.n) if rows is None else rows
        for i in range(0, len(idx), size):
            j = idx[i:i + size]
            yield j, torch.tensor(self.img[j].astype(np.float32), device=self.dev), torch.tensor(self.doff[j], device=self.dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='NPC'); ap.add_argument('--files', type=int, nargs='+', required=True)
    ap.add_argument('--ksum', type=int, default=8); ap.add_argument('--kstride', type=int, default=25)
    ap.add_argument('--nframes', type=int, default=0); ap.add_argument('--chunk', type=int, default=2048); ap.add_argument('--psf', default='NPC_f_dxy15')
    ap.add_argument('--field', default='poly+tile'); ap.add_argument('--order_npc', type=int, default=3); ap.add_argument('--order_mt', type=int, default=3)
    ap.add_argument('--tile', type=float, default=12.0); ap.add_argument('--tblk', type=int, default=25); ap.add_argument('--pix', type=int, default=1)
    a = ap.parse_args()
    from simcode2.util.proctitle import auto; auto('4pi-cl-ksum')
    dev = 'cuda'; os.makedirs(OUT, exist_ok=True)
    K, S = a.ksum, a.kstride
    a.tag = f'_k{K}'
    RC.OUT = OUT                                              # redirect the classical outputs
    cl4pi.V_PX = K * 1.14 ** 2 + (0.2 * K) ** 2               # read variance of the summed pixel
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/{a.psf}.json'))['psf']
    lut = LUT4Pi(st, P=15, dev=dev)
    c4 = Classic4Pi(lut, delta=(0.0, RC.DELTA[a.cell]), dev=dev); fr = Front(a.cell, dev)
    for fi in a.files:
        fn = [f for f in movie_files(a.cell) if file_index(f) == fi][0]
        nfr = a.nframes or Dcimg(fn).nframes; nblk = nfr // (6 * K * S)
        q = queue.Queue(maxsize=2)

        def sum_reader():
            d = Dcimg(fn)
            for b in range(nblk):
                r = d.raw(b * 6 * K * S, 6 * K * S).astype(np.float32)
                q.put((b * 6 * S, r.reshape(K, 6 * S, *r.shape[1:]).sum(0)))
            q.put(None)
        threading.Thread(target=sum_reader, daemon=True).start()
        acc = {k: [] for k in ('w', 'cx', 'cy', 'doff', 'img', 'score', 'thr')}; t0 = time.time()
        while True:
            it = q.get()
            if it is None: break
            a0, raw = it
            with torch.no_grad():
                E = (torch.tensor(crops(raw), device=dev) - K * fr.off) * GAIN
                w, cy, cx, sc, thr = FCL.detect_wavelet(fr, E)
                if len(w) == 0: continue
                img, doff, ok = fr.patches(E, w, cy, cx)
                s = torch.nonzero(ok)[:, 0]
                for k, v in (('w', w[s] + a0 // 6), ('cx', cx[s]), ('cy', cy[s]), ('doff', doff[s]), ('img', img[s].half()), ('score', sc[s]), ('thr', thr[s])):
                    acc[k].append(v.cpu().numpy())
        R = {k: np.concatenate(v) for k, v in acc.items()}; n = len(R['w'])
        print(f'{a.cell} {fi} K = {K}: {n} wavelet candidates ({n / (nblk * S):.1f} per summed window), read + detect {time.time() - t0:.0f}s', flush=True)
        src = RamSource(R['img'], R['doff'], R['w'].astype(np.int64), R['cx'].astype(np.float64), R['cy'].astype(np.float64), dev)
        t1 = time.time(); A = {}
        for j, im, df in src.chunks(a.chunk):
            r = M.stage_a(c4, im, df, refine=(10, 25))
            for k, v in r.items(): A.setdefault(k, []).append(v)
        A = {k: np.concatenate(v) for k, v in A.items()}
        a.order = a.order_npc if a.cell == 'NPC' else a.order_mt
        RC.process(c4, src, a.cell, fi, a, dev, A=A, pix_mask=np.ones(n, bool), extra=dict(score=R['score'], det_thr=R['thr'], box=np.full(n, -1)), tA=time.time() - t1)
        print(f'{a.cell} {fi} K = {K}: classical table written to {OUT}, total {time.time() - t0:.0f}s', flush=True)
        del R, src


if __name__ == '__main__':
    main()
