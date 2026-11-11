"""4Pi-SIMFLUX front end on raw dcimg: detection on the registered 4-channel 6-frame sum, unwarped 15 x 15 channel
patches at the mapped positions, the 4Pi head (simcode2/fourpi/head4pi.py). Writes the head outputs per candidate.

Windows: frames 6w .. 6w+5 of each file (every file starts at pattern step 0, SMLM_REPORT 1.3).
Detection (same rule as the classical pass 1, process.py, on GPU): DoG (1.5 / 5 px) on the registered sum, 7 x 7
local maximum, > 8 robust sd and > 12 e, x in [55, 305) of the crop, away from the border.
Per candidate (integer ch0 pixel cx, cy) and channel c: mapped position pm.inv(c, cx, cy) (in-situ polynomial map),
patch centre = round(mapped), doff_c = mapped - centre (x, y).
Camera: offset map (out/4pi/smlm/camera), gain 0.23 e/ADU.
Output npz per file: w, cx, cy, doff [n,4,2], T, C [n,6,4], L [n,6,4,3] (l11, l22, l21), bg [n,6,4], dxy [n,2],
z [n], ls [n,3], dog, nn (px to the nearest other detection in the window).
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/front4pi.py --cell NPC --files 10 --model v2 [--nframes 6000]
"""
import argparse, os, sys, time, threading, queue
import numpy as np, torch
import torch.nn.functional as F
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from dc import Dcimg, movie_files, crops, file_index
from register import PolyMap
from simcode2.fourpi.head4pi import Head4Pi

GAIN = 0.23
XLIM = (55, 305)
P = 15
OUT = f'{REPO}/out/4pi/net'


def gauss_kernel(s, dev):
    r = int(np.ceil(3 * s)); x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / s) ** 2); return k / k.sum()


def blur(img, s):
    k = gauss_kernel(s, img.device); r = len(k) // 2
    x = F.conv2d(F.pad(img[:, None], (r, r, 0, 0), mode='replicate'), k[None, None, None, :])
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), k[None, None, :, None])
    return x[:, 0]


class Front:
    def __init__(self, cell, dev='cuda'):
        self.dev = dev
        self.off = torch.tensor(np.load(f'{REPO}/out/4pi/smlm/camera/camera_{cell}.npz')['off'], device=dev, dtype=torch.float32)
        self.pm = pm = PolyMap.load(f'{REPO}/out/4pi/smlm/reg/reg_insitu_{cell}.json')
        yy, xx = np.mgrid[0:200, 0:360].astype(np.float64)
        self.grid = [None]
        for c in (1, 2, 3):
            xs, ys = pm.inv(c, xx, yy)
            g = torch.tensor(np.stack([xs / 359 * 2 - 1, ys / 199 * 2 - 1], -1), device=dev, dtype=torch.float32)
            self.grid.append(g[None])
        self.maps = {c: (torch.tensor(pm.inv(c, xx, yy)[0], device=dev, dtype=torch.float32),
                         torch.tensor(pm.inv(c, xx, yy)[1], device=dev, dtype=torch.float32)) for c in (1, 2, 3)}

    def to_e(self, raw):
        return (torch.tensor(crops(raw), device=self.dev) - self.off) * GAIN            # [n,4,200,360]

    def detect(self, E):
        nW = E.shape[0] // 6
        S6 = E[:nW * 6].reshape(nW, 6, 4, 200, 360).sum(1)                               # [nW,4,H,W]
        ssum = S6[:, 0].clone()
        for c in (1, 2, 3):
            ssum += F.grid_sample(S6[:, c:c + 1], self.grid[c].expand(nW, -1, -1, -1), mode='bilinear',
                                  padding_mode='border', align_corners=True)[:, 0]
        fl = blur(ssum, 1.5) - blur(ssum, 5.0)
        reg = fl[:, :, XLIM[0]:XLIM[1]].reshape(nW, -1)
        med = reg.median(1).values; sd = 1.4826 * (reg - med[:, None]).abs().median(1).values
        mx = F.max_pool2d(fl[:, None], 7, 1, 3)[:, 0]
        pk = (fl == mx) & (fl > torch.maximum(8 * sd, torch.full_like(sd, 12.0))[:, None, None])
        pk[:, :, :XLIM[0]] = False; pk[:, :, XLIM[1]:] = False
        h = P // 2 + 1
        pk[:, :h] = False; pk[:, -h:] = False
        w, cy, cx = torch.nonzero(pk, as_tuple=True)
        return w, cy, cx, fl[w, cy, cx]

    def patches(self, E, w, cy, cx):
        """-> img [n,6,4,P,P], doff [n,4,2], ok [n]"""
        n = len(w); h = P // 2
        cxs, cys = [cx.float()], [cy.float()]
        for c in (1, 2, 3):
            mx, my = self.maps[c]; cxs.append(mx[cy, cx]); cys.append(my[cy, cx])
        mxy = torch.stack([torch.stack(cxs, 1), torch.stack(cys, 1)], -1)                 # [n,4,2]
        ctr = torch.round(mxy); doff = mxy - ctr
        ix = ctr[..., 0].long() - h; iy = ctr[..., 1].long() - h                           # [n,4]
        ok = ((ix >= 0) & (ix + P <= 360) & (iy >= 0) & (iy + P <= 200)).all(1)
        ix = ix.clamp(0, 360 - P); iy = iy.clamp(0, 200 - P)
        o = torch.arange(P, device=E.device)
        t = (6 * w[:, None] + torch.arange(6, device=E.device)[None])                      # [n,6]
        chn = torch.arange(4, device=E.device)
        Y = (iy[:, :, None] + o[None, None])                                               # [n,4,P]
        X = (ix[:, :, None] + o[None, None])
        img = E[t[:, :, None, None, None], chn[None, None, :, None, None], Y[:, None, :, :, None], X[:, None, :, None, :]]
        return img, doff, ok


def reader(fn, f0, nfr, chunk, q):
    d = Dcimg(fn)
    for a in range(f0, f0 + nfr, chunk):
        q.put((a, d.raw(a, min(chunk, f0 + nfr - a))))
    q.put(None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='NPC'); p.add_argument('--files', type=int, nargs='+', required=True)
    p.add_argument('--model', default='v2'); p.add_argument('--nframes', type=int, default=0)
    p.add_argument('--f0', type=int, default=0); p.add_argument('--chunk', type=int, default=600)
    p.add_argument('--out', default=OUT); p.add_argument('--batch', type=int, default=4096)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(f'{REPO}/out/4pi/models/head4pi_{a.model}.pt', weights_only=False, map_location='cuda')
    net = Head4Pi(ch=ck['args']['ch'], nblk=ck['args']['nblk']).cuda().eval(); net.load_state_dict(ck['state'])
    fr = Front(a.cell)
    for fi in a.files:
        fn = [f for f in movie_files(a.cell) if file_index(f) == fi][0]
        nfr = a.nframes or Dcimg(fn).nframes
        q = queue.Queue(maxsize=2)
        th = threading.Thread(target=reader, args=(fn, a.f0, nfr, a.chunk, q), daemon=True); th.start()
        acc = {k: [] for k in ('w', 'cx', 'cy', 'doff', 'T', 'C', 'L', 'bg', 'dxy', 'z', 'ls', 'dog', 'nn', 'Tlin', 'Clin')}
        t0 = time.time(); nc = 0
        while True:
            it = q.get()
            if it is None: break
            a0, raw = it
            with torch.no_grad():
                E = fr.to_e(raw)
                w, cy, cx, dog = fr.detect(E)
                if len(w) == 0: continue
                # nearest other detection in the same window
                key = w.double() * 1e4
                dd = torch.full((len(w),), 99.0, device='cuda')
                for wv in torch.unique(w):
                    s = torch.nonzero(w == wv)[:, 0]
                    if len(s) > 1:
                        D = torch.cdist(torch.stack([cx[s], cy[s]], 1).float(), torch.stack([cx[s], cy[s]], 1).float())
                        D.fill_diagonal_(99.0); dd[s] = D.min(1).values
                img, doff, ok = fr.patches(E, w, cy, cx)
                sel = torch.nonzero(ok)[:, 0]
                outs = []
                for i in range(0, len(sel), a.batch):
                    j = sel[i:i + a.batch]
                    outs.append(net(img[j], doff[j]))
                o = {k: torch.cat([u[k] for u in outs]) for k in outs[0]}
            wg = w[sel] + (a0 - a.f0) // 6 + a.f0 // 6
            for k, v in (('w', wg), ('cx', cx[sel]), ('cy', cy[sel]), ('doff', doff[sel]), ('T', o['T']), ('C', o['C']),
                         ('L', torch.stack([o['L'][..., 0, 0], o['L'][..., 1, 1], o['L'][..., 1, 0]], -1)), ('bg', o['bg']),
                         ('dxy', o['dxy']), ('z', o['z']), ('ls', o['ls']), ('dog', dog[sel]), ('nn', dd[sel]),
                         ('Tlin', o['T_lin']), ('Clin', o['C_lin'])):
                acc[k].append(v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy())
            nc += 1
            if nc % 10 == 0:
                print(f'{a.cell} {fi}: frame {a0 + a.chunk}/{a.f0 + nfr}, {sum(len(x) for x in acc["w"])} cand, {time.time() - t0:.0f}s', flush=True)
        R = {k: np.concatenate(v) for k, v in acc.items()}
        fn_out = f'{a.out}/front_{a.cell}_{fi:03d}_{a.model}.npz'
        np.savez(fn_out, **R, cell=a.cell, file_index=fi, f0=a.f0, nframes=nfr, model=a.model)
        print(f'{a.cell} {fi}: {len(R["w"])} candidates in {time.time() - t0:.0f}s -> {fn_out}', flush=True)


if __name__ == '__main__':
    main()
