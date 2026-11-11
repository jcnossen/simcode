"""Pass 1 (front end) on a simulated 4Pi movie (analysis/fourpi/simmovie4pi.py): the same steps as front4pi.py on dcimg
(window sums aligned to step 0, registered 4-channel sum with the registration map, DoG detection with the same
thresholds, unwarped 15 x 15 channel patches at the mapped positions, head4pi), written in the front4pi output format so
the same solve code runs on it. New file; front4pi.py is unchanged.
The sensor layout [T, H, 4 W] is cropped and channels 2/3 un-mirrored as in dc.crops; offset 100 ADU, gain 0.23.
The registration map is the simulator's (the real pipeline uses the in-situ map in the same way).
Usage: CUDA_VISIBLE_DEVICES=2 python analysis/fourpi/simfront4pi.py --tag d1 --model v2
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from front4pi import blur
from simmovie4pi import reg_map, GAIN, OFF
from simcode2.fourpi.head4pi import Head4Pi

P = 15


def detect(E, grid, border, noise='mad', k=8.0, nms=7, s1=1.5):
    nW = E.shape[0] // 6; H, W = E.shape[-2:]
    S6 = E[:nW * 6].reshape(nW, 6, 4, H, W).sum(1)
    ssum = S6[:, 0].clone()
    for c in (1, 2, 3):
        ssum += F.grid_sample(S6[:, c:c + 1], grid[c].expand(nW, -1, -1, -1), mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
    fl = blur(ssum, s1) - blur(ssum, 5.0 if s1 >= 1.5 else 4.0)
    reg = fl.reshape(nW, -1)
    med = reg.median(1).values
    if noise == 'mad':     # the real-data rule (front4pi / process.py)
        sd = 1.4826 * (reg - med[:, None]).abs().median(1).values
    else:                  # 'low': noise from the lower tail (median - 16 % quantile); emitters only lift the upper tail
        sd = med - torch.quantile(reg, 0.16, dim=1)
    mx = F.max_pool2d(fl[:, None], nms, 1, nms // 2)[:, 0]
    pk = (fl == mx) & (fl > torch.maximum(k * sd, torch.full_like(sd, 12.0))[:, None, None])
    pk[:, :border] = False; pk[:, -border:] = False; pk[:, :, :border] = False; pk[:, :, -border:] = False
    w, cy, cx = torch.nonzero(pk, as_tuple=True)
    return w, cy, cx, fl[w, cy, cx], ssum


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='d1'); p.add_argument('--model', default='v2'); p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--chunk', type=int, default=600)
    p.add_argument('--noise', default='low'); p.add_argument('--k', type=float, default=8.0); p.add_argument('--out', default=''); p.add_argument('--nms', type=int, default=7); p.add_argument('--s1', type=float, default=1.5)
    a = p.parse_args()
    from simcode2.util.proctitle import auto; auto('4pi-sim')
    d = f'{REPO}/out/4pi/simmovie/{a.tag}'
    meta = json.load(open(f'{d}/meta.json')); H, W, T = meta['H'], meta['W'], meta['T']
    raw = np.load(f'{d}/movie.npy', mmap_mode='r')
    dev = 'cuda'
    ck = torch.load(f'{REPO}/out/4pi/models/head4pi_{a.model}.pt', weights_only=False, map_location=dev)
    net = Head4Pi(ch=ck['args']['ch'], nblk=ck['args']['nblk']).cuda().eval(); net.load_state_dict(ck['state'])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    grid = [None]; maps = {}
    for c in (1, 2, 3):
        xs, ys = reg_map(c, xx, yy, H, W)
        grid.append(torch.tensor(np.stack([xs / (W - 1) * 2 - 1, ys / (H - 1) * 2 - 1], -1), device=dev, dtype=torch.float32)[None])
        maps[c] = (torch.tensor(xs, device=dev, dtype=torch.float32), torch.tensor(ys, device=dev, dtype=torch.float32))
    acc = {k: [] for k in ('w', 'cx', 'cy', 'doff', 'T', 'C', 'L', 'bg', 'dxy', 'z', 'ls', 'dog', 'nn', 'Tlin', 'Clin', 'roi_sum')}
    t0 = time.time(); h = P // 2
    for a0 in range(0, T, a.chunk):
        r = np.asarray(raw[a0:a0 + a.chunk]).astype(np.float32)
        cr = np.stack([r[:, :, 0:W], r[:, :, W:2 * W], r[:, :, 2 * W:3 * W][:, :, ::-1], r[:, :, 3 * W:4 * W][:, :, ::-1]], 1)
        E = (torch.tensor(cr.copy(), device=dev) - OFF) * GAIN
        with torch.no_grad():
            w, cy, cx, dog, ssum = detect(E, grid, h + 1, a.noise, a.k, a.nms, a.s1)
            if len(w) == 0: continue
            dd = torch.full((len(w),), 99.0, device=dev)
            for wv in torch.unique(w):
                s = torch.nonzero(w == wv)[:, 0]
                if len(s) > 1:
                    pts = torch.stack([cx[s], cy[s]], 1).float(); D = torch.cdist(pts, pts); D.fill_diagonal_(99.0); dd[s] = D.min(1).values
            cxs, cys = [cx.float()], [cy.float()]
            for c in (1, 2, 3):
                mx_, my_ = maps[c]; cxs.append(mx_[cy, cx]); cys.append(my_[cy, cx])
            mxy = torch.stack([torch.stack(cxs, 1), torch.stack(cys, 1)], -1)
            ctr = torch.round(mxy); doff = mxy - ctr
            ix = ctr[..., 0].long() - h; iy = ctr[..., 1].long() - h
            ok = ((ix >= 0) & (ix + P <= W) & (iy >= 0) & (iy + P <= H)).all(1)
            ix = ix.clamp(0, W - P); iy = iy.clamp(0, H - P)
            o = torch.arange(P, device=dev)
            t = 6 * w[:, None] + torch.arange(6, device=dev)[None]
            chn = torch.arange(4, device=dev)
            Y = iy[:, :, None] + o[None, None]; X = ix[:, :, None] + o[None, None]
            img = E[t[:, :, None, None, None], chn[None, None, :, None, None], Y[:, None, :, :, None], X[:, None, :, None, :]]
            sel = torch.nonzero(ok)[:, 0]
            outs = [net(img[sel[i:i + a.batch]], doff[sel[i:i + a.batch]]) for i in range(0, len(sel), a.batch)]
            oo = {k: torch.cat([u[k] for u in outs]) for k in outs[0]}
            # registered-sum ROI for the classical baseline (same detections)
            ws, ys_, xs_ = w[sel], cy[sel], cx[sel]
            rs = ssum[ws[:, None, None], (ys_[:, None, None] + o[None, :, None] - h).clamp(0, H - 1), (xs_[:, None, None] + o[None, None, :] - h).clamp(0, W - 1)]
        for k, v in (('w', w[sel] + a0 // 6), ('cx', cx[sel]), ('cy', cy[sel]), ('doff', doff[sel]), ('T', oo['T']), ('C', oo['C']),
                     ('L', torch.stack([oo['L'][..., 0, 0], oo['L'][..., 1, 1], oo['L'][..., 1, 0]], -1)), ('bg', oo['bg']),
                     ('dxy', oo['dxy']), ('z', oo['z']), ('ls', oo['ls']), ('dog', dog[sel]), ('nn', dd[sel]),
                     ('Tlin', oo['T_lin']), ('Clin', oo['C_lin']), ('roi_sum', rs)):
            acc[k].append(v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy())
    R = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez(f'{d}/front_SIM_000_{a.model}{a.out}.npz', **R, cell='SIM', file_index=0, f0=0, nframes=T, model=a.model)
    print(f'{len(R["w"])} candidates ({len(R["w"]) / (T // 6):.1f} per window) in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
