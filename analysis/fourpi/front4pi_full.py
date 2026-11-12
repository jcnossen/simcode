"""Pass-1 front end of the full 4Pi pipeline (SIMCODE v2 + 4Pi) on raw dcimg: learned detector det4pi on the registered
4-channel sum of each 6-frame window (6 frame planes, det_n1 input format), sub-pixel NMS (det/up2.select, p >= 0.3),
then the pass-1 head head4pi_p1 with the neighbour splat (other candidates, p-weighted, sd 1 px) and the
detector-probability crop as extra inputs. Same reader, camera conversion, registration and patch geometry as
front4pi.py (imported); output in the front4pi format plus U [n,48,4] (float16, low-rank covariance factor), p, and the
detector sub-pixel position.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/front4pi_full.py --cell NPC --files 10 --det out/4pi/models/det4pi/final.pt \
           --head out/4pi/models/head4pi_p1/final.pt --model p1 [--nframes 6000]
"""
import argparse, os, sys, time, threading, queue
import numpy as np, torch
import torch.nn.functional as F
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from front4pi import Front, reader, XLIM, P, OUT
from dc import Dcimg, movie_files, file_index


def registered_frames(fr, E):
    """[nW, 6, H, W] registered 4-channel sum per frame (bilinear warp of channels 1-3 onto ch0)"""
    n = E.shape[0]
    s = E[:, 0].clone()
    for c in (1, 2, 3):
        s += F.grid_sample(E[:, c:c + 1], fr.grid[c].expand(n, -1, -1, -1), mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
    nW = n // 6
    return s[:nW * 6].reshape(nW, 6, *s.shape[-2:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='NPC'); p.add_argument('--files', type=int, nargs='+', required=True)
    p.add_argument('--det', required=True); p.add_argument('--head', required=True); p.add_argument('--model', default='p1')
    p.add_argument('--nframes', type=int, default=0); p.add_argument('--f0', type=int, default=0)
    p.add_argument('--chunk', type=int, default=600); p.add_argument('--out', default=OUT); p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--p_min', type=float, default=0.3); p.add_argument('--det_batch', type=int, default=25)
    a = p.parse_args()
    from simcode2.util.proctitle import auto; auto('4pi-front1')
    from simcode2.det.up2 import load as dload, select
    from simcode2.fourpi.head4pi_p1 import Head4PiP1
    os.makedirs(a.out, exist_ok=True)
    det, _ = dload(a.det)
    ck = torch.load(a.head, weights_only=False, map_location='cuda')
    init = torch.load(f"{REPO}/{ck['args']['init']}", weights_only=False)
    head = Head4PiP1(ch=init['args']['ch'], nblk=init['args']['nblk']).cuda().eval(); head.load_state_dict(ck['state'])
    fr = Front(a.cell)
    h = P // 2
    oo = torch.arange(P, device='cuda')
    yy, xx = torch.meshgrid(oo.float() - h, oo.float() - h, indexing='ij')
    for fi in a.files:
        fn = [f for f in movie_files(a.cell) if file_index(f) == fi][0]
        nfr = a.nframes or Dcimg(fn).nframes
        q = queue.Queue(maxsize=2)
        th = threading.Thread(target=reader, args=(fn, a.f0, nfr, a.chunk, q), daemon=True); th.start()
        keys = ('w', 'cx', 'cy', 'doff', 'T', 'C', 'L', 'U', 'bg', 'dxy', 'z', 'ls', 'p', 'xd', 'yd', 'nn', 'Tlin', 'Clin')
        acc = {k: [] for k in keys}
        t0 = time.time(); nc = 0
        while True:
            it = q.get()
            if it is None: break
            a0, raw = it
            with torch.no_grad():
                E = fr.to_e(raw)
                Rf = registered_frames(fr, E)                                     # [nW,6,200,360]
                ws, cxs, cys, ps, xds, yds, pmaps = [], [], [], [], [], [], []
                for i in range(0, Rf.shape[0], a.det_batch):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = det(Rf[i:i + a.det_batch])
                    out = {k: v.float() for k, v in out.items()}
                    c = select(out, 1, k=4096, p_min=a.p_min)
                    ok = torch.isfinite(c['logit'])
                    ws.append(c['b'][ok] + i); cxs.append(c['ix'][ok]); cys.append(c['iy'][ok]); ps.append(torch.sigmoid(c['logit'][ok]))
                    xds.append(c['x'][ok]); yds.append(c['y'][ok]); pmaps.append(torch.sigmoid(out['logit']))
                w = torch.cat(ws); cx = torch.cat(cxs); cy = torch.cat(cys); pc = torch.cat(ps); xd = torch.cat(xds); yd = torch.cat(yds)
                pmap = torch.cat(pmaps)
                ins = (cx >= XLIM[0]) & (cx < XLIM[1]) & (cy >= h + 1) & (cy < 200 - h - 1)
                w, cx, cy, pc, xd, yd = w[ins], cx[ins], cy[ins], pc[ins], xd[ins], yd[ins]
                if len(w) == 0: continue
                img, doff, ok = fr.patches(E, w, cy, cx)
                sel = torch.nonzero(ok)[:, 0]
                # neighbour splat (other candidates of the window, p-weighted) and the detector-probability crop
                n = len(sel); ws_, cx_, cy_ = w[sel], cx[sel], cy[sel]
                splat = torch.zeros(n, P, P, device='cuda')
                same = (ws_[:, None] == ws_[None]) & ((cx_[:, None] - cx_[None]).abs() <= h + 2) & ((cy_[:, None] - cy_[None]).abs() <= h + 2)
                same.fill_diagonal_(False)
                i_, j_ = torch.nonzero(same, as_tuple=True)
                if len(i_):
                    dxn = (xd[sel][j_] - cx_[i_].float()); dyn = (yd[sel][j_] - cy_[i_].float())
                    splat.index_add_(0, i_, torch.exp(-((xx[None] - dxn[:, None, None]) ** 2 + (yy[None] - dyn[:, None, None]) ** 2) / 2.0) * pc[sel][j_, None, None])
                Yd = (cy_[:, None] + oo[None] - h).clamp(0, 199); Xd = (cx_[:, None] + oo[None] - h).clamp(0, 359)
                pcrop = pmap[ws_[:, None, None], Yd[:, :, None], Xd[:, None, :]]
                extra = torch.stack([splat, pcrop], 1)
                dd = torch.full((n,), 99.0, device='cuda')
                for wv in torch.unique(ws_):
                    s_ = torch.nonzero(ws_ == wv)[:, 0]
                    if len(s_) > 1:
                        D = torch.cdist(torch.stack([xd[sel][s_], yd[sel][s_]], 1), torch.stack([xd[sel][s_], yd[sel][s_]], 1)); D.fill_diagonal_(99.0); dd[s_] = D.min(1).values
                outs = [head(img[sel[i:i + a.batch]], doff[sel[i:i + a.batch]], extra[i:i + a.batch]) for i in range(0, n, a.batch)]
                o = {k: torch.cat([u[k] for u in outs]) for k in outs[0]}
            for k, v in (('w', ws_ + a0 // 6), ('cx', cx_), ('cy', cy_), ('doff', doff[sel]), ('T', o['T']), ('C', o['C']),
                         ('L', torch.stack([o['L'][..., 0, 0], o['L'][..., 1, 1], o['L'][..., 1, 0]], -1)), ('U', o['U'].half()),
                         ('bg', o['bg']), ('dxy', o['dxy']), ('z', o['z']), ('ls', o['ls']), ('p', pc[sel]), ('xd', xd[sel]), ('yd', yd[sel]),
                         ('nn', dd), ('Tlin', o['T_lin']), ('Clin', o['C_lin'])):
                acc[k].append(v.cpu().numpy() if v.dtype == torch.float16 else (v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy()))
            nc += 1
            if nc % 10 == 0:
                print(f'{a.cell} {fi}: frame {a0 + a.chunk}/{a.f0 + nfr}, {sum(len(x) for x in acc["w"])} cand, {time.time() - t0:.0f}s', flush=True)
        R = {k: np.concatenate(v) for k, v in acc.items()}
        R['dog'] = R['p']                                                         # solve_real4pi reads 'dog' (unused there)
        fn_out = f'{a.out}/front_{a.cell}_{fi:03d}_{a.model}.npz'
        np.savez(fn_out, **R, cell=a.cell, file_index=fi, f0=a.f0, nframes=nfr, model=a.model)
        print(f'{a.cell} {fi}: {len(R["w"])} candidates in {time.time() - t0:.0f}s -> {fn_out}', flush=True)


if __name__ == '__main__':
    main()
