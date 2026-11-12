"""Train head4pi_p1 (simcode2/fourpi/head4pi_p1.py) on detector candidates in full 4Pi scenes (realsim4pi), as the main
pipeline's sc4 head is trained on det_n1 candidates in realsim scenes.

Per batch: realsim4pi scenes -> det4pi on the registered sum (or GT-jitter candidates with --det '') -> candidates with
p >= 0.3 inside the scene (8 px border) -> per channel the unwarped 15 x 15 patch at the mapped candidate position (the
scene's registration), the neighbour splat (other candidates, p-weighted, sd 1 px) and the detector-probability crop ->
matched GT emitter (nearest, < 1 px, >= 1 on-frame, >= 60 window photons) -> targets T, C [6,4], x, y (rel. to the
candidate pixel), z. Unmatched candidates carry no read-out target (as sc4).
Loss: low-rank-covariance NLL / 48 + prior NLL / 3 + w_mr x mean-removed term; nn-bin weights (sc4.nn_weights).
Init: head4pi v3 (new input planes and the low-rank outputs zero-initialised).
Usage: CUDA_VISIBLE_DEVICES=2 python analysis/fourpi/train_head4pi_p1.py --out out/4pi/models/head4pi_p1 --det out/4pi/models/det4pi/final.pt
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
REPO = '/home/jc/simcode2'
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.ckpt import save as safe_save   # atomic, disk-guarded
P = 15


def worker(q, seed, psf, B, size, dens):
    import torch
    from simcode2.fourpi.realsim4pi import RealSim4Pi
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/{psf}.json'))['psf']
    rs = RealSim4Pi(st, H=size, W=size, seed=seed, n_psf=6)
    while True:
        S = rs.sample(B=B, dens=dens)
        keep = ('img', 'origin', 'ox', 'oy', 'det_in')
        o = {k: S[k].cpu() for k in keep}; o['cell'] = S['cell']
        o['gt'] = {k: v.cpu() for k, v in S['gt'].items()}
        q.put(o)


def candidates(o, det, dev, p_min=0.3, jit=0.35, border=8):
    """-> b, cx, cy (integer candidate pixel, ch0 scene coords), p, and the probability map [B,H,W]"""
    fr = o['det_in']; B, T, H, W = fr.shape
    if det is not None:
        from simcode2.det.up2 import select
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = det(fr)
        out = {k: v.float() for k, v in out.items()}
        c = select(out, 1, k=640, p_min=p_min)
        ok = torch.isfinite(c['logit'])
        b, iy, ix, lg = c['b'][ok], c['iy'][ok], c['ix'][ok], c['logit'][ok]
        p = torch.sigmoid(lg); pmap = torch.sigmoid(out['logit'])
    else:
        g = o['gt']
        on = (g['q'].sum(1) >= 1) & (g['Nt'].sum(1) >= 60)
        b = g['b'][on]
        ix = torch.round(g['x'][on] + jit * torch.randn(int(on.sum()), device=dev)).long()
        iy = torch.round(g['y'][on] + jit * torch.randn(int(on.sum()), device=dev)).long()
        p = torch.ones(len(b), device=dev); pmap = torch.zeros(B, H, W, device=dev)
        pmap[b.clamp(0, B - 1), iy.clamp(0, H - 1), ix.clamp(0, W - 1)] = 1.0
    ins = (ix >= border) & (ix < W - border) & (iy >= border) & (iy < H - border)
    return b[ins], ix[ins], iy[ins], p[ins], pmap


def build_inputs(o, reg, b, cx, cy, p, pmap, dev):
    """channel patches [n,6,4,P,P], doff [n,4,2], extra [n,2,P,P]"""
    h = P // 2
    img = o['img']; B, T, C, H, W = img.shape
    n = len(b)
    X = cx.float() + o['ox'][b]; Y = cy.float() + o['oy'][b]
    pt = torch.zeros(n, 6, 4, P, P, device=dev); doff = torch.zeros(n, 4, 2, device=dev)
    oo = torch.arange(P, device=dev)
    for cname in set(o['cell']):
        msc = torch.tensor([cc == cname for cc in o['cell']], device=dev)
        me = msc[b]
        if not me.any(): continue
        for c in range(4):
            xc, yc = reg[cname](c, X[me], Y[me])
            lx = xc - o['origin'][b[me], c, 0]; ly = yc - o['origin'][b[me], c, 1]
            rx, ry = torch.round(lx), torch.round(ly)
            doff[me, c, 0] = lx - rx; doff[me, c, 1] = ly - ry
            Yi = (ry.long()[:, None] + oo[None] - h).clamp(0, H - 1); Xi = (rx.long()[:, None] + oo[None] - h).clamp(0, W - 1)
            bb = b[me]
            pt[me, :, c] = img[bb[:, None, None, None], torch.arange(6, device=dev)[None, :, None, None], c, Yi[:, None, :, None], Xi[:, None, None, :]]
    # neighbour splat (other candidates, p-weighted, sd 1 px) and the detector-probability crop, in ch0 patch coordinates
    yy, xx = torch.meshgrid(oo.float() - h, oo.float() - h, indexing='ij')
    splat = torch.zeros(n, P, P, device=dev)
    same = (b[:, None] == b[None]) & ((cx[:, None] - cx[None]).abs() <= h + 2) & ((cy[:, None] - cy[None]).abs() <= h + 2)
    same.fill_diagonal_(False)
    i, j = torch.nonzero(same, as_tuple=True)
    if len(i):
        dxn = (cx[j] - cx[i]).float(); dyn = (cy[j] - cy[i]).float()
        blob = torch.exp(-((xx[None] - dxn[:, None, None]) ** 2 + (yy[None] - dyn[:, None, None]) ** 2) / 2.0) * p[j, None, None]
        splat.index_add_(0, i, blob)
    Yd = (cy[:, None] + oo[None] - h).clamp(0, pmap.shape[1] - 1); Xd = (cx[:, None] + oo[None] - h).clamp(0, pmap.shape[2] - 1)
    pc = pmap[b[:, None, None], Yd[:, :, None], Xd[:, None, :]]
    return pt, doff, torch.stack([splat, pc], 1)


def match(o, b, cx, cy, dev, r=1.0):
    g = o['gt']
    act = (g['q'].sum(1) >= 1) & (g['Nt'].sum(1) >= 60)
    gb, gx, gy = g['b'][act], g['x'][act], g['y'][act]
    idx = torch.nonzero(act)[:, 0]
    d2 = (cx.float()[:, None] - gx[None]) ** 2 + (cy.float()[:, None] - gy[None]) ** 2
    d2 = torch.where(b[:, None] == gb[None], d2, torch.full_like(d2, 1e9))
    dm, j = d2.min(1)
    ok = dm < r * r
    return ok, idx[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--det', default='')
    ap.add_argument('--init', default='out/4pi/models/head4pi_v3.pt'); ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--pct', type=float, default=0.05)
    ap.add_argument('--w_mr', type=float, default=2.0); ap.add_argument('--psf', default='NPC_f_dxy15')
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--dens', default='0.01,2.0')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ckpt_every', type=int, default=5000)
    a = ap.parse_args()
    from simcode2.util.proctitle import auto; auto('4pi-head1')
    from simcode2.fourpi.head4pi_p1 import Head4PiP1, nll_lowrank, mean_removed
    from simcode2.fourpi.head4pi import nll_prior
    from simcode2.fourpi.realsim4pi import RegMap
    from simcode2.hd.sc4 import nn_weights
    dev = 'cuda'
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    ck = torch.load(f'{REPO}/{a.init}' if not a.init.startswith('/') else a.init, weights_only=False)
    net = Head4PiP1.from_v3(ck).to(dev)
    det = None
    if a.det:
        from simcode2.det.up2 import load as dload
        det, _ = dload(a.det)
    reg = {c: RegMap(c, dev) for c in ('MT', 'NPC')}
    ctx = mp.get_context('spawn'); q = ctx.Queue(maxsize=6)
    dens = tuple(float(v) for v in a.dens.split(','))
    ps = [ctx.Process(target=worker, args=(q, a.seed * 100 + i, a.psf, a.batch, a.size, dens), daemon=True) for i in range(a.workers)]
    for p in ps: p.start()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a'); t0 = time.time(); acc = []
    for step in range(a.steps + 1):
        o = q.get()
        o = {k: (v.to(dev) if torch.is_tensor(v) else ({kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in o.items()}
        with torch.no_grad():
            b, cx, cy, p, pmap = candidates(o, det, dev)
            ok, gi = match(o, b, cx, cy, dev)
            b, cx, cy, p, gi = b[ok], cx[ok], cy[ok], p[ok], gi[ok]
            if len(b) < 8: continue
            pt, doff, extra = build_inputs(o, reg, b, cx, cy, p, pmap, dev)
            g = o['gt']
            Tg, Cg = g['T'][gi], g['C'][gi]
            tx, ty, tz = g['x'][gi] - cx.float(), g['y'][gi] - cy.float(), g['z'][gi]
            gx, gy = g['x'], g['y']
            d2 = (gx[gi][:, None] - gx[None]) ** 2 + (gy[gi][:, None] - gy[None]) ** 2
            d2 = torch.where((g['b'][gi][:, None] == g['b'][None]) & (d2 > 1e-6), d2, torch.full_like(d2, 1e6))
            wt = nn_weights(b, d2.min(1).values.sqrt())
        out = net(pt, doff, extra)
        l_i = nll_lowrank(Tg, Cg, out) / 48
        l_p = nll_prior(tx, ty, tz, out) / 3
        l_mr = mean_removed(Tg, out)
        loss = (wt * (l_i + l_p + a.w_mr * l_mr)).sum() / wt.sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        acc.append([loss.item(), l_i.mean().item(), l_p.mean().item(), l_mr.mean().item(), len(b)])
        if step % 250 == 0:
            m = torch.tensor(acc).mean(0).tolist(); acc = []
            with torch.no_grad():
                ez = (out['z'] - tz).abs().median().item() * 1000
                ex = ((out['dxy'][:, 0] - tx) * 128).abs().median().item()
            r = dict(step=step, t=round(time.time() - t0), loss=m[0], nll_I=m[1], nll_p=m[2], mr=m[3], n_cand=m[4],
                     prior_dx_med_nm=ex, prior_dz_med_nm=ez, lr=sched.get_last_lr()[0], qsize=q.qsize())
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if step % a.ckpt_every == 0 and step > 0:
            safe_save(dict(state=net.state_dict(), args=vars(a), step=step), a.out + f'/ckpt_{step}.pt')
    safe_save(dict(state=net.state_dict(), args=vars(a), step=a.steps), a.out + '/final.pt')
    for p in ps: p.terminate()


if __name__ == '__main__':
    main()
