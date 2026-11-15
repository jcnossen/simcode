"""
Training targets and loss weights from the simulated ground truth (detector and emitter-network trainers).

  assign / detector_targets  detector cell targets: GT emitters with >= 1 on-frame (sum of on-fractions) and >= min_phot window
                             photons take their nearest pixel; when two fall in the same pixel the brighter keeps it and the
                             other takes the free 8-neighbour pixel nearest to it (offset target up to ~1.2 px), so close pairs
                             get two positives; still-colliding ones are ignored. Emitters with 0 < on-frames < 1 or fewer
                             photons are 'don't care' (their pixel is excluded from the negatives).
  gt_nn                      nearest-neighbour distance among the active GT emitters of a window
  nn_bin / nn_weights        per-target weights: the nn bins < 1.5 / 1.5-2.5 / 2.5-4 / > 4 px carry 0.15 / 0.25 / 0.25 / 0.35 of
                             each window's weight (empty bins renormalised away); each window's weights sum to its target count
  match_gt                   one-to-one matching (< 1 px) of detector candidates to the active GT
  emitter_targets / ref_var  GT quantities of matched candidates for the emitter-network losses
"""
import numpy as np, torch
from simcode.layers.blocks import extract_patches

dev = 'cuda'
NN_EDGES = (1.5, 2.5, 4.0)
NN_W = (0.15, 0.25, 0.25, 0.35)


# ------------------------------------------------------------------------------------------------ detector
def assign(gt, B, H, W, min_on=1.0, min_phot=60.0, dontcare=None):
    """-> label [B,H,W] (1 pos, 0 neg), ignore [B,H,W] bool, tidx [B,H,W] (GT index, -1), off [B,2,H,W] targets."""
    tot = gt['phot'].sum(1); on = gt['onf'].sum(1)
    px, py = torch.round(gt['x']).long(), torch.round(gt['y']).long()
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    pos = inside & (on >= min_on) & (tot >= min_phot)
    if dontcare is not None:     # e.g. the dim carpet: never a positive, its pixel is excluded from the negatives
        pos = pos & ~dontcare
    dc = inside & (on > 0) & ~pos
    HW = H * W
    ign = torch.zeros(B * HW, dtype=torch.bool, device=dev)
    ign[(gt['fov'] * HW + py * W + px)[dc]] = True
    tid = torch.full((B * HW,), -1, dtype=torch.long, device=dev)
    idx = torch.nonzero(pos)[:, 0]
    key = tot[idx] + 1e-6 * torch.arange(len(idx), device=dev)          # unique keys: deterministic winner
    lin = gt['fov'][idx] * HW + py[idx] * W + px[idx]
    best = torch.full((B * HW,), -1.0, device=dev).scatter_reduce(0, lin, key, 'amax')
    win = key == best[lin]
    tid[lin[win]] = idx[win]
    lose = idx[~win]
    if len(lose):
        o = torch.tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)], device=dev)
        cy = py[lose][:, None] + o[None, :, 0]; cx = px[lose][:, None] + o[None, :, 1]
        okc = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
        cl = gt['fov'][lose][:, None] * HW + cy.clamp(0, H - 1) * W + cx.clamp(0, W - 1)
        okc &= tid[cl] < 0
        dd = (gt['x'][lose][:, None] - cx) ** 2 + (gt['y'][lose][:, None] - cy) ** 2
        dd = torch.where(okc, dd, torch.full_like(dd, 1e9))
        j = dd.argmin(1); ok = dd.min(1).values < 1e8
        lin2 = cl[torch.arange(len(lose), device=dev), j][ok]; l2 = lose[ok]
        k2 = tot[l2] + 1e-6 * torch.arange(len(l2), device=dev)
        best2 = torch.full((B * HW,), -1.0, device=dev).scatter_reduce(0, lin2, k2, 'amax')
        w2 = k2 == best2[lin2]
        tid[lin2[w2]] = l2[w2]
    tid = tid.reshape(B, H, W)
    label = (tid >= 0).float()
    ign = ign.reshape(B, H, W) & (tid < 0)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    tc = tid.clamp(min=0)
    off = torch.stack([gt['x'][tc] - xx, gt['y'][tc] - yy], 1)
    return label, ign, tid, off


def detector_targets(o, up, H, W, **kw):
    """cell targets of a batch o (o['gt'], o['nb']) on the up x grid (up = 1: the detector's native grid; offsets in native px);
    kw: min_on / min_phot of assign"""
    gt = o['gt']
    g2 = dict(gt, x=(gt['x'] + 0.5) * up - 0.5, y=(gt['y'] + 0.5) * up - 0.5)
    label, ign, tid, off = assign(g2, o['nb'], H * up, W * up, **kw)
    return label, ign, tid, off / up                                    # offsets in native px


def gt_nn(gt):
    """nearest-neighbour distance (px) among the active GT emitters of the same window"""
    act = gt['onf'].sum(1) > 0
    x, y, f = gt['x'], gt['y'], gt['fov']
    nn_ = torch.full_like(x, 1e3)
    D = (x[:, None] - x[None]) ** 2 + (y[:, None] - y[None]) ** 2
    ok = (f[:, None] == f[None]) & act[None] & ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    D = torch.where(ok, D, torch.full_like(D, 1e6))
    if len(x): nn_ = D.min(1).values.sqrt()
    return nn_


# ------------------------------------------------------------------------------------------------ weights
def nn_bin(nnd):
    b = torch.zeros_like(nnd, dtype=torch.long)
    for e in NN_EDGES: b = b + (nnd >= e).long()
    return b


def nn_weights(fov, nnd):
    """per-target weights: within each window the nn bins get NN_W of the weight (empty bins renormalised), split
    equally among the bin's targets; each window's weights sum to its target count"""
    bn = nn_bin(nnd)
    key = fov * 4 + bn
    cnt = torch.zeros(int(key.max()) + 1, device=fov.device).index_add_(0, key, torch.ones_like(nnd))
    wb = torch.tensor(NN_W, device=fov.device)
    w = wb[bn] / cnt[key]
    nf = int(fov.max()) + 1
    tot = torch.zeros(nf, device=fov.device).index_add_(0, fov, w)
    nw = torch.zeros(nf, device=fov.device).index_add_(0, fov, torch.ones_like(w))
    return w / tot[fov] * nw[fov]


# ------------------------------------------------------------------------------------------------ emitter network
def match_gt(o, c, dev=dev):
    """one-to-one matching (< 1 px) of detector candidates to the active GT"""
    from scipy.optimize import linear_sum_assignment
    nb = o['frames'].shape[0]
    keep = torch.isfinite(c['logit'])
    cb, cy_, cx_, xx, yy = c['b'][keep], c['iy'][keep], c['ix'][keep], c['x'][keep], c['y'][keep]
    G = o['gt']; act = (G['onf'].sum(1) > 0) & (G['phot'].sum(1) >= 50)
    gi = torch.nonzero(act)[:, 0]
    mi, mg = [], []
    gb, gx, gy = G['fov'][gi].cpu().numpy(), G['x'][gi].cpu().numpy(), G['y'][gi].cpu().numpy()
    pb, px, py = cb.cpu().numpy(), xx.cpu().numpy(), yy.cpu().numpy()
    for f in range(nb):
        ip = np.nonzero(pb == f)[0]; ig = np.nonzero(gb == f)[0]
        if len(ip) == 0 or len(ig) == 0: continue
        Dm = (px[ip, None] - gx[None, ig]) ** 2 + (py[ip, None] - gy[None, ig]) ** 2
        Cc = np.where(Dm < 1, Dm, 1e6); r_, c_ = linear_sum_assignment(Cc); ok = Cc[r_, c_] < 1
        mi.append(ip[r_[ok]]); mg.append(ig[c_[ok]])
    if not mi: return None
    mi = torch.tensor(np.concatenate(mi), device=dev); mg = gi[torch.tensor(np.concatenate(mg), device=dev)]
    tx, ty = G['x'][mg], G['y'][mg]
    Dg = torch.cdist(torch.stack([tx, ty], 1), torch.stack([G['x'][gi], G['y'][gi]], 1))
    Dg = torch.where(G['fov'][mg][:, None] == G['fov'][gi][None], Dg, torch.full_like(Dg, 1e9))
    nnd = Dg.sort(1).values[:, 1] if Dg.shape[1] > 1 else torch.full_like(tx, 1e9)
    return dict(b=cb[mi], iy=cy_[mi], ix=cx_[mi], x=xx[mi], y=yy[mi], g=mg, nn=nnd)


def ref_var(Ig, g, bgp, rn2p):
    """per-frame intensity Fisher variance with known PSF template g [n,P,P] (true offset), bg [n,T,P,P], rn^2 [n,1,P,P]:
    1 / sum_q g^2 / (I g + bg + rn^2)"""
    mu = Ig.clamp(min=0)[..., None, None] * g[:, None] + bgp.clamp(min=0.3) + rn2p
    return 1.0 / ((g[:, None] ** 2) / mu).sum((2, 3)).clamp(min=1e-9)


def emitter_targets(o, m, col=None):
    """GT quantities for matched candidates: Ig, tx, ty, pt, t0, t1, bg patch [n,T,P,P], rn2 at the target pixel,
    and the bg at the rounded true pixel (the bg target)"""
    G = o['gt']; g = m['g']; nb = o['frames'].shape[0]
    tx, ty = G['x'][g], G['y'][g]
    bgp = extract_patches(o['bg'], m['b'], m['iy'], m['ix'])
    rn = o['rn'].expand(nb, 1, *o['rn'].shape[-2:])
    rn2 = (rn ** 2)[G['fov'][g], 0, torch.round(ty).long().clamp(0, 63), torch.round(tx).long().clamp(0, 63)]
    bgt_px = o['bg'][G['fov'][g], :, torch.round(ty).long().clamp(0, 63), torch.round(tx).long().clamp(0, 63)]
    rn2p = extract_patches((rn ** 2).float(), m['b'], m['iy'], m['ix'])
    C = ref_var(G['phot'][g], col(torch.stack([tx - m['ix'].float(), ty - m['iy'].float()], 1)), bgp, rn2p)
    return dict(C=C, Ig=G['phot'][g], tx=tx, ty=ty, pt=o['patt'][G['fov'][g]], t0=G['t0'][g], t1=G['t1'][g], bgp=bgp,
                rn2=rn2[:, None], bgt_px=bgt_px)
