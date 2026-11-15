"""
Candidate selection on the detector maps (detector/model.Detector outputs).

select_subpixel: every pixel with p > p_min proposes the point pixel + offset; greedy NMS on those POINTS with a sub-pixel
radius (default 0.7 px; computed with shifted maps over a (2 win + 1)^2 neighbourhood, iterated to the greedy fixed point),
so two candidates can come from neighbouring pixels; then top-k per image (static shapes).
select_points: the same on a detector output dict, returning the points with their pixel (iy, ix) for the patch gather.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def select_subpixel(logit, dxy, k, p_min=0.05, radius=0.7, win=3, iters=4, extra=None):
    """Greedy NMS on sub-pixel points (pixel + offset) + top-k per image.
    logit [B,H,W], dxy [B,2,H,W] -> dict b, iy, ix, x, y, logit (each [B*k]; entries below p_min have logit -inf
    and should be dropped by thresholding). extra: dict of [B,(C,)H,W] maps read out at the candidates."""
    B, H, W = logit.shape
    dev = logit.device
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
    X = xx + dxy[:, 0].float(); Y = yy + dxy[:, 1].float()
    s = logit.float()
    valid = s > math.log(p_min / (1 - p_min))
    s = torch.where(valid, s, torch.full_like(s, -1e9))
    pad = lambda t, v: F.pad(t, (win, win, win, win), value=v)
    sP, XP, YP = pad(s, -1e9), pad(X, 1e6), pad(Y, 1e6)
    shifts = [(oy, ox) for oy in range(-win, win + 1) for ox in range(-win, win + 1) if (oy, ox) != (0, 0)]
    # which neighbours (shift) suppress this pixel if they are kept: higher score and point within radius
    sup_by = []
    r2 = radius * radius
    for oy, ox in shifts:
        sl = (slice(None), slice(win + oy, win + oy + H), slice(win + ox, win + ox + W))
        q_s, q_x, q_y = sP[sl], XP[sl], YP[sl]
        higher = (q_s > s) | ((q_s == s) & ((oy, ox) < (0, 0)))
        sup_by.append(higher & ((q_x - X) ** 2 + (q_y - Y) ** 2 < r2) & (q_s > -1e8))
    keep = valid.clone()
    for _ in range(iters):
        kP = F.pad(keep, (win, win, win, win), value=False)
        sup = torch.zeros_like(keep)
        for (oy, ox), m in zip(shifts, sup_by):
            sup |= m & kP[:, win + oy: win + oy + H, win + ox: win + ox + W]
        new = valid & ~sup
        if torch.equal(new, keep):
            break
        keep = new
    sc = torch.where(keep, s, torch.full_like(s, -float('inf')))
    top = sc.reshape(B, -1).topk(min(k, H * W), dim=1)
    kk = top.indices.shape[1]
    b = torch.arange(B, device=dev)[:, None].expand(B, kk).reshape(-1)
    idx = top.indices.reshape(-1)
    iy, ix = idx // W, idx % W
    out = dict(b=b, iy=iy, ix=ix, x=X[b, iy, ix], y=Y[b, iy, ix], logit=top.values.reshape(-1))
    for nm, mp in (extra or {}).items():
        out[nm] = mp[b, ..., iy, ix] if mp.dim() == 3 else mp[b, :, iy, ix]
    return out


@torch.no_grad()
def select_points(out, k=640, p_min=0.2, radius=0.7, win=3):
    """detector output dict -> dict b, iy, ix (the NMS pixel), x, y (px), logit; entries with logit -inf are below p_min.
    (x, y = the NMS point through the cell-centre conversion (c + 0.5) - 0.5 of the earlier code, kept for bit-identical
    points.)"""
    lg = out['logit'].float(); dxy = out['dxy'].float()
    c = select_subpixel(lg, dxy, k, p_min=p_min, radius=radius, win=win)
    x = (c['x'] + 0.5) / 1 - 0.5; y = (c['y'] + 0.5) / 1 - 0.5
    return dict(b=c['b'], iy=c['iy'], ix=c['ix'], x=x, y=y, logit=c['logit'])
