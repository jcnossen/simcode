"""Multi-channel front end (the 4Pi network pipeline and the classical 4Pi pipeline): camera conversion of the channel
crops, the registered channel sum (detector input), the classical detection on it, and the unwarped P x P patches of
every channel at the mapped candidate position.

  ChannelFront(regmap, offset, gain, C, H, W, P, xlim, weights, dev)
      regmap  PolyMap (simcode/channels/registration.py): channel c <-> reference channel 0
      offset  camera offset (ADU): scalar or a map [C, H, W] of the channel crops; gain e / ADU
      to_e(crops ADU [n, C, H, W] numpy) -> photo-electrons (torch, dev)
      registered_sum(X [n, C, H, W]) -> [n, H, W]: channel 0 + the other channels resampled into channel-0 coordinates
                     (bilinear), each times its weight (1.0: unchanged)
      detect(E, window)  classical detection on the registered sum of each window of `window` frames: DoG 1.5 / 5 px,
                     7 x 7 local maximum, > 8 robust sd and > 12 e, x inside xlim -> (window, row, col, score)
      patches(E, w, cy, cx, window)  per channel the P x P patch centred on round(mapped position) of every candidate
                     -> img [n, window, C, P, P], doff [n, C, 2] (sub-pixel rest of the mapped position), ok [n]
  nn_dist(w, cx, cy)   distance to the nearest other candidate of the same window
  blur(img, s)         separable Gaussian blur (replicate padding)
"""
import numpy as np, torch
import torch.nn.functional as F


def gauss_kernel(s, dev):
    r = int(np.ceil(3 * s)); x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / s) ** 2); return k / k.sum()


def blur(img, s):
    k = gauss_kernel(s, img.device); r = len(k) // 2
    x = F.conv2d(F.pad(img[:, None], (r, r, 0, 0), mode='replicate'), k[None, None, None, :])
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), k[None, None, :, None])
    return x[:, 0]


def nn_dist(w, cx, cy, far=99.0):
    """distance (px) of every candidate to the nearest other candidate of the same window (far if alone)"""
    dd = torch.full((len(w),), far, device=w.device)
    for wv in torch.unique(w):
        s = torch.nonzero(w == wv)[:, 0]
        if len(s) > 1:
            pts = torch.stack([cx[s], cy[s]], 1).float(); D = torch.cdist(pts, pts); D.fill_diagonal_(far); dd[s] = D.min(1).values
    return dd


class ChannelFront:
    def __init__(self, regmap, offset, gain, C, H, W, P=15, xlim=None, weights=None, dev='cuda'):
        self.dev, self.pm, self.gain, self.C, self.H, self.W, self.P = dev, regmap, gain, C, H, W, P
        self.xlim = tuple(xlim) if xlim is not None else (0, W)
        self.weights = list(weights) if weights is not None else [1.0] * C
        self.off = torch.tensor(offset, device=dev, dtype=torch.float32) if np.ndim(offset) else float(offset)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        self.grid = [None]
        for c in range(1, C):
            xs, ys = regmap.inv(c, xx, yy)
            g = torch.tensor(np.stack([xs / (W - 1) * 2 - 1, ys / (H - 1) * 2 - 1], -1), device=dev, dtype=torch.float32)
            self.grid.append(g[None])
        self.maps = {c: (torch.tensor(regmap.inv(c, xx, yy)[0], device=dev, dtype=torch.float32),
                         torch.tensor(regmap.inv(c, xx, yy)[1], device=dev, dtype=torch.float32)) for c in range(1, C)}

    def to_e(self, crops_adu):
        """channel crops [n, C, H, W] in ADU (float32 numpy, as ChannelReader.read) -> photo-electrons (torch)"""
        return (torch.tensor(crops_adu, device=self.dev) - self.off) * self.gain

    def registered_sum(self, X):
        """[n, C, H, W] -> [n, H, W]: channel 0 + channels 1..C-1 resampled into channel-0 coordinates (bilinear)"""
        s = X[:, 0].clone()
        if self.weights[0] != 1.0: s = s * self.weights[0]
        for c in range(1, self.C):
            r = F.grid_sample(X[:, c:c + 1], self.grid[c].expand(X.shape[0], -1, -1, -1), mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
            s += r if self.weights[c] == 1.0 else self.weights[c] * r
        return s

    def detect(self, E, window=6):
        H, W, x0, x1 = self.H, self.W, *self.xlim
        nW = E.shape[0] // window
        S6 = E[:nW * window].reshape(nW, window, self.C, H, W).sum(1)                   # [nW,C,H,W]
        ssum = self.registered_sum(S6)
        fl = blur(ssum, 1.5) - blur(ssum, 5.0)
        reg = fl[:, :, x0:x1].reshape(nW, -1)
        med = reg.median(1).values; sd = 1.4826 * (reg - med[:, None]).abs().median(1).values
        mx = F.max_pool2d(fl[:, None], 7, 1, 3)[:, 0]
        pk = (fl == mx) & (fl > torch.maximum(8 * sd, torch.full_like(sd, 12.0))[:, None, None])
        pk[:, :, :x0] = False; pk[:, :, x1:] = False
        h = self.P // 2 + 1
        pk[:, :h] = False; pk[:, -h:] = False
        w, cy, cx = torch.nonzero(pk, as_tuple=True)
        return w, cy, cx, fl[w, cy, cx]

    def patches(self, E, w, cy, cx, window=6):
        """-> img [n, window, C, P, P], doff [n, C, 2], ok [n]"""
        P, C, H, W = self.P, self.C, self.H, self.W
        h = P // 2
        cxs, cys = [cx.float()], [cy.float()]
        for c in range(1, C):
            mx, my = self.maps[c]; cxs.append(mx[cy, cx]); cys.append(my[cy, cx])
        mxy = torch.stack([torch.stack(cxs, 1), torch.stack(cys, 1)], -1)                 # [n,C,2]
        ctr = torch.round(mxy); doff = mxy - ctr
        ix = ctr[..., 0].long() - h; iy = ctr[..., 1].long() - h                           # [n,C]
        ok = ((ix >= 0) & (ix + P <= W) & (iy >= 0) & (iy + P <= H)).all(1)
        ix = ix.clamp(0, W - P); iy = iy.clamp(0, H - P)
        o = torch.arange(P, device=E.device)
        t = (window * w[:, None] + torch.arange(window, device=E.device)[None])           # [n,window]
        chn = torch.arange(C, device=E.device)
        Y = (iy[:, :, None] + o[None, None])                                               # [n,C,P]
        X = (ix[:, :, None] + o[None, None])
        img = E[t[:, :, None, None, None], chn[None, None, :, None, None], Y[:, None, :, :, None], X[:, None, :, None, :]]
        self.last_corner = torch.stack([ix, iy], -1)                                       # [n,C,2] patch top-left (x, y)
        return img, doff, ok
