"""4Pi front end shared by the network pipeline and the classical pipeline: camera conversion, the channel registration
map, detection on the registered 4-channel 6-frame sum, and the unwarped P x P channel patches of a candidate.

  PolyMap    polynomial registration channel c <-> channel 0 (mirrored-crop pixel coordinates, x = column, y = row):
             fwd(c, x, y) -> ch0 coordinates; inv(c, x0, y0) -> channel c coordinates (separately fitted inverse)
  Front      per cell: camera offset map (<4Pi work dir>/smlm/camera/camera_<cell>.npz), in-situ registration
             (<4Pi work dir>/smlm/reg/reg_insitu_<cell>.json); to_e (ADU crops -> photo-electrons), detect (DoG 1.5 / 5 px
             on the registered sum, 7 x 7 local maximum, > 8 robust sd and > 12 e, x inside XLIM), patches (per channel
             centred on round(mapped position), doff = the sub-pixel rest)
Windows: frames 6w .. 6w+5 of each file (every file starts at pattern step 0).
"""
import json
import numpy as np, torch
import torch.nn.functional as F
from simcode.fourpi.setup import P4, GAIN, XLIM, P


def feats(x, y, order):
    u, v = (np.asarray(x) - 180.0) / 180.0, (np.asarray(y) - 100.0) / 100.0
    return np.stack([u ** i * v ** j for i in range(order + 1) for j in range(order + 1 - i)], -1)


class PolyMap:
    def __init__(self, coef_fwd, coef_inv, order):
        self.cf, self.ci, self.order = coef_fwd, coef_inv, order    # {c: [nf, 2]}

    def fwd(self, c, x, y):
        if c == 0: return np.asarray(x, float), np.asarray(y, float)
        F_ = feats(x, y, self.order) @ self.cf[c]; return F_[..., 0], F_[..., 1]

    def inv(self, c, x, y):
        if c == 0: return np.asarray(x, float), np.asarray(y, float)
        F_ = feats(x, y, self.order) @ self.ci[c]; return F_[..., 0], F_[..., 1]

    def save(self, fn, **meta):
        json.dump(dict(order=self.order, fwd={c: v.tolist() for c, v in self.cf.items()},
                       inv={c: v.tolist() for c, v in self.ci.items()}, **meta), open(fn, 'w'), indent=1)

    @staticmethod
    def load(fn):
        d = json.load(open(fn))
        return PolyMap({int(c): np.array(v) for c, v in d['fwd'].items()}, {int(c): np.array(v) for c, v in d['inv'].items()}, d['order'])


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


class Front:
    def __init__(self, cell, dev='cuda'):
        self.dev = dev
        self.off = torch.tensor(np.load(f'{P4}/smlm/camera/camera_{cell}.npz')['off'], device=dev, dtype=torch.float32)
        self.pm = pm = PolyMap.load(f'{P4}/smlm/reg/reg_insitu_{cell}.json')
        yy, xx = np.mgrid[0:200, 0:360].astype(np.float64)
        self.grid = [None]
        for c in (1, 2, 3):
            xs, ys = pm.inv(c, xx, yy)
            g = torch.tensor(np.stack([xs / 359 * 2 - 1, ys / 199 * 2 - 1], -1), device=dev, dtype=torch.float32)
            self.grid.append(g[None])
        self.maps = {c: (torch.tensor(pm.inv(c, xx, yy)[0], device=dev, dtype=torch.float32),
                         torch.tensor(pm.inv(c, xx, yy)[1], device=dev, dtype=torch.float32)) for c in (1, 2, 3)}

    def to_e(self, crops_adu):
        """channel crops [n, 4, 200, 360] in ADU (float32 numpy, as DcimgMovie.read with the layout) -> photo-electrons (GPU)"""
        return (torch.tensor(crops_adu, device=self.dev) - self.off) * GAIN

    def registered_sum(self, X):
        """[n, 4, H, W] -> [n, H, W]: channel 0 + channels 1-3 resampled into channel-0 coordinates (bilinear)"""
        s = X[:, 0].clone()
        for c in (1, 2, 3):
            s += F.grid_sample(X[:, c:c + 1], self.grid[c].expand(X.shape[0], -1, -1, -1), mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
        return s

    def detect(self, E):
        nW = E.shape[0] // 6
        S6 = E[:nW * 6].reshape(nW, 6, 4, 200, 360).sum(1)                               # [nW,4,H,W]
        ssum = self.registered_sum(S6)
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
        h = P // 2
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
        self.last_corner = torch.stack([ix, iy], -1)                                       # [n,4,2] patch top-left (x, y)
        return img, doff, ok
