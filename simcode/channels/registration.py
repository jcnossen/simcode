"""Registration of the detection channels to the reference channel 0.

  PolyMap   polynomial map per channel c (channel crop coordinates, x = column, y = row, pixel centres at integers):
            fwd(c, x, y) -> channel-0 coordinates; inv(c, x0, y0) -> channel-c coordinates (a separately fitted inverse).
            Features u^i v^j (i + j <= order), u = (x - cx) / sx, v = (y - cy) / sy; the normalisation is stored in
            the json ('feats_norm'); files without it use the crop centre (cx, cy, sx, sy) = (W/2, H/2, W/2, H/2).
            json: {order, fwd: {c: [nf, 2]}, inv: {c: [nf, 2]}, feats_norm: {cx, cy, sx, sy}, ...meta}
"""
import json
import numpy as np


class PolyMap:
    def __init__(self, coef_fwd, coef_inv, order, norm):
        """coef_fwd / coef_inv: {c: [nf, 2]} for the channels c >= 1; norm = (cx, cy, sx, sy)"""
        self.cf, self.ci, self.order, self.norm = coef_fwd, coef_inv, order, tuple(float(v) for v in norm)

    @staticmethod
    def default_norm(W, H):
        return (W / 2, H / 2, W / 2, H / 2)

    def feats(self, x, y, order=None):
        cx, cy, sx, sy = self.norm
        o = self.order if order is None else order
        u, v = (np.asarray(x) - cx) / sx, (np.asarray(y) - cy) / sy
        return np.stack([u ** i * v ** j for i in range(o + 1) for j in range(o + 1 - i)], -1)

    def fwd(self, c, x, y):
        if c == 0: return np.asarray(x, float), np.asarray(y, float)
        F_ = self.feats(x, y) @ self.cf[c]; return F_[..., 0], F_[..., 1]

    def inv(self, c, x, y):
        if c == 0: return np.asarray(x, float), np.asarray(y, float)
        F_ = self.feats(x, y) @ self.ci[c]; return F_[..., 0], F_[..., 1]

    def to_dict(self, **meta):
        cx, cy, sx, sy = self.norm
        return dict(order=self.order, fwd={c: np.asarray(v).tolist() for c, v in self.cf.items()},
                    inv={c: np.asarray(v).tolist() for c, v in self.ci.items()}, feats_norm=dict(cx=cx, cy=cy, sx=sx, sy=sy), **meta)

    def save(self, fn, **meta):
        from simcode.io.work import atomic_json
        atomic_json(fn, self.to_dict(**meta), indent=1, default=str)

    @staticmethod
    def from_dict(d, size=None):
        """size = (W, H) of the channel crops: the normalisation of a map without 'feats_norm'"""
        if d.get('feats_norm'):
            n = d['feats_norm']; norm = (n['cx'], n['cy'], n['sx'], n['sy'])
        elif size is not None:
            norm = PolyMap.default_norm(*size)
        else:
            raise ValueError("registration map without 'feats_norm': give the channel crop size (W, H)")
        return PolyMap({int(c): np.array(v) for c, v in d['fwd'].items()}, {int(c): np.array(v) for c, v in d['inv'].items()}, d['order'], norm)

    @staticmethod
    def load(fn, size=None):
        return PolyMap.from_dict(json.load(open(fn)), size)
