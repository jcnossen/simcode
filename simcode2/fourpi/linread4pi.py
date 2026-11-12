"""Linear fixed-window read-outs of a candidate's 24 channel patches, with their analytic covariance (2026-11-12).

    Y[t, c, k] = gain[c, k] * sum_px w_k(px; p_c) (img[t, c, px] - bg[c]),   p_c = (x, y) + doff_c, z
windows from windows4pi (functions of the position only), the same for the 6 frames; one background per channel.
Covariance per (t, c): K x K, sum_px w_k w_l var(px), var = Y_0 template + bg + read variance (pixel noise only; errors
of bg / gain are common to the 6 frames). No learned or data-dependent weights: a frame's read-out depends on that
frame's pixels only, so there is no cross-talk between frames.
"""
import torch
from simcode2.fourpi.windows4pi import windows_at

VREAD = 1.14 ** 2 + 0.2 ** 2


def linear_readout(img, doff, lut, wins, xy, z, bgc, gain=None, vread=VREAD, return_w=False):
    """img [n,6,4,P,P] e; doff [n,4,2]; xy [n,2] px, z [n] um (window position); bgc [n,4] e/px; gain [n,4,K] or None
    -> Y [n,6,4,K], Lb [n,6,4,K,K] (lower Cholesky), Wsum [n,4,K] (window sums: a residual background b = true - bgc adds b Wsum to Y)"""
    n, T, C, P, _ = img.shape; dev = img.device; K = len(wins)
    o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
    oy, ox = torch.meshgrid(o, o, indexing='ij')
    Y = torch.zeros(n, T, C, K, device=dev); S = torch.zeros(n, T, C, K, K, device=dev); Ws = torch.zeros(n, C, K, device=dev)
    Wall = []
    for c in range(C):
        pos = torch.stack([xy[:, 0] + doff[:, c, 0], xy[:, 1] + doff[:, c, 1], z], 1)
        w, g = windows_at(lut, c, pos, wins, ox, oy)                               # [n,K,P,P], [n,P,P]
        if gain is not None:
            w = w * gain[:, c, :, None, None]
        y = img[:, :, c] - bgc[:, None, c, None, None]                             # [n,6,P,P]
        Yc = torch.einsum('nkuv,ntuv->ntk', w, y)
        vp = Yc[..., 0].clamp(min=0)[..., None, None] * g[:, None] + bgc[:, None, c, None, None].clamp(min=0.05) + vread
        Y[:, :, c] = Yc; Ws[:, c] = w.sum((-1, -2))
        if return_w: Wall.append(w)
        S[:, :, c] = torch.einsum('nkuv,nluv,ntuv->ntkl', w, w, vp)
    S = S + 1e-6 * torch.diagonal(S, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(K, device=dev)
    if return_w:
        return Y, torch.linalg.cholesky(S), Ws, torch.stack(Wall, 1)
    return Y, torch.linalg.cholesky(S), Ws


def border_bg(img):
    """[n,4]: median over the border ring of the frame-mean image per channel (one background per channel)"""
    border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
    return border.mean(1).median(-1).values
