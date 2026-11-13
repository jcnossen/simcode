"""Fixed ("canonical") read-out windows of the linear 4Pi head and of the K-read-out solve (2026-11-12).

A read-out k of a (frame, channel) image is  Y_k = sum_px w_k(px; p) (img(px) - bg),  with a window w_k that is a fixed
function of a 3D position p = (x, y, z) in that channel's patch (no learned, data-dependent weights):
  ('mf', beta)  : w = u / (sum u inc) * (sum inc),  u = inc / (inc + beta),  inc = the phase-averaged (incoherent) channel
                  PSF at p. beta -> 0: flat over the PSF (Poisson-optimal); large beta: w ~ inc (background-optimal).
                  Normalised so that the read-out of the incoherent PSF itself is its patch sum (photons in the patch).
  ('inc', 0)    : w = inc / (sum inc^2) * (sum inc)   (plain matched filter, the beta -> infinity limit)
  ('gauss', sd) : w = exp(-r^2 / 2 sd^2) centred at p (the v2..v8 'C' read-out)
  ('flat', 0)   : w = 1 on the whole patch
  ('cr', beta), ('ci', beta) : w = cr / (inc + beta), ci / (inc + beta): quadrature windows on the 4Pi cross term
The solve models E[Y_k] = N_t sum_px w_k(px; p) g_c(px; p, Phi) exactly (tables over p), so a read-out placed at the
emitter position is unbiased by construction, without knowing the 4Pi phase.
"""
import torch

DEFAULT = (('mf', 0.003), ('gauss', 1.0))
WINS_K = (('mf', 0.003), ('cr', 0.003), ('ci', 0.003), ('flat', 0.0))   # the K = 4 read-outs of the learned-window head (head4pi_w)


def needs_coh(wins):
    return any(k in ('cr', 'ci') for k, _ in wins)


def window(kind, par, inc, ex, ey, coh=None):
    """inc [n,P,P] incoherent channel PSF at p (clamped positive); ex, ey [n,P,P] pixel - p offsets; coh = (cr, ci) the
    coherent cross-term components at p (only for 'cr' / 'ci') -> w [n,P,P]"""
    if kind in ('cr', 'ci'):   # quadrature windows on the 4Pi cross term: the image is inc + cos(Phi) cr - sin(Phi) ci, so
        # these two read the interference amplitude and phase without knowing Phi (Poisson weights 1 / (inc + beta))
        return coh[0 if kind == 'cr' else 1] / (inc + par)
    if kind == 'mf':
        u = inc / (inc + par)
        return u / (u * inc).sum((-1, -2), keepdim=True) * inc.sum((-1, -2), keepdim=True)
    if kind == 'inc':
        return inc / (inc * inc).sum((-1, -2), keepdim=True) * inc.sum((-1, -2), keepdim=True)
    if kind == 'flat':      # all pixels of the patch (with the background nuisance in the solve: the background estimator)
        return torch.ones_like(inc)
    if kind == 'gauss':
        return torch.exp(-(ex ** 2 + ey ** 2) / (2 * par ** 2))
    raise ValueError(kind)


def windows_at(lut, c, pos, wins, ox, oy):
    """all K windows of channel c at positions pos [n,3] -> [n,K,P,P], and the normalised incoherent template [n,P,P]"""
    inc = lut.luts[0][c].evaluate(pos).clamp(min=1e-7)
    coh = (lut.luts[1][c].evaluate(pos), lut.luts[2][c].evaluate(pos)) if needs_coh(wins) else None
    ex = ox[None] - pos[:, 0, None, None]; ey = oy[None] - pos[:, 1, None, None]
    return torch.stack([window(k, p, inc, ex, ey, coh) for k, p in wins], 1), inc / inc.sum((-1, -2), keepdim=True)
