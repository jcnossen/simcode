"""Covariance ablation: the solve variants. Each variant = a transform of the head's Cholesky factor L (what the solve is given as the
network covariance) + the c terms of the calibration (cmult = c, cmult_ang = c_a). Nothing in the package is changed: sc_solve builds
Sigma = L L^T * kap_I2(photon bin) + c^2 diag(mu^2) + c_a^2 (mu mu^T within each angle) from these inputs.
  A   full Sigma_net, c = c_a = 0.10                    (production)
  B   diagonal of Sigma_net only, c = c_a = 0.10
  C   full Sigma_net, c = c_a = 0
  D   diagonal only, c = c_a = 0
  E   no network covariance at all: Sigma = sigma^2(N) I, one variance per frame from a table over the window photon count N (the table
      is the kap_I2 bins of the per-variant calibration fit, i.e. fitted on the tuning seed like every other variant), c = 0
      = unweighted least squares on the six intensities (+ the position prior, whose weight needs a noise scale)
  E1  per-emitter scalar from the network: Sigma = (trace(Sigma_net) / 6) I, c = 0
  F   full Sigma_net, c = 0.10 only (c_a = 0)
  G   full Sigma_net, c_a = 0.10 only (c = 0)
  H   diagonal only, c = 0.10 only (c_a = 0)        (the simplest candidate that keeps a brightness-fluctuation term)
  *n  (suffix) the same without the position prior: the variant's calibration with the prior sigma x 1000 (kap_p), sigma factor refitted"""
import numpy as np

VARS = {
    'A': dict(L='full', c=0.10, ca=0.10), 'B': dict(L='diag', c=0.10, ca=0.10), 'C': dict(L='full', c=0.0, ca=0.0),
    'D': dict(L='diag', c=0.0, ca=0.0), 'E': dict(L='ident', c=0.0, ca=0.0), 'E1': dict(L='scalar', c=0.0, ca=0.0),
    'F': dict(L='full', c=0.10, ca=0.0), 'G': dict(L='full', c=0.0, ca=0.10), 'H': dict(L='diag', c=0.10, ca=0.0),
}
for k in ('A', 'D'):
    VARS[k + 'n'] = dict(VARS[k], noprior=True)
ORDER = ['A', 'B', 'C', 'D', 'E', 'E1', 'F', 'G', 'H', 'An', 'Dn']
LABEL = dict(A='A full + c + c_a (production)', B='B diagonal + c + c_a', C='C full, no c', D='D diagonal, no c', E='E identity (variance table by photons)',
             E1='E1 network scalar x identity', F='F full + c only', G='G full + c_a only', H='H diagonal + c only', An='A without prior', Dn='D without prior')


def L_variant(L, kind):
    """L [n,6,6] numpy (lower Cholesky of Sigma_net) -> the factor given to the solve"""
    if kind == 'full': return L
    d2 = (L.astype(np.float64) ** 2).sum(-1)                      # diag(L L^T)
    out = np.zeros_like(L); i = np.arange(L.shape[1])
    if kind == 'diag': out[:, i, i] = np.sqrt(d2)
    elif kind == 'scalar': out[:, i, i] = np.sqrt(d2.mean(1, keepdims=True))
    elif kind == 'ident': out[:, i, i] = 1.0
    else: raise ValueError(kind)
    return out.astype(L.dtype)


def cal_variant(cal, v):
    c = dict(cal); c['cmult'] = float(v['c']); c['cmult_ang'] = float(v['ca'])
    return c


def lsxy_variant(lsxy, v):
    return lsxy      # the no-prior variants act through the calibration (kap_p x 1000 after the fit; gate_abl.fit), not through lsxy:
                     # plib.fit_cal rescales the prior sigma (kap_p), which would undo an inflation of lsxy
