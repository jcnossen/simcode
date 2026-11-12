"""Solve-aware training loss for the learned K-read-out head (2026-11-12).

The explicit solve is linearised at the truth: with the read-out model  E[Y] = F q_t P_t(x, y) A(theta) + b_c Wsum  (exact
window model of solvek4pi, the head's windows), whitened Jacobian Ju and whitened read-out error ru (analytic pixel-noise
covariance times the head's scale cs), prior rows for (x, y, z) and the residual background b,
    dtheta = H^-1 g,   H = Ju^T Ju / cs^2 + P_prior,   g = Ju^T ru / cs^2 + prior terms,
is the error of the solve's estimate to first order and H^-1 its reported covariance. The loss is the Gaussian NLL of
(dx, dy, dz) under that covariance: it penalises what the solve gets wrong (and mis-reports), not the error of each
read-out. Errors of the read-outs that the solve cannot see (common gain, anything orthogonal to the Jacobian) cost
nothing; frame / channel mixing that moves the pattern phase or the 4Pi phase costs directly.
"""
import torch
from simcode2.fourpi.solve4pi import ANG


def solve_loss(sv, o, S, Ystar, sb=0.2, clamp=50.0):
    """sv: SolveK (pattern k of the simulator); o: head output; S: Sim4Pi sample; Ystar [n,6,4,K] targets.
    -> nll [n], dict(u [n,3] = dtheta / reported sigma, r2 [n] = mean whitened read-out residual^2)"""
    n = Ystar.shape[0]; dev = Ystar.device; K = Ystar.shape[-1]
    with torch.no_grad():
        sv._ctx = dict(W=o['W'], p0=None, Wsum=None)
        A, _ = sv.readout_k(S['x'], S['y'], S['z'], S['Phi0'], S['doff'])
        A = A[:, torch.tensor(ANG, device=dev)]                                   # [n,6,4,K,5]
        from simcode2.fourpi.solve4pi import psi_from_eps
        Pt, dPx, dPy = sv.pattern(S['x'], S['y'], S['m'], psi_from_eps(S['eps']))
        Fq = (S['F'][:, None] * S['q'])                                           # [n,6]
        A0 = A[..., 0]
        J = torch.zeros(n, 6, 4, K, 8, device=dev)
        J[..., 0] = (Fq * dPx)[:, :, None, None] * A0 + (Fq * Pt)[:, :, None, None] * A[..., 1]
        J[..., 1] = (Fq * dPy)[:, :, None, None] * A0 + (Fq * Pt)[:, :, None, None] * A[..., 2]
        J[..., 2] = (Fq * Pt)[:, :, None, None] * A[..., 3]
        J[..., 3] = (S['q'] * Pt)[:, :, None, None] * A0
        for c in range(4):
            J[:, :, c, :, 4 + c] = o['Wsum'][:, None, c, :]
        Ju = torch.linalg.solve_triangular(o['Lb0'], J, upper=False).reshape(n, -1, 8)
        JJ = torch.einsum('nmp,nmq->npq', Ju, Ju)
        sp = torch.exp(o['ls'])
        bt = S['bg'][:, 0].mean((-1, -2))
    bstar = bt - o['bgc']                                                         # true residual background [n,4]
    r0 = o['Y'] - Ystar - (bstar[:, :, None] * o['Wsum'])[:, None]
    ru = torch.linalg.solve_triangular(o['Lb0'], r0[..., None], upper=False)[..., 0].reshape(n, -1)
    w = 1.0 / o['cs'] ** 2
    Pp = torch.zeros(n, 8, device=dev); Pp[:, :3] = 1.0 / sp.detach() ** 2; Pp[:, 4:] = 1.0 / sb ** 2
    H = JJ * w[:, None, None] + torch.diag_embed(Pp) + 1e-8 * torch.eye(8, device=dev)
    g = torch.einsum('nmp,nm->np', Ju, ru) * w[:, None]
    ep = torch.stack([o['dxy'][:, 0] - S['x'], o['dxy'][:, 1] - S['y'], o['z'] - S['z']], 1).detach()
    g = g + torch.cat([ep * Pp[:, :3], torch.zeros(n, 1, device=dev), -bstar * Pp[:, 4:]], 1)
    Hi = torch.linalg.inv(H)
    dth = torch.einsum('npq,nq->np', Hi, g)[:, :3]
    var = torch.diagonal(Hi, dim1=1, dim2=2)[:, :3].clamp(min=1e-12)
    u2 = (dth ** 2 / var)
    nll = 0.5 * (u2.clamp(max=clamp) + torch.log(var)).sum(1)
    return nll, dict(u=(dth / var.sqrt()).detach(), r2=(ru ** 2).mean(1) * w, dth=dth.detach(), var=var.detach())
