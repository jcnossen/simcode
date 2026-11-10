"""Closed-form (weighted linear least squares) solution of the linear parameters of a track sample:
   mu[b,t,:] = I[b,t] * psf[b,:] + bg[b, group[b,t]]
Per-frame intensities, background shared per group (e.g. per pattern angle). Iteratively reweighted with
var = mu + rn2. Used as variable projection: nonlinear params get gradients with I/bg treated as constants."""
import torch


@torch.no_grad()
def solve_linear(psf, data, mask, group, rn2, ngroups=2, iters=3, I_min=0.0):
    """psf [B,P], data [B,T,P], mask [B,T] bool, group [B,T] long in [0,ngroups). Returns I [B,T], bg [B,ngroups]."""
    B, T, P = data.shape
    n = T + ngroups
    m = mask.float()
    G = torch.nn.functional.one_hot(group.clamp(min=0), ngroups).float() * m[..., None]   # [B,T,ng]
    # initial weights: uniform
    w = torch.ones(B, T, P, device=data.device) * m[..., None]
    I = torch.zeros(B, T, device=data.device); bg = torch.zeros(B, ngroups, device=data.device)
    for _ in range(iters):
        # normal equations. columns: I_t (psf on frame t), bg_g (ones on frames in group g)
        pw = psf[:, None, :] * w                                     # [B,T,P]
        A_II = torch.diag_embed((pw * psf[:, None, :]).sum(-1))      # [B,T,T]
        A_Ib = (pw.sum(-1)[..., None] * G)                           # [B,T,ng]
        A_bb = torch.diag_embed((w.sum(-1)[..., None] * G).sum(1))   # [B,ng,ng]
        rhs_I = (pw * data).sum(-1)                                  # [B,T]
        rhs_b = ((w * data).sum(-1)[..., None] * G).sum(1)           # [B,ng]
        A = torch.zeros(B, n, n, device=data.device)
        A[:, :T, :T] = A_II; A[:, :T, T:] = A_Ib; A[:, T:, :T] = A_Ib.transpose(1, 2); A[:, T:, T:] = A_bb
        A = A + torch.eye(n, device=data.device) * 1e-6 * A.diagonal(dim1=1, dim2=2).mean(1)[:, None, None]
        # masked frames: make the system trivially solvable
        pad = torch.cat([1 - m, torch.zeros(B, ngroups, device=data.device)], 1)
        A = A + torch.diag_embed(pad)
        x = torch.linalg.solve(A, torch.cat([rhs_I, rhs_b], 1))
        I = x[:, :T].clamp(min=I_min) * m; bg = x[:, T:]
        mu = I[..., None] * psf[:, None, :] + (G * bg[:, None, :]).sum(-1, keepdim=True)     # [B,T,P] (no bmm: ng=1 fails on cuda)
        w = m[..., None] / (mu.clamp(min=1e-3) + rn2)
    return I, bg
