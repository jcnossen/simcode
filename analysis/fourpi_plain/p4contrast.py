"""Is the remaining mismatch per spot rather than a PSF shape? Extra per-spot freedoms added to the spline model of a
PSF, deviance of the test spots compared (LM with forward-mode Jacobians):
  base     x, y, z, N, bg_c (Phi = the SIMCODE field)
  phi      + Phi free
  contrast + Phi free + the interference contrast m (g = inc + m (cos Phi cr - sin Phi ci)): a cavity phase that moves
           during the 50 ms exposure lowers m per spot
  tau      + Phi, m free + per-spot channel weights (3 free ratios)

  python analysis/fourpi_plain/p4contrast.py [--state <Model4Pi state json>] [--n 6000]
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from insitu4pi import load_spots, theta0, localize, nll, deviance, load_psf0, Model4Pi, dev
from p4insitu_fit import select, split


def model_fn(spl, pos0, th, mode):
    """th [n, 10 + 3]: x, y, z, Phi, N, b0..b3, m, w1, w2, w3"""
    C = 4
    xy = pos0 + th[:, None, :2]
    z, Phi = th[:, 2], th[:, 3]
    cP, sP = torch.cos(Phi)[:, None, None], torch.sin(Phi)[:, None, None]
    out = []
    for c in range(C):
        pos = torch.cat([xy[:, c], z[:, None]], 1)
        inc, cr, ci = (spl.luts[k][c].render(pos) for k in range(3))
        out.append(inc + th[:, 9, None, None] * (cP * cr - sP * ci))
    g = torch.stack(out, 1)
    w = torch.cat([torch.ones_like(th[:, :1]), torch.exp(th[:, 10:13])], 1)
    w = w / w.mean(1, keepdim=True)
    return th[:, 4, None, None, None] * g * w[:, :, None, None] + th[:, 5:9, None, None]


def lm(spl, D, th, free, iters=25):
    var = D['var']; th = th.clone()
    lam = torch.full((len(th),), 1e-2, device=dev)
    f = lambda t: model_fn(spl, D['pos0'], t, None)
    cost = lambda t: nll(f(t), D['d'], var).sum((1, 2, 3))
    cur = cost(th)
    for it in range(iters):
        Js = []
        for k in free:
            tg = torch.zeros_like(th); tg[:, k] = 1
            mu, d = torch.func.jvp(f, (th,), (tg,)); Js.append(d)
        J = torch.stack(Js, -1).reshape(len(th), -1, len(free))
        w = (1.0 / (mu + var).clamp(min=1e-3)).reshape(len(th), -1); r = (D['d'] - mu).reshape(len(th), -1)
        A = torch.einsum('npi,np,npj->nij', J, w, J); b = torch.einsum('npi,np->ni', J, w * r)
        A = A + lam[:, None, None] * torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2) + 1e-6)
        st = torch.linalg.solve(A, b[..., None])[..., 0]
        tn = th.clone(); tn[:, free] += st
        tn[:, :2] = tn[:, :2].clamp(-2.5, 2.5); tn[:, 4] = tn[:, 4].clamp(min=10); tn[:, 5:9] = tn[:, 5:9].clamp(min=0.01)
        tn[:, 9] = tn[:, 9].clamp(0.0, 1.5); tn[:, 10:13] = tn[:, 10:13].clamp(-1, 1)
        new = cost(tn); acc = new < cur
        th = torch.where(acc[:, None], tn, th); cur = torch.where(acc, new, cur)
        lam = torch.where(acc, lam * 0.3, lam * 10).clamp(1e-6, 1e6)
    return th


def main():
    p = argparse.ArgumentParser(); p.add_argument('--state', default=''); p.add_argument('--n', type=int, default=6000)
    p.add_argument('--knots', default='')
    a = p.parse_args()
    D = load_spots(sel=select)
    j = split(D)['test']; j = np.sort(np.random.default_rng(3).choice(j, a.n, replace=False))
    D = {k: (v[j] if isinstance(v, (np.ndarray, torch.Tensor)) and len(v) == len(D['x']) else v) for k, v in D.items()}
    if a.knots:
        from p4spline_refine import load_refined
        S = load_refined(a.knots)
    else:
        m = Model4Pi.from_state(json.load(open(a.state)), roisize=17) if a.state else load_psf0(17)
        from insitu4pi import baked_spline, SplineModel
        S = SplineModel(baked_spline(m, oversample=4), torch.zeros(4, 2, device=dev), 0.0)
    t0, _ = localize(S, D, theta0(D), iters=25)
    th = torch.cat([t0, torch.ones(len(t0), 1, device=dev), torch.zeros(len(t0), 3, device=dev)], 1)
    N = th[:, 4].cpu().numpy()
    bins = [(2500, 5000), (5000, 10000), (10000, 1e9)]
    res = {}
    for nm, free in (('base', [0, 1, 2, 4, 5, 6, 7, 8]), ('phi', [0, 1, 2, 3, 4, 5, 6, 7, 8]), ('contrast', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                     ('tau', list(range(13)))):
        th = lm(S.spl, D, th, free)
        with torch.no_grad():
            dv = deviance(model_fn(S.spl, D['pos0'], th, None), D['d'], D['var'])
        e = (dv.sum((1, 2, 3)) - dv[0].numel()).cpu().numpy()
        res[nm] = dict(excess_bins=[float(np.median(e[(N >= lo) & (N < hi)])) for lo, hi in bins], ctr=float(dv[:, :, 4:13, 4:13].mean((1, 2, 3)).median()),
                       m_median=float(th[:, 9].median()), m_sd=float(th[:, 9].std()))
        print(nm, json.dumps(res[nm]), flush=True)


if __name__ == '__main__':
    main()
