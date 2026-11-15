"""
Emitter layouts for the gate scenes (simcode/benchmark/gate.py) on top of simcode.simulate.crlb (scene generator, multi-emitter CRLB,
one-to-one matching): isolated grids, close pairs, Poisson scenes, wide brightness, nearest-neighbour distances, and the
pattern-estimate error model (common per-angle phase error U(0, 0.05) rad sd, per-step 0.03 rad, depth +-10 % per FOV) so
the estimators see an imperfect pattern while the CRLB uses the true one.
"""
import math
import torch
import simcode.simulate.crlb as CD

dev = 'cuda'
S, P, h, R = CD.S, CD.P, CD.h, CD.R


def iso_emitters(nb, T, zmax, Brange=(60.0, 1200.0), tau=50.0, min_on=3.0):
    """9 emitters per FOV on an 18 px grid (isolated), brightness log-uniform per emitter, exponential on-times."""
    g = lambda *s: torch.rand(*s, device=dev)
    c = torch.tensor([14.0, 32.0, 50.0], device=dev)
    cy, cx = torch.meshgrid(c, c, indexing='ij')
    n = nb * 9
    pos = torch.stack([cx.reshape(-1), cy.reshape(-1)], 1).repeat(nb, 1) + (g(n, 2) - 0.5)
    fov = torch.arange(nb, device=dev).repeat_interleave(9)
    bright = torch.exp(math.log(Brange[0]) + (math.log(Brange[1]) - math.log(Brange[0])) * g(n))
    z = zmax * g(n) ** 1.5
    t0, t1 = kin_times(n, T, tau, min_on)
    return dict(fov=fov, x=pos[:, 0], y=pos[:, 1], z=z, bright=bright, t0=t0, t1=t1)


def kin_times(n, T, tau, min_on):
    """on-intervals from a stationary binding process (exponential on-time tau), conditioned on >= min_on on-frames."""
    g = lambda *s: torch.rand(*s, device=dev)
    fr = torch.arange(T, device=dev, dtype=torch.float32)
    t0 = torch.zeros(n, device=dev); t1 = torch.zeros(n, device=dev); todo = torch.ones(n, dtype=torch.bool, device=dev)
    for _ in range(500):
        if not todo.any(): break
        m_ = int(todo.sum()); s0 = -5 * tau + (T + 5 * tau) * g(m_); s1 = s0 - torch.log(g(m_).clamp(min=1e-9)) * tau
        onfr = (torch.minimum(s1[:, None], fr + 1) - torch.maximum(s0[:, None], fr)).clamp(0, 1).sum(1)
        ok = onfr >= min_on
        idx = torch.nonzero(todo)[:, 0][ok]
        t0[idx] = s0[ok]; t1[idx] = s1[ok]; todo[idx] = False
    return t0, t1


def pair_emitters(nb, sep, T, zmax, B=150.0, tau=50.0, min_on=3.0):
    em, role = CD.pair_emitters(nb, sep, '1:1', 'always-on', T, zmax)
    em['bright'] = torch.full_like(em['bright'], B)
    em['t0'], em['t1'] = kin_times(len(em['x']), T, tau, min_on)
    return em, role


def nn_dist(G, tg, minphot=100.0):
    n = len(tg)
    vis = G['phot'].sum(1) >= minphot
    D = torch.cdist(torch.stack([G['x'][tg], G['y'][tg]], 1), torch.stack([G['x'], G['y']], 1))
    D = torch.where((G['fov'][tg][:, None] == G['fov'][None]) & vis[None], D, torch.full_like(D, 1e9))
    D[torch.arange(n, device=dev), tg] = 1e9
    return D.min(1).values


def wide_brightness(em, T, tot=(100.0, 20000.0)):
    """per-emitter brightness such that the expected total window photons (mean illumination) are log-uniform in tot."""
    fr = torch.arange(T, device=dev, dtype=torch.float32)
    onf = (torch.minimum(em['t1'][:, None], fr[None] + 1) - torch.maximum(em['t0'][:, None], fr[None])).clamp(0, 1).sum(1)
    N = len(onf)
    lt = math.log(tot[0]) + (math.log(tot[1]) - math.log(tot[0])) * torch.rand(N, device=dev)
    em['bright'] = torch.exp(lt) / onf.clamp(min=1.0)
    return em
