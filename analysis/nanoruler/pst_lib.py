"""
Per-step pattern analysis of the 80 nm nanoruler movie: shared library.

Data: out/nanoruler/psf/bl_int_intensities.pt (per-frame PSF-fit intensities of isolated emitter tracks, 12-frame samples).
Stored (reordered) frame f: step s = f % 6, orientation a = s // 3, acquisition time 6 (f // 6) + [0,2,4,1,3,5][s].

Pattern models (all: I[b,f] = A[b,a] rel_s (1 + m_s cos(k_a . r_b - phi_s - D_s(f))), A per 12-frame sample and orientation,
Huber loss (delta 3) on (pred - I) / sqrt(I + 100), as simcode/calibration/fit_pattern.py):
  shared   D_s(f) = eps_a(f)                 one piecewise-linear drift per orientation (the pipeline's model)
  perstep  D_s(f) = d_s(f)                   six independent piecewise-linear drifts
  hybrid   D_s(f) = eps_a(f) + d_s(f)        shared drift on fine knots (kf) + per-step drift on coarse knots (kf2)
Smoothness penalty lam * sum(diff^2) per curve with lam = 10 * 1500 / knot spacing (= fit_pattern.py at 1500 frames;
random-walk scaling for other spacings).
"""
import math
import numpy as np, torch

ROOT = '/home/jc/simcode2'
INT = ROOT + '/out/nanoruler/psf/bl_int_intensities.pt'
SHARED = ROOT + '/out/nanoruler/pattern_nr80.pt'
OUT = ROOT + '/out/nanoruler/perstep/'
PX_NM = 65.0
NFRAMES = 30000
ACQ = np.array([0, 2, 4, 1, 3, 5])           # acquisition slot of stored step s within the cycle


def load_data():
    """fully-on frames as flat arrays + sample / track / site ids."""
    d = torch.load(INT)
    I, r, frames, mask, full = [d[k].numpy() for k in ('I', 'r', 'frames', 'mask', 'full')]
    B = len(I)
    # track identity: consecutive samples, same position (< 0.5 px), contiguous frames
    last = np.array([frames[b][mask[b]].max() for b in range(B)]); first = np.array([frames[b][mask[b]].min() for b in range(B)])
    newtr = np.ones(B, bool)
    newtr[1:] = ~((np.hypot(*(r[1:] - r[:-1]).T) < 0.5) & (first[1:] == last[:-1] + 1))
    track = np.cumsum(newtr) - 1
    # sites: tracks whose mean positions are within 1 px (same binding site region / ruler), connected components
    nt = track.max() + 1
    tr_r = np.stack([np.bincount(track, r[:, i], nt) / np.bincount(track, minlength=nt) for i in (0, 1)], 1)
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pr = cKDTree(tr_r).query_pairs(1.5, output_type='ndarray')
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(nt, nt))
    nsite, site_of_track = connected_components(g, directed=False)
    bi, ti = np.nonzero(full)
    fr = frames[bi, ti]
    D = dict(I=I[bi, ti].astype(np.float64), r=r[bi].astype(np.float64), fr=fr, b=bi, s=fr % 6, a=(fr % 6) // 3,
             track=track[bi], site=site_of_track[track[bi]], B=B, ntrack=nt, nsite=nsite,
             raw=dict(I=I, r=r, frames=frames, mask=mask, full=full, track=track, site=site_of_track[track]))
    D['g'] = D['b'] * 2 + D['a']                   # amplitude group (sample, orientation)
    return D


def site_halves(D, seed=0):
    """boolean [n] : datum belongs to half 0 (random split over sites)."""
    rng = np.random.default_rng(seed)
    h = rng.permutation(D['nsite']) % 2
    return h[D['site']] == 0


def interp(knots, kf, fr):
    """knots [C,K] torch on frames j*kf; curve index c [n], frames fr [n] (torch) -> value [n]."""
    def f(c):
        t = fr.double() / kf
        i0 = t.floor().long().clamp(0, knots.shape[1] - 2); w = (t - i0).clamp(0, 1)
        return knots[c, i0] * (1 - w) + knots[c, i0 + 1] * w
    return f


class Pattern:
    """parameter container; model in ('shared', 'perstep', 'hybrid')."""
    def __init__(self, model, kf, kf2=None, init=None, dev='cpu'):
        self.model, self.kf, self.kf2, self.dev = model, kf, kf2, dev
        S = torch.load(SHARED) if init is None else init
        t = lambda v: torch.as_tensor(v, dtype=torch.float64, device=dev).clone()
        self.k = t(S['k']); self.phi = t(S['phi']); self.m = t(S['m']); self.logrel = t(S['logrel'])
        nk = (NFRAMES - 1) // kf + 2
        self.eps = torch.zeros(2, nk, dtype=torch.float64, device=dev)          # shared drift (shared, hybrid)
        nk2 = (NFRAMES - 1) // (kf2 if model == "hybrid" else kf) + 2
        self.d = torch.zeros(6, nk2, dtype=torch.float64, device=dev)           # per-step drift (perstep, hybrid)

    def params(self):
        p = [self.phi, self.m, self.logrel]
        if self.model in ('shared', 'hybrid'): p.append(self.eps)
        if self.model in ('perstep', 'hybrid'): p.append(self.d)
        return p

    def drift(self, fr, s):
        """total phase drift D_s(f) [n] (torch)."""
        a = s // 3; out = 0
        if self.model in ('shared', 'hybrid'): out = out + interp(self.eps, self.kf, fr)(a)
        if self.model in ('perstep', 'hybrid'): out = out + interp(self.d, self.kf2 if self.model == 'hybrid' else self.kf, fr)(s)
        return out

    def theta(self, r, fr, s):
        return (self.k[s // 3] * r).sum(1) - self.phi[s] - self.drift(fr, s)

    def x(self, r, fr, s, deriv=False):
        """unit-amplitude prediction rel_s (1 + m_s cos theta); with deriv also d x / d(phase error) = rel m sin theta."""
        rel = torch.exp(self.logrel - self.logrel.mean()); th = self.theta(r, fr, s)
        x = rel[s] * (1 + self.m[s] * torch.cos(th))
        return (x, rel[s] * self.m[s] * torch.sin(th), th) if deriv else x

    def penalty(self):
        p = 0
        if self.model in ('shared', 'hybrid'):
            p = p + 10.0 * 1500 / self.kf * (self.eps[:, 1:] - self.eps[:, :-1]).pow(2).sum()
        if self.model in ('perstep', 'hybrid'):
            kf = self.kf2 if self.model == 'hybrid' else self.kf
            p = p + 10.0 * 1500 / kf * (self.d[:, 1:] - self.d[:, :-1]).pow(2).sum()
        return p

    def gauge(self):
        with torch.no_grad():
            self.m.clamp_(0, 1)
            if self.model == 'shared':
                self.eps -= self.eps.mean(1, keepdim=True)
            elif self.model == 'perstep':
                mu = self.d.mean(1, keepdim=True); self.d -= mu; self.phi += mu[:, 0]
            else:
                self.eps -= self.eps.mean(1, keepdim=True)
                mu = self.d.mean(1, keepdim=True); self.d -= mu; self.phi += mu[:, 0]
                # hybrid: the per-step curves carry only the step-relative part (orientation mean -> eps is not representable
                # on a different knot grid; it is left free, the penalty and the data decide)

    def total_drift_np(self, frames):
        """[6, len(frames)] total drift D_s at the given frames (numpy), evaluated for every step at every frame."""
        f = torch.as_tensor(frames, device=self.dev)
        with torch.no_grad():
            return np.stack([self.drift(f, torch.full_like(f, s)).cpu().numpy() for s in range(6)])

    def state(self):
        return dict(model=self.model, knot_frames=self.kf, knot_frames2=self.kf2, k=self.k.detach().cpu(), phi=self.phi.detach().cpu(),
                    m=self.m.detach().cpu(), logrel=self.logrel.detach().cpu(), eps=self.eps.detach().cpu(), d=self.d.detach().cpu())

    @staticmethod
    def from_state(S, dev='cpu'):
        P = Pattern(S['model'], S['knot_frames'], S['knot_frames2'], dev=dev)
        for k in ('k', 'phi', 'm', 'logrel', 'eps', 'd'): setattr(P, k, S[k].double().to(dev).clone())
        return P


def huber(z, delta=3.0):
    az = z.abs()
    return torch.where(az < delta, 0.5 * z * z, delta * (az - 0.5 * delta))


def tens(D, sel, dev='cpu'):
    t = lambda v, dt=torch.float64: torch.as_tensor(v[sel], dtype=dt, device=dev)
    gi = np.unique(D['g'][sel], return_inverse=True)[1]
    return dict(I=t(D['I']), r=t(D['r']), fr=t(D['fr'], torch.long), s=t(D['s'], torch.long), g=torch.as_tensor(gi, device=dev),
                ng=int(gi.max()) + 1)


def solve_A(T, x, iters=30):
    """Huber-IRLS amplitude per group for fixed pattern (the loss of fit_pattern in A): returns A [ng]."""
    w = 1.0 / (T['I'] + 100.0); I = T['I']; g = T['g']; ng = T['ng']
    z = torch.zeros(ng, dtype=torch.float64, device=I.device)
    A = (z.index_add(0, g, w * x * I) / z.index_add(0, g, w * x * x).clamp(min=1e-12)).clamp(min=1.0)
    for _ in range(iters):
        zr = (A[g] * x - I) * w.sqrt()
        hw = torch.where(zr.abs() < 3, torch.ones_like(zr), 3 / zr.abs().clamp(min=1e-9)) * w
        A = (z.index_add(0, g, hw * x * I) / z.index_add(0, g, hw * x * x).clamp(min=1e-12)).clamp(min=1.0)
    return A


def fit(D, sel, model, kf, kf2=None, iters=3000, fit_k=True, init=None, dev='cpu', verbose=False):
    """fit a pattern model on the data subset sel (bool [n]). Returns (Pattern, final loss per datum)."""
    T = tens(D, sel, dev)
    P = Pattern(model, kf, kf2, init=init, dev=dev)
    with torch.no_grad():
        A0 = solve_A(T, P.x(T['r'], T['fr'], T['s']))
    logA = torch.log(A0.clamp(min=10)).requires_grad_(True)
    for p in P.params(): p.requires_grad_(True)
    grp = [{'params': [logA] + P.params(), 'lr': 0.02}]
    if fit_k: P.k.requires_grad_(True); grp.append({'params': [P.k], 'lr': 2e-4})
    opt = torch.optim.Adam(grp)
    w = (1.0 / (T['I'] + 100.0)).sqrt()
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.7), int(iters * 0.9)], 0.3)
    for it in range(iters):
        opt.zero_grad()
        res = torch.exp(logA)[T['g']] * P.x(T['r'], T['fr'], T['s']) - T['I']
        data = huber(res * w).sum(); loss = data + P.penalty()
        loss.backward(); opt.step(); sched.step(); P.gauge()
        if verbose and (it % 500 == 0 or it == iters - 1):
            print(f'  {model} kf {kf} it {it:5d} loss {loss.item():.1f} data/n {data.item() / len(res):.5f}', flush=True)
    for p in P.params() + [P.k]: p.requires_grad_(False)
    return P, data.item() / len(res)


def heldout(D, sel, P):
    """robust residual of pattern P on the subset sel: amplitudes refitted (Huber IRLS), pattern frozen.
    Returns per-datum Huber loss [n_sel] (numpy), standardized residual z, and the count of residual dof."""
    T = tens(D, sel, P.dev)
    with torch.no_grad():
        x = P.x(T['r'], T['fr'], T['s']); A = solve_A(T, x)
        z = (A[T['g']] * x - T['I']) / (T['I'] + 100.0).sqrt()
        return huber(z).cpu().numpy(), z.cpu().numpy(), len(z) - T['ng']
