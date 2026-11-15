"""
Output modes for the per-window sc pipeline (2026-11-09, duplicates / overlapping windows; run_uncalibrated --pass2_mode solve).
The pass-2 list has one localization per (emitter, 6-frame window) at stride 3, so consecutive windows w, w + 3 share 3
frames of photons and an emitter that is on for many frames appears in every window. The per-window (dynamic) list is
kept unchanged (pass2_<tag>_<f0>.npz); these functions derive
  'disjoint'  photon-disjoint windows only: wstart % 6 == phase (every other W3 window). Each emitter at most once per
              6 frames; no linking, statistics of each row unchanged.
  'merge'     link the same emitter across consecutive windows (w -> w + 3, and w -> w + 6 across one missed window):
              mutual-nearest pairs within min(r_max, k sqrt(s_a^2 + s_b^2)) -> chains (union-find) -> one localization per
              chain: inverse-variance weighted position; variance = (2n / (n + 1)) / sum(1 / s_i^2) for n stride-3 windows
              (each interior frame is in two windows, so the naive inverse-variance sum double counts; exact for n = 1, 2 when
              the windows are photon-disjoint is not assumed), photons = sum over the photon-disjoint members, frame range
              [first window start, last window start + 6), n_windows. Dynamics within the chain are averaged out.
"""
import numpy as np
from scipy.spatial import cKDTree


def disjoint(O, phase=0):
    m = (np.asarray(O['wstart']) % 6) == phase
    return {k: (v[m] if np.ndim(v) and len(v) == len(m) else v) for k, v in O.items()}


def _chains(w, x, y, s, r_max=0.5, k=3.0, gaps=(3, 6)):
    n = len(w); par = np.arange(n)

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    linked_next = np.zeros(n, bool); linked_prev = np.zeros(n, bool)
    uw = np.unique(w); idx = {u: np.nonzero(w == u)[0] for u in uw}
    trees = {u: cKDTree(np.c_[x[i], y[i]]) for u, i in idx.items()}
    for g in gaps:
        for u in uw:
            if u + g not in idx: continue
            ia, ib = idx[u], idx[u + g]
            ia = ia[~linked_next[ia]]; ib = ib[~linked_prev[ib]]
            if len(ia) == 0 or len(ib) == 0: continue
            ta, tb = cKDTree(np.c_[x[ia], y[ia]]), cKDTree(np.c_[x[ib], y[ib]])
            d1, j1 = tb.query(np.c_[x[ia], y[ia]], distance_upper_bound=r_max)
            d2, j2 = ta.query(np.c_[x[ib], y[ib]], distance_upper_bound=r_max)
            for a_, (d, j) in enumerate(zip(d1, j1)):
                if not np.isfinite(d) or j2[j] != a_: continue
                A, Bb = ia[a_], ib[j]
                if d > k * np.sqrt(s[A] ** 2 + s[Bb] ** 2): continue
                par[find(A)] = find(Bb); linked_next[A] = True; linked_prev[Bb] = True
    return np.array([find(i) for i in range(n)])


def merge(O, r_max=0.5, k=3.0, xk='x_hy0', yk='y_hy0'):
    w = np.asarray(O['wstart']).astype(np.int64)
    x, y = np.asarray(O[xk], np.float64), np.asarray(O[yk], np.float64)
    sx, sy = np.asarray(O['sx_cal'], np.float64), np.asarray(O['sy_cal'], np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sx) & np.isfinite(sy) & (sx > 0) & (sy > 0)
    ii = np.nonzero(ok)[0]
    root = _chains(w[ii], x[ii], y[ii], np.sqrt(0.5 * (sx[ii] ** 2 + sy[ii] ** 2)), r_max, k)
    u, lab = np.unique(root, return_inverse=True)
    nC = len(u)
    wx, wy = 1 / sx[ii] ** 2, 1 / sy[ii] ** 2
    Sx = np.bincount(lab, wx, nC); Sy = np.bincount(lab, wy, nC)
    n = np.bincount(lab, minlength=nC).astype(float)
    f = 2 * n / (n + 1)
    w0 = np.full(nC, np.iinfo(np.int64).max); np.minimum.at(w0, lab, w[ii])
    w1 = np.full(nC, -1); np.maximum.at(w1, lab, w[ii])
    dis = ((w[ii] - w0[lab]) % 6) == 0                       # photon-disjoint members (from the chain start)
    ph = np.bincount(lab, np.where(dis, np.asarray(O['phot'], np.float64)[ii], 0), nC) if 'phot' in O else np.zeros(nC)
    out = dict(x=np.bincount(lab, wx * x[ii], nC) / Sx, y=np.bincount(lab, wy * y[ii], nC) / Sy,
               sx=np.sqrt(f / Sx), sy=np.sqrt(f / Sy), n_windows=n.astype(int), wstart=w0, frame0=w0, frame1=w1 + 6, phot=ph)
    out['member_chain'] = np.full(len(w), -1); out['member_chain'][ii] = lab
    return out
