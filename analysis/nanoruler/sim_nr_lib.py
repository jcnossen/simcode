"""Helpers of the simulated nanoruler movie (sim_nr.py): frame-order maps and exact ground-truth windows.

TIME BASES
  acquisition frame t : the order in which the camera recorded. Pattern index t % 6, orientation (t % 6) % 2,
                        phase step (t % 6) // 2 (the two orientations are interleaved).
  stored frame f      : the order of the frames in sim_movie.tif (and of the cached real movie, analysis/nanoruler/reorder.py).
                        Every 6-frame cycle is stored in the order PERM = [0, 2, 4, 1, 3, 5]:
                            stored frame 6 c + j  =  acquisition frame 6 c + PERM[j]
                        so that stored step s = f % 6 has orientation s // 3 and phase step s % 3 (the simcode / calib convention).
Only stored windows that start at a multiple of 6 are 6 consecutive acquisition frames. A stored window starting at 6 c + 3
holds the acquisition frames 6 c + {1, 3, 5} and 6 c + 6 + {0, 2, 4}."""
import numpy as np

PERM = np.array([0, 2, 4, 1, 3, 5])      # stored position j in the cycle -> acquisition index in the cycle
INV = np.argsort(PERM)                   # acquisition index in the cycle -> stored position ([0, 3, 1, 4, 2, 5])


def acq_of_stored(f):
    f = np.asarray(f); return 6 * (f // 6) + PERM[f % 6]


def stored_of_acq(t):
    t = np.asarray(t); return 6 * (t // 6) + INV[t % 6]


def stored_onf(t0, t1, f):
    """exact on-fraction of events with acquisition on / off times t0, t1 (broadcastable) in the stored frames f"""
    ta = acq_of_stored(f)
    return np.clip(np.minimum(t1, ta + 1) - np.maximum(t0, ta), 0, 1)


def gt_windows_exact(G, grid=3, drift=True, margin=None):
    """Exact counterpart of simcode.evalproto.lib.gt_windows_fast for the reordered movie.
    Rows (event, stored window start w, w % grid == 0) with >= 1 on-frame; columns
      0 w (stored), 1 event, 2 x, 3 y (px, drift-free site/event position), 4 P (pattern-weighted window photons without the
      per-frame fluctuation, as gt_windows_fast; with the sample drift in the illumination phase if drift), 5 target flag
      (photon time centroid over the stored window in [1, 4); inside the FOV minus `margin` px if given),
      6 minimum stored on-fraction of the 6 frames (== 1: fully on during the 6 stored frames), 7 summed on-fraction."""
    x, y, br, t0, t1, ea, es, patt = (G[k] for k in ('x', 'y', 'bright', 't0', 't1', 'ea', 'es', 'patt'))
    NF = int(G['nf'])
    c0 = np.floor(t0 / 6).astype(np.int64); c1 = np.floor(t1 / 6).astype(np.int64)
    wlo = np.maximum(np.ceil((6 * c0 - 5) / grid), 0).astype(np.int64) * grid
    whi = np.minimum(np.floor((6 * c1 + 5) / grid), (NF - 6) // grid).astype(np.int64) * grid
    n = np.maximum((whi - wlo) // grid + 1, 0)
    ev = np.repeat(np.arange(len(x)), n)
    w = wlo[ev] + grid * (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    fr = w[:, None] + np.arange(6)[None]
    onf = stored_onf(t0[ev, None], t1[ev, None], fr)
    pt = patt[fr]
    xe, ye = x[ev, None], y[ev, None]
    if drift and 'drift' in G:
        xe = xe + G['drift'][fr, 0]; ye = ye + G['drift'][fr, 1]
    il = pt[..., 4] * (1 + pt[..., 3] * np.cos(pt[..., 0] * xe + pt[..., 1] * ye - pt[..., 2]))
    s = fr % 6
    ph = br[ev, None] * il * onf * np.take_along_axis(ea[ev], s // 3, 1) * np.take_along_axis(es[ev], s, 1)
    P = ph.sum(1); tc = (ph * np.arange(6)).sum(1) / np.maximum(P, 1e-6)
    tgt = (tc >= 1.0) & (tc < 4.0) & (onf.sum(1) >= 1)
    if margin is not None:
        H, W = (int(v) for v in G['shape'])
        tgt &= (x[ev] >= margin) & (x[ev] < W - margin) & (y[ev] >= margin) & (y[ev] < H - margin)
    k = onf.sum(1) > 0
    return np.c_[w, ev, x[ev], y[ev], P, tgt, onf.min(1), onf.sum(1)][k]


def frame_photons(G, ev):
    """dense per-stored-frame GT photons of one event from the sparse table: (stored frames, photons, on-fraction)"""
    m = G['fr_ev'] == ev; o = np.argsort(G['fr_frame'][m])
    return G['fr_frame'][m][o], G['fr_ph'][m][o], G['fr_onf'][m][o]
