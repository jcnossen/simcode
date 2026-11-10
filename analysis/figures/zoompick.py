"""Pick zoom regions at filament crossings: render at 20 nm, smooth, threshold, skeletonize, find junction pixels
(>= 3 skeleton neighbours), score by local density, return non-overlapping zoom boxes (px units of the input).
Skeleton: skel.py (numpy port of skimage.morphology.skeletonize, identical output)."""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, convolve, label, center_of_mass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skel import skeletonize
def pick_crossings(x, y, lim=(5, 115, 5, 115), n=3, size=10.0, px=0.2, margin=None):
    a, b, c, d = lim
    H, _, _ = np.histogram2d(y, x, bins=[int((d - c) / px), int((b - a) / px)], range=[[c, d], [a, b]])
    S = gaussian_filter(H, 2.0)
    m = S > np.percentile(S[S > 0], 70)
    sk = skeletonize(m)
    nb = convolve(sk.astype(int), np.ones((3, 3), int), mode='constant') - sk
    J = sk & (nb >= 3)
    lab, nl = label(gaussian_filter(J.astype(float), 1.5) > 0.05)
    cands = []
    for i in range(1, nl + 1):
        cy, cx = center_of_mass(lab == i)
        X, Y = a + (cx + 0.5) * px, c + (cy + 0.5) * px
        r = int(1.5 / px); iy, ix = int(cy), int(cx)
        dens = S[max(0, iy - r):iy + r, max(0, ix - r):ix + r].mean()
        cands.append((dens, X, Y))
    cands.sort(reverse=True)
    out, half = [], size / 2
    margin = half if margin is None else margin
    for dens, X, Y in cands:
        if X - margin < a or X + margin > b or Y - margin < c or Y + margin > d: continue
        if all(max(abs(X - u), abs(Y - v)) > size for u, v in out):
            out.append((X, Y))
        if len(out) == n: break
    return [(X - half, X + half, Y - half, Y + half) for X, Y in out]
