import numpy as np
from scipy.spatial import cKDTree


def nn_overlap(x, y, wstart, T=12, rmax=8.0, same_site=1.0):
    """Distance to the nearest other localization whose window overlaps in time (|dw| < T), excluding the same site seen
    in another window (distance < same_site px, dw != 0). Returns inf if none within rmax (critic review 12 #5)."""
    pts = np.stack([x, y], 1)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(rmax, output_type='ndarray')
    nn = np.full(len(x), np.inf)
    if len(pairs) == 0:
        return nn
    i, j = pairs[:, 0], pairs[:, 1]
    dw = np.abs(wstart[i] - wstart[j]); d = np.hypot(x[i] - x[j], y[i] - y[j])
    ok = (dw < T) & ~((dw > 0) & (d < same_site))
    i, j, d = i[ok], j[ok], d[ok]
    np.minimum.at(nn, i, d); np.minimum.at(nn, j, d)
    return nn
