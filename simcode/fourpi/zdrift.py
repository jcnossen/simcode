"""Sample axial drift estimated from the structure itself (4Pi tables; no fiducials needed)."""
import numpy as np


def zdrift_struct(x, y, z, w, bin_nm=60.0, blk_w=500, n_it=8, clip=150.0):
    """Sample axial drift from the localizations (generalises the NPC per-pore step to any sample): per block of blk_w
    windows, the median of z - (median z of the same bin_nm lateral bin over the whole movie), smoothed over blocks
    (sd 1 block), iterated n_it times; zero mean. x, y, z in nm, w = window index per localization. Returns the drift (nm)
    per block; subtract drift[w // blk_w]. Blocks with fewer than 200 rows within clip nm get no update."""
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d
    ix = np.floor((x - x.min()) / bin_nm).astype(np.int64); iy = np.floor((y - y.min()) / bin_nm).astype(np.int64)
    pix = np.unique(ix * (iy.max() + 1) + iy, return_inverse=True)[1]
    blk = (w // blk_w).astype(np.int64); nb = int(blk.max()) + 1; drift = np.zeros(nb)
    for _ in range(n_it):
        zc = z - drift[blk]
        ref = pd.Series(zc).groupby(pix).median().to_numpy()
        r = zc - ref[pix]; m = np.abs(r) < clip
        g = pd.Series(r[m]).groupby(blk[m]); md = np.array(g.median().reindex(range(nb)).fillna(0).to_numpy(), dtype=float)
        md[np.array(g.size().reindex(range(nb)).fillna(0).to_numpy()) < 200] = 0
        drift = drift + gaussian_filter1d(md, 1.0); drift -= drift.mean()
    return drift
