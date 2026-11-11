"""Shared pieces of the experimental paper figures (fig_exp_new.py): matplotlib settings, the zoom boxes of the two paper
movies, draw_rect, and compute_pc = the RSP panel (Pearson correlation of each render with the widefield mean image;
`pearson_corr.compute_correlation` is imported from ~/smlmtorch/scripts). compute_pc caches <tif>_mean.npy and its result
next to the movie path, so that path must be writable (out/paper2/<ds>/movie/)."""
import os, sys, pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from smlmtorch import Dataset
from smlmtorch.util.multipart_tiff import tiff_read_all

sys.path.append(os.path.expanduser('~/smlmtorch/scripts'))   # for `from pearson_corr import compute_correlation` only
mpl.rcParams.update({"font.size": 15, "font.serif": [], "svg.fonttype": 'none'})    # svg text stays text

# zoom boxes ((x, y) corner px, width px, render args) and overview render args, as in the preprint's Fig 3 / Fig 4
ZOOMS = {
    '1809_2': dict(renders=[((75, 28), 9, {'scalebar_nm': 500, 'clip_percentile': 97}),
                            ((75, 76), 13, {'scalebar_nm': 500, 'clip_percentile': 97})],
                   render_args={'clip_percentile': 96}),
    '2.5nM_2': dict(renders=[((50, 15), 30, {'scalebar_nm': 500, 'clip_percentile': 98}),
                             ((85, 40), 15, {'scalebar_nm': 500, 'clip_percentile': 98})],
                    render_args={'clip_percentile': 98}),
}


def draw_rect(roi, zoom, ax):
    zroi = np.array(roi) * zoom
    rect = patches.Rectangle(zroi[0], zroi[1][0] - zroi[0][0], zroi[1][1] - zroi[0][1],
                             linewidth=5, edgecolor='white', facecolor='none', transform=ax.transData)
    ax.add_patch(rect)
    return rect


def compute_pc(tif_path, datasets, border=2, zoom=20, sigma=1.3):
    """-> pc [n], rmsd [n], error maps; cached per set of datasets (key: sum of positions)"""
    checksum = 1
    for ds in datasets:
        checksum += ds.pos.sum()
    checksum += border * zoom * sigma
    cache_fn = tif_path + f'_rse_cache_{checksum}.pkl'
    if os.path.exists(cache_fn):
        with open(cache_fn, 'rb') as f:
            return pickle.load(f)
    from pearson_corr import compute_correlation
    mean_img_fn = tif_path + '_mean.npy'
    if os.path.exists(mean_img_fn):
        mean_img = np.load(mean_img_fn)
    else:
        mean_img = np.mean(tiff_read_all(tif_path), axis=0, dtype=np.float32)
        np.save(mean_img_fn, mean_img)
    pc = np.zeros(len(datasets)); rmsd = np.zeros(len(datasets)); error_maps = []
    for i in range(len(datasets)):
        pc[i], rmsd[i], error_map = compute_correlation(datasets[i].render(zoom=zoom), mean_img, sigma=sigma, align=False)
        error_maps.append(error_map)
    with open(cache_fn, 'wb') as f:
        pickle.dump([pc, rmsd, error_maps], f)
    return pc, rmsd, error_maps
