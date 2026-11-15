"""SI figure 'si_simulated_live_smlm' (moving microtubules) in the preprint's layout (smlmtorch 5f7fdcb scripts/fig_moving_tubules.py,
the '2x3 middle block' panel + the 'mean Jaccard' bar chart), redrawn with the simcode2 results.

A  one reconstructed 1000-frame block, 2 x 3: Input frame | SMLM | SIMFLUX / Ground Truth | DECODE | SIMCODE. Render style of the
   preprint: every localization of the block (frames strictly inside the block, as the preprint's Jaccard) drawn as a Gaussian of
   sigma 1 render pixel at zoom 10 (10.8 nm pixels), clipped at the 99th percentile, inferno, 500 nm scale bar (Dataset.renderFigure);
   input frame = the block's first frame (gray); ground truth = all trajectory points of the block splatted bilinearly at zoom 30
   (render_frames_to_bitmaps; here the trajectory stored every 10 frames, clipped at the 99.5th percentile instead of uint8).
   SIMCODE = SIMCODE v2 with its paper selection (p >= 0.63, calibrated max(sigma) <= 12 nm); the other methods = all their output
   rows (= the rows the Jaccard scores; metrics.py moving).
B  mean Jaccard over the blocks at 10 / 20 / 40 nm, bars red / green / blue (alpha 0.7), legend 'Jaccard distance threshold = ...'.
--internal 1: also the preprint SIMCODE (extra column in A, extra group in B) -> <name>_internal.*
Numbers: the json written by metrics.py (metrics_<model>.json) and fig_moving_main.py's GT-matched errors (moving_gterr_<model>.json).
usage: python analysis/papersim/fig_moving_pp.py [--movie out/papersim/moving_exp_v2psf] [--model m1809v2] [--block 32] [--internal 0|1]
output: out/paper_figs_v2/si_simulated_live_smlm{,_internal}.{pdf,svg,png}"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter


def render_locs(x, y, W, zoom=10, sigma=1.0, clip=99):
    img = np.histogram2d(y * zoom, x * zoom, bins=W * zoom, range=[[0, W * zoom], [0, W * zoom]])[0]
    img = gaussian_filter(img, sigma) * 2 * np.pi * sigma ** 2          # = summed unit Gaussians (fastpsf Draw)
    return np.minimum(img, np.percentile(img, clip))


def render_gt(P, W, zoom=30, clip=99.5):
    p = P * zoom; ok = (p[:, 0] >= 0) & (p[:, 0] < W * zoom) & (p[:, 1] >= 0) & (p[:, 1] < W * zoom); p = p[ok]
    x0 = np.floor(p[:, 0]).astype(int); y0 = np.floor(p[:, 1]).astype(int); wx = p[:, 0] - x0; wy = p[:, 1] - y0
    x1 = np.minimum(x0 + 1, W * zoom - 1); y1 = np.minimum(y0 + 1, W * zoom - 1)
    img = np.zeros((W * zoom, W * zoom))
    for yy, xx, w in ((y0, x0, (1 - wx) * (1 - wy)), (y0, x1, wx * (1 - wy)), (y1, x0, (1 - wx) * wy), (y1, x1, wx * wy)):
        np.add.at(img, (yy, xx), w)
    return np.minimum(img, np.percentile(img, clip))


def scalebar(ax, W, nm=500, fs=11):
    ax.add_patch(Rectangle(xy=(0.05, 0.05), width=nm / (W * C.PX_NM), height=0.03, transform=ax.transAxes, color='white'))
    ax.text(0.05, 0.08, f'{nm} nm', transform=ax.transAxes, color='white', va='bottom', fontsize=fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movie', default=os.path.join(C.OUT, 'moving_exp_v2psf')); ap.add_argument('--model', default='m1809v2')
    ap.add_argument('--name', default='si_simulated_live_smlm'); ap.add_argument('--outdir', default=os.path.join(os.path.dirname(C.OUT), 'paper_figs_v2'))
    ap.add_argument('--block', type=int, default=32, help='the reconstructed block shown (default: the middle block, as the preprint)')
    ap.add_argument('--internal', type=int, default=0)
    a = ap.parse_args()
    md = os.path.abspath(a.movie); run = os.path.join(md, 'full')
    R = json.load(open(os.path.join(run, f'metrics_{a.model}.json'))); g0 = json.load(open(os.path.join(md, 'gen.json')))
    W, bs = g0['W'], R['block']; b = a.block; f0, f1 = b * bs, (b + 1) * bs
    traj = np.load(os.path.join(md, 'traj.npy'), mmap_mode='r'); ts = int(g0['traj_step'])
    meth = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2sig', 'SIMCODE')]
    if a.internal: meth = meth[:3] + [('v1', 'SIMCODE (preprint)'), ('v2sig', 'SIMCODE v2')]
    ncol = 3 if not a.internal else 4
    fig = plt.figure(figsize=(7.4 if not a.internal else 9.6, 8.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2 * 7.4 / 3 * 0.95, 3.4], hspace=0.12, left=0.1, right=0.98, top=0.95, bottom=0.07)
    ga = gs[0].subgridspec(2, ncol, wspace=0.04, hspace=0.12)
    import tifffile
    with tifffile.TiffFile(os.path.join(md, 'movie.tif')) as tf: fr = tf.pages[f0].asarray().astype(np.float32)
    A = fig.add_subplot(ga[0, 0]); A.imshow(fr, cmap='gray'); A.axis('off'); A.set_title('Input frame', fontsize=12)
    A.text(-0.08, 1.06, 'A', transform=A.transAxes, fontsize=16, fontweight='bold', ha='right', va='bottom')
    G = fig.add_subplot(ga[1, 0]); G.imshow(render_gt(np.asarray(traj[f0 // ts:f1 // ts]).reshape(-1, 2), W), cmap='gray'); G.axis('off')
    G.set_title('Ground Truth', fontsize=12)
    slots = [(0, 1), (0, 2), (1, 1), (1, 2)] if not a.internal else [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]
    nloc = {}
    for (k, lab), (r, c) in zip(meth, slots):
        T = M.load_method(k, R['methods'][k]['path']); s = (T['f'] > f0) & (T['f'] < f1); nloc[k] = int(s.sum())
        X = fig.add_subplot(ga[r, c]); X.imshow(render_locs(T['x'][s], T['y'][s], W), cmap='inferno'); X.axis('off'); X.set_title(lab, fontsize=12)
        scalebar(X, W)
    if a.internal: fig.add_subplot(ga[1, 3]).axis('off')
    B = fig.add_subplot(gs[1])
    n = len(meth); x = (np.arange(3)[:, None] + np.arange(n)[None, :] * 4).flatten()
    y = np.array([[R['methods'][k]['jaccard_mean'][t] for k, _ in meth] for t in range(3)]).flatten()
    colors = ['r', 'g', 'b']
    B.bar(x, y, color=np.repeat(colors, n), alpha=0.7)
    B.set_xticks(x.reshape(3, n).mean(0)); B.set_xticklabels([l for _, l in meth]); B.set_ylabel('Jaccard Index')
    B.set_title('Jaccard index for different methods and thresholds')
    for i, c in enumerate(colors): B.bar(0, 0, color=c, label=f'Jaccard distance threshold = {R["thresholds_nm"][i]} nm', alpha=0.7)
    B.legend(fontsize=9, loc='upper left'); B.set_ylim(0, 1.15 if a.internal else 1.1)
    B.text(-0.06, 1.04, 'B', transform=B.transAxes, fontsize=16, fontweight='bold', ha='right', va='bottom')
    os.makedirs(a.outdir, exist_ok=True); out = os.path.join(a.outdir, a.name + ('_internal' if a.internal else ''))
    for ext, kw in (('svg', dict(dpi=300)), ('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        fig.savefig(f'{out}.{ext}', **kw); print(f'{out}.{ext}')
    json.dump(dict(block=b, frames=[f0, f1], localizations_in_block=nloc,
                   jaccard_mean={k: R['methods'][k]['jaccard_mean'] for k, _ in meth}, thresholds_nm=R['thresholds_nm']),
              open(out + '_values.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
