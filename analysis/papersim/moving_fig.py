"""SI S8 figure (moving microtubules) from metrics.py's moving metrics: per-1000-frame-block Jaccard at 10 / 20 / 40 nm (curves) and
their means (grouped bars, as the preprint's figure), per-block FRC, and renders of three blocks (start, middle, end) for the
ground truth (trajectory points of the block) and every method. Also a hand-count check of the Jaccard on one block (brute-force
distances instead of KD-trees) written to the json.
usage: python analysis/papersim/moving_fig.py <movie dir> [--model m1809]
outputs: <movie dir>/full/moving_fig_<model>.png, moving_table_<model>.md, moving_check_<model>.json"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import metrics as M

ORDER = ['smlm', 'sf', 'sf_gtp', 'decode', 'decode_pre', 'decode_sim', 'v1', 'v1gt', 'v2sig']
LAB = dict(smlm='SMLM', sf='SIMFLUX (est. pattern)', sf_gtp='SIMFLUX (GT pattern)', decode='DECODE dec_1809_2', decode_sim='DECODE dec_sim1809', decode_pre='DECODE (preprint model)',
           v1='SIMCODE v1 (preprint)', v1gt='SIMCODE v1, GT pattern (diag.)', v2sig='SIMCODE v2, σ <= 12 nm')
COL = dict(smlm='tab:blue', sf='tab:green', sf_gtp='olive', decode='tab:orange', decode_sim='darkorange', decode_pre='orangered', v1='tab:brown', v1gt='peru', v2sig='crimson')


def hand_check(md, run, model, R, block=10, thr_nm=20):
    """Jaccard of one block recomputed by brute force (all pairwise distances) for v2sig and compared with metrics.py"""
    g = json.load(open(os.path.join(md, 'gen.json'))); traj = np.load(os.path.join(md, 'traj.npy'), mmap_mode='r'); ts = int(g['traj_step'])
    T = M.load_method('v2sig', M.method_paths(run, model)['v2sig']); bs = R['block']
    s = (T['f'] > block * bs) & (T['f'] < (block + 1) * bs)
    A = np.c_[T['x'][s], T['y'][s]]; B = np.asarray(traj[block * bs // ts:(block + 1) * bs // ts]).reshape(-1, 2)
    t = thr_nm / C.PX_NM
    a_hit = np.zeros(len(A), bool); b_hit = np.zeros(len(B), bool)
    for i in range(0, len(A), 500):
        d = np.sqrt(((A[i:i + 500, None, :] - B[None]) ** 2).sum(-1)); a_hit[i:i + 500] = (d <= t).any(1); b_hit |= (d <= t).any(0)
    jac = (a_hit.sum() + b_hit.sum()) / (len(A) + len(B))
    k = R['thresholds_nm'].index(thr_nm)
    return dict(block=block, threshold_nm=thr_nm, n_locs=int(len(A)), n_gt_points=int(len(B)), locs_within=int(a_hit.sum()), gt_within=int(b_hit.sum()),
                jaccard_bruteforce=float(jac), jaccard_metrics=float(R['methods']['v2sig']['jaccard'][k][block]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('movie'); ap.add_argument('--model', default='m1809'); a = ap.parse_args()
    md = os.path.abspath(a.movie); run = os.path.join(md, 'full')
    R = json.load(open(os.path.join(run, f'metrics_{a.model}.json')))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = [k for k in ORDER if k in R['methods']]
    g = json.load(open(os.path.join(md, 'gen.json'))); traj = np.load(os.path.join(md, 'traj.npy'), mmap_mode='r'); ts = int(g['traj_step']); W = g['W']
    nb, bs = R['blocks'], R['block']; show = [1, nb // 2, nb - 2]
    fig = plt.figure(figsize=(2.2 * (len(ks) + 1), 2.2 * 3 + 8))
    gs = fig.add_gridspec(3 + 2, len(ks) + 1, height_ratios=[1, 1, 1, 1.6, 1.6])
    from scipy.ndimage import gaussian_filter
    z = 10
    def rend(x, y):
        im = np.histogram2d(y, x, bins=W * z, range=[[0, W], [0, W]])[0]; im = gaussian_filter(im, 1.0); return np.clip(im / (np.percentile(im, 99.7) + 1e-9), 0, 1)
    for r, b in enumerate(show):
        P = np.asarray(traj[b * bs // ts:(b + 1) * bs // ts]).reshape(-1, 2)
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(rend(P[:, 0], P[:, 1]), cmap='gray'); ax.axis('off'); ax.set_title(f'GT, frames {b * bs}-{(b + 1) * bs}', fontsize=7)
        for i, k in enumerate(ks):
            T = M.load_method(k, R['methods'][k]['path']); s = (T['f'] > b * bs) & (T['f'] < (b + 1) * bs)
            ax = fig.add_subplot(gs[r, i + 1]); ax.imshow(rend(T['x'][s], T['y'][s]), cmap='inferno'); ax.axis('off')
            if r == 0: ax.set_title(LAB[k], fontsize=7)
    ax = fig.add_subplot(gs[3, :(len(ks) + 1) // 2])
    for k in ks:
        J = np.array(R['methods'][k]['jaccard']); ax.plot(np.arange(nb) * bs / 1000, J[0], color=COL[k], lw=1.2, label=LAB[k])
    ax.set_xlabel('block start (x1000 frames)'); ax.set_ylabel('Jaccard 10 nm'); ax.set_ylim(0, 1); ax.legend(fontsize=6, ncol=2); ax.set_title('per-block Jaccard (10 nm)', fontsize=8)
    ax = fig.add_subplot(gs[3, (len(ks) + 1) // 2:])
    for k in ks:
        ax.plot(np.arange(nb) * bs / 1000, R['methods'][k]['frc_block_nm'], color=COL[k], lw=1)
    ax.set_ylabel('FRC per block (nm)'); ax.set_xlabel('block start (x1000 frames)'); ax.set_title('per-block FRC (disjoint windows, sigma < 0.15 px, 60-frame halves)', fontsize=8)
    ax = fig.add_subplot(gs[4, :])
    x = np.arange(len(ks)); w = 0.27
    for j, (t, c) in enumerate(zip(R['thresholds_nm'], ('r', 'g', 'b'))):
        ax.bar(x + (j - 1) * w, [R['methods'][k]['jaccard_mean'][j] for k in ks], w, color=c, alpha=0.7, label=f'{t} nm')
    ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in ks], fontsize=7, rotation=15); ax.set_ylabel('mean Jaccard over blocks'); ax.legend(fontsize=7); ax.set_ylim(0, 1)
    fig.suptitle(f"moving microtubules ({g.get('cond', 'preprint')} photons / bg), {g['nframes']} frames, {nb} blocks of {bs}; v2 net set {a.model}", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(run, f'moving_fig_{a.model}.png'), dpi=100); plt.close(fig)
    L = ['| method | rows | Jaccard 10 / 20 / 40 nm (mean over blocks) | FRC per block, median (nm) |', '|---|---|---|---|']
    for k in ks:
        r = R['methods'][k]
        L.append(f"| {LAB[k]} | {r['n_all']} | {' / '.join(f'{v:.3f}' for v in r['jaccard_mean'])} | {r['frc_block_median_nm'] if r['frc_block_median_nm'] is None else round(r['frc_block_median_nm'], 1)} |")
    open(os.path.join(run, f'moving_table_{a.model}.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
    chk = hand_check(md, run, a.model, R)
    json.dump(chk, open(os.path.join(run, f'moving_check_{a.model}.json'), 'w'), indent=1); print('hand check', chk)


if __name__ == '__main__':
    main()
