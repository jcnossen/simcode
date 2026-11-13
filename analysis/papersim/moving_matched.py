"""Matched-count Jaccard for the moving-microtubule movie (S8): the preprint's Jaccard (GT = all trajectory points of the block)
rewards how many localizations a method outputs, so here every method is subsampled, per 1000-frame block, to the same number
of localizations: the smallest count among the reference methods (DECODE, SIMCODE v1, SIMCODE v2 sigma <= 12 nm (the main row since 2026-11-13; it was u <= 0.45 before); methods with fewer
rows than that are used whole and flagged). Also DECODE restricted to window starts every 6 frames (as the preprint's DECODE,
6-frame sums from frame 0). Random subsampling with a fixed seed; the mean over 3 draws.
usage: python analysis/papersim/moving_matched.py <movie dir> [--netset m1809]
output: <movie>/full/moving_matched_<netset>.json / .md"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('movie'); ap.add_argument('--netset', default='m1809'); a = ap.parse_args()
    md = os.path.abspath(a.movie); run = os.path.join(md, 'full')
    R = json.load(open(os.path.join(run, f'metrics_{a.netset}.json'))); g = json.load(open(os.path.join(md, 'gen.json')))
    traj = np.load(os.path.join(md, 'traj.npy'), mmap_mode='r'); ts = int(g['traj_step']); bs, nb = R['block'], R['blocks']; thr = R['thresholds_nm']
    P = M.method_paths(run, a.netset)
    T = {k: M.load_method(k, P[k]) for k in P if k in R['methods']}
    for dk in [k for k in ('decode', 'decode_pre', 'decode_sim') if k in T]:
        t = T[dk]; s = t['f'] % 6 == 0
        if not s.all(): T[dk + '_w6'] = {kk: v[s] for kk, v in t.items()}
    refs = [k for k in ('decode_pre', 'decode', 'decode_sim', 'v1', 'v2sig') if k in T]
    nrow = {k: len(T[k]['x']) for k in refs}
    refs = [k for k in refs if nrow[k] >= 0.2 * max(nrow.values())]     # a failed method (e.g. v1 when its pattern estimate fails) is not a reference
    rng = np.random.default_rng(0)
    J = {k: np.full((len(thr), nb), np.nan) for k in T}; nmatch = np.zeros(nb, int); short = {k: 0 for k in T}
    for i in range(nb):
        sel = {k: np.nonzero((t['f'] > i * bs) & (t['f'] < (i + 1) * bs))[0] for k, t in T.items()}
        n = min(len(sel[k]) for k in refs if len(sel[k]) > 0); nmatch[i] = n
        G = np.asarray(traj[i * bs // ts:(i + 1) * bs // ts]).reshape(-1, 2)
        for k, t in T.items():
            idx = sel[k]
            if not len(idx): continue
            if len(idx) < n: short[k] += 1
            vals = []
            for rep in range(3):
                ii = idx if len(idx) <= n else rng.choice(idx, n, replace=False)
                vals.append([M.compute_jaccard_points(np.c_[t['x'][ii], t['y'][ii]], G, th / C.PX_NM) for th in thr])
            J[k][:, i] = np.mean(vals, 0)
    out = dict(movie=md, netset=a.netset, reference_methods=refs, matched_count_per_block=dict(mean=float(nmatch.mean()), min=int(nmatch.min()), max=int(nmatch.max())),
               thresholds_nm=thr, methods={k: dict(jaccard_mean=np.nanmean(J[k], 1).tolist(), blocks_with_fewer_rows=short[k]) for k in T})
    json.dump(out, open(os.path.join(run, f'moving_matched_{a.netset}.json'), 'w'), indent=1)
    L = [f'| method | Jaccard 10 / 20 / 40 nm at matched count (mean {nmatch.mean():.0f} localizations per block) | blocks with fewer rows |', '|---|---|---|']
    for k in T: L.append(f"| {k} | {' / '.join(f'{v:.3f}' for v in out['methods'][k]['jaccard_mean'])} | {short[k]} |")
    open(os.path.join(run, f'moving_matched_{a.netset}.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
