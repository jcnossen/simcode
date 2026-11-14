"""Movie of the moving-microtubule simulation (SI S8): ground truth and the methods side by side, rendered from existing outputs.
Time axis: a sliding window of 1000 frames (the preprint's block length) advanced by 250 frames (253 movie frames for 64 000
frames; 12 movie frames / s -> ~21 s). The tubules move ~0.1 px (11 nm) per 250 frames, so consecutive movie frames change smoothly,
and a 1000-frame window holds ~5 000-20 000 localizations per method. Ground truth = all trajectory points of the window
(traj.npy, every 10th frame). Every panel: same 50 x 50 px field, 10.8 nm pixels (zoom 10), 1-pixel Gaussian blur, inferno colour
map, intensity scale fixed per panel over the movie (the median over windows of the 99.5th percentile), 500 nm scale bar, label
with the window's localization count. Rows used = each method's scored rows (SMLM / SIMFLUX / DECODE: all output rows; SIMCODE
v2: sigma <= 12 nm; SIMCODE v1: its merged rows).
usage: python analysis/papersim/make_movie.py <movie dir> [--model m1809] [--decode decode_sim] [--step 250] [--block 1000] [--fps 12]
outputs: <movie dir>/full/movie_<model>.mp4 (h264, yuv420p), movie_<model>_still.png"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('movie'); ap.add_argument('--model', default='m1809'); ap.add_argument('--decode', default='decode_sim')
    ap.add_argument('--step', type=int, default=250); ap.add_argument('--block', type=int, default=1000); ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--zoom', type=int, default=10)
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    import imageio.v2 as imageio
    md = os.path.abspath(a.movie); run = os.path.join(md, 'full'); g = json.load(open(os.path.join(md, 'gen.json')))
    W, nf, z = g['W'], g['nframes'], a.zoom
    traj = np.load(os.path.join(md, 'traj.npy'), mmap_mode='r'); ts = int(g['traj_step'])
    P = M.method_paths(run, a.model)
    dec_lab = {'decode_sim': 'DECODE (retrained, dec_sim1809)', 'decode': 'DECODE (dec_1809_2)', 'decode_pre': 'DECODE (preprint model)'}
    panels = [('smlm', 'SMLM'), (a.decode, dec_lab.get(a.decode, a.decode)), ('sf_gtp', 'SIMFLUX (true pattern)'), ('v1', 'SIMCODE (preprint)'),
              ('v2sig', 'SIMCODE v2 (σ ≤ 12 nm)')]
    T = {}
    for k, _ in panels:
        if k in P and os.path.exists(P[k]):
            t = M.load_method(k, P[k]); o = np.argsort(t['f'], kind='stable'); T[k] = {kk: v[o] for kk, v in t.items()}
    v1_failed = 'v1' in T and len(T['v1']['x']) < 0.2 * max(len(T[k]['x']) for k in T if k != 'v1')
    starts = np.arange(0, nf - a.block + 1, a.step)
    def rows(k, s):
        t = T[k]; i0, i1 = np.searchsorted(t['f'], [s, s + a.block]); return t['x'][i0:i1], t['y'][i0:i1]
    def rend(x, y):
        im = np.histogram2d(y, x, bins=W * z, range=[[0, W], [0, W]])[0]; return gaussian_filter(im, 1.0)
    def gt(s):
        p = np.asarray(traj[s // ts:(s + a.block) // ts]).reshape(-1, 2); return rend(p[:, 0], p[:, 1]), len(p)
    # fixed per-panel intensity scale
    probe = starts[np.linspace(0, len(starts) - 1, 9).astype(int)]
    vmax = {'gt': np.median([np.percentile(gt(s)[0], 99.5) for s in probe])}
    for k in T: vmax[k] = np.median([np.percentile(rend(*rows(k, s)), 99.5) for s in probe]) + 1e-9
    out = os.path.join(run, f'movie_{a.model}.mp4')
    wr = imageio.get_writer(out, fps=a.fps, codec='libx264', quality=7, pixelformat='yuv420p', macro_block_size=16)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8.4), dpi=100); axs = axs.ravel()
    cond = {'1809_2': 'experimental 1809/2 photons and background', None: 'preprint photons (500 / 6 frames) and background'}.get(g.get('cond'), 'preprint photons')
    psf = 'Gaussian 1.3 px PSF' if str(g.get('render_psf', '')).startswith('gauss') else 'measured (averaged 1809/2) PSF'
    still = None
    for fi, s in enumerate(starts):
        for ax in axs: ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        im, n = gt(s); axs[0].imshow(np.clip(im / vmax['gt'], 0, 1), cmap='inferno', vmin=0, vmax=1); axs[0].set_title('ground truth (all label positions)', fontsize=11)
        for ax, (k, lab) in zip(axs[1:], panels):
            if k not in T or (k == 'v1' and v1_failed):
                ax.set_facecolor('k'); ax.text(0.5, 0.5, f'{lab}:\n' + ('pattern estimate failed,\nno usable output' if k == 'v1' else 'no output'), color='w', ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(lab, fontsize=11); continue
            x, y = rows(k, s); ax.imshow(np.clip(rend(x, y) / vmax[k], 0, 1), cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f'{lab}\n{len(x):,} localizations', fontsize=10)
        for ax in axs:
            sb = 500 / C.PX_NM * z; ax.plot([15, 15 + sb], [W * z - 20] * 2, 'w-', lw=4); ax.text(15, W * z - 32, '500 nm', color='w', fontsize=9)
        fig.suptitle(f'moving microtubules, {cond}, {psf}: frames {s}-{s + a.block}', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        wr.append_data(frame)
        if fi == len(starts) // 2: still = frame
    wr.close(); plt.close(fig)
    imageio.imwrite(os.path.join(run, f'movie_{a.model}_still.png'), still)
    print(out, f'{os.path.getsize(out) / 1e6:.1f} MB', len(starts), 'frames')


if __name__ == '__main__':
    main()
