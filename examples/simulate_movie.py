"""
Minimal example data for docs/QUICKSTART.md: a small simulated DMD-SIMFLUX DNA-PAINT movie plus a matching pipeline config.

    python examples/simulate_movie.py /data/simcode_example            # writes movie.tif and quickstart.yaml there (~10 s)
    python -m simcode.pipeline /data/simcode_example/quickstart.yaml   # calibration, in-situ PSF, localization with m1809

The movie uses the simulator the networks are trained with (simcode/simulate/acquisition.py, preset '0.5nM': the PSF of our
setup, blinking DNA-PAINT emitters on filaments, the SIMCODE pattern of 2 orientations x 3 phase steps, fluctuating
background, sCMOS read noise) and is written like a real recording: uint16 ADU, gain 0.56 e/ADU, offset 100 ADU.
No bead stack: the pipeline estimates the PSF from the movie itself. The ground truth (emitter positions, on-times, the
pattern per frame) goes to ground_truth.npz. Needs a CUDA GPU.
"""
import argparse, os
import numpy as np, torch, tifffile
from simcode.simulate.acquisition import AcquisitionSim
from simcode.simulate.batches import paper_bright

GAIN, OFFSET = 0.56, 100.0      # e- / ADU, ADU (written into the config)
dev = 'cuda'


def simulate(n_frames=6000, size=120, seed=1):
    torch.manual_seed(seed)
    rs = AcquisitionSim('0.5nM', size, size, n_frames, seed=seed)
    P = rs.pattern(1)
    while int(P['steps'][0, 0]) != 0:                       # the movie starts at phase step 0 of orientation 0
        P = rs.pattern(1)
    patt = P['patt'][0]                                     # [frames, 5]: kx, ky, phase, depth, relative intensity
    em = rs.emitters(1)
    x, y, z, t0, t1 = (em[k] for k in ('x', 'y', 'z', 't0', 't1'))
    bright = paper_bright('0.5nM', 1, em['fov'], z, rs, jitter=None)
    psf = rs.sim.render_psf(x, y, z)                        # [N, R, R] unit-sum PSF per emitter
    R = psf.shape[-1]; hr = R // 2; pad = hr + 8                 # emitters may sit a few px outside the frame
    bg = rs.background(1, (torch.arange(n_frames, device=dev) % 6)[None], mode='fluct')[0][0]
    rn = rs.rn_map[:size, :size]                            # read noise (e-) per pixel
    cx = torch.round(x).long() + pad; cy = torch.round(y).long() + pad
    oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
    movie = np.empty((n_frames, size, size), np.uint16)
    for a in range(0, n_frames, 60):                         # 60 frames at a time
        fr = torch.arange(a, min(a + 60, n_frames), device=dev, dtype=torch.float32); T = len(fr)
        img = torch.zeros(T, size + 2 * pad, size + 2 * pad, device=dev)
        act = torch.nonzero((t1 > a) & (t0 < a + T))[:, 0]
        if len(act):
            on = (torch.minimum(t1[act, None], fr[None] + 1) - torch.maximum(t0[act, None], fr[None])).clamp(0, 1)
            p = patt[a:a + T]
            illum = p[None, :, 4] * (1 + p[None, :, 3] * torch.cos(p[None, :, 0] * x[act, None] + p[None, :, 1] * y[act, None] - p[None, :, 2]))
            ph = bright[act, None] * illum * on                                     # photons per emitter per frame
            iy = (cy[act, None, None] - hr + oy).reshape(len(act), -1); ix = (cx[act, None, None] - hr + ox).reshape(len(act), -1)
            Wp = img.shape[-1]
            lin = torch.arange(T, device=dev)[None, :, None] * Wp * Wp + (iy * Wp + ix)[:, None, :]
            img.view(-1).index_add_(0, lin.reshape(-1), (ph[:, :, None] * psf[act].reshape(len(act), 1, -1)).reshape(-1))
        mu = img[:, pad:pad + size, pad:pad + size] + bg[a:a + T]
        e = torch.poisson(mu) + torch.randn_like(mu) * rn
        movie[a:a + T] = torch.round(e / GAIN + OFFSET).clamp(0, 65535).cpu().numpy().astype(np.uint16)
    gt = dict(x=x, y=y, z=z, t0=t0, t1=t1, photons=bright, pattern=patt)
    return movie, {k: v.cpu().numpy() for k, v in gt.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', help='folder for movie.tif, ground_truth.npz and quickstart.yaml')
    ap.add_argument('--frames', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    movie, gt = simulate(a.frames, seed=a.seed)
    tifffile.imwrite(os.path.join(out, 'movie.tif'), movie)
    np.savez_compressed(os.path.join(out, 'ground_truth.npz'), **gt)
    with open(os.path.join(out, 'quickstart.yaml'), 'w') as f:
        f.write(f'name: "example"\nmovie: {out}/movie.tif\nout: {out}/simcode_out\n'
                f'camera:\n  gain_e_per_adu: {GAIN}\n  offset_adu: {OFFSET}\n'
                f'psf:                       # a short movie has fewer isolated emitters than the default PSF fit uses (8000)\n'
                f'  nsamples: 2000\n  heldout_nsamples: 2000\n'
                f'train:                     # the shipped model fits this simulated setup: no training. For your own\n'
                f'  model: m1809             # microscope leave this out, and the pipeline trains a model for its PSF.\n')
    print(f'{out}/movie.tif: {movie.shape[0]} frames of {movie.shape[1]} x {movie.shape[2]} px, {len(gt["x"])} emitter events;'
          f' config {out}/quickstart.yaml')


if __name__ == '__main__':
    main()
