"""The preprint's SIMCODE (v1) on a papersim movie: analysis/baselines/run_simcode.py (the paper's pipeline: network
sf_conv_g1.3_tio2_L48 checkpoint_3, detector p >= 0.7, crlb_filter 0.2 px, its pattern estimation with depth fixed 0.9, the
mod-error < 0.1 selection, NDI fit, |NDI - net| < 0.2 px, MVU merge 'sfhd-ndi+smlm'), imported unchanged, run with its paper
defaults (--model L48 --ckpt 3 --pattern est --fix_sign 0, i.e. including the drift-sign behaviour of the published code).
Only the camera differs: run_simcode.py's paper camera is the real Zyla's global gain 0.56 / offset 106.34; this movie's
camera is gain 1 / offset 100 (calib_sim.json), so the module constants GAIN / OFFSET are set to the movie's values in this
process (otherwise every photon count would be off by 0.56x and the offset by 6 ADU).
Note: the v1 network was trained on a Gaussian sigma 1.3 px PSF; these movies use the measured vectorial PSF (the same
mismatch v1 has on the real data). gen_density.py --psf gauss1.3 makes the preprint's own condition (control).
The runner expects <dir>/sim_movie.tif (+ sim_gt.npz 'patt' for --pattern gt); a work dir with a symlink is made.
usage: CUDA_VISIBLE_DEVICES=0 python analysis/papersim/run_simcode_v1.py <movie dir> [--out <movie dir>/full/v1/locs.npz]
          [--maxframes N] [--pattern est|gt] [--fix_sign 0]
output: the runner's npz (net rows, ndi_idx, ndi_*, mrg_* = 'sfhd-ndi+smlm'), + v1_info.json"""
import os, sys, json, argparse, importlib.util, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('movie_dir'); ap.add_argument('--out', default=None); ap.add_argument('--maxframes', type=int, default=0)
    ap.add_argument('--pattern', default='est'); ap.add_argument('--fix_sign', type=int, default=0)
    a = ap.parse_args()
    from simcode.util.proctitle import set_title; set_title('psim-v1')
    md = os.path.abspath(a.movie_dir)
    out = os.path.abspath(a.out or os.path.join(md, 'full', 'v1', 'locs.npz'))
    wd = os.path.dirname(out); os.makedirs(wd, exist_ok=True)
    src = os.path.join(md, 'movie.tif')
    if a.maxframes:            # the runner processes the whole tif: write the cut
        import tifffile
        src = os.path.join(wd, f'movie_{a.maxframes}.tif')
        if not os.path.exists(src):
            with tifffile.TiffFile(os.path.join(md, 'movie.tif')) as t:
                tifffile.imwrite(src, np.stack([t.pages[i].asarray() for i in range(a.maxframes)]))
    lk = os.path.join(wd, 'sim_movie.tif')
    if os.path.lexists(lk): os.remove(lk)
    os.symlink(src, lk)
    g = json.load(open(os.path.join(md, 'gen.json')))
    if a.pattern == 'gt':     # per-frame true pattern (kx, ky, phi, m, rel) in this movie's cos convention
        gt = np.load(os.path.join(md, 'gt.npz')); nf = a.maxframes or g['nframes']; s = np.arange(nf) % 6
        k = gt['k'][s // 3]
        np.savez(os.path.join(wd, 'sim_gt.npz'), patt=np.c_[k, gt['phi'][s], gt['m'][s], gt['rel'][s]])
    _s = importlib.util.spec_from_file_location('run_simcode_v1base', os.path.join(REPO, 'analysis', 'baselines', 'run_simcode.py'))
    RS = importlib.util.module_from_spec(_s); _s.loader.exec_module(RS)
    cam = np.load(os.path.join(md, 'camera_sim.npz'))
    RS.GAIN = float(cam['gain']); RS.OFFSET = float(cam['offset_adu'].mean())
    t0 = time.time()
    sys.argv = ['run_simcode.py', wd, out, '--model', 'L48', '--ckpt', '3', '--pattern', a.pattern, '--cam', 'paper', '--fix_sign', str(a.fix_sign)]
    RS.main()
    json.dump(dict(movie=md, maxframes=a.maxframes, gain=RS.GAIN, offset=RS.OFFSET, model='sf_conv_g1.3_tio2_L48/checkpoint_3', pattern=a.pattern,
                   fix_sign=a.fix_sign, settings='analysis/baselines/run_simcode.py paper defaults (p >= 0.7, crlb 0.2 px, depth 0.9, mod err 0.1, |ndi-net| < 0.2 px, MVU merge)',
                   s=round(time.time() - t0, 1)), open(os.path.join(wd, 'v1_info.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
