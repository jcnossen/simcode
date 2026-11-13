"""Classical baselines (2D SMLM on 6-frame sums, SIMFLUX) on the 80 nm nanoruler movie sim4_1 (reordered cache) with the
movie's in-situ PSF: a wrapper around analysis/baselines/paper_pipeline.py. paper_pipeline is imported as a module and run
through its own main(); nothing on disk is modified, every change below is an in-memory override.

Overrides (what / why):
  1. pp.DATASETS['nr80'] added: tif = out/nanoruler/raw/sim4_1_reord.tif, cam = ABSOLUTE path of
     out/nanoruler/camera_maps_nr80.npz (os.path.join(REPO, 'models', 'insitu', <absolute>) returns the absolute path; the npz
     has 'gain' = 0.45 e/ADU and the per-pixel 'offset_adu' map, the two keys main() reads), nframes = 30000.
     Why: the script only knows the three DMD-setup movies.
  2. pp.PAPER (module-level settings dict) updated:
       gain 0.56 -> 0.45 e/ADU, offset 106.34 -> 100 ADU (only used with --camera paper; the default --camera maps takes
         the gain and the offset map of the npz), pixelsize 108.33 -> 65.0 nm          [other camera / magnification]
       roisize 8 -> --roi (default 10; the PSF is sigma 1.84 px instead of 1.3 px; 10 = the ROI of the original processing
         of this movie; must be even: SpotDetector asserts an even template size)
       detection_threshold 3.5 -> --thr (default 3.5 = unchanged; it is setup dependent, see the report of the run)
       fix_depth: --fix_depth, default 0.9 = the script's value, NOT changed (the estimated depths on this movie are
         0.89-0.97, the calib json has 0.88 / 0.92); --fix_depth 0.85 = the fixed depth of the original processing of this
         movie (delivered as the alternative table); 0 = None = keep the estimated per-pattern depths
     Unchanged: pattern_frames [[0,1,2],[3,4,5]], smlm_max_chisq 4, mod_error_threshold 0.1, smlm_spots_per_bin 20000,
     sf_iterations 50, distFromSMLM 0.5, zrange.
  3. pp.PSFLocalizer replaced by the subclass NRLocalizer:
       a. detection template: Gaussian sigma --det_sigma (default 1.84 px) instead of the hard-coded sigma 1.3 px, built the
          same way (fastpsf CreatePSF_XYIBg(roisize, [s, s]).ExpectedValue at the ROI centre). Why: PSF width of this setup.
       b. psf_calib keyword (used only as part of the ROI-cache key when det_template = 'paper') [1.3, 1.3] -> [s, s].
       c. detect_spots(moving_window=...) forced to --moving_window (default 0 = block mode: disjoint windows of 6 frames,
          window start = 6 x cycle, written as the frame number). Why: the cache movie is reordered within each 6-frame
          cycle, so only windows starting at a multiple of 6 are six consecutive acquisition frames; block mode gives exactly
          these. The pattern estimator is called by the script with moving_window=True; with all window starts multiples of 6
          its index (pattern - frame) % 6 = pattern, i.e. identical to its block mode.
  4. pp.link_movie replaced: main() would create out/paper2/nr80/movie/; the symlink to the movie is placed in <out>/movie/
     instead. Why: outputs only below out/nanoruler/baselines/.
  5. pp.crlb_summary wrapped with px = 65 nm (its default 108.33 nm is only used for the median_crlb_nm of the info json).
  6. pp.limits (fit-parameter limits (x, y) in (1.5, P - 2.5), I > 10, bg > 1): the x / y limits already follow the ROI
     size P (ROI 10: (1.5, 7.5); ROI 12: (1.5, 9.5)) and are not changed. The bg lower limit can be replaced by --bg_min;
     the default 1.0 is the script's value (no change in effect). Why the option: the limits are shared by the summed fit
     (bg about 10 photons/px) and the SIMFLUX fit, whose bg is PER FRAME (about 1.6 photons/px/frame here, close to the
     limit). Checked: --bg_min 0.2 gives the same SIMFLUX table (79749 rows, minimum fitted bg 1.035), so the limit rejects
     nothing on this movie.
  7. process title 'nr-base'.
  8. batch sizes (GPU memory only, results unchanged): pp.TorchSMLMFitter subclassed with batch = --batch (script default
     40000 ROIs) and pp.fit_simflux_torch wrapped with batch = --batch (script default 20000). Why: the spline-PSF lookup
     needs about 160 kB per ROI for a 10-px ROI and the GPUs are shared.
Not changed: the torch LM-MLE fitters, the per-frame EstimateIBg (its Gaussian sigma argument 1.3 is not used: the PSF images
are passed in), chi-square filter (chi^2 <= 4 x ROI^2), the verbatim pattern estimate (pitch search 150-500 nm, depth fixed
after the estimate), modulation-error filter < 0.1, SIMFLUX 50 iterations, |SIMFLUX - SMLM| < 0.5 px, --sf_init fixed.
smlmtorch: the copy imported in this environment (~/simcode/smlmtorch) counts frames with len(tif.pages); the wrapper checks
that reader count = tifffile count = 30000 before running.

usage:
  CUDA_VISIBLE_DEVICES=<gpu> python analysis/nanoruler/bl_run.py [--roi 10] [--thr 3.5] [--fix_depth 0.9] [--det_sigma 1.84]
        [--bg_min 1.0] [--batch 8000] [--moving_window 0] [--maxframes N] [--tag run] [--psf out/nanoruler/calib_nr80.json] [--psf_z um]
outputs: out/nanoruler/baselines/<tag>/ = the --out directory of paper_pipeline (smlm/smlm.npy + .hdf5, mp_smlm.pickle,
  mp_sf.pickle, info.json; sf/sf.hdf5, info.json; run.json; work/<run>/ ROI cache and result figures; movie/ symlink) and
  wrapper.json (the overrides and the frame-count check)."""
import os, sys, json, argparse, time
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
REPO = '/home/jc/simcode2'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis', 'baselines'))
from simcode2.util.proctitle import set_title
set_title('nr-base')
import numpy as np, tifffile

MOVIE = f'{REPO}/out/nanoruler/raw/sim4_1_reord.tif'
CAM = f'{REPO}/out/nanoruler/camera_maps_nr80.npz'
CALIB = f'{REPO}/out/nanoruler/calib_nr80.json'
OUTROOT = f'{REPO}/out/nanoruler/baselines'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roi', type=int, default=10)
    ap.add_argument('--thr', type=float, default=3.5)
    ap.add_argument('--fix_depth', type=float, default=0.9)
    ap.add_argument('--det_sigma', type=float, default=1.84)
    ap.add_argument('--bg_min', type=float, default=1.0)
    ap.add_argument('--moving_window', type=int, default=0)
    ap.add_argument('--maxframes', type=int, default=None)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--psf', default=CALIB)
    ap.add_argument('--psf_z', type=float, default=None)
    ap.add_argument('--steps', default='smlm,sf')
    ap.add_argument('--batch', type=int, default=8000)
    a = ap.parse_args()
    assert a.roi % 2 == 0, 'SpotDetector needs an even ROI / template size'
    out = os.path.join(OUTROOT, a.tag)
    os.makedirs(out, exist_ok=True)

    # frame count: tifffile vs the smlmtorch reader
    from smlmtorch.util.multipart_tiff import tiff_get_movie_size, tiff_get_filenames
    with tifffile.TiffFile(MOVIE) as t: n_tf = len(t.pages)
    shape, n_rd = tiff_get_movie_size(MOVIE)
    files = tiff_get_filenames(MOVIE)
    print(f'frames: tifffile {n_tf}, smlmtorch reader {n_rd}, shape {shape}, parts {len(files)}', flush=True)
    assert n_tf == n_rd == 30000 and len(files) == 1 and tuple(shape) == (400, 400)

    import paper_pipeline as pp
    from fastpsf import GaussianPSFMethods

    pp.DATASETS['nr80'] = dict(tif=MOVIE, cam=CAM, nframes=n_tf)                                           # 1
    pp.PAPER.update(gain=0.45, offset=100.0, pixelsize=65.0, roisize=a.roi, detection_threshold=a.thr,     # 2
                    fix_depth=(a.fix_depth if a.fix_depth > 0 else None))

    class NRLocalizer(pp.PSFLocalizer):                                                                    # 3
        def __init__(self, tif_path, psf, iters=50, det_template='paper', **kw):
            if det_template == 'paper': kw['psf_calib'] = [a.det_sigma, a.det_sigma]                       # 3b
            super().__init__(tif_path, psf, iters=iters, det_template=det_template, **kw)

        def create_psf_model(self):
            super().create_psf_model()
            if self._det_template == 'paper':                                                              # 3a
                g = GaussianPSFMethods(self.fastpsf_ctx).CreatePSF_XYIBg(self.roisize, [a.det_sigma, a.det_sigma], cuda=True)
                self.smlm_fitter.detection_template = g.ExpectedValue([[self.roisize / 2, self.roisize / 2, 1, 0]])
                self.detection_template = self.smlm_fitter.detection_template

        def detect_spots(self, *args, moving_window=False, **kw):                                          # 3c
            return super().detect_spots(*args, moving_window=bool(a.moving_window), **kw)
    pp.PSFLocalizer = NRLocalizer

    _link = pp.link_movie
    pp.link_movie = lambda tif, movie_dir: _link(tif, os.path.join(out, 'movie'))                          # 4
    _cs = pp.crlb_summary
    pp.crlb_summary = lambda ds, px=65.0: _cs(ds, px)                                                      # 5
    _lim = pp.limits
    def limits(P):                                                                                         # 6
        lo, hi = _lim(P); lo = lo.copy(); lo[3] = a.bg_min
        return lo, hi
    pp.limits = limits

    class NRFitter(pp.TorchSMLMFitter):                                                                    # 8
        def __init__(self, psf, roisize, ctx, iters=50, batch=40000, dev='cuda'):
            super().__init__(psf, roisize, ctx, iters, a.batch, dev)
    pp.TorchSMLMFitter = NRFitter
    _sf = pp.fit_simflux_torch
    pp.fit_simflux_torch = lambda sfloc, psf, mp, ds, iterations=50, distFromSMLM=0.5, batch=20000, init='fixed': _sf(
        sfloc, psf, mp, ds, iterations, distFromSMLM, a.batch, init)

    over = dict(args=vars(a), frames=dict(tifffile=n_tf, reader=n_rd), PAPER={k: (v.tolist() if hasattr(v, 'tolist') else v)
                                                                             for k, v in pp.PAPER.items()},
                limits_lo=limits(a.roi)[0].tolist(), limits_hi=limits(a.roi)[1].tolist(), movie=MOVIE, cam=CAM,
                smlmtorch=os.path.dirname(sys.modules['smlmtorch'].__file__))
    json.dump(over, open(os.path.join(out, 'wrapper.json'), 'w'), indent=1)
    argv = ['paper_pipeline.py', 'nr80', '--psf', a.psf, '--out', out, '--sf_init', 'fixed', '--camera', 'maps',
            '--det_template', 'paper', '--steps', a.steps]
    if a.maxframes: argv += ['--maxframes', str(a.maxframes)]
    if a.psf_z is not None: argv += ['--psf_z', str(a.psf_z)]
    print('paper_pipeline argv:', ' '.join(argv[1:]), flush=True)
    sys.argv = argv
    t0 = time.time()
    pp.main()
    over['wall_s'] = time.time() - t0
    json.dump(over, open(os.path.join(out, 'wrapper.json'), 'w'), indent=1)
    print('wrapper done, wall %.1f s' % over['wall_s'], flush=True)


if __name__ == '__main__':
    main()
