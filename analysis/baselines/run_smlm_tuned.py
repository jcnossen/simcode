"""'SMLM tuned' baseline (critic 35 #6): the paper's classical SMLM pipeline (analysis/baselines/run_classic.py, smlmtorch SFLocalizer, 6-frame
moving-window sums, fixed-sigma 1.3 px 2D Gaussian MLE) with dense-data settings instead of the 0.5 nM paper settings:
  ROI <roisize> px (paper 8) and local-max detection spacing <maxfilt> px (paper = ROI = 8; ~100 candidates per window at
  any density), detection threshold 3.5 (paper), all fits kept (max_chisq=None); the chi^2 threshold is applied in
  analysis/hd_compare/score.py (kind 'smlmt:<chi2 per px>') so that one fit serves several thresholds.
usage: [DT=<det. threshold, 3.5>] [BGF=<bg disc radius px, default ROI>] CUDA_VISIBLE_DEVICES=1 python analysis/baselines/run_smlm_tuned.py <sim_dir> <work_dir> <out.npz> <roisize> <maxfilt> [maxframes]
Output npz as run_classic.py (smlm part only): frame, x, y, sx, sy, photons, bg, chisq (total over the ROI), chisq_px, timing."""
import os, sys, time, json
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
import numpy as np, torch
import smlmtorch.util.progbar
smlmtorch.util.progbar.USE_AUTO_TQDM = False
import matplotlib
matplotlib.use('Agg')
import smlmtorch.smlm.detection as detection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_simcode as RS      # analysis/baselines/run_simcode.py
from simcode2 import config

CAP = int(os.environ.get('CAP', 2000))
_orig = detection.SpotDetector
class _Det(_orig):
    def __init__(self, *a, spots_per_frame=100, **k):
        super().__init__(*a, spots_per_frame=CAP, **k)
detection.SpotDetector = _Det
from smlmtorch.simflux.localizer import SFLocalizer

sd, wd, out, roisize, maxfilt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
maxframes = int(sys.argv[6]) if len(sys.argv) > 6 and int(sys.argv[6]) > 0 else None
DT = float(os.environ.get('DT', 3.5)); BGF = int(os.environ['BGF']) if os.environ.get('BGF') else None
os.makedirs(wd, exist_ok=True)
tif = f'{wd}/sim_movie.tif'
if not os.path.exists(tif): os.symlink(f'{sd}/sim_movie.tif', tif)
cam = np.load(config.model('insitu/camera_maps_0.5nM.npz'))
T = dict(roisize=roisize, maxfilt=maxfilt, det_threshold=DT, bg_filter=BGF)
torch.cuda.synchronize(); t0 = time.time()
sfl = SFLocalizer(tif, pattern_frames=RS.PF, pixelsize=RS.PX, detection_threshold=DT, psf_calib=[1.3, 1.3], roisize=roisize,
                  device='cuda:0', zrange=[-0.3, 0.3], gain=float(cam['gain']), offset=cam['offset_adu'].astype(np.float32),
                  maxframes=maxframes, psf_sigma_binsize=None, result_dir='results')
sfl.detect_spots(ignore_cache=True, moving_window=True, max_filter_size=maxfilt, bg_filter_size=BGF)
torch.cuda.synchronize(); T['detect_s'] = time.time() - t0
info, _ = detection.load_rois(sfl.rois_path) if sfl.numrois < 6e6 else (None, None)
if info is not None:
    _, cnt = np.unique(info['id'], return_counts=True)
    T['rois'] = int(len(info)); T['rois_per_window_med'] = float(np.median(cnt)); T['rois_per_window_max'] = int(cnt.max())
    del _
t1 = time.time()
smlm = sfl.fit_smlm(max_crlb_xy=None, ignore_cache=True, max_chisq=None)
torch.cuda.synchronize(); T['smlm_fit_s'] = time.time() - t1; T['n_smlm'] = len(smlm); T['total_s'] = T['detect_s'] + T['smlm_fit_s']
print(json.dumps(T), flush=True)
fr = smlm.frame.astype(int)
np.savez_compressed(out, frame=fr, x=smlm.pos[:, 0], y=smlm.pos[:, 1], sx=smlm.crlb.pos[:, 0], sy=smlm.crlb.pos[:, 1],
                    photons=smlm.photons, bg=smlm.background, chisq=smlm.chisq, chisq_px=smlm.chisq / roisize ** 2, timing=json.dumps(T))
print('saved', out, flush=True)
