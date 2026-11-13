"""Classical baselines (SMLM, SIMFLUX 'sf') on a sim movie, with the paper's pipeline code imported read-only from ~/simcode
(smlmtorch.simflux.localizer.SFLocalizer), settings as in ~/smlmtorch/scripts/run_pipeline.py process() called by
process_paper_datasets.ipynb (and the real results' config.pickle): 2D Gaussian PSF sigma 1.3 px (fixed), ROI 8 px, detection
threshold 3.5 on the 6-frame summed image, moving window (stride 1, every window start), fit_smlm(max_crlb_xy=None,
max_chisq=4) -> 'smlm'; mod_error < 0.1 (on the per-frame ibg fits) -> fit_simflux(iterations=50, lambda_=500,
normalizeWeights=True), |sf - smlm| < 0.5 px -> 'sf'.
Deviations (documented): (1) spots_per_frame cap 100 -> CAP (default 2000; the paper's cap is FOV-specific; the number the
paper's cap would have dropped is reported); (2) camera = the sim's per-pixel offset map + gain 0.56 (correct calibration);
(3) SF pattern = the TRUE per-frame pattern (gt_mp of analysis/baselines/run_simcode.py, true depths), converted at the measured
smlm-vs-GT coordinate offset (pass 1 of this script), i.e. the known pattern (our pass 1 uses its own estimate, which scores
identically to the true pattern on these movies).
usage: CUDA_VISIBLE_DEVICES=1 python analysis/baselines/run_classic.py <sim_dir> <work_dir> <out.npz> [maxframes]
"""
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
from simcode import config

CAP = int(os.environ.get('CAP', 2000))
_orig = detection.SpotDetector
class _Det(_orig):
    def __init__(self, *a, spots_per_frame=100, **k):
        super().__init__(*a, spots_per_frame=CAP, **k)
detection.SpotDetector = _Det
from smlmtorch.simflux.localizer import SFLocalizer

sd, wd, out = sys.argv[1], sys.argv[2], sys.argv[3]
maxframes = int(sys.argv[4]) if len(sys.argv) > 4 else None
os.makedirs(wd, exist_ok=True)
tif = f'{wd}/sim_movie.tif'
if not os.path.exists(tif): os.symlink(f'{sd}/sim_movie.tif', tif)
cam = np.load(config.model('insitu/camera_maps_0.5nM.npz'))
PF = RS.PF; PX = RS.PX
T = {}
torch.cuda.synchronize(); t0 = time.time()
sfl = SFLocalizer(tif, pattern_frames=PF, pixelsize=PX, detection_threshold=3.5, psf_calib=[1.3, 1.3], roisize=8,
                  device='cuda:0', zrange=[-0.3, 0.3], gain=float(cam['gain']), offset=cam['offset_adu'].astype(np.float32),
                  maxframes=maxframes, psf_sigma_binsize=None, result_dir='results')
sfl.detect_spots(ignore_cache=True, moving_window=True)
torch.cuda.synchronize(); T['detect_s'] = time.time() - t0
info, _ = detection.load_rois(sfl.rois_path) if sfl.numrois < 3e6 else (None, None)
if info is not None:
    # how many the paper's 100-per-window cap would have dropped
    _, cnt = np.unique(info['id'], return_counts=True)
    T['rois'] = int(len(info)); T['rois_per_window_med'] = float(np.median(cnt)); T['rois_per_window_max'] = int(cnt.max())
    T['frac_dropped_by_cap100'] = float(np.maximum(cnt - 100, 0).sum() / len(info))
    del _
t1 = time.time()
smlm = sfl.fit_smlm(max_crlb_xy=None, ignore_cache=True, max_chisq=4)
torch.cuda.synchronize(); T['smlm_fit_s'] = time.time() - t1; T['n_smlm'] = len(smlm)
# coordinate offset smlm vs GT (nearest GT on-emitter within 1.5 px, windows starting at 0 mod 6; 1000-20000 photons)
G = dict(np.load(f'{sd}/sim_gt.npz'))
fr = smlm.frame.astype(int); x, y = smlm.pos[:, 0], smlm.pos[:, 1]
dx, dy = [], []
for w in np.unique(fr[fr % 60 == 0])[:400]:
    a = np.nonzero((G['t1'] > w) & (G['t0'] < w + 6))[0]
    i = np.nonzero(fr == w)[0]
    if len(a) == 0 or len(i) == 0: continue
    D = np.hypot(x[i, None] - G['x'][a][None], y[i, None] - G['y'][a][None])
    j = D.argmin(1); ok = D[np.arange(len(i)), j] < 1.5
    dx.append(G['x'][a][j][ok] - x[i][ok]); dy.append(G['y'][a][j][ok] - y[i][ok])
off = [float(np.median(np.concatenate(dx))), float(np.median(np.concatenate(dy)))]
T['coord_off_gt_minus_smlm_px'] = off
print('coord offset GT - smlm (px):', off, flush=True)
mp = RS.gt_mp(sd, int(smlm.numFrames), off)
t1 = time.time()
me = mp.mod_error(smlm)
sel = me < RS.MOD_ERR
T['n_moderr'] = int(sel.sum())
sf = sfl.fit_simflux(mp, smlm[sel], iterations=50, lambda_=500, normalizeWeights=True, ignore_cache=True)
torch.cuda.synchronize(); T['sf_fit_s'] = time.time() - t1; T['n_sf'] = len(sf)
T['total_smlm_s'] = T['detect_s'] + T['smlm_fit_s']; T['total_sf_s'] = T['total_smlm_s'] + T['sf_fit_s']
print(json.dumps(T), flush=True)
# sf rows -> smlm rows (fit_simflux keeps roi_id of the input dataset rows)
sub = np.nonzero(sel)[0]
# fit_simflux sets sf_ds.roi_id = index into the INPUT dataset (smlm[sel]); map to smlm rows
sf_rows = sub[sf.roi_id.astype(int)]
assert np.all(smlm.frame[sf_rows] == sf.frame)
np.savez_compressed(out, frame=fr, x=x, y=y, sx=smlm.crlb.pos[:, 0], sy=smlm.crlb.pos[:, 1], photons=smlm.photons,
                    bg=smlm.background, chisq=smlm.chisq, moderr=me, ibg=smlm.ibg[:, :, 0],
                    sf_idx=sf_rows, sf_x=sf.pos[:, 0], sf_y=sf.pos[:, 1], sf_sx=sf.crlb.pos[:, 0], sf_sy=sf.crlb.pos[:, 1],
                    sf_photons=sf.photons, timing=json.dumps(T), coord_off=np.array(off))
print('saved', out, flush=True)
