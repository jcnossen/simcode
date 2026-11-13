"""Assemble the nanoruler calibration json (format of models/insitu/calib_*.json) from the in-situ fits.
usage: python make_calib.py <fit prefix (out/nanoruler/psf/bl)> <out json> [avg]   (avg: mean of the per-angle dipole tensors)
Measured: PSF (Zernike 5-21, focus, blur, photoselection) and pattern (fit_pattern.py) from the movie; per-pixel offset
deviation and read-noise maps (camera_maps.py). Assumed: pixel size 65 nm, NA 1.49, emission 680 nm, n 1.518 / 1.33,
TIRF depth 82 nm, gain 0.45 e/ADU and offset 100 ADU (the SIMFLUX paper's processing values; not measurable from this movie)."""
import sys, json, shutil, os, torch, numpy as np
pre, outp = sys.argv[1:3]; AVG = len(sys.argv) > 3 and sys.argv[3] == 'avg'
# optional (other movies of the setup): argv[4] = pattern json, argv[5] = camera-map npz, argv[6] = movie label
PAT = sys.argv[4] if len(sys.argv) > 4 else '/home/jc/simcode2/out/nanoruler/pattern_nr80.json'
CAM = sys.argv[5] if len(sys.argv) > 5 else '/home/jc/simcode2/out/nanoruler/camera_maps_nr80.npz'
LAB = sys.argv[6] if len(sys.argv) > 6 else 'sim4_1 (80 nm)'
S = torch.load(pre + '_psf.pt', weights_only=False); J = json.load(open(pre + '_psf.json')); ex = S.get('extras', {})
P = json.load(open(PAT))
c = dict(source='in-situ calibration of the SIMFLUX-paper nanoruler movie ' + LAB + ' (reordered cache, cycle order [0,2,4,1,3,5]); fit ' + pre,
         pixelsize_um=0.065, wavelength_um=0.68, NA=float(S['NA']), n_imm=1.518, n_med=1.33, apod=[float(v) for v in S['apod'].tolist()],
         zernike_noll=list(range(5, 22)), zernike_waves=[float(v) for v in S['zcoef'].tolist()], focus_um=float(S['focus']),
         camera=dict(gain_e_per_adu=0.45, offset_adu=100.0, readnoise_e=1.48, maps=os.path.basename(CAM),
                     note='Hamamatsu ORCA-Flash4.0 V2; gain 0.45 e/ADU and offset 100 ADU ASSUMED (paper processing values); maps from the movie (frames 0-6000): offset deviation, read noise (median 1.48 e)'),
         pattern=dict(k_rad_per_px=P['k'], pitch_nm=[float(2 * np.pi / np.hypot(*k) * 65.0) for k in P['k']], depth=[0.88] * 3 + [0.92] * 3,
                      phase=P['phase'], relint=P['relint'], depth_regression_diluted=P['depth'],
                      steps='REORDERED frame % 6; angle = step // 3; I = A relint (1 + depth cos(k.r - phase)); depth = paper estimate (0.82-0.93), the regression depth is diluted'),
         tirf_depth_um=0.082, emitter_z_median_um=0.01)
if 'blur_px2' in ex and len(ex['blur_px2']) == 1: c['psf_blur_px2'] = ex['blur_px2'][0]
if 'dipole_M' in ex:
    M = ex['dipole_M']
    if AVG: c['psf_dipole_M'] = [[(M[0][i][j] + M[1][i][j]) / 2 for j in range(3)] for i in range(3)]
    else: c['psf_dipole_M_angle'] = M
c['psf3d'] = dict(fit=pre, gamma=ex.get('gamma'), zeta=ex.get('zeta'), resid_var=J.get('resid_var'), avg=AVG)
json.dump(c, open(outp, 'w'), indent=1)
d = os.path.join(os.path.dirname(os.path.abspath(outp)), os.path.basename(CAM))
if not os.path.exists(d): shutil.copy(CAM, d)
print('wrote', outp)
