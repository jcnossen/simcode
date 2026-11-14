"""Gain / offset of the paper movie's camera from Calibration/bright_2 (2000 frames) and dark_1 (200 frames), as
photonpy.smlm.process_movie.create_calib_obj does (offset = per-pixel dark mean, gain = mean / var of bright - offset,
in e-/ADU), plus a version with the frame-mean fluctuation removed (lamp flicker inflates var) and a version from
frame differences (removes the fixed pattern and slow drifts). Output out/zimflux/inventory/camera_jan13.json."""
import json, tifffile, numpy as np
C = '/mnt/data/dmdsf/simcode2_out/zimflux/surfdrive/Zimflux/Calibration/'
dark = tifffile.imread(C + 'dark_1_MMStack_Default.ome.tif').astype(np.float64)
light = tifffile.imread(C + 'bright_2_MMStack_Default.ome.tif').astype(np.float64)
off = dark.mean(0); rn = dark.std(0)
sig = light - off
m, v = sig.mean(0), sig.var(0)
g_photonpy = m / v
fm = sig.mean((1, 2)); sig2 = sig * (fm.mean() / fm)[:, None, None]      # remove global frame-to-frame flicker
v2 = sig2.var(0); g_flat = m / (v2 - rn ** 2)
d = np.diff(sig2, axis=0); vd = d.var(0) / 2; g_diff = m / (vd - rn ** 2)
q = lambda a: np.percentile(a, [10, 50, 90]).round(4).tolist()
res = {'dark_frames': len(dark), 'bright_frames': len(light), 'shape': list(off.shape),
       'offset_adu_mean_median': [float(off.mean()), float(np.median(off))], 'offset_p10_50_90': q(off),
       'readnoise_adu_median': float(np.median(rn)), 'readnoise_e_median': float(np.median(rn) * np.median(g_diff)),
       'bright_signal_adu_p10_50_90': q(m), 'bright_frame_mean_rel_std': float(fm.std() / fm.mean()),
       'gain_e_per_adu_photonpy_mean': float(np.mean(g_photonpy)), 'gain_photonpy_p10_50_90': q(g_photonpy),
       'gain_flickercorr_p10_50_90': q(g_flat), 'gain_framediff_p10_50_90': q(g_diff),
       'gain_framediff_mean': float(np.mean(g_diff)), 'runfile_values': {'gain': 0.36, 'offset': 100}}
print(json.dumps(res, indent=1))
json.dump(res, open('/home/jc/simcode2/out/zimflux/inventory/camera_jan13.json', 'w'), indent=1)
