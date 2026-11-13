"""Cycle-length check on a cached sample: per-frame mean level vs frame index modulo m (m = 2..12) after removing the slow
trend, the spectrum of the mean-level series, frame interval and dropped-frame check. Input: frames_info.npz of the sample."""
import argparse, json, numpy as np
from scipy.ndimage import median_filter
ap = argparse.ArgumentParser(); ap.add_argument('--info', required=True); ap.add_argument('--n', type=int, default=3000); ap.add_argument('--out', default='')
a = ap.parse_args()
z = np.load(a.info); m = z['mean'][:a.n]; el = z['elapsed_ms'][:a.n]; fi = z['frameindex'][:a.n]; N = len(m)
res = {'n': int(N), 'mean_level': float(m.mean()), 'median_of_frame_medians': float(np.median(z['median'][:a.n]))}
d = np.diff(el)
res['dt_ms'] = {'median': float(np.median(d)), 'mean': float((el[-1] - el[0]) / (N - 1)), 'min': float(d.min()), 'max': float(d.max()), 'n_gt_1.5x': int((d > 1.5 * np.median(d)).sum())}
res['frameindex_contiguous'] = bool((np.diff(fi) == 1).all())
print('level mean %.3f ADU, median of medians %.1f; dt %s; frameindex contiguous %s' % (res['mean_level'], res['median_of_frame_medians'], res['dt_ms'], res['frameindex_contiguous']))
r = m - median_filter(m, 241, mode='nearest')          # slow trend (blinking density, bleaching) removed
res['mod'] = {}
for mod in range(2, 13):
    cm = np.array([r[s::mod].mean() for s in range(mod)]); n_c = N // mod
    within = np.mean([r[s::mod].var() for s in range(mod)])
    F = cm.var(ddof=1) * n_c / within               # ~1 if no periodic structure with this period
    res['mod'][mod] = {'class_means_adu': [float(v) for v in cm - cm.mean()], 'F': float(F)}
    print('mod %2d  F %8.1f  class means (ADU, minus mean): %s' % (mod, F, np.array2string(cm - cm.mean(), precision=3, suppress_small=True)))
sp = np.abs(np.fft.rfft(r * np.hanning(N))) ** 2; fr = np.fft.rfftfreq(N)
med = np.median(sp[10:]); idx = np.argsort(sp[10:])[::-1][:8] + 10
res['spectrum_peaks'] = [{'period_frames': float(1 / fr[i]), 'power_over_median': float(sp[i] / med)} for i in idx]
print('spectrum peaks (period in frames, power / median):', ', '.join('%.3f (%.0f)' % (1 / fr[i], sp[i] / med) for i in idx))
if a.out: json.dump(res, open(a.out, 'w'), indent=1)
