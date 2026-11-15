#!/bin/bash
# In-situ calibration of another movie of the SIMFLUX setup (same recipe and scripts as for sim4_1; STATUS.md 01:40):
# quick SMLM on 6-frame moving sums -> camera maps -> isolated tracks -> PSF (base, line blur, photoselection 3; all samples
# and checkerboard-parity-0 for the held-out check) -> intensities -> pattern (init: sim4_1's in-situ pattern) -> calib json.
# usage: GPU=<g> bash analysis/nanoruler/insitu_any.sh <name nr40|nr20> [offset ADU, default 100] [max frame for tracks, default 30000]
N=$1; OFF=${2:-100}; MAXF=${3:-30000}
cd /home/jc/simcode2; source ~/miniconda3/bin/activate simcode-py311
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPU:-0} NR_PX=0.065 NR_WL=0.68 NR_OFFSET=$OFF NR_RN=1.6
M=/mnt/data/dmdsf/simcode2_out/nanoruler/raw_$N/movie_reord.tif; O=$PWD/out/nanoruler/insitu_$N; mkdir -p $O
set -e
[ -f $O/quick_w6.hdf5 ] || python analysis/nanoruler/quick_smlm.py --movie $M --sum 6 --f1 $MAXF --offset $OFF --out $O/quick_w6.hdf5 > $O/quick.log 2>&1
python - <<PY
import h5py, numpy as np
L = h5py.File('$O/quick_w6.hdf5')['locs'][:]
g = (L['chi2'] < 3.6) & (L['snr'] > 8) & (L['sx'] > 1.4) & (L['sx'] < 2.4); G = L[g]
with h5py.File('$O/quick_w6_good.hdf5', 'w') as h: h.create_dataset('locs', data=G)
nf = int(L['frame'].max()) + 1
print('quick w6: good per window %.2f, photons p5/p50/p95 %s, bg/px/frame %.3f, sigma %.3f' % (len(G) / nf, np.percentile(G['photons'], [5, 50, 95]).round(0), np.median(G['bg']) / 6, np.median(G['sx'])))
PY
[ -f $O/camera_maps_$N.npz ] || python -m simcode.calibration.camera_maps --movie $M --nframes 6000 --gain 0.45 --offset $OFF --locs $O/quick_w6_good.hdf5 --out $O/camera_maps_$N.npz > $O/camera_maps.log 2>&1
[ -f $O/tracks.npz ] || python -m simcode.calibration.extract_tracks --movie $M --locs $O/quick_w6_good.hdf5 --maxframe $MAXF --roi 15 --isolation 10 --out $O/tracks.npz > $O/tracks.log 2>&1
tail -2 $O/tracks.log
for X in "" _cb0; do
  SP=""; [ "$X" = "_cb0" ] && SP="--split cb0"
  F="python -u analysis/nanoruler/nr_fit_insitu2.py --tracks $O/tracks.npz --gain 0.45 --intensities 0 $SP"
  [ -f $O/base${X}_psf.pt ] || $F --out $O/base$X > $O/base$X.log 2>&1
  [ -f $O/bl${X}_psf.pt ] || $F --blur line --out $O/bl$X > $O/bl$X.log 2>&1
  [ -f $O/blps3${X}_psf.pt ] || $F --init-fit $O/bl${X}_psf.pt --freeze-base 1 --photosel 3 --pol-calib $PWD/out/nanoruler/calib_nr80.json --out $O/blps3$X > $O/blps3$X.log 2>&1
done
for f in base bl blps3; do [ -f $O/held_$f.npz ] || python analysis/nanoruler/nr_psf_heldout.py --tracks $O/tracks.npz --fit $O/${f}_cb0_psf.pt --test cb1 --out $O/held_$f.npz > $O/held_$f.log 2>&1; done
python -m simcode.calibration.psf_compare $O/held_base.npz $O/held_bl.npz $O/held_blps3.npz > $O/held_compare.txt 2>&1
[ -f $O/bl_int_intensities.pt ] || python -u analysis/nanoruler/nr_fit_insitu2.py --tracks $O/tracks.npz --gain 0.45 --init-fit $O/bl_psf.pt --freeze-base 1 --iters 200 --intensities 1 --out $O/bl_int > $O/bl_int.log 2>&1
[ -f $O/pattern.json ] || python -m simcode.calibration.fit_pattern --intensities $O/bl_int_intensities.pt --oldmod $PWD/out/nanoruler/oldmod_$N.pickle --out $O/pattern > $O/pattern.log 2>&1
echo "insitu $N done $(date)"
