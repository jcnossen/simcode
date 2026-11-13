#!/bin/bash
# in-situ PSF of the nanoruler movie: the psf3d recipe (out/psf3d/run_final.sh / run_final_ps.sh) with the parametrized copy
# nr_fit_insitu2.py; zero init (no bead fit of this setup). Stages: base, bl (line blur), then (needs the draft calib for k)
# blps3 (frozen + photoselection 3). $1 = suffix ('' = all samples, '_cb0' = checkerboard-parity-0 training for the held-out check)
cd /home/jc/simcode2; source ~/miniconda3/bin/activate simcode-py311
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${GPU:-2} NR_PX=0.065 NR_WL=0.68 NR_OFFSET=100.0 NR_RN=1.6
O=out/nanoruler/psf; T=out/nanoruler/tracks_nr80.npz; X=$1; SP=""; [ "$X" = "_cb0" ] && SP="--split cb0"
F="python -u analysis/nanoruler/nr_fit_insitu2.py --tracks $T --gain 0.45 --intensities 0 $SP"
[ -f $O/base${X}_psf.pt ] || $F --out $O/base$X > $O/base$X.log 2>&1
[ -f $O/bl${X}_psf.pt ] || $F --blur line --out $O/bl$X > $O/bl$X.log 2>&1
if [ -f out/nanoruler/calib_nr80_draft.json ]; then
  [ -f $O/blps3${X}_psf.pt ] || $F --init-fit $O/bl${X}_psf.pt --freeze-base 1 --photosel 3 --pol-calib $PWD/out/nanoruler/calib_nr80_draft.json --out $O/blps3$X > $O/blps3$X.log 2>&1
fi
echo "psf $X done $(date)"
