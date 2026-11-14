#!/bin/bash
# m20check (2026-11-12): 20nM_2 production with the m25 model (PSF-matched to 2.5nM_2; no 20 nM-matched model exists).
# = runs/psfmatch/prod.sh m25 (python -m simcode.run --model m25: its networks, _real calibrations and sigma rescale; the 12 nm
# selection file locs_p2_sig12.npz) with the 20 nM movie + its own in-situ calib and the output in
# out/paper2/20nM_2/v2_m25 -> symlink to /mnt/data/dmdsf/simcode2_out/paper2/20nM_2/v2_m25 (large; / must not fill up).
# Resumable (finished chunks are skipped by simcode.run).
# usage: cd ~/simcode2; PREF="1 2 0" setsid nohup bash analysis/psfmatch/prod_m20_m25.sh > runs/psfmatch/prod_20nM_2_m25.log 2>&1 < /dev/null &
S=m25; ds=20nM_2
source /home/jc/simcode2/runs/psfmatch/env.sh
MOV=/mnt/data/dmdsf/bn_dnapaint/dnapaint/20nM/2/cover1_500mW_5.5ms_cf660.tif
CAL=insitu/calib_20nM_m2_psf3d.json
BIG=/mnt/data/dmdsf/simcode2_out/paper2/$ds/v2_m25; OD=out/paper2/$ds/v2_m25
mkdir -p $BIG; [ -e $OD ] || ln -s $BIG $OD
[ "$(readlink -f $OD)" = "$BIG" ] || { echo "$OD is not the symlink to $BIG"; exit 1; }
[ $(df --output=avail -BG $BIG | tail -1 | tr -dc 0-9) -ge 150 ] || { echo "less than 150 GB free on $BIG"; exit 1; }
G=$(pickgpu ${NEED_GB:-8} "${PREF:-1 2 0}")
if [ ! -f $OD/locs_p2.npz ]; then
  echo "$(date) $ds GPU $G: python -m simcode.run --movie $MOV --calib $CAL --out $OD --model $S" | tee -a $OD/command.txt
  CUDA_VISIBLE_DEVICES=$G python -u -m simcode.run --movie $MOV --calib $CAL --out $OD --model $S > $OD/run.log 2>&1 || { echo "FAIL $ds"; exit 1; }
fi
echo "$(date) prod $ds ($S) done"
