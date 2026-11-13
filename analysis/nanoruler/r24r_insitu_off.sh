#!/bin/bash
# R24 reuse check, part D: base in-situ PSF fits with offset 99 / 100 (read noise 1.6 as in the calibration fits, and 1.48 = the
# camera-map median) on the existing tracks of nr40, nr20 and nr80 (control). One short GPU call per fit through r24_gpu.sh.
# Outputs: out/nanoruler/<name>/reuse/insitu_off<offset>[_rn<rn>]/base_* (nr80: under out/nanoruler/nr40/reuse/insitu_nr80_off...).
source /home/jc/simcode2/runs/psfmatch/env.sh
export NR_PX=0.065 NR_WL=0.68
for N in ${1:-nr40 nr20 nr80}; do for OFF in 100 99; do for RN in 1.6 1.48; do
  D=out/nanoruler/$N/reuse/insitu_off${OFF}_rn$RN; [ $N = nr80 ] && D=out/nanoruler/nr40/reuse/insitu_nr80_off${OFF}_rn$RN
  [ -f $D/base_summary.json ] && continue
  mkdir -p $D
  NR_OFFSET=$OFF NR_RN=$RN bash analysis/nanoruler/r24_gpu.sh 3 python -u analysis/nanoruler/r24r_insitu_off.py $N $PWD/$D/base > $D/base.log 2>&1 || echo "FAILED $D"
done; done; done
echo "insitu_off done $(date +%F_%T)"
