#!/bin/bash
# R24 reuse check, part C (analysis driver): calibration refit, acceptance numbers per scene (this movie and the nr80 reference),
# table, 0.31 px pair analysis. One short GPU call per step through r24_gpu.sh (the explicit pattern solve runs on the GPU);
# resumable (existing jsons are skipped). Needs the dumps of r24r_gates.sh.
# usage: bash analysis/nanoruler/r24r_accept.sh <nr40|nr20> [front s1|s2, default s1]
source /home/jc/simcode2/runs/psfmatch/env.sh
N=$1; export R24R_FRONT=${2:-s1}; F=$R24R_FRONT; A=analysis/nanoruler; O=out/nanoruler/$N/reuse; LG=$O/logs; mkdir -p $LG
say() { echo "$(date +%F_%T) $*"; }
run() { bash $A/r24_gpu.sh 3 python -u $A/r24r_accept.py "$@" >> $LG/accept_$F.log 2>&1 || { say "FAILED r24r_accept.py $* (see $LG/accept_$F.log)"; exit 1; }; }
[ -f $O/calib_${F}_refit_$N.json ] || { say "fit $N $F"; run fit $N; }
for sc in iso rs05qm pair1 pair2; do
  [ -f out/nanoruler/nr40/reuse/accept_ref_nr80_${F}_$sc.json ] || { say "scene nr80 $sc"; run scene nr80 $sc; }
  [ -f $O/accept_${F}_$sc.json ] || { say "scene $N $sc"; run scene $N $sc; }
done
CUDA_VISIBLE_DEVICES= R24R_DEV=cpu python $A/r24r_accept.py table $N > $LG/table_$F.log 2>&1 || { say "FAILED table"; exit 1; }
if [ -f $O/fe_$N/pair0.31__s301_$F.npz ] && [ ! -f $O/pair031_${F}_$N.txt ]; then
  say "pair0.31 $N"; bash $A/r24_gpu.sh 3 python -u $A/r24r_pair031.py $N >> $LG/pair031_$F.log 2>&1 || { say "FAILED pair031 (see $LG/pair031_$F.log)"; exit 1; }
fi
say "accept $N $F done"
