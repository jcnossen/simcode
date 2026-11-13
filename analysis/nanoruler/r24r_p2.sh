#!/bin/bash
# R24 reuse check, part C repeated for PASS 2 of the nr80 set (front s2 = p2det_nr80 + p2headL_nr80, calibration calib_p2L_nr80.json)
# on the nr40 / nr20 gate scenes (the scenes of the pass-1 check are reused; only the dumps and the analysis are new).
# Refuses to start before the nr80 evaluation chain has finished ("eval nr80 done" in out/nanoruler/eval/chain.log), because it
# needs the pass-2 nets, their calibration and the nr80 reference dumps out/nanoruler/eval/fe_nr80/*_s2.npz of chain step 3.
# Resumable: every existing dump / json is skipped. Every GPU call goes through r24_gpu.sh (one scene per call, ~4 GB).
# usage: cd ~/simcode2; setsid nohup bash analysis/nanoruler/r24r_p2.sh >> out/nanoruler/nr40/reuse/p2_chain.log 2>&1 < /dev/null &
# outputs: out/nanoruler/<name>/reuse/fe_<name>/*_s2.npz, calib_s2_refit_<name>.json, accept_s2_<scene>.json, accept_s2_<name>.txt,
#          pair031_s2_<name>.txt; nr80 reference: out/nanoruler/nr40/reuse/accept_ref_nr80_s2_<scene>.json
source /home/jc/simcode2/runs/psfmatch/env.sh
A=analysis/nanoruler; MD=$PWD/out/nanoruler/models
grep -q "eval nr80 done" out/nanoruler/eval/chain.log || { echo "$(date +%F_%T) r24r_p2: 'eval nr80 done' not in out/nanoruler/eval/chain.log yet; not starting"; exit 2; }
for f in $MD/p2det_nr80.pt $MD/p2headL_nr80.pt $MD/calib/calib_p2L_nr80.json; do [ -f $f ] || { echo "r24r_p2: $f missing"; exit 2; }; done
for sc in iso pair1 pair2 rs05qm; do for sd in 202 301 302 303; do
  [ -f out/nanoruler/eval/fe_nr80/${sc}__s${sd}_s2.npz ] || { echo "r24r_p2: nr80 reference dump ${sc}__s${sd}_s2.npz missing"; exit 2; }
done; done
for N in ${1:-nr40 nr20}; do
  bash $A/r24r_gates.sh $N "s2=$MD/p2det_nr80.pt:$MD/p2headL_nr80.pt" pair0.31 || exit 1
  bash $A/r24r_accept.sh $N s2 || exit 1
done
echo "$(date +%F_%T) r24r_p2 done"
