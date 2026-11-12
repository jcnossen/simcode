#!/bin/bash
# Dense simulated movies with K-summed noise conditions (simmovie4pi.py --ksum): detector + head + in-situ pattern / phase field + solve.
# Usage: DET=out/4pi/models/det4pi_k8/final.pt CUDA_VISIBLE_DEVICES=1 bash analysis/fourpi/gate_movies_k.sh v11 > out/4pi/sim/gate_movies_k_v11.log 2>&1
M=${1:-v11}; cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH; D=${DET:-out/4pi/models/det4pi/final.pt}; DT=${DTAG:-det}
LIST=${TAGS:-ks1_d011:1 ks4_d045:4 ks8_d09:8}; for tk in $LIST; do t=${tk%%:*}; K=${tk##*:}
  echo "== movie $t (K = $K conditions; head $M, detector $D)"
  python analysis/fourpi/simfront4pi.py --tag $t --model v3L --det $D --out _$DT --save_img 1 2>&1 | tail -1
  for m in "oracle oracle oracle${M}_phiO" "iter field iter${M}"; do set -- $m
    python analysis/fourpi/simlin4pi.py --tag $t --front v3L_$DT --head $M --ksum $K --pattern $1 --phi $2 --chi2_max ${CHI2:-2.5} 2>&1 | grep -v Warn | tail -1 | cut -c1-190
    echo "-- $3"; python analysis/fourpi/simeval4pi.py --tag $t --model v3L_$DT --pattern $3 2>&1 | grep -E "^network \(head|e: n" | head -6
  done
done
echo MOVIES_DONE
