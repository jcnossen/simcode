#!/bin/bash
# Reference curve of the FIXED linear read-out (not the method; user decision 13:00) on the simulated movies, det4pi candidates:
# true pattern + field and estimated pattern + field. Usage: CUDA_VISIBLE_DEVICES=0 bash analysis/fourpi/ref_lin_movies.sh > out/4pi/sim/ref_lin_movies.log 2>&1
cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH; D=out/4pi/models/det4pi/final.pt
for t in iso d03 d1; do
  echo "== movie $t (fixed linear read-out, det4pi candidates, v3L prior)"
  python analysis/fourpi/simfront4pi.py --tag $t --model v3L --det $D --out _det --save_img 1 2>&1 | tail -1
  for m in "oracle oracle oraclelin_phiO" "iter field iterlin"; do set -- $m
    python analysis/fourpi/simlin4pi.py --tag $t --front v3L_det --pattern $1 --phi $2 2>&1 | grep -v Warn | tail -1 | cut -c1-200
    echo "-- $3"; python analysis/fourpi/simeval4pi.py --tag $t --model v3L_det --pattern $3 2>&1 | grep -E "^network \(head|e: n" | head -6
  done
  echo "-- v3L trained head, same candidates, true pattern + field"
  python analysis/fourpi/simsolve4pi.py --tag $t --model v3L_det --pattern oracle --phi oracle --tile 6 --tblk 25 2>&1 | tail -1 | cut -c1-80
  python analysis/fourpi/simeval4pi.py --tag $t --model v3L_det --pattern oracle_phiO 2>&1 | grep -E "^network \(head|e: n" | head -6
done
echo REF_DONE
