#!/bin/bash
# Movie gates with the learned detector: simulated movies iso (0.03 /um^2), d03, d1 through det4pi candidates + head M;
# pattern + phase field estimated in situ ('iter') and given ('oracle' pattern + 'phiO' true field); offsets and GT metrics.
# Usage: CUDA_VISIBLE_DEVICES=0 bash analysis/fourpi/gate_movies_det.sh v6 > out/4pi/sim/gate_movies_det_v6.log 2>&1
set -u
M=${1:-v6}; D=out/4pi/models/det4pi/final.pt
cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
for t in iso d03 d1; do
  echo "== movie $t ($M + det4pi)"
  python analysis/fourpi/simfront4pi.py --tag $t --model $M --det $D --out _det | tail -1
  python analysis/fourpi/simsolve4pi.py --tag $t --model ${M}_det --pattern iter --tile 6 --tblk 25 | tail -1 | cut -c1-120
  python analysis/fourpi/simsolve4pi.py --tag $t --model ${M}_det --pattern oracle --phi oracle --tile 6 --tblk 25 | tail -1 | cut -c1-80
  python analysis/fourpi/simoffset4pi.py --tag $t --model ${M}_det | grep -A2 "^pattern-solved, in-situ\|^network prior" | head -6
  echo "-- estimated pattern + field"; python analysis/fourpi/simeval4pi.py --tag $t --model ${M}_det --pattern iter 2>&1 | grep -E "^network \(head|e: n" | head -6
  echo "-- true pattern + field";      python analysis/fourpi/simeval4pi.py --tag $t --model ${M}_det --pattern oracle_phiO 2>&1 | grep -E "^network \(head|e: n" | head -6
done
echo MOVIES_DONE
