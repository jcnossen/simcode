#!/bin/bash
# Movie gates on simulation for a learned-window head (v10 ...): det4pi candidates, the head's intensities + explicit solve
# (simlin4pi.py --head), with the true pattern + field and with both estimated in situ. GT metrics from simeval4pi.py.
# Usage: CUDA_VISIBLE_DEVICES=1 bash analysis/fourpi/gate_movies_w.sh v10 > out/4pi/sim/gate_movies_v10.log 2>&1
M=${1:-v10}; cd "$(dirname "$0")/../.."; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH; O4=$(python -c "from simcode.fourpi.setup import P4; print(P4)"); D=$O4/models/det4pi/final.pt
for t in ${TAGS:-iso d03 d1}; do
  echo "== movie $t ($M, det4pi candidates)"
  [ -f $O4/simmovie/$t/front_SIM_000_v3L_det.npz ] && python -c "import numpy as np,sys; sys.exit(0 if 'img' in np.load('$O4/simmovie/$t/front_SIM_000_v3L_det.npz').files else 1)" || python analysis/fourpi/simfront4pi.py --tag $t --model v3L --det $D --out _det --save_img 1 2>&1 | tail -1
  for m in "oracle oracle oracle${M}_phiO" "iter field iter${M}"; do set -- $m
    python analysis/fourpi/simlin4pi.py --tag $t --front v3L_det --head $M --pattern $1 --phi $2 2>&1 | grep -v Warn | tail -1 | cut -c1-190
    echo "-- $3"; python analysis/fourpi/simeval4pi.py --tag $t --model v3L_det --pattern $3 2>&1 | grep -E "^network \(head|e: n" | head -6
  done
done
echo MOVIES_DONE
