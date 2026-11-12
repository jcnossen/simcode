#!/bin/bash
# Isolated-emitter CRLB gate for a head (default v4), run after its training finishes:
#  1. decomposition on the bench emitters (pattern + phase field given; row-by-row vs the pixel MLE)
#  2. sparse simulated movie (iso, 0.03 /um^2): pattern + phase field ESTIMATED in situ; GT RMS vs per-emitter CRLB
#  3. d03 / d1: lateral offsets vs GT and the GT metrics
# Usage: STEPS=40000 CUDA_VISIBLE_DEVICES=1 bash analysis/fourpi/gate_v5.sh v5 > out/4pi/sim/gate_v4.log 2>&1
set -u
M=${1:-v4}
cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
until grep -q "it ${STEPS:-60000} " out/4pi/models/head4pi_$M.log 2>/dev/null; do sleep 60; done; sleep 30
echo "== decomposition ($M)"; python analysis/fourpi/decomp4pi.py --head $M 2>&1 | grep -v Warning
for t in iso d03 d1; do
  echo "== movie $t ($M)"
  python analysis/fourpi/simfront4pi.py --tag $t --model $M --noise low --k 2 --nms 3 --s1 1.0 | tail -1
  python analysis/fourpi/simsolve4pi.py --tag $t --model $M --pattern iter --tile 6 --tblk 25 | tail -1 | cut -c1-160
  python analysis/fourpi/simsolve4pi.py --tag $t --model $M --pattern oracle --phi oracle --tile 6 --tblk 25 | tail -1 | cut -c1-100
  python analysis/fourpi/simoffset4pi.py --tag $t --model $M | grep -A2 "^pattern-solved\|^network prior"
  python analysis/fourpi/simeval4pi.py --tag $t --model $M --pattern iter 2>&1 | grep -E "^network \(head|e: n" | head -6
  python analysis/fourpi/simeval4pi.py --tag $t --model $M --pattern oracle_phiO 2>&1 | grep -E "^network \(head|e: n" | head -6
done
echo GATE_DONE
