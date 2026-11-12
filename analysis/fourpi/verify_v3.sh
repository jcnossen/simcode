#!/bin/bash
# Step-3 verification of head4pi v3 (pattern phase uniform over 2 pi in training) on the simulations that exposed the
# prior bug: patch-sim prior slope, isolated-emitter gates, simulated movies d03 / d1 (offset vs GT, pattern / free rows).
# Usage: CUDA_VISIBLE_DEVICES=0 bash analysis/fourpi/verify_v3.sh > out/4pi/sim/verify_v3.log 2>&1
set -u
cd /home/jc/simcode2
export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
python analysis/fourpi/prior_slope4pi.py v2 v3
python analysis/fourpi/eval_head4pi.py --tag v3 > out/4pi/sim/eval_head4pi_v3.txt 2>&1; grep -v "^\s*$" out/4pi/sim/eval_head4pi_v3.txt | tail -30
for t in d03 d1; do
  python analysis/fourpi/simfront4pi.py --tag $t --model v3 --noise low --k 2 --nms 3 --s1 1.0
  python analysis/fourpi/simsolve4pi.py --tag $t --model v3 --pattern iter --tile 6 --tblk 25 | tail -1 | cut -c1-200
  python analysis/fourpi/simsolve4pi.py --tag $t --model v3 --pattern oracle --tile 6 --tblk 25 | tail -1 | cut -c1-100
  python analysis/fourpi/simoffset4pi.py --tag $t --model v3
  python analysis/fourpi/simeval4pi.py --tag $t --model v3 --pattern iter > out/4pi/simmovie/$t/eval_v3.txt 2>&1
  grep -E "locs, precision|2000-4000|4000-8000|render" out/4pi/simmovie/$t/eval_v3.txt
done
echo VERIFY_DONE
