#!/bin/bash
# the x1 hollow120k movie: its base step (pid given) is running; afterwards DECODE, v2, metrics, hollow
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
while kill -0 $1 2>/dev/null; do sleep 60; done
d=out/papersim/hollow120k/1809_2_R25_x1
python analysis/papersim/run_all.py $d --netset m1809 --gt_diag 0 --steps base,decode,v2,metrics,hollow >> out/papersim/logs/full_h120k_1809_2_R25_x1.log 2>&1
rm -rf $d/full/base/work
echo "$(date) done hollow120k 1809_2:25:1 (all methods) rc $?" >> out/papersim/logs/full_stream.log
