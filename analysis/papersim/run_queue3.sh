#!/bin/bash
# after the Dexp1 redo: v2 default + metrics for D1 (its default pass stopped at the failed v1 step), then hollow120k x2, x8
cd ~/simcode2
until grep -q "done Dexp1 (redo" out/papersim/logs/full_stream.log; do sleep 60; done
source ~/miniconda3/bin/activate simcode-py311
python analysis/papersim/run_all.py out/papersim/D1 --steps v2,metrics --netset default --gt_diag 0 >> out/papersim/logs/full_D1.log 2>&1
echo "$(date) done D1:v2,metrics:default rc $?" >> out/papersim/logs/full_stream.log
bash analysis/papersim/run_hollow_final.sh "1809_2:25:2 1809_2:25:8"
