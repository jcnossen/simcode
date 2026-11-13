#!/bin/bash
# SIMCODE v2 (g13) on the Gaussian-PSF reproduction movies except D1_gauss (run by the retrain worker), then S10 Gaussian
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
for D in D4_gauss D2_gauss D0.5_gauss moving_gauss; do
  python analysis/papersim/run_all.py out/papersim/$D --steps v2,metrics --netset g13 --gt_diag 0 --redo metrics >> out/papersim/logs/full_$D.log 2>&1
  echo "$(date) done g13 $D rc $?" >> out/papersim/logs/full_stream.log
done
python analysis/papersim/s10.py --netset g13 --psf gauss1.3 --out out/papersim/s10 > out/papersim/logs/s10_g13.log 2>&1
echo "$(date) done g13 s10 rc $?" >> out/papersim/logs/full_stream.log
