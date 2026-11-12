#!/bin/bash
# Gaussian-PSF reproduction of the preprint's simulations, baselines as the preprint ran them (SMLM / SIMFLUX on the Gaussian
# sigma 1.3 px model, DECODE = the preprint's own model, SIMCODE v1 as published). v2 (g13) later.
# usage: CUDA_VISIBLE_DEVICES=2 bash analysis/papersim/run_gauss_base.sh "D4_gauss D1_gauss"
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
for D in $1; do
  while [ ! -f out/papersim/$D/gen.json ]; do sleep 30; done
  python analysis/papersim/run_all.py out/papersim/$D --steps base,decode_pre,v1,metrics --psf gauss:1.3 --netset g13 >> out/papersim/logs/full_$D.log 2>&1
  rc=$?
  [ $rc -ne 0 ] && python analysis/papersim/run_all.py out/papersim/$D --steps metrics --netset g13 --redo metrics >> out/papersim/logs/full_$D.log 2>&1
  rm -rf out/papersim/$D/full/base/work
  echo "$(date) done gauss-base $D rc $rc" >> out/papersim/logs/full_stream.log
done
