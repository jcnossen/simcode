#!/bin/bash
# free-model CRLB dumps on GPU 1, one scene / seed at a time, only when >= 3 GB is free (shared GPU)
source /home/jc/simcode2/runs/psfmatch/env.sh >/dev/null 2>&1
export SIMCODE_GATE_CALIB=models/insitu/calib_1809_2_psf3d_avg.json CUDA_VISIBLE_DEVICES=1 GATE_PSF_CHUNK=48 GATE_CRLB_DIV=12
for job in "iso 302" "iso 303" "rs05qm 301" "rs05qm 302" "rs05qm 303" "pair2 301" "pair2 302" "pair2 303" "rs05pm 301" "rs05pm 302" "rs05pm 303"; do
  set -- $job
  [ -f out/crlbcmp/crlbfree/$1__s$2.npz ] && continue
  for try in 1 2 3; do
    until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1) -ge 3072 ]; do sleep 20; done
    python analysis/crlbcmp/crlb_free.py $1 $2 2>&1 | grep -v CUDACachingAllocator && [ -f out/crlbcmp/crlbfree/$1__s$2.npz ] && break
    sleep 30
  done
done
echo done
