#!/bin/bash
# Queue for the remaining showcase stages: waits for the running d2 network pass, then runs on the GPU (0 or 1) with the
# most free memory: d4 network (patches in host memory), evals, the 25 nm tube movie. Resumable through showcase4pi_run.sh.
cd /home/jc/simcode2; L=out/4pi/showcase/logs
pick() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1<2' | sort -t, -k2 -nr | head -1 | cut -d, -f1; }
while [ "$(grep -c RUN_DONE $L/run_d.log)" -lt 2 ]; do sleep 30; done
g=$(pick); echo "d4 net on GPU $g"; CUDA_VISIBLE_DEVICES=$g IMG_CPU=1 TAGS=sc_d4 STAGES=net bash analysis/fourpi/showcase4pi_run.sh
g=$(pick); echo "evals on GPU $g"; CUDA_VISIBLE_DEVICES=$g TAGS="sc_d2 sc_d4" STAGES=eval bash analysis/fourpi/showcase4pi_run.sh
while [ ! -f out/4pi/showcase/cl/sc_d03_r125/cl_sc_d03_r125.npz ] && ! grep -q Traceback $L/cl_sc_d03_r125.log; do sleep 30; done
g=$(pick); echo "r125 on GPU $g"; CUDA_VISIBLE_DEVICES=$g TAGS=sc_d03_r125 STAGES="net eval" bash analysis/fourpi/showcase4pi_run.sh
echo QUEUE_DONE
