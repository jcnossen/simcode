#!/bin/bash
# waits for dec_m25v3's epoch-300 snapshot (metrics flat since ~epoch 165, the m1809 criterion), freezes it as
# model_e0300_converged.pt and runs DECODE v3 on the 2.5 nM/2 real data (GPU 1, next to the m25 training)
M=~/simcode2/out/paper2/decode/lutpsf/dec_m25v3
until [ -f $M/snapshots/model_e0300.pt ]; do sleep 120; done
sleep 30; cp $M/snapshots/model_e0300.pt $M/model_e0300_converged.pt
GPU=1 bash ~/simcode2-decodev3/analysis/decode_checks/v3/rerun_real.sh 2.5nM_2 dec_m25v3 $M/model_e0300_converged.pt
