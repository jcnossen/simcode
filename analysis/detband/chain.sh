#!/bin/bash
# sequential evaluation chain on GPU 1 (one job at a time; GPU 1 memory is shared with the trainings and other jobs,
# so every job is retried after an OOM until its outputs exist)
cd /home/jc/simcode2; A=analysis/detband; R=out/detband/runs
export GATE_PSF_CHUNK=500
w() { until [ -f $1 ]; do sleep 30; done; }
g() { local t=$1; shift; for i in $(seq 20); do
  [ $(ls out/detband/gate/*__s30?_$t.npz 2>/dev/null | wc -l) -ge 12 ] && return 0
  bash $A/gate_ab.sh $t "$@"; sleep 60; done; }
mv_() { for i in $(seq 10); do [ -f out/detband/movie_Dexp1_$1/locs_p2_sim.npz ] && return 0
  (source runs/psfmatch/env.sh; CUDA_VISIBLE_DEVICES=1 python -u $A/movie.py Dexp1 $1 $2 > out/detband/movie_Dexp1_$1.log 2>&1); sleep 60; done; }
g h05 models/det_m1809.pt $R/h05/final.pt 0.5
g h10 models/det_m1809.pt $R/h10/final.pt 1.0
mv_ d0 models/det_m1809.pt
for t in b00 b10 b20; do w $R/$t/final.pt; sleep 20; g $t $R/$t/final.pt; done
for t in b10 b20; do
  g ${t}h05 $R/$t/final.pt $R/h05/final.pt 0.5
  g ${t}h10 $R/$t/final.pt $R/h10/final.pt 1.0
done
for t in b00 b10 b20; do mv_ $t $R/$t/final.pt; done
echo chain done > out/detband/chain.done
