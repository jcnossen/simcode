#!/bin/bash
# after chain.sh: rerun the control gate d0 (its first run lost pair2 iterations to OOM)
cd /home/jc/simcode2; A=analysis/detband; export GATE_PSF_CHUNK=500
until [ -f out/detband/chain.done ]; do sleep 30; done
for i in $(seq 20); do [ $(ls out/detband/gate/*__s30?_d0.npz 2>/dev/null | wc -l) -ge 12 ] && break; bash $A/gate_ab.sh d0 models/det_m1809.pt; sleep 60; done
echo done > out/detband/chain2.done
