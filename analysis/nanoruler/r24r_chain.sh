#!/bin/bash
# R24 reuse check, part C: the whole pass-1 sequence for both movies (what was run on 2026-11-13, there as two detached pieces):
# gate scenes + dumps (r24r_gates.sh), then refit / acceptance / pair0.31 analysis (r24r_accept.sh). Resumable.
# usage: cd ~/simcode2; setsid nohup bash analysis/nanoruler/r24r_chain.sh >> out/nanoruler/nr40/reuse/chain.log 2>&1 < /dev/null &
cd /home/jc/simcode2
for N in ${1:-nr40 nr20}; do
  bash analysis/nanoruler/r24r_gates.sh $N || exit 1
  bash analysis/nanoruler/r24r_accept.sh $N s1 || exit 1
done
