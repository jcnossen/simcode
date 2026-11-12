#!/bin/bash
# FINAL hollow runs, photon levels above the experimental one (2026-11-12 06:45): SIMCODE v2 (m1809, u <= 0.45) + DECODE only.
# SMLM / SIMFLUX are run at x1 only (run_hollow_final.sh): their smlmtorch detection takes ~1.7 h per 120 000-frame movie and the
# 30 000-frame sweep (out/papersim/hollow) already has them at every level.
# usage: CUDA_VISIBLE_DEVICES=<gpu> bash analysis/papersim/run_hollow_final2.sh "<cond:R:scale> ..."
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
for spec in $1; do
  free=$(df --output=avail -BG / | tail -1 | tr -d 'G ')
  if [ "$free" -lt 15 ]; then echo "$(date) DISK GUARD: ${free} GB free, stopping before $spec" >> out/papersim/logs/full_stream.log; exit 1; fi
  IFS=: read cond R s <<< "$spec"
  d=out/papersim/hollow120k/${cond}_R${R}_x${s}
  [ -f $d/movie.tif ] || python analysis/papersim/gen_hollow.py --cond $cond --radius_nm $R --phot_scale $s --nframes 120000 --out $d > out/papersim/logs/gen120k_${cond}_R${R}_x${s}.log 2>&1
  python analysis/papersim/run_all.py $d --netset m1809 --gt_diag 0 --steps decode,v2,metrics,hollow >> out/papersim/logs/full_h120k_${cond}_R${R}_x${s}.log 2>&1
  echo "$(date) done hollow120k $spec (v2 + DECODE) rc $?" >> out/papersim/logs/full_stream.log
done
