#!/bin/bash
# FINAL hollow-microtubule runs (user 2026-11-12): R = 25 nm (antibody labelling), 120 000 frames (the real 1809/2 length),
# v2 net set m1809, v2 selection u <= 0.45 (metrics 'v2q45'); base (SMLM / SIMFLUX) + DECODE dec_1809_2 (in range x1-x2).
# usage: CUDA_VISIBLE_DEVICES=<gpu> bash analysis/papersim/run_hollow_final.sh "<cond:R:scale> ..."
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
for spec in $1; do
  free=$(df --output=avail -BG / | tail -1 | tr -d 'G ')
  if [ "$free" -lt 15 ]; then echo "$(date) DISK GUARD: ${free} GB free, stopping before $spec" >> out/papersim/logs/full_stream.log; exit 1; fi
  IFS=: read cond R s <<< "$spec"
  d=out/papersim/hollow120k/${cond}_R${R}_x${s}
  [ -f $d/movie.tif ] || python analysis/papersim/gen_hollow.py --cond $cond --radius_nm $R --phot_scale $s --nframes 120000 --out $d > out/papersim/logs/gen120k_${cond}_R${R}_x${s}.log 2>&1
  python analysis/papersim/run_all.py $d --netset m1809 --gt_diag 0 --steps base,decode,v2,metrics,hollow >> out/papersim/logs/full_h120k_${cond}_R${R}_x${s}.log 2>&1
  rc=$?; rm -rf $d/full/base/work
  echo "$(date) done hollow120k $spec rc $rc" >> out/papersim/logs/full_stream.log
done
