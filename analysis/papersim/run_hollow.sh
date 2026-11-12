#!/bin/bash
# hollow-microtubule sweep: one stream of movies (generate + all methods + metrics), resumable.
# usage: CUDA_VISIBLE_DEVICES=2 bash analysis/papersim/run_hollow.sh <netset> "<cond:R:scale> ..."
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
NS=$1; shift
for spec in $1; do
  IFS=: read cond R s <<< "$spec"
  d=out/papersim/hollow/${cond}_R${R}_x${s}
  [ -f $d/movie.tif ] || python analysis/papersim/gen_hollow.py --cond $cond --radius_nm $R --phot_scale $s > out/papersim/hollow/logs/gen_${cond}_R${R}_x${s}.log 2>&1
  python analysis/papersim/run_all.py $d --netset $NS --gt_diag 0 --steps base,decode,v2,metrics >> out/papersim/hollow/logs/run_${cond}_R${R}_x${s}_${NS}.log 2>&1
  echo "$(date) done $spec rc $?" >> out/papersim/hollow/logs/stream_${NS}.log
done
