#!/bin/bash
# The *_wsm3d sets (2026-11-15): the webSMLM GT of the *_wsm sets (wsm:11:3:3) with every label at its own webSMLM height
# (wsm_gt.heights_um) instead of a v2-rule height; every method exactly as for _v2psf_wsm (out/papersim/logs/run_wsm.sh), except
# the SMLM / SIMFLUX fit plane = the median GT height (run_all --psf_z gt). One chain per call, densities in order:
#   gen (CPU render, same random streams as the _wsm movie) -> base (+ SIMFLUX) -> v1, v1gt (preprint SIMCODE, internal only)
#   -> DECODE (DEC, default dec_sim1809v2; frame fix = main's dec_common) -> SIMCODE v2 (m1809 weights, name m1809v2) -> metrics, gtfrc
# Movies on the SSD (MOVROOT, default ~/simcode2/out/papersim_ssd), symlinked into out/papersim. GPU steps wait for one of GPUS
# (default "0 1"; GPU 2 = a long DECODE training) with < 50 % utilisation (mean of 5 samples) and >= 8 GB free.
# usage: [REPO=<checkout>] [P=Dexp|D] [DEC=dec_sim1809v2] bash analysis/papersim/run_wsm3d.sh "<densities, e.g. 1 4>"
# DECODE swap later: DEC=<new model> ... and run_all --steps decode --redo decode, then metrics,gtfrc --redo (resumable steps).
source ~/miniconda3/bin/activate simcode-py311; cd ${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
O=/home/jc/simcode2/out/papersim; MOVROOT=${MOVROOT:-/home/jc/simcode2/out/papersim_ssd}; P=${P:-Dexp}; DEC=${DEC:-dec_sim1809v2}
GPUS=${GPUS:-0 1}; SPEC=wsm3d:11:3:3
MV2=/home/jc/simcode2/out/psf_v2/eval/m1809v2/model_m1809      # = models.json m1809 (identical weights), as the _wsm runs
mkdir -p $MOVROOT $O/logs
pick() {
  while true; do
    exec 9>/tmp/wsm3d_gpu.lock; flock 9
    for g in $GPUS; do
      [ -f /tmp/wsm3d_gpu_$g.last ] && [ $(( $(date +%s) - $(cat /tmp/wsm3d_gpu_$g.last) )) -lt 120 ] && continue
      u=0; for i in 1 2 3 4 5; do u=$((u + $(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits))); sleep 2; done
      f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ $((u / 5)) -lt 50 ] && [ $f -ge 8000 ]; then date +%s > /tmp/wsm3d_gpu_$g.last; flock -u 9; PICK=$g; return; fi
    done
    flock -u 9; sleep 45
  done
}
step() {   # step <movie> <run_all args...>
  local M=$1; shift; pick; local g=$PICK
  echo "$(date) start ${M} $* on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python analysis/papersim/run_all.py $O/${M} "$@" >> $O/logs/full_${M}.log 2>&1; local rc=$?
  echo "$(date) ${M} $1 $2 rc $rc GPU $g$( [ $rc -ne 0 ] && echo "  FAILED, see $O/${M}/full/logs/$2.log" )"
}
for D in $1; do
  M=${P}${D}_v2psf_wsm3d
  [ -e $O/$M ] || { mkdir -p $MOVROOT/$M; ln -s $MOVROOT/$M $O/$M; }
  if [ ! -f $O/$M/gen.json ]; then
    echo "$(date) gen $M"
    if [ $P = Dexp ]; then CUDA_VISIBLE_DEVICES= python analysis/papersim/gen_hollow.py --radius_nm 0 --density $D --nframes 60000 --v2psf 1 --gt $SPEC --device cpu --out $O/$M > $O/logs/gen_$M.log 2>&1
    else CUDA_VISIBLE_DEVICES= python analysis/papersim/gen_density.py --density $D --nframes 60000 --psf v2 --gt $SPEC --device cpu --out $O/$M > $O/logs/gen_$M.log 2>&1; fi
    echo "$(date) gen $M rc $?"
  fi
  step $M --steps base --psf models/insitu/calib_1809_2_psf_v2.json --psf_z gt; rm -rf $O/$M/full/base/work
  step $M --steps decode --decode_name $DEC
  step $M --steps v2 --model $MV2 --name m1809v2 --gt_diag 0
  step $M --steps v1; step $M --steps v1gt
  echo "$(date) $M metrics,gtfrc"
  CUDA_VISIBLE_DEVICES= nice -n 19 python analysis/papersim/run_all.py $O/$M --steps metrics,gtfrc --model $MV2 --name m1809v2 >> $O/logs/full_$M.log 2>&1
  echo "$(date) $M metrics,gtfrc rc $?"
done
