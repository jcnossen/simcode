#!/bin/bash
# hd head experiment (2026-11-15): network stage (simlin4pi_hd.py, in-situ pattern + phase field; NETB=1 also true pattern + field)
# and the sweep metrics (sweep4pi_eval_hd.py) of one head on the existing sweep movies. Resumable (finished outputs skipped).
# Inputs (read-only): <4Pi work folder>/simmovie/<tag>/ (movie, GT, det4pi front with patches). Outputs: $OUTROOT/<tag>/.
# Usage: CUDA_VISIBLE_DEVICES=0 HEAD=<W4>/models/head4pi_hd1/head4pi_hd1.pt NAME=hd1 OUTROOT=<W4>/models/head4pi_hd1/sweep \
#        TAGS="mtc_d03 mtc_d075" bash analysis/fourpi_sim/hd_run.sh <W4 = 4Pi work folder>
O4=${1:?usage: $0 <4Pi work folder>}; O4=$(realpath "$O4")
cd "$(dirname "$0")/../.."; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
for t in ${TAGS:-mtc_d03 mtc_d075 mtc_d15 mtc_d05 mtc_d01}; do
  M=$O4/simmovie/$t; O=$OUTROOT/$t; mkdir -p $O; echo "== $NAME $t $(date +%H:%M)"
  NM=("iter field iter$NAME"); [ "${NETB:-0}" = 1 ] && NM+=("oracle oracle oracle${NAME}_phiO")
  for m in "${NM[@]}"; do set -- $m
    [ -f $O/locs_SIM_000_v3L_det_$3.npz ] || python analysis/fourpi_sim/simlin4pi_hd.py --work $O4 --tag $t --front v3L_det --head $HEAD --name $NAME \
        --pattern $1 --phi $2 --img_cpu 0 --out $O > $O/net_$3.log 2>&1; tail -n 1 $O/net_$3.log | cut -c1-250
  done
  B=$O/locs_SIM_000_v3L_det_oracle${NAME}_phiO.npz
  [ -f $O/eval.json ] || python analysis/fourpi_sim/sweep4pi_eval_hd.py --work $O4 --tag $t --locs $O/locs_SIM_000_v3L_det_iter$NAME.npz \
      $([ -f $B ] && echo --bound $B) --out $O > $O/eval.log 2>&1; grep -E "^net_(base|sig0|sig1|bound)" $O/eval.log | cut -c1-200
done
echo RUN_DONE
