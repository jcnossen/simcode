#!/bin/bash
# 4Pi-SIMCODE density sweep on the microtubule cell field (resumable: finished outputs are skipped).
# STAGES: movie (sweep4pi_movie.py), front (det4pi candidates + patches), net (head v10 + explicit solve, in-situ pattern
# + phase field; NETB=1 also the true pattern + field), cl (classical, analysis/fourpi_classical/sim_cl.py), eval (sweep4pi_eval.py).
# Outputs: movies / front / locs in $P4/simmovie/<tag>, classical + eval in $P4/sweep4pi/<tag> (P4 = simcode.fourpi.setup.P4).
# Usage: CUDA_VISIBLE_DEVICES=0 TAGS="mtc_d03" STAGES="movie front net cl eval" bash analysis/fourpi_sim/sweep4pi_run.sh
cd "$(dirname "$0")/../.."; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
O4=$(python -c "from simcode.config import root; print(root('fourpi'))"); D=$O4/models/det4pi/final.pt; NF=${FRAMES:-30000}
mkdir -p $O4/sweep4pi/logs
declare -A DENS=([mtc_d01]=0.1 [mtc_d05]=0.5 [mtc_d03]=0.3 [mtc_d075]=0.75 [mtc_d15]=1.5)
for t in ${TAGS:-mtc_d03 mtc_d075 mtc_d15 mtc_d01}; do
  M=$O4/simmovie/$t; L=$O4/sweep4pi/logs; echo "== $t $(date +%H:%M)"
  for s in ${STAGES:-movie front net cl eval}; do
    case $s in
      movie) [ -f $M/meta.json ] || python analysis/fourpi_sim/sweep4pi_movie.py --tag $t --dens ${DENS[$t]} --frames $NF --r 0.0246 > $L/movie_$t.log 2>&1; tail -1 $L/movie_$t.log | cut -c1-200 ;;
      front) [ -f $M/front_SIM_000_v3L_det.npz ] || python analysis/fourpi/simfront4pi.py --tag $t --model v3L --det $D --out _det --save_img 1 2>&1 | grep -v Warn | tail -1 ;;
      net) NM=("iter field iterv10"); [ "${NETB:-0}" = 1 ] && NM+=("oracle oracle oraclev10_phiO")
           for m in "${NM[@]}"; do set -- $m
             [ -f $M/locs_SIM_000_v3L_det_$3.npz ] || python analysis/fourpi/simlin4pi.py --tag $t --front v3L_det --head v10 --pattern $1 --phi $2 --img_cpu ${IMG_CPU:-0} > $L/net_${t}_$3.log 2>&1; tail -1 $L/net_${t}_$3.log | cut -c1-250
           done ;;
      cl) O=$O4/sweep4pi/$t/cl; mkdir -p $O; [ -f $O/cl_$t.npz ] || python analysis/fourpi_classical/sim_cl.py --dir $M --front $M/front_SIM_000_v3L_det.npz --out $O > $L/cl_$t.log 2>&1; grep -E "candidates|field:|kept" $L/cl_$t.log | head -4 ;;
      eval) python analysis/fourpi_sim/sweep4pi_eval.py --tag $t > $L/eval_$t.log 2>&1; tail -25 $L/eval_$t.log ;;
    esac
  done
done
echo RUN_DONE
