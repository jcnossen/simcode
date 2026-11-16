#!/bin/bash
# 4Pi-SIMFLUX high-density showcase: all methods on the showcase movies (resumable: finished outputs are skipped).
# STAGES: front (det4pi candidates + patches), net (head v10 + explicit solve: true and in-situ pattern/field),
# cl (classical published-style + pixel MLE, analysis/fourpi_classical/sim_cl.py, outputs under <4Pi work folder>/showcase/cl),
# eval (GT metrics, astigmatism-only baseline, FRC, tube profiles; showcase4pi_eval.py).
# Usage: CUDA_VISIBLE_DEVICES=1 TAGS="sc_d01 sc_d03" STAGES="front net cl eval" bash analysis/fourpi/showcase4pi_run.sh <4Pi work folder>
O4=${1:?usage: $0 <4Pi work folder>}; O4=$(realpath "$O4"); cd "$(dirname "$0")/../.."; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH; D=$O4/models/det4pi/final.pt
for t in ${TAGS:-sc_d01 sc_d03 sc_d1 sc_d2 sc_d4}; do
  M=$O4/simmovie/$t; echo "== $t $(date +%H:%M)"
  for s in ${STAGES:-front net cl eval}; do
    case $s in
      front) [ -f $M/front_SIM_000_v3L_det.npz ] || python analysis/fourpi/simfront4pi.py --work $O4 --tag $t --model v3L --det $D --out _det --save_img 1 2>&1 | grep -v Warn | tail -1 ;;
      net) for m in "oracle oracle oraclev10_phiO" "iter field iterv10"; do set -- $m
             [ -f $M/locs_SIM_000_v3L_det_$3.npz ] || python analysis/fourpi/simlin4pi.py --work $O4 --tag $t --front v3L_det --head v10 --pattern $1 --phi $2 --img_cpu ${IMG_CPU:-0} 2>&1 | grep -v Warn | tail -1 | cut -c1-200
           done ;;
      cl) O=$O4/showcase/cl/$t; [ -f $O/cl_$t.npz ] || python analysis/fourpi_classical/sim_cl.py --work $O4 --dir $M --front $M/front_SIM_000_v3L_det.npz --out $O > $O4/showcase/logs/cl_$t.log 2>&1; grep -E "candidates|field:|kept" $O4/showcase/logs/cl_$t.log | head -4 ;;
      eval) while [ ! -f $O4/showcase/cl/$t/cl_$t.npz ] && ! grep -q Traceback $O4/showcase/logs/cl_$t.log 2>/dev/null; do sleep 30; done; python analysis/fourpi/showcase4pi_eval.py --work $O4 --tag $t 2>&1 | grep -v Warn | tail -30 ;;
    esac
  done
done
echo RUN_DONE
