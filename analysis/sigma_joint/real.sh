#!/bin/bash
# joint prior / intensity covariance (2026-11-17): full rerun of a paper movie (Figs 3 / 4: 1809_2 with m1809, 2.5nM_2 with m25)
# with the fine-tuned emitter networks (the repo's models/<m>/emitter_pass{1,2}.pt) and the joint solve; several GPUs share the
# chunks (--claim 1). Nothing is fitted on real data.
# usage: SRC=<paper repo> GPUS="1 2" bash real.sh <1809_2|2.5nM_2> <out dir>
source ~/miniconda3/bin/activate simcode-py311
export CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$SRC
DS=$1; OUT=$2; mkdir -p $OUT; cd $SRC
for g in ${GPUS:-1}; do
  CUDA_VISIBLE_DEVICES=$g python -m simcode.run --dataset $DS --data-root /mnt/data/dmdsf/bn_dnapaint --out $OUT --claim 1 > $OUT.gpu$g.log 2>&1 &
  sleep 20
done
wait
