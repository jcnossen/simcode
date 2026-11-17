#!/bin/bash
# sigma layer (2026-11-17): the kap = 1 precision check. Pipeline runs of the held-out sims with the k1 model folders (solve
# weights kap_p = kap_I2 = 1; mk_models.py k1). Pass 1 (network outputs) and the estimated pattern do not depend on kap: they are
# hard-linked from the id run of the same movie (which must be finished), so the run redoes the pattern solve and pass 2.
# usage: PKG=<package repo> GPU=<g> bash runs_k1.sh <model>:<sim> [...]   (sim names as runs.sh; ps:<movie> for papersim)
source ~/miniconda3/bin/activate simcode-py311
export CUDA_VISIBLE_DEVICES=${GPU:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PKG
O=$HOME/simcode2/out/sigma_layer
for spec in "$@"; do
  m=${spec%%:*}; s=${spec#*:}; n=${s#ps:}
  src=$O/runs/${m}_id/$n; out=$O/runs/${m}_k1/$n
  until [ -f $src/locs_p2.hdf5 ]; do sleep 20; done
  [ -f $out/locs_p2.hdf5 ] && { echo "k1 $m $n exists"; continue; }
  mkdir -p $out; for f in $src/pass1_* $src/pattern_est.*; do ln -f $f $out/; done
  cmd=$(grep -o 'python -m simcode[.]run .*' $src/command.txt | head -1 | sed "s#--out [^ ]*#--out $out#; s#--model [^ ]*#--model $O/models/${m}_k1#")
  t0=$(date +%s)
  eval $cmd > $out.log 2>&1 && echo "k1 $m $n done $(( $(date +%s) - t0 )) s" || echo "k1 $m $n FAILED"
done
