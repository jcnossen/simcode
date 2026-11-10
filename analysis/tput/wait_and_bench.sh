#!/bin/bash
# Runs bench_tput.py per dataset on an otherwise idle GPU (1 or 2): waits until a GPU has had no compute process for two
# consecutive polls (60 s apart), runs the dataset while sampling the GPU's compute-process list every 15 s; if any other
# process appeared during the run the result is moved to bench_<d>_contaminated_<n>.json and the dataset is retried (<= 6 tries).
#  same logic, simcode2 paths.
# usage: [DATASETS="0.5nM 2.5nM"] [VARIANTS=orig,fast,fastc,heads] [GPUS="1 2"] bash analysis/tput/wait_and_bench.sh
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISABLE_NATIVE_JIT=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
T=${TPUT_OUT:-${SIMCODE2_OUT:-simcode2_out}/tput}; mkdir -p $T; export TPUT_OUT=$T
napps() { nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -v "^$2$" | grep -c . ; }
pick() {
  while true; do
    for g in ${GPUS:-1 2}; do
      if [ "$(napps $g x)" = "0" ]; then sleep 60; [ "$(napps $g x)" = "0" ] && { echo $g; return; }; fi
    done
    sleep 60
  done
}
for d in ${DATASETS:-0.5nM 2.5nM}; do
  for try in 1 2 3 4 5 6; do
    G=$(pick)
    echo "== $d try $try on GPU $G start $(date +%T)"
    rm -f $T/bench_$d.json
    CUDA_VISIBLE_DEVICES=$G $PY $REPO/analysis/tput/bench_tput.py $d ${VARIANTS:-orig,fast,fastc,heads} > $T/bench_$d.log 2>&1 &
    BP=$!; FOREIGN=0; seen=""
    while kill -0 $BP 2>/dev/null; do
      others=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -v "^$BP$" | tr '\n' ' ')
      [ -n "${others// /}" ] && { FOREIGN=1; seen="$seen $others"; }
      sleep 15
    done
    if [ $FOREIGN = 0 ]; then echo "== $d end $(date +%T) VALID (no other process on GPU $G)"; break; fi
    mv $T/bench_$d.json $T/bench_${d}_contaminated_$try.json 2>/dev/null
    echo "== $d end $(date +%T) CONTAMINATED (other pids on GPU $G:$seen) -> retry"
  done
done
echo "WAIT_BENCH DONE"
