#!/bin/bash
# Gate scenes (simulation benchmark against the multi-emitter CRLB; simcode2/hd/gate.py) for the production pass 1 and the
# report. Scenes: iso, pair1 / pair1.5 / pair2 / pair3, rs05x / rs25x (exact model), rs05 / rs25 (full mismatch), rs05p / rs25p,
# rs05pm / rs25pm, rs05q / rs25q, rs05qm / rs25qm, rs05r / rs25r (+ b), rs25s.  Seeds 301-303 (tuning seed 202 is separate).
# Expected (production sc5nb + calib_sc_hd_sc5nbb15_pm_bright): iso all-target RMS / CRLB_SF_multi 1.17 [1.14, 1.20].
# usage: GPU=1 bash analysis/gate/run_gate.sh [scenes] [out_dir]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export CUDA_VISIBLE_DEVICES=${GPU:-1} TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
SC=${1:-iso}; OUT=${2:-${SIMCODE2_OUT:-simcode2_out}/gate}; TAG=_sc5nb_b10
$PY -u -m simcode2.hd.gate --scen $SC --fronts sc --sc hd_sc5nb.pt --sccal calib/calib_sc_hd_sc5nbb15_pm_bright.json --tag $TAG --out $OUT --ntarget 3500
$PY -m simcode2.hd.gate_report_xdet --dir $OUT --scen $SC --tag $TAG --phbins 1 --out $OUT/gate_report$TAG.json
