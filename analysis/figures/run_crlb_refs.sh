#!/bin/bash
# Evaluation-only reference lines of crlb_all_v2.png (development sc7/run_evalA.sh, run_lm.sh, c34/run_sf.sh, c34/run_sf_rs.sh,
# c34/run_scode_gate.sh). Needs the gate npz of the networks in <out>/gate (analysis/gate/run_gate.sh; GT is copied from the
# _hd_sc5nbb15pm files) and writes <out>/sc7/icrlb_*.npz and <out>/sc7/gate_lm/*.npz. Then:
#   python analysis/figures/crlb_all_v2.py sc5nb,sc6n     (and lm_robust.py, icrlb_report.py for the text tables)
# usage: GPU=1 bash analysis/figures/run_crlb_refs.sh [icrlb|lm|sf|scode|all]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=${GPU:-1} TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
WHAT=${1:-all}
f() { "$PY" "$@" 2>&1 | grep --line-buffered -v "Warn\|CUDACachingAllocator" || true; }
if [[ $WHAT == icrlb || $WHAT == all ]]; then          # per-frame I_t / bg vs CRLB of the networks 
  f "$HERE/icrlb.py" hd_sc5nb.pt sc5nb iso,rs05q,rs05qm 301,302
fi
if [[ $WHAT == lm || $WHAT == all ]]; then             # basic single-emitter LM (11 x 11; z fixed 50 nm, then z free; 15 x 15 for the I_t text)
  f "$HERE/lm_icrlb.py" 11 iso,rs05q,rs05qm 301,302
  f "$HERE/lm_gate.py" iso,rs05q,rs05qm 301,302,303 11
  f "$HERE/lm_icrlb.py" 15 iso,rs05q,rs05qm 301,302
  LM_ZFREE=1 f "$HERE/lm_icrlb.py" 11 iso,rs05q,rs05qm 301,302
  LM_ZFREE=1 f "$HERE/lm_gate.py" iso,rs05q,rs05qm 301,302,303 11
fi
if [[ $WHAT == sf || $WHAT == all ]]; then             # joint SIMFLUX MLE (pattern known, one bg per angle, kinetics off)
  f "$HERE/sf_gate.py" iso,rs05q,rs05qm 301,302,303 11 angle
fi
if [[ $WHAT == scode || $WHAT == all ]]; then          # repaired SIMCODE on the gate (needs smlmtorch; see simcode/eff/simcode_gate.py)
  (cd "$REPO" && f -m simcode.eff.simcode_gate --model ni6_3 --scen ${SCEN:-rs05q,rs05qm})
fi
echo DONE
