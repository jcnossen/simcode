#!/bin/bash
# nr80 evaluation, step-order test: gate scenes iso / rs05qm (seeds 301-303) rendered with DECREASING phase steps
# (preset out/nanoruler/eval/nr80_preset_rev.json, spacing [4.06, 4.43]), nr80 pass 1 / 2 dumped, pass 2 - pass 1 on identical
# targets (ev_p12.py) next to the same table on the existing forward-order scenes. One GPU job at a time.
# usage: setsid nohup bash analysis/nanoruler/ev_revtest.sh [tag=rev] >> out/nanoruler/eval/revtest.log 2>&1 < /dev/null &
source /home/jc/simcode2/runs/psfmatch/env.sh; set -o pipefail
t=${1:-rev}; A=analysis/nanoruler; EV=out/nanoruler/eval; MD=$PWD/out/nanoruler/models; C=$PWD/out/nanoruler/calib_nr80.json
SC=${EV_SCENES:-iso,rs05qm}; SEEDS=301,302,303; G=$EV/gate_nr80$t; FE=$EV/fe_nr80$t
export NR_PRESET=$PWD/$EV/nr80_preset_$t.json NR_TITLE=nr-eval SIMCODE_GATE_CALIB=$C
export CUDA_VISIBLE_DEVICES=$(NEED_GB=4 pickgpu 4 "2 0 1"); mkdir -p $G $FE
say() { echo "$(date +%F_%T) $*"; }
say "revtest $t preset $NR_PRESET GPU $CUDA_VISIBLE_DEVICES"
last=${SC##*,}
[ -f $G/${last}__s303_hd_cur.npz ] || python -u $A/nrwrap.py simcode.hd.gate --scen $SC --fronts sc --sc hd_sc5nb.pt --sccal calib/calib_sc_hd_sc5nbb15_pm_bright.json \
  --tag _hd_cur --out $G --ntarget 3500 --seeds $SEEDS || { say FAILED gate; exit 1; }
SIMCODE_GATE_DIR=$G python -u $A/nrwrap.py simcode.p2.dump --scen $SC --seeds $SEEDS --fronts s1=$MD/det_nr80.pt:$MD/hd_nr80.pt,s2=$MD/p2det_nr80.pt:$MD/p2headL_nr80.pt --out $FE \
  > $EV/logs/dump_$t.log 2>&1 || { say FAILED dump; exit 1; }
grep -q "CRLB from None" $EV/logs/dump_$t.log && { say "FAILED: CRLB reference not found"; exit 1; }
python $A/ev_p12.py $FE $SC > $EV/p12_gate_$t.txt.tmp && mv $EV/p12_gate_$t.txt.tmp $EV/p12_gate_$t.txt || { say FAILED p12; exit 1; }
[ -f $EV/p12_gate_fwd.txt ] || { NR_PRESET=$PWD/out/nanoruler/nr80_preset.json python $A/ev_p12.py $EV/fe_nr80 $SC > $EV/p12_gate_fwd.txt.tmp && mv $EV/p12_gate_fwd.txt.tmp $EV/p12_gate_fwd.txt; }
say "revtest $t done"
