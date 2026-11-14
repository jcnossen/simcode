#!/bin/bash
# R24 reuse check, part C (GPU part): gate scenes rendered with THIS movie's calib (PSF, read-noise map, k) and the nr80 network
# front(s) dumped on them. Same commands as analysis/nanoruler/ev_chain.sh step 1 (gate) and step 2 (dump s1), with
# SIMCODE_GATE_CALIB = out/nanoruler/calib_<name>.json and NR_PRESET = out/nanoruler/<name>/preset_<name>.json.
# Every GPU call goes through r24_gpu.sh (global lock) and is ONE gate / dump call for ONE scene (4 seeds), a few minutes each.
# Resumable: a scene whose last-seed output exists is skipped. Extra scene pair0.31 (two emitters 0.31 px = 20 nm apart; gate.py
# takes the separation from the scene name: pair<sep px>), seed 301 only.
# usage: bash analysis/nanoruler/r24r_gates.sh <nr40|nr20> [front spec, default s1=<nr80 pass-1 det>:<head>] [extra scenes, default pair0.31]
source /home/jc/simcode2/runs/psfmatch/env.sh
N=$1; A=analysis/nanoruler; TR=$PWD/out/nanoruler/train
FR=${2:-s1=$TR/det/final.pt:$TR/hd/final.pt}; XS=${3-pair0.31}
O=out/nanoruler/$N/reuse; G=$O/gate_$N; FE=$O/fe_$N; LG=$O/logs; mkdir -p $G $FE $LG
C=$PWD/out/nanoruler/calib_$N.json; export NR_PRESET=$PWD/out/nanoruler/$N/preset_$N.json NR_TITLE=nr-r24r
B1=calib/calib_sc_hd_sc5nbb15_pm_bright.json; W="python -u $A/nrwrap.py"; SEEDS=202,301,302,303
fn=$(echo $FR | sed 's/=.*//')
say() { echo "$(date +%F_%T) $*"; }
for sc in iso pair1 pair2 rs05qm $XS; do
  sd=$SEEDS; [[ $sc == pair0* ]] && sd=301
  last=${sd##*,}
  if [ ! -f $G/${sc}__s${last}_hd_cur.npz ]; then
    say "gate $N $sc"
    SIMCODE_GATE_CALIB=$C bash $A/r24_gpu.sh 4 $W simcode.hd.gate --scen $sc --fronts sc --sc hd_sc5nb.pt --sccal $B1 \
      --tag _hd_cur --out $G --ntarget 3500 --seeds $sd >> $LG/gate_$sc.log 2>&1 || { say "FAILED gate $sc (see $LG/gate_$sc.log)"; exit 1; }
  fi
  if [ ! -f $FE/${sc}__s${last}_$fn.npz ]; then
    say "dump $N $sc $fn"
    SIMCODE_GATE_CALIB=$C SIMCODE_GATE_DIR=$G bash $A/r24_gpu.sh 4 $W simcode.p2.dump --scen $sc --seeds $sd --fronts $FR --out $FE \
      >> $LG/dump_${fn}_$sc.log 2>&1 || { say "FAILED dump $sc (see $LG/dump_${fn}_$sc.log)"; exit 1; }
    grep -q "CRLB from None" $LG/dump_${fn}_$sc.log && { say "FAILED dump $sc: CRLB reference not found"; exit 1; }
  fi
done
say "gates $N $fn done"
