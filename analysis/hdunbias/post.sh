#!/bin/bash
# after the fine-tunes: gate + movies for every arm (GPU $1 for gates, $2 for movies), then the evaluations
cd /home/jc/simcode2; G=${1:-2}; M=${2:-2}; TAGS=${TAGS:-"ctl J P JP"}
( for t in $TAGS; do bash analysis/hdunbias/gate.sh $t out/hdunbias/runs/$t/final.pt $G; done ) &
( for t in $TAGS; do bash analysis/hdunbias/movie.sh $t out/hdunbias/runs/$t/final.pt $M; done ) &
wait
source runs/psfmatch/env.sh
python analysis/hdunbias/ev_gate.py base $TAGS > out/hdunbias/ev_gate.log 2>&1
for D in D1 Dexp1; do python analysis/hdunbias/ev_mv.py $D base $TAGS > out/hdunbias/ev_mv_$D.log 2>&1; done
echo post done
