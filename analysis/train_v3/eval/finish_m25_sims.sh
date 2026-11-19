#!/bin/bash
# m25 held-out sims after the v3b2 failure (new model: 1422 clean emitter-windows < the pipeline's 2000 for the brightness-noise
# estimate in 4800 frames of the dimmest sim; current: 2546): v3b3 (MED 800, dens 0.016) replaces v3b2; same runs as eval.sh
T=/home/jc/simcode2/out/train_v3; source $T/env.sh 2; cd $SRC; E=$T/eval/m25
for s in v3b3 v3b4; do for w in new cur; do
  MD=$E/model_m25; [ $w = cur ] && MD=/home/jc/simcode2/models/archive_prev_v2psf/m25
  RD=$E/runs/${s}_$w
  [ -f $RD/locs_p2.hdf5 ] || python -m simcode.run --movie $T/eval/sims/$s/sim_movie.tif --calib $T/eval/calibs/calib_2p5nM_m2_psf_v3_simcam.json --out $RD --model $MD --nframes 4800 > $RD.log 2>&1 || echo "run $s $w FAILED"
done; done
python -u $T/eval/eval_sims.py $E/sims_m25.json m25 $E/runs v3b1 v3b3 v3b4 > $E/sims_m25.txt 2>&1 && echo "$(date +%F_%T) m25: eval sims done (v3b1 v3b3 v3b4)" >> $T/stages.log
