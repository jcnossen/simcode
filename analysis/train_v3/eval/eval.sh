#!/bin/bash
# v3 retrain acceptance of one model (after chain.sh): the recipe's acceptance (simcode/pipeline/training.py: v3-PSF gate scenes,
# dumps, the solve calibrations of the new networks fitted on seed 202, sigma zcal gates [0.9, 1.15] and pass-2 / pass-1 RMS/CRLB
# <= 1 on the realistic scene at the held-out seed 301: simcode.pipeline.acceptance) + new vs current (models/archive_prev_v2psf/<m>)
# on the SAME v3 scenes, each with its own calibration fitted on seed 202 of them (analysis/z3d/accept_z.py, seeds 301-303; pass 2
# with the table z from pass 1) + both models' pipeline runs (python -m simcode.run) on the held-out v3 sim movies (sims.sh;
# the current model with its shipped calibrations, as production would run it) -> eval_sims.py (calibration per photon bin,
# precision, brightness noise estimate) + the staged model folder model_<m>/ (NOT copied into models/).
# Fronts: s1 / s2 = new pass 1 (detector + emitter_joint) / pass 2 (detector_pass2_cont + emitter_pass2_joint), c1 / c2 = current.
# usage: setsid nohup bash eval.sh <m1809|m25> <gpu> > eval_<m>.log 2>&1 &
M=$1; G=$2; T=/home/jc/simcode2/out/train_v3; source $T/env.sh $G; cd $SRC
P=/home/jc/simcode2; D=$T/runs/$M; E=$T/eval/$M; GD=$T/eval/gate_$M; FE=$E/fe; CUR=$P/models/archive_prev_v2psf/$M
case $M in
  m1809) C=$T/calib/calib_1809_2_psf_v3.json; SC=iso,pair1,pair2,rs05pm,rs05qm; FIT=rs05qm,rs05pm; SIMS="v3a1 v3a2 v3a5"; SCAL=calib_1809_2_psf_v3_simcam.json ;;
  m25)   C=$T/calib/calib_2p5nM_m2_psf_v3.json; SC=iso,pair1,pair2,rs25qm; FIT=rs25qm; SIMS="v3b1 v3b2 v3b4"; SCAL=calib_2p5nM_m2_psf_v3_simcam.json ;;
esac
# dry run of the procedure (DRY=1): the current networks stand in for the new ones, output eval/<m>_dry, first sim only, no wait
if [ "$DRY" = 1 ]; then
  D=$T/eval/dry_$M; E=$T/eval/${M}_dry; FE=$E/fe; SIMS=${SIMS%% *}
  for p in detector:detector_pass1 emitter_joint:emitter_pass1 detector_pass2_cont:detector_pass2 emitter_pass2_joint:emitter_pass2; do
    mkdir -p $D/${p%%:*}; ln -sf $CUR/${p##*:}.pt $D/${p%%:*}/final.pt; done
fi
say() { echo "$(date +%F_%T) eval $M: $*"; echo "$(date +%F_%T) $M: eval $*" >> $T/stages.log; }
[ "$DRY" = 1 ] || until grep -q "$M: chain done" $T/stages.log 2>/dev/null; do sleep 60; done
until [ -f $GD/done ]; do sleep 60; done
for s in $SIMS; do until [ -f $T/eval/sims/$s/sim_gt.npz ]; do sleep 60; done; done
mkdir -p $E/calib $E/runs
say start on GPU $G
# ---- dumps of the new and the current networks on the v3 gate scenes (CRLB from the gate files)
if [ ! -f $FE/done ]; then
  python -u -m simcode.processing.gate_dump --scen $SC --seeds 202,301,302,303 --ntarget 3500 --calib $C --gate_dir $GD --out $FE \
    --fronts s1=$D/detector/final.pt:$D/emitter_joint/final.pt,s2=$D/detector_pass2_cont/final.pt:$D/emitter_pass2_joint/final.pt,c1=$CUR/detector_pass1.pt:$CUR/emitter_pass1.pt,c2=$CUR/detector_pass2.pt:$CUR/emitter_pass2.pt \
    > $E/dump.log 2>&1 && touch $FE/done || { say "dump FAILED"; exit 1; }
fi
say dumps done
# ---- solve calibrations on seed 202 (templates of models/calib)
fit() { [ -f $3 ] || python -u -m simcode.calibration.fit_cal_set $1 $FE $FIT $2 $3 > $3.log 2>&1 || { say "fit $1 FAILED"; exit 1; }; }
fit s1 template_calib_emitter_pass1_sim.json $E/calib/calib_emitter_pass1_sim.json
fit s2 template_calib_emitter_pass2_sim.json $E/calib/calib_emitter_pass2_sim.json
fit c1 template_calib_emitter_pass1_sim.json $E/calib/cur_calib_emitter_pass1_sim.json
fit c2 template_calib_emitter_pass2_sim.json $E/calib/cur_calib_emitter_pass2_sim.json
say calibrations done
# ---- recipe acceptance (held-out seed 301) and new vs current (seeds 301-303)
python -u $SRC/simcode/pipeline/acceptance.py $FE $E/calib/calib_emitter_pass1_sim.json $E/calib/calib_emitter_pass2_sim.json $SC $E/accept_recipe.json > $E/accept_recipe.log 2>&1
python -u $P/analysis/z3d/accept_z.py $E/accept_p1 $FE s1 $E/calib/calib_emitter_pass1_sim.json $FE c1 $E/calib/cur_calib_emitter_pass1_sim.json $C $SC > $E/accept_p1.log 2>&1
python -u $P/analysis/z3d/accept_z.py $E/accept_p2 $FE s2 $E/calib/calib_emitter_pass2_sim.json $FE c2 $E/calib/cur_calib_emitter_pass2_sim.json $C $SC \
  s1 $E/calib/calib_emitter_pass1_sim.json > $E/accept_p2.log 2>&1
say accept done
# ---- staged model folder (the names of config.MODEL_FILES; PSF = the v3 calibration the model was trained on)
MF=$E/model_$M; mkdir -p $MF/calib
cp -p $D/detector/final.pt $MF/detector_pass1.pt; cp -p $D/emitter_joint/final.pt $MF/emitter_pass1.pt
cp -p $D/detector_pass2_cont/final.pt $MF/detector_pass2.pt; cp -p $D/emitter_pass2_joint/final.pt $MF/emitter_pass2.pt
cp -p $E/calib/calib_emitter_pass1_sim.json $E/calib/calib_emitter_pass2_sim.json $MF/calib/
PB=$(basename $C .json); cp -p $T/calib/$PB.json $T/calib/$PB.psf.zip $MF/
cp -p $T/calib/$(python -c "import json; print(json.load(open('$C'))['camera']['maps'])") $MF/
python - $MF $M $PB <<'EOF'
import json, sys
d, m, pb = sys.argv[1:4]
json.dump({m: dict(psf=f'{pb}.psf.zip', detector='detector_pass1.pt', emitter='emitter_pass1.pt', cal_sim='calib/calib_emitter_pass1_sim.json',
                   detector_pass2='detector_pass2.pt', emitter_pass2='emitter_pass2.pt', cal_sim_pass2='calib/calib_emitter_pass2_sim.json'),
           '_recipe': dict(trainer_chain=f'/home/jc/simcode2/out/train_v3/chain.sh {m} (logs out/train_v3/runs/{m}/; frozen source out/train_v3/src_e8fc62f4 = psfclean e8fc62f4, torch-psf out/psf_v3/tpsf_64871e5)',
                           evaluation=f'/home/jc/simcode2/out/train_v3/eval/eval.sh {m} (out/train_v3/eval/{m}/accept_*.txt, sims_{m}.txt)',
                           note=('in-situ PSF v3 (out/psf_v3: Zernike 5-21 + focus' + (' + general blur covariance' if m == 'm25' else '') + ' + photoselection, '
                                 'angle-averaged, fit sampling npupil 64 / oversample 2), camera v2 (gain 0.4256, per-pixel read-noise map, no extra white noise), '
                                 'training heights from the calibration (non-pinned fitted heights, clipped 0-0.2 um), brightness noise of the dataset '
                                 + ('c 0.093 / c_a 0.084' if m == 'm1809' else 'c 0.190 / c_a 0.065') + ' (--bnoise); the full recipe of models/README.md: '
                                 'detector 40k, emitter 18k, detector_pass2 (dedicated) 40k -> _mix 40k -> _cont 40k, emitter_pass2 24k -> _long 50k -> _final 50k, '
                                 'joint-covariance fine-tunes emitter_joint 12k / emitter_pass2_joint 20k; solve calibrations fitted on seed 202 of the v3 gate scenes'))},
          open(f'{d}/model.json', 'w'), indent=1)
EOF
say staged
# ---- pipeline runs on the held-out sims (new: the staged folder; current: the archived model as shipped)
for s in $SIMS; do for w in new cur; do
  MD=$MF; [ $w = cur ] && MD=$CUR
  RD=$E/runs/${s}_$w
  [ -f $RD/locs_p2.hdf5 ] || python -m simcode.run --movie $T/eval/sims/$s/sim_movie.tif --calib $T/eval/calibs/$SCAL --out $RD --model $MD \
    --nframes 4800 > $RD.log 2>&1 || { say "sim run $s $w FAILED"; exit 1; }
done; done
say sim runs done
python -u $T/eval/eval_sims.py $E/sims_$M.json $M $E/runs $SIMS > $E/sims_$M.txt 2>&1 || { say "eval_sims FAILED"; exit 1; }
say "done"
echo "$(date) eval $M done"
