#!/bin/bash
# v3-PSF gate scenes (scenes + CRLB of the v3 PSF + the pretraining reference front), = the pipeline training stage's gate step
# (simcode/pipeline/training.py) with the paper scene sets: m1809 iso,pair1,pair2,rs05qm,rs05pm; m25 iso,pair1,pair2,rs25qm; seeds
# 202 (calibration fit) and 301-303 (evaluation). usage: bash gate.sh <m1809|m25> <gpu>   (out: out/train_v3/eval/gate_<m>)
M=$1; G=$2; T=/home/jc/simcode2/out/train_v3; source $T/env.sh $G; cd $SRC
case $M in m1809) C=$T/calib/calib_1809_2_psf_v3.json; SC=iso,pair1,pair2,rs05qm,rs05pm ;; m25) C=$T/calib/calib_2p5nM_m2_psf_v3.json; SC=iso,pair1,pair2,rs25qm ;; esac
O=$T/eval/gate_$M; mkdir -p $O
python -u -m simcode.benchmark.gate_scenes --scen $SC --fronts sc --detector /home/jc/simcode2/models/archive_prev_v2psf/m1809/detector_pass1.pt --emitter pretrain_emitter_pass1.pt --cal template_calib_emitter_pass1_sim.json \
  --tag _hd_cur --calib $C --out $O --ntarget 3500 --seeds 202,301,302,303 && touch $O/done
