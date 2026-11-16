#!/bin/bash
# DECODE model 1 (dec_m1809v3: PSF v2 of 1809_2, sims + 1809_2) = the from-scratch trial arm (v3t_m1809_scratch, copied at
# epoch 28 and resumed: same checkpoint, RNG state, schedule), + the sim-only reference (preset sim1809 = the dec_sim1809v2 recipe,
# same schedule, from scratch) for ~100 epochs next to it (matched-epoch cost of the wider ranges), + the progress evaluation.
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2-decodev3
export CUDA_VISIBLE_DEVICES=${GPU:-2} TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=~/simcode2/out/decode_checks/v3/logs; T=analysis/baselines/decode/train_v3.py
nohup python -u $T --preset m1809 --name dec_m1809v3 --snap_every 4 --warmup 512 >> $L/train_dec_m1809v3.log 2>&1 &
nohup python -u $T --preset sim1809 --name v3t_sim1809_scratch --snap_every 4 --warmup 512 --epochs ${REF_EPOCHS:-120} >> $L/train_v3t_sim1809_scratch.log 2>&1 &
sleep 60
nohup python -u analysis/decode_checks/v3/eval_watch.py --names dec_m1809v3,v3t_sim1809_scratch --real 1809_2 --tag dec_m1809v3 --poll 300 >> $L/eval_dec_m1809v3.log 2>&1 &
wait
