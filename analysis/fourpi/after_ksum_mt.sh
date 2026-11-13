#!/bin/bash
# After the K-summed MT box run: ground truth vs the K = 1 MT tables inside the boxes, classical pipeline on the summed data for a few files,
# merged table (DME, z drift), showcase and selection curves with the K-summed network rows.
# Usage: K=4 HEAD=v11b setsid nohup bash analysis/fourpi/after_ksum_mt.sh > out/4pi/logs/after_MT_v11b_k4.log 2>&1 &
K=${K:-4}; HEAD=${HEAD:-v11b}; SUB=${SUB:-"5 35 65 95"}; cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
T=${HEAD}_k${K}cal; RG=out/4pi/patches/regions_MT.json
until [ $(grep -l FULL_DONE out/4pi/logs/full_MT_${HEAD}_k${K}_*.log 2>/dev/null | wc -l) -ge ${NW:-2} ]; do sleep 120; done
FILES=$(ls out/4pi/net/locs_MT_[0-9][0-9][0-9]_$T.npz | sed "s/.*MT_0*\([0-9]*\)_$T.npz/\1/" | tr '\n' ' ')
date; echo "== files: $FILES"
echo "== ground truth: K = 1 network table, inside the boxes"; python analysis/fourpi/eval_ksum4pi.py --cell MT --files $FILES --tag $T --K $K --ref net --regions $RG --out out/4pi/net/eval_MT_${T}_refnet.json 2>&1 | tail -22
echo "== classical pipeline on the summed data, files $SUB"
CUDA_VISIBLE_DEVICES=${GPU:-1} python analysis/fourpi/cl_ksum4pi.py --cell MT --files $SUB --ksum $K 2>&1 | grep -E "K = |kept" | cut -c1-220
python analysis/fourpi/eval_ksum4pi.py --cell MT --files $SUB --K $K --ref net --regions $RG --table "out/4pi/net/cl_ksum/cl_{cell}_{file:03d}_k$K.npz" --out out/4pi/net/eval_MT_cl_k${K}_refnet.json 2>&1 | tail -22 | grep -E "all|summed|fringe errors by"
python analysis/fourpi/eval_ksum4pi.py --cell MT --files $SUB --tag $T --K $K --ref net --regions $RG 2>&1 | tail -22 | grep -E "all|summed|fringe errors by"
echo "== post"; python analysis/fourpi/post_w4pi.py --cell MT --head $T --ksum $K 2>&1 | grep -v Warn | tail -8
echo "== showcase + selection curves with the K-summed network rows"
python analysis/fourpi/showcase_real_mt.py --net_all out/4pi/net/net_MT_${T}_all.npz --head "${HEAD} K=${K} summed" --tag ${HEAD}_k${K} 2>&1 | tail -8
python analysis/fourpi/select_curve_mt.py --net_all out/4pi/net/net_MT_${T}_all.npz --head "${HEAD} K=${K} summed" --tag ${HEAD}_k${K} 2>&1 | grep -E "OP |->" | tail -14
date; echo AFTER_KSUM_MT_DONE
