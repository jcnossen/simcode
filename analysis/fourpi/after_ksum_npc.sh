#!/bin/bash
# After the K-summed NPC run: ground-truth evaluation over all files (references: K = 1 network table and K = 1 classical pixel MLE), merged table
# (DME, z drift), classical pipeline on the same summed data for a file subset + its evaluation, in-situ variant on the same subset.
# Usage: K=4 HEAD=v11b setsid nohup bash analysis/fourpi/after_ksum_npc.sh > out/4pi/logs/after_NPC_v11b_k4.log 2>&1 &
K=${K:-4}; HEAD=${HEAD:-v11b}; SUB=${SUB:-"10 5 20 30"}; cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
T=${HEAD}_k${K}cal
until [ $(grep -l FULL_DONE out/4pi/logs/full_NPC_${HEAD}_k${K}_*.log 2>/dev/null | wc -l) -ge ${NW:-3} ]; do sleep 120; done
FILES=$(ls out/4pi/net/locs_NPC_[0-9][0-9][0-9]_$T.npz | sed "s/.*NPC_0*\([0-9]*\)_$T.npz/\1/" | tr '\n' ' ')
date; echo "== files: $FILES"
echo "== ground truth: K = 1 network table"; python analysis/fourpi/eval_ksum4pi.py --cell NPC --files $FILES --tag $T --K $K --ref net --out out/4pi/net/eval_NPC_${T}_refnet.json 2>&1 | tail -19
echo "== ground truth: K = 1 classical pixel MLE"; python analysis/fourpi/eval_ksum4pi.py --cell NPC --files $FILES --tag $T --K $K --ref clp --out out/4pi/net/eval_NPC_${T}_refclp.json 2>&1 | tail -19
echo "== post"; python analysis/fourpi/post_w4pi.py --cell NPC --head $T --ksum $K 2>&1 | grep -v Warn | tail -8
echo "== classical pipeline on the summed data, files $SUB"
CUDA_VISIBLE_DEVICES=${GPU:-1} python analysis/fourpi/cl_ksum4pi.py --cell NPC --files $SUB --ksum $K 2>&1 | grep -E "K = |kept" | cut -c1-220
python analysis/fourpi/eval_ksum4pi.py --cell NPC --files $SUB --K $K --ref net --table "out/4pi/net/cl_ksum/cl_{cell}_{file:03d}_k$K.npz" --out out/4pi/net/eval_NPC_cl_k${K}_refnet.json 2>&1 | tail -19
echo "== network on the same file subset (for the row next to the classical one)"
python analysis/fourpi/eval_ksum4pi.py --cell NPC --files $SUB --tag $T --K $K --ref net 2>&1 | tail -19 | grep -E "all|summed"
echo "== in-situ pattern + phase field variant, files $SUB"
GPU=${GPU:-1} K=$K HEAD=$HEAD CAL=0 FILES="$SUB" bash analysis/fourpi/run_full_ksum4pi.sh NPC
python analysis/fourpi/eval_ksum4pi.py --cell NPC --files $SUB --tag ${HEAD}_k${K}insitu --K $K --ref net --out out/4pi/net/eval_NPC_${HEAD}_k${K}insitu_refnet.json 2>&1 | tail -19 | grep -E "all|summed"
date; echo AFTER_KSUM_DONE
