#!/bin/bash
# v3 reprocessing of both movies (head4pi v3 = pattern-phase-randomised training; the prior-bug fix), detached, resumable.
#   pass 1: front4pi.py with raw-patch saving for the classical worker (NPC all candidates; MT regions_MT.json boxes)
#   pass 2: solve_real4pi.py with phase-field time blocks of 25 windows (sim: halves the field error vs 50)
#   post  : post4pi.py (merge, DME, render) with the DAFL-based sigma recalibration
# Usage: source ~/miniconda3/etc/profile.d/conda.sh && conda activate simcode-py311 &&
#        GPU=0 setsid nohup bash analysis/fourpi/run_full4pi_v3.sh "NPC MT" > out/4pi/logs/full_v3.log 2>&1 < /dev/null &
set -u
CELLS=${1:-"NPC MT"}; MODEL=v3
REPO=/home/jc/simcode2; cd $REPO
export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p out/4pi/logs out/4pi/net out/4pi/patches
for CELL in $CELLS; do
  FILES=$(python -c "import sys; sys.path.insert(0,'analysis/fourpi'); from dc import movie_files, file_index; print(' '.join(str(file_index(f)) for f in movie_files('$CELL')))")
  REG=""; [ "$CELL" = "MT" ] && REG="--patch_regions out/4pi/patches/regions_MT.json"
  TODO=""
  for f in $FILES; do o=$(printf "out/4pi/net/front_%s_%03d_%s.npz" $CELL $f $MODEL); [ -f "$o" ] || TODO="$TODO $f"; done
  echo "$(date) $CELL pass 1 files:$TODO"
  [ -n "$TODO" ] && python analysis/fourpi/front4pi.py --cell $CELL --files $TODO --model $MODEL --save_patches out/4pi/patches $REG
  TODO=""
  for f in $FILES; do o=$(printf "out/4pi/net/locs_%s_%03d_%s.npz" $CELL $f $MODEL); [ -f "$o" ] || TODO="$TODO $f"; done
  echo "$(date) $CELL pass 2 files:$TODO"
  [ -n "$TODO" ] && python analysis/fourpi/solve_real4pi.py --cell $CELL --files $TODO --model $MODEL --tblk 25
  python analysis/fourpi/post4pi.py --cell $CELL --model $MODEL --sig_dafl 1
  echo "$(date) $CELL done"
done
echo "$(date) ALL DONE"
