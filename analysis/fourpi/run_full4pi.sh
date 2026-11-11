#!/bin/bash
# Full 4Pi-SIMFLUX processing, both movies, all files (detached; GPU 2).
#   pass 1: front4pi.py  (dcimg -> detection -> 4Pi head outputs per candidate), one file after the other, single reader
#   pass 2: solve_real4pi.py (per-file SIM pattern from the network intensities, in-situ 4Pi phase field, final solve)
# Usage: setsid nohup bash analysis/fourpi/run_full4pi.sh v2 "NPC MT" > out/4pi/logs/full_v2.log 2>&1 &
# Resumable: files whose output exists are skipped.
set -u
MODEL=${1:-v2}; CELLS=${2:-"NPC MT"}
REPO=/home/jc/simcode2; cd $REPO
export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
export CUDA_VISIBLE_DEVICES=2
mkdir -p out/4pi/logs out/4pi/net
for CELL in $CELLS; do
  FILES=$(python -c "import sys; sys.path.insert(0,'analysis/fourpi'); from dc import movie_files, file_index; print(' '.join(str(file_index(f)) for f in movie_files('$CELL')))")
  TODO=""
  for f in $FILES; do
    o=$(printf "out/4pi/net/front_%s_%03d_%s.npz" $CELL $f $MODEL)
    [ -f "$o" ] || TODO="$TODO $f"
  done
  echo "$(date) $CELL pass 1 files:$TODO"
  [ -n "$TODO" ] && python analysis/fourpi/front4pi.py --cell $CELL --files $TODO --model $MODEL
  TODO=""
  for f in $FILES; do
    o=$(printf "out/4pi/net/locs_%s_%03d_%s.npz" $CELL $f $MODEL)
    [ -f "$o" ] || TODO="$TODO $f"
  done
  echo "$(date) $CELL pass 2 files:$TODO"
  [ -n "$TODO" ] && python analysis/fourpi/solve_real4pi.py --cell $CELL --files $TODO --model $MODEL
  python analysis/fourpi/post4pi.py --cell $CELL --model $MODEL
  echo "$(date) $CELL done"
done
echo "$(date) ALL DONE"
