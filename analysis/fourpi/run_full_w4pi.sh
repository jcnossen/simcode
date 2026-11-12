#!/bin/bash
# Full-movie run of the learned-window 4Pi pipeline (real_w4pi.py), resumable; several workers may run on different GPUs
# (a file is claimed with a marker; finished files are skipped; a crashed worker leaves locs_<cell>_<file>_<head>.claim: delete it).
# Usage: GPU=1 ORDER=fwd setsid nohup bash analysis/fourpi/run_full_w4pi.sh NPC v10 1.56 > out/4pi/logs/full_NPC_v10_a.log 2>&1 &
CELL=${1:-NPC}; HEAD=${2:-v10}; COH=${3:-1.56}
cd "$(dirname "$0")/../.."; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH; O4=$(python -c "from simcode.fourpi.setup import P4; print(P4)"); export CUDA_VISIBLE_DEVICES=${GPU:-1}
FILES=$(python -c "
import sys; sys.path.insert(0, 'analysis/fourpi')
from dc import movie_files, file_index
f = sorted(file_index(x) for x in movie_files('$CELL'))
print(' '.join(str(i) for i in (f[::-1] if '${ORDER:-fwd}' == 'rev' else f)))")
for f in $FILES; do
  o=$(printf "$O4/net/locs_%s_%03d_%s" $CELL $f $HEAD)
  [ -f $o.npz ] && continue
  ( set -o noclobber; echo $$ > $o.claim ) 2>/dev/null || continue
  if [ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -lt 12 ]; then echo "disk below 12 GB free: stop"; rm -f $o.claim; exit 1; fi
  python analysis/fourpi/real_w4pi.py --cell $CELL --files $f --head $HEAD --coh $COH 2>&1 | grep -v -E "Warn|solved"
  rm -f $o.claim
done
echo FULL_DONE $CELL
