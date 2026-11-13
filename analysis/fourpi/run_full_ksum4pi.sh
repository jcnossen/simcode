#!/bin/bash
# Full-movie run on K-summed real data (real dense data by summing K source windows), resumable, several workers possible (claim files).
# Pattern and 4Pi phase field: calibration from the K = 1 run of the same file (CAL=1, default) or estimated from the dense data (CAL=0).
# Usage: GPU=2 K=4 HEAD=v11b setsid nohup bash analysis/fourpi/run_full_ksum4pi.sh NPC > out/4pi/logs/full_NPC_v11b_k4_a.log 2>&1 &
CELL=${1:-NPC}; HEAD=${HEAD:-v11b}; K=${K:-4}; COH=${COH:-1.56}; CAL=${CAL:-1}; REF=${REF:-v10}
cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH; export CUDA_VISIBLE_DEVICES=${GPU:-2}
TAG=_k${K}$([ "$CAL" = 1 ] && echo cal || echo insitu)
EXTRA=""; [ "$CAL" = 1 ] && EXTRA="--pattern_from out/4pi/net/locs_${CELL}_${REF}_summary.json --field_from out/4pi/net/locs_${CELL}_{file:03d}_${REF}.npz"
FILES=${FILES:-$(python -c "
import sys; sys.path.insert(0, 'analysis/fourpi')
from dc import movie_files, file_index
f = sorted(file_index(x) for x in movie_files('$CELL'))
print(' '.join(str(i) for i in (f[::-1] if '${ORDER:-fwd}' == 'rev' else f)))")}
for f in $FILES; do
  o=$(printf "out/4pi/net/locs_%s_%03d_%s%s" $CELL $f $HEAD $TAG)
  [ -f $o.npz ] && continue
  ( set -o noclobber; echo $$ > $o.claim ) 2>/dev/null || continue
  if [ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -lt 12 ]; then echo "disk below 12 GB free: stop"; rm -f $o.claim; exit 1; fi
  python analysis/fourpi/real_w4pi.py --cell $CELL --files $f --head $HEAD --det out/4pi/models/det4pi_k8/final.pt --coh $COH --ksum $K --tblk 25 $EXTRA $([ -n "${REGIONS:-}" ] && echo "--regions $REGIONS") --tag $TAG 2>&1 | grep -v -E "Warn|solved"
  rm -f $o.claim
done
echo FULL_DONE $CELL K=$K
