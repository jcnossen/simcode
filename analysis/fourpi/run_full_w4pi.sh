#!/bin/bash
# Full-movie run of the learned-window 4Pi pipeline (simcode/fourpi/run.py), resumable: finished files are skipped, a file
# is claimed by the worker that processes it, and the claim of a worker that died is taken over at the next start.
# One worker per GPU (the same command, sharing the files through the claims):
# Usage: GPUS=1,2 setsid nohup bash analysis/fourpi/run_full_w4pi.sh NPC v10 1.56 > /dev/null 2>&1 &
#        (logs: <4Pi work dir>/logs/full_<cell>_<head>_gpu<g>.log; ORDER=rev processes the files backwards)
CELL=${1:-NPC}; HEAD=${2:-v10}; COH=${3:-1.56}
cd "$(dirname "$0")/../.."; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH; O4=$(python -c "from simcode.fourpi.setup import P4; print(P4)")
mkdir -p $O4/logs
python -m simcode.fileio.launch --gpus ${GPUS:-${GPU:-1}} --log $O4/logs/full_${CELL}_${HEAD} -- \
  python -m simcode.fourpi.run --cell $CELL --all --head $HEAD --coh $COH --claim 1 --order ${ORDER:-fwd}
echo FULL_DONE $CELL
