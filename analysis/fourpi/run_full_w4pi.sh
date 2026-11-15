#!/bin/bash
# Whole-movie run of 4Pi-SIMCODE through the pipeline (setup fourpi, simcode/pipeline/configs/fourpi.py): per file the
# in-situ registration, then the processing of every file (simcode/processing/fourpi.py; one worker per GPU, files
# claimed, finished files skipped, the claim of a dead worker taken over). Settings (network, coh, ...): the config.
# Usage: GPUS=1,2 setsid nohup bash analysis/fourpi/run_full_w4pi.sh NPC > /dev/null 2>&1 &
#        (log: <4Pi work dir>/pipeline/<cell>/auto.log and process/process.log)
CELL=${1:-NPC}
cd "$(dirname "$0")/../.."; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
python -m simcode.pipeline simcode/pipeline/configs/fourpi.py --only fourpi_$CELL --gpus ${GPUS:-1}
echo FULL_DONE $CELL
