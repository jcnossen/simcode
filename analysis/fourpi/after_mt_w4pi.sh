#!/bin/bash
# Runs by itself after the full MT run of the learned-window pipeline: waits for all MT tables, then post-processing (DME, z drift,
# sigma), whole-movie comparison with the classical table, the real-MT showcase figure and the selection curves on the full table.
# Usage: setsid nohup bash analysis/fourpi/after_mt_w4pi.sh v10 > out/4pi/logs/after_MT_v10.log 2>&1 &
H=${1:-v10}; cd "$(dirname "$0")/../.."; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH; O4=$(python -c "from simcode.fourpi.setup import P4; print(P4)")
NF=$(python -c "
import sys; sys.path.insert(0, 'analysis/fourpi')
from dc import movie_files; print(len(movie_files('MT')))")
until [ $(ls $O4/net/locs_MT_[0-9][0-9][0-9]_$H.npz 2>/dev/null | wc -l) -ge $NF ]; do sleep 120; done
sleep 60; date; echo "== post"
python analysis/fourpi/post_w4pi.py --cell MT --head $H 2>&1 | grep -v Warn | tail -12
echo "== comparison with the classical table (all kept rows, then validated range)"
python analysis/fourpi/cmp_full_w4pi.py --cell MT --head $H 2>&1 | grep -v Warn | tail -40
python analysis/fourpi/cmp_full_w4pi.py --cell MT --head $H --sel 1 2>&1 | grep -v Warn | tail -40
echo "== showcase + selection curves on the full table"
python analysis/fourpi/showcase_real_mt.py --net_all $O4/net/net_MT_${H}_all.npz --tag ${H}_full 2>&1 | tail -15
python analysis/fourpi/select_curve_mt.py --net_all $O4/net/net_MT_${H}_all.npz --tag ${H}_full 2>&1 | tail -40
date; echo AFTER_MT_DONE
