#!/bin/bash
# Showcase movies (resumable: existing movies are skipped). Usage: CUDA_VISIBLE_DEVICES=0 bash analysis/fourpi/showcase4pi_movies.sh
cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
for dt in "d01 0.1 20000" "d03 0.3 10000" "d1 1.0 6000" "d2 2.0 4000" "d4 4.0 4000"; do set -- $dt
  python analysis/fourpi/showcase4pi_movie.py --tag sc_$1 --dens $2 --frames $3 2>&1 | grep -v Warn
done
echo MOVIES_DONE
