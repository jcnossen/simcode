#!/bin/bash
# usage: movie.sh <tag> <head> <gpu> [D ...]  (default D1 Dexp1)
source /home/jc/simcode2/runs/psfmatch/env.sh
T=$1; H=$2; G=$3; shift 3; DS=${@:-D1 Dexp1}
for D in $DS; do CUDA_VISIBLE_DEVICES=$G python -u analysis/hdunbias/movie.py $D $T $H > out/hdunbias/movie_${D}_$T.log 2>&1; done
