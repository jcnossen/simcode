#!/bin/bash
# optics sensitivity of the summed-PSF in-situ fit (MT): same samples/seed as the base fit, fewer iterations.
cd /home/jc/simcode2/analysis/fourpi
source ~/miniconda3/etc/profile.d/conda.sh; conda activate simcode-py311
L=/home/jc/simcode2/out/4pi/smlm/psf
export OMP_NUM_THREADS=3
V="--iters 150 --ntrain 2000 --threads 3"
(python run_psf.py MT b150 $V > $L/MT_b150.log 2>&1
 python run_psf.py MT na130 --NA 1.30 $V > $L/MT_na130.log 2>&1
 python run_psf.py MT lam057 --lam 0.57 $V > $L/MT_lam057.log 2>&1
 python run_psf.py MT px120 --px 0.120 $V > $L/MT_px120.log 2>&1
 python run_psf.py MT nmed133 --nmed 1.33 --depth 1.0 $V > $L/MT_nmed133.log 2>&1) &
(python run_psf.py MT na140 --NA 1.40 $V > $L/MT_na140.log 2>&1
 python run_psf.py MT lam063 --lam 0.63 $V > $L/MT_lam063.log 2>&1
 python run_psf.py MT px108 --px 0.1083 $V > $L/MT_px108.log 2>&1
 python run_psf.py MT twoarm --two_arm 1 $V > $L/MT_twoarm.log 2>&1) &
wait
