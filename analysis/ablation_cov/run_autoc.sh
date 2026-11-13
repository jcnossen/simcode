#!/bin/bash
# excess-variance estimates as 4 parallel streams on GPU 2 (each estimate ~5 GPU-minutes uncontended)
cd /home/jc/simcode2; export PYTHONPATH=/home/jc/simcode2 PATH=/home/jc/miniconda3/envs/simcode-py311/bin:$PATH CUDA_VISIBLE_DEVICES=2
L=out/ablation_cov/logs
python analysis/ablation_cov/autoc_est.py sim ctl,f10,f20,a10 > $L/autoc_sim1.log 2>&1 &
python analysis/ablation_cov/autoc_est.py sim f10a10,mis,full > $L/autoc_sim2.log 2>&1 &
python analysis/ablation_cov/autoc_est.py real 1809_2,20nM_2 > $L/autoc_real1.log 2>&1 &
python analysis/ablation_cov/autoc_est.py real,simpipe 2.5nM_2,D1 > $L/autoc_real2.log 2>&1 &
wait
