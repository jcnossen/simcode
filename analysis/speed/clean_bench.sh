#!/bin/bash
# Clean benchmark of the speed branch: run on an IDLE GPU (check nvidia-smi first; no other compute process on it).
# usage: bash analysis/speed/clean_bench.sh <gpu index>      (~40 min; outputs in /home/jc/simcode2/out/speed/clean/)
set -u
G=$1
source ~/miniconda3/bin/activate simcode-py311
export PYTHONPATH=/home/jc/simcode2-speed CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G TORCHINDUCTOR_COMPILE_THREADS=1
O=/home/jc/simcode2/out/speed/clean; mkdir -p $O; cd /home/jc/simcode2-speed
nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv > $O/gpu_apps_start.txt
S=analysis/speed
python $S/prof_stages.py 1809 6000 > $O/prof_1809.log 2>&1
python $S/prof_stages.py 20nM 3000 > $O/prof_20nM.log 2>&1
python $S/bench_head.py 1809 6000 base,noise,mode:trunk 3 > $O/head_1809.log 2>&1
python $S/bench_head.py 20nM 1800 base,noise,mode:trunk 3 > $O/head_20nM.log 2>&1
python $S/bench_p2.py 1809 6000 base,noise,trunk 3 > $O/p2_1809.log 2>&1
python $S/bench_p2.py 20nM 1800 base,noise,trunk 3 > $O/p2_20nM.log 2>&1
python $S/batch_sweep.py 1809 3000 2 > $O/sweep_1809.log 2>&1
python $S/batch_sweep.py 20nM 900 2 > $O/sweep_20nM.log 2>&1
bash $S/e2e_prefetch.sh /mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/simflux_5.5ms_500mW.tif insitu/calib_1809_2.json 1809 12000 6000 2 > $O/prefetch_1809.log 2>&1
python $S/pattern_kd.py /home/jc/simcode2/out/paper/1809_2/v2/run_default /mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/simflux_5.5ms_500mW.tif 2 > $O/pattern_kd_1809.log 2>&1
python tests/test_fast_paths.py > $O/tests.log 2>&1
nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv > $O/gpu_apps_end.txt
echo CLEAN BENCH DONE
