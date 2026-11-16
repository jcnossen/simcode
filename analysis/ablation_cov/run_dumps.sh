#!/bin/bash
# extra gate dumps (GPU 2): excess-variance control / recovery scenes (pass 1 only, seed 301) and the dense scene rs25qm (CRLB via gate.py first)
OUT=$(realpath "${1:?usage: $0 <out root, e.g. ~/simcode2/out>}")
cd /home/jc/simcode2; export PYTHONPATH=/home/jc/simcode2; export PATH=/home/jc/miniconda3/envs/simcode-py311/bin:$PATH
export CUDA_VISIBLE_DEVICES=2 TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=$OUT/ablation_cov/fe_ctl; mkdir -p $O
EX="psf_var=0,cam_err=0,exact_est=1,bgmode=struct"
D=python\ analysis/ablation_cov/gate_dump.py
GM=$OUT/psfmatch/gate_m1809a
$D rs05qm 301 $O _ctl - "emitter_var=0,$EX" s1 6000 --gate_dir $GM
$D rs05qm 301 $O _f10 "ang_sd=0,step_sd=0,frame_sd=0.10,halo=0" "emitter_var=1,$EX" s1 6000 --gate_dir $GM
$D rs05qm 301 $O _f20 "ang_sd=0,step_sd=0,frame_sd=0.20,halo=0" "emitter_var=1,$EX" s1 6000 --gate_dir $GM
$D rs05qm 301 $O _a10 "ang_sd=0.10,step_sd=0,frame_sd=0,halo=0" "emitter_var=1,$EX" s1 6000 --gate_dir $GM
$D rs05qm 301 $O _f10a10 "ang_sd=0.10,step_sd=0,frame_sd=0.10,halo=0" "emitter_var=1,$EX" s1 6000 --gate_dir $GM
$D rs05qm 301 $O _mis - "emitter_var=0,psf_var=1,cam_err=1,exact_est=0" s1 6000 --gate_dir $GM
$D rs05qm 301 $O _full - - s1 6000 --gate_dir $GM
# dense scene: CRLB (gate.py, tag _hd_cur) then the dump
G=$OUT/ablation_cov/gate_m1809a; mkdir -p $G
for s in 301 302; do
python -c "
from simcode.util import proctitle; proctitle.set_title('covabl')
import sys; sys.argv=['gate','--scen','rs25qm','--seeds','$s','--emitter','archive_2d/m1809/emitter_pass1.pt','--cal','archive_2d/m1809/calib/calib_emitter_pass1_sim.json','--tag','_hd_cur','--out','$G','--ntarget','3500','--calib','insitu/calib_1809_2_psf3d_avg.json']
from simcode.benchmark import gate_scenes; gate_scenes.main()"
$D rs25qm $s $OUT/ablation_cov/fe_m1809a "" - - s1,s2 3500 --gate_dir $G
done
echo ALLDONE
