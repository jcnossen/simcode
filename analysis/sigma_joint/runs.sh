#!/bin/bash
# joint covariance (2026-11-17): pipeline runs (python -m simcode.run, full movie: pass 1, pattern estimation, pattern solve, pass 2)
# of the sim movies of analysis/sigma_layer/sims.sh (and the paper simulations) with a model folder of mk_models.py.
# usage: PKG=<package repo> GPU=<g> LABEL=<joint|...> bash runs.sh <model>:<sim> [...]
#   sim: a name of analysis/sigma_layer/sims.sh (out/sigma_layer/sims/<name>), ah1 / bh1 (out/psf_v2/eval/sims m05r3v2 / m25rc3v2) or ps:<papersim movie>
source ~/miniconda3/bin/activate simcode-py311
export CUDA_VISIBLE_DEVICES=${GPU:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PKG
O=$HOME/simcode2/out/sigma_joint; S=$HOME/simcode2/out/sigma_layer; C=$S/calibs; L=${LABEL:-joint}
for spec in "$@"; do
  m=${spec%%:*}; s=${spec#*:}; crop=''; nf='--nframes 4800'
  case $m in m1809) cal=$C/calib_1809_2_psf_v2_simcam.json ;; m25) cal=$C/calib_2p5nM_m2_psf_v2_simcam.json ;; g13) cal=$C/calib_gauss1.3_simcam.json ;; esac
  case $s in
    ah1) mov=$HOME/simcode2/out/psf_v2/eval/sims/m05r3v2/sim_movie.tif ;;
    bh1) mov=$HOME/simcode2/out/psf_v2/eval/sims/m25rc3v2/sim_movie.tif ;;
    ps:*) p=${s#ps:}; mov=$HOME/simcode2/out/papersim/$p/movie.tif; cal=$HOME/simcode2/out/papersim/$p/calib_sim.json; crop='--crop 0,0,40'; nf='' ;;
    *) mov=$S/sims/$s/sim_movie.tif ;;
  esac
  out=$O/runs/${m}_$L/${s#ps:}
  [ -f $out/locs_p2.hdf5 ] && { echo "run $m $s exists"; continue; }
  t0=$(date +%s)
  python -m simcode.run --movie $mov --calib $cal --out $out --model $O/models/${m}_$L $crop $nf > $out.log 2>&1 \
    && echo "run $m $s done $(( $(date +%s) - t0 )) s" || echo "run $m $s FAILED"
done
