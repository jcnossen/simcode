#!/bin/bash
# DECODE v3 (dec_m1809v3, frozen converged snapshot) on every paper simulation + the isolated-emitter movie:
# <movie>/full/decode/locs_v3.npz next to the old locs.npz / locs_fixed.npz (frame-label fix: dec_common.process_padded).
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2-decodev3
export PYTHONPATH=/home/jc/simcode2/external/torch-psf-040c3f3${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${GPU:-2} TQDM_DISABLE=1
MF=${MF:-$HOME/simcode2/out/paper2/decode/lutpsf/dec_m1809v3/model_e0300_converged.pt}
PS=~/simcode2/out/papersim; L=~/simcode2/out/decode_checks/v3/logs/rerun_sims.log
for md in $PS/{D,Dexp}{0.5,1,2,4}_v2psf $PS/{D,Dexp}{0.5,1,2,4}_v2psf_wsm $PS/{D,Dexp}{0.5,1,2,4}_v2psf_wsm3d $PS/Dexp{0.5,1,2,4}_v2psf_wsmaxis \
          $PS/moving_exp_v2psf $PS/moving_exp_v2psf_wsm ~/simcode2/out/decode_checks/iso_v2psf; do
  out=$md/full/decode/locs_v3.npz
  [ -f $md/movie.tif ] || { echo "skip (no movie) $md" >> $L; continue; }
  [ -f $out ] && { echo "done already $out" >> $L; continue; }
  echo "$(date) $md" >> $L
  python -u analysis/papersim/run_decode.py $md --name dec_m1809v3 --model_file $MF --out $out >> $L 2>&1 || echo "FAILED $md" >> $L
done
echo "$(date) all done" >> $L
