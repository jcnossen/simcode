#!/bin/bash
# SIMCODE v2 with the Gaussian-PSF network set g13 on the preprint-reproduction movies, once g13 is registered in netsets.json
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
until python -c "import json,sys,os; S=json.load(open('analysis/papersim/netsets.json')); n=S.get('g13'); sys.exit(0 if n and all(os.path.exists('models/'+n[k]) for k in ('det','head','p2_det','p2_head','cal_sim','p2_cal_sim')) else 1)"; do sleep 120; done
echo "$(date) g13 registered, starting v2 runs" >> out/papersim/logs/full_stream.log
for D in D1_gauss D4_gauss D2_gauss D0.5_gauss moving_gauss; do
  python analysis/papersim/run_all.py out/papersim/$D --steps v2,metrics --netset g13 --gt_diag 0 --redo metrics >> out/papersim/logs/full_$D.log 2>&1
  echo "$(date) done g13 $D rc $?" >> out/papersim/logs/full_stream.log
done
python analysis/papersim/s10.py --netset g13 --psf gauss1.3 --out out/papersim/s10 > out/papersim/logs/s10_g13.log 2>&1
echo "$(date) done g13 s10 rc $?" >> out/papersim/logs/full_stream.log
