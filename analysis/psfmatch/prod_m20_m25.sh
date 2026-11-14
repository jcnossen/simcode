#!/bin/bash
# m20check (2026-11-12): 20nM_2 production with the m25 network set (PSF-matched to 2.5nM_2; no 20 nM-matched set exists).
# = runs/psfmatch/prod.sh m25 (the 2.5nM_2 matched production: frozen v2 flags runs/psf3d/frozen_config.txt, m25 nets, _real calibrations,
# strict with the m25 sigma rescale, strict-only and u045 files) with the 20 nM movie + its own in-situ calib and the output in
# out/paper2/20nM_2/v2_m25 -> symlink to /mnt/data/dmdsf/simcode2_out/paper2/20nM_2/v2_m25 (large; / must not fill up).
# NOT started by the m20check worker. Resumable (finished chunks are skipped by simcode2.run).
# usage: cd ~/simcode2; PREF="1 2 0" setsid nohup bash analysis/psfmatch/prod_m20_m25.sh > runs/psfmatch/prod_20nM_2_m25.log 2>&1 < /dev/null &
S=m25; ds=20nM_2
source /home/jc/simcode2/runs/psfmatch/env.sh
MOV=/mnt/data/dmdsf/bn_dnapaint/dnapaint/20nM/2/cover1_500mW_5.5ms_cf660.tif
CAL=insitu/calib_20nM_m2_psf3d.json
BIG=/mnt/data/dmdsf/simcode2_out/paper2/$ds/v2_m25; OD=out/paper2/$ds/v2_m25
mkdir -p $BIG; [ -e $OD ] || ln -s $BIG $OD
[ "$(readlink -f $OD)" = "$BIG" ] || { echo "$OD is not the symlink to $BIG"; exit 1; }
[ $(df --output=avail -BG $BIG | tail -1 | tr -dc 0-9) -ge 150 ] || { echo "less than 150 GB free on $BIG"; exit 1; }
CFG=$(cat runs/psf3d/frozen_config.txt)
NETS="--det det_$S.pt --head hd_$S.pt --cal calib/calib_sc_hd_${S}_real.json --p2_det p2det_$S.pt --p2_head p2headL_$S.pt --p2_cal calib/calib_p2L_${S}_real.json --pass2"
G=$(pickgpu ${NEED_GB:-8} "${PREF:-1 2 0}")
if [ ! -f $OD/locs_p2.npz ]; then
  echo "$(date) $ds GPU $G: python -m simcode.run --movie $MOV --calib $CAL --out $OD $CFG $NETS" | tee -a $OD/command.txt
  CUDA_VISIBLE_DEVICES=$G python -u -m simcode.run --movie $MOV --calib $CAL --out $OD $CFG $NETS > $OD/run.log 2>&1 || { echo "FAIL $ds"; exit 1; }
fi
for p in 1 2; do tg=est; [ $p = 2 ] && tg=est2
  echo "$(date) python -m simcode.strict $OD --tag $tg --sig_u models/calib/sigu/sigu_p${p}_$S.json" | tee -a $OD/command.txt
  python -m simcode.strict $OD --tag $tg --sig_u models/calib/sigu/sigu_p${p}_$S.json || { echo "FAIL strict $p"; exit 1; }
done
python - $OD <<'PY'
import sys, numpy as np
d = sys.argv[1]; T = np.load(f'{d}/locs_p2.npz')
c = lambda k: dict(sel=np.ones(int(k.sum()), bool), x=T['x'][k], y=T['y'][k], frame=T['frame'][k], sx=T['sx'][k], sy=T['sy'][k], photons=T['photons'][k])
k = T['strict'].astype(bool); np.savez(f'{d}/locs_p2_strictonly.npz', **c(k)); print('strictonly rows', int(k.sum()), 'of', len(k))
u = T['strict_u']; k = np.isfinite(u) & (u <= 0.45); np.savez(f'{d}/locs_p2_u045.npz', **c(k)); print('u045 rows', int(k.sum()))
PY
echo "$(date) prod $ds ($S) done"
