"""m20ghost (2026-11-13): m25 on the first 6000 frames of the real 20nM_2 movie, production flags of
analysis/psfmatch/prod_m20_m25.sh + --nframes 6000 --sig_u (m25 rescale), then the strict step as in that script.
Process title 'simcode:m20ghost' (proctitle.auto patched so the package's own 'run'/'pipeline' titles do not overwrite it).
usage: source runs/psfmatch/env.sh; CUDA_VISIBLE_DEVICES=<g> python -u analysis/psfmatch/m20ghost_run.py > out/psfmatch/m20check/real6k_m25.log 2>&1
"""
import os, sys
from simcode2.util import proctitle
proctitle.set_title('m20ghost')
proctitle.auto = lambda name: proctitle.set_title('m20ghost')
from simcode2 import run, strict

S = 'm25'
MOV = '/mnt/data/dmdsf/bn_dnapaint/dnapaint/20nM/2/cover1_500mW_5.5ms_cf660.tif'
CAL = 'insitu/calib_20nM_m2_psf3d.json'
OD = 'out/psfmatch/m20check/real6k_m25'
CFG = open('runs/psf3d/frozen_config.txt').read().split()
NETS = ['--det', f'det_{S}.pt', '--head', f'hd_{S}.pt', '--cal', f'calib/calib_sc_hd_{S}_real.json', '--p2_det', f'p2det_{S}.pt',
        '--p2_head', f'p2headL_{S}.pt', '--p2_cal', f'calib/calib_p2L_{S}_real.json', '--pass2']
EXTRA = ['--nframes', '6000', '--sig_u', f'models/calib/sigu/sigu_p1_{S}.json,models/calib/sigu/sigu_p2_{S}.json']
os.makedirs(OD, exist_ok=True)
args = ['--movie', MOV, '--calib', CAL, '--out', OD] + CFG + NETS + EXTRA
if not os.path.exists(f'{OD}/locs_p2.npz'):
    open(f'{OD}/command.txt', 'a').write('python -m simcode2.run ' + ' '.join(args) + '\n')
    run.main(args)
for p, tg in ((1, 'est'), (2, 'est2')):
    sa = [OD, '--tag', tg, '--sig_u', f'models/calib/sigu/sigu_p{p}_{S}.json']
    open(f'{OD}/command.txt', 'a').write('python -m simcode2.strict ' + ' '.join(sa) + '\n')
    strict.main(sa)
print('m20ghost run done', flush=True)
