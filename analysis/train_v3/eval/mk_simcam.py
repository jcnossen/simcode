"""processing calibrations of the v3 held-out sim movies (sims.sh; as out/psf_v2/eval/mk_simcam.py): the v3 calibration the movie
was rendered with + the camera of analysis/simreal/sim_movie_r.py (0.5 nM offset map, the calibration's gain and read-noise map).
usage: python mk_simcam.py   (-> calibs/calib_<n>_psf_v3_simcam.json + camera_maps_<n>_v3_simcam.npz)"""
import json, os
import numpy as np
from simcode import config
T = '/home/jc/simcode2/out/train_v3'; O = f'{T}/eval/calibs'; os.makedirs(O, exist_ok=True)
off = np.load(config.model('insitu/camera_maps_0.5nM.npz'))['offset_adu']
for n in ('1809_2', '2p5nM_m2'):
    c = json.load(open(f'{T}/calib/calib_{n}_psf_v3.json'))
    rn = np.load(f'{T}/calib/{c["camera"]["maps"]}')['readnoise_e'][:120, :120]
    np.savez_compressed(f'{O}/camera_maps_{n}_v3_simcam.npz', offset_adu=off.astype(np.float32), readnoise_e=rn.astype(np.float32),
                        gain=np.float32(c['camera']['gain_e_per_adu']))
    c['camera'] = dict(gain_e_per_adu=c['camera']['gain_e_per_adu'], offset_adu=float(np.median(off)), readnoise_e=0.9,
                       maps=f'camera_maps_{n}_v3_simcam.npz',
                       note='SIM-ANALYSIS COPY (train_v3 eval): camera of the v3 sim movies (sim_movie_r.py: 0.5 nM offset map, the calibration gain and read-noise map)')
    c['source'] = c['source'] + ' | SIM-ANALYSIS COPY (out/train_v3/eval/mk_simcam.py): camera block of the sim movies'
    json.dump(c, open(f'{O}/calib_{n}_psf_v3_simcam.json', 'w'), indent=1)
    print(n, 'ok')
