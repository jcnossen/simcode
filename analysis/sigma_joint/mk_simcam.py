"""sigma study (2026-11-17): processing calibrations of the sim movies (sims.sh) = the PSF calibration the movie was rendered
with + the camera of analysis/simreal/sim_movie_r.py (0.5 nM offset map, the calibration's gain and read-noise map; as
out/psf_v2/eval/mk_simcam.py, which made the m1809 / m25 ones in out/psf_v2/eval/calibs/). Here: the Gaussian sigma 1.3 px PSF
(models/insitu/calib_gauss1.3.json, the PSF of g13).
usage: python mk_simcam.py <paper repo> <out dir>"""
import json, os, sys
import numpy as np
R, O = sys.argv[1:3]; M = os.path.join(R, 'models', 'insitu'); os.makedirs(O, exist_ok=True)
off = np.load(f'{M}/camera_maps_0.5nM.npz')['offset_adu']
c = json.load(open(f'{M}/calib_gauss1.3.json'))
rn = np.load(f'{M}/{c["camera"]["maps"]}')['readnoise_e'][:120, :120]
np.savez_compressed(f'{O}/camera_maps_gauss1.3_simcam.npz', offset_adu=off.astype(np.float32), readnoise_e=rn.astype(np.float32),
                    gain=np.float32(c['camera']['gain_e_per_adu']))
c['camera'] = dict(gain_e_per_adu=c['camera']['gain_e_per_adu'], offset_adu=float(np.median(off)), readnoise_e=0.9,
                   maps='camera_maps_gauss1.3_simcam.npz', note='SIM-ANALYSIS COPY (sigma layer): camera of the sim movies (sim_movie_r.py)')
c['source'] = c['source'] + ' | SIM-ANALYSIS COPY (analysis/sigma_layer/mk_simcam.py): camera block of the sim movies'
json.dump(c, open(f'{O}/calib_gauss1.3_simcam.json', 'w'), indent=1)
print('ok')
