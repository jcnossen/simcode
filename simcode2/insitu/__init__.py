"""In-situ calibration of the PSF, camera and illumination pattern from the DNA-PAINT movie itself (recipe: README.md
in this directory). Default inputs are relative to the data root, default outputs go to <out>/insitu."""
import os
from simcode2 import config

MOVIE_05 = 'dnapaint/0.5nM/cover1/1/simflux_5.5ms_500mW.tif'                     # 0.5 nM movie (data root)
SMLM_05 = 'dnapaint/0.5nM/cover1/1/results/simflux_5.5ms_500mW/smlm.hdf5'        # its SIMCODE-pipeline 'smlm' locs (Picasso hdf5)
SFCFG_05 = 'dnapaint/0.5nM/cover1/1/cache/simflux_5.5ms_500mW/sf_fits_cfg.pickle'  # its SIMCODE-pipeline pattern (init of fit_pattern)
BEAD_PSF = 'insitu/beads/psf_fit_sign-1_na149_ball_drift.pt'                      # bead z-stack fit (psf/fit_beads.py), models dir


def data(p):
    """movie / locs path: absolute, or relative to the data root"""
    return config.resolve(p, 'data')


def out(*f):
    """default output path under <out>/insitu"""
    return os.path.join(config.out_dir('insitu'), *f)


def cam_maps(calib_path, cam):
    """camera-map path of a calibration json (relative paths are relative to the json, as in sim.simulator)"""
    mp = cam['maps']
    return mp if os.path.isabs(mp) else os.path.join(os.path.dirname(os.path.abspath(calib_path)), mp)
