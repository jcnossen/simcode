"""In-situ calibration of the PSF, camera and illumination pattern from the DNA-PAINT movie itself (recipe: README.md
in this directory). Default inputs are relative to the data root, default outputs go to <out>/insitu (the output directory keeps its name)."""
import os
from simcode import config

MOVIE_05 = 'dnapaint/0.5nM/cover1/1/simflux_5.5ms_500mW.tif'                     # 0.5 nM movie (data root)
SMLM_05 = 'dnapaint/0.5nM/cover1/1/results/simflux_5.5ms_500mW/smlm.hdf5'        # its SIMCODE-pipeline 'smlm' locs (Picasso hdf5)


def data(p):
    """movie / locs path: absolute, or relative to the data root"""
    return config.resolve(p, 'data')


def out(*f):
    """default output path under <out>/insitu"""
    return os.path.join(config.out_dir('insitu'), *f)
