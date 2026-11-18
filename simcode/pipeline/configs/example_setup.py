"""Example of acquisitions defined in Python: one base config for the DMD-SIMFLUX setup and the movies derived from it (here
one, the 1809/2 movie with the example model m1809; add more with acquisition(name, movie, model)).

  python -m simcode.pipeline simcode/pipeline/configs/example_setup.py --show            print the resolved configs, run nothing
  python -m simcode.pipeline simcode/pipeline/configs/example_setup.py                   run the acquisitions, one after the other
  python -m simcode.pipeline simcode/pipeline/configs/example_setup.py --only 1809_2     one of them

  from simcode.pipeline import LocalizationPipeline
  pipes = LocalizationPipeline.from_file('simcode/pipeline/configs/example_setup.py')       # [LocalizationPipeline, ...]

The file is ordinary Python; the pipeline reads `configs` (a list) or `config` (one). Every config is checked against the
schema of simcode/pipeline/acquisition_config.py (unknown keys are errors, defaults are filled in), exactly as a yaml file is.
"""
import os
from simcode.util.config_dict import ConfigDict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = '/mnt/data/dmdsf/bn_dnapaint'
OUT = '/mnt/data/dmdsf/simcode2_out/auto'
GAIN = 0.56                                # e- / ADU, used for the movies and the bead stacks

# ---- the setup: TIRF-SIM with a DMD, NA 1.49 oil, Zyla 4.2 sCMOS, 642 nm excitation, 108.33 nm pixels
base = ConfigDict(
    camera=dict(type='scmos', gain_e_per_adu=GAIN, offset_adu=99.3,
                readnoise_from=f'{REPO}/models/insitu/camera_maps_0.5nM.npz'),       # read-noise map of the same sensor
    optics=dict(pixel_um=0.10833, NA=1.49, wavelength_um=0.70, n_imm=1.518, n_med=1.33, tirf_depth_um=0.082),
    pattern=dict(pattern_frames=[[0, 1, 2], [3, 4, 5]], pitch_px=[1.6, 2.8]),
    beads=dict(stacks=[f'{DATA}/PSF_calib/raw/zstack{i}-1.tif' for i in range(2, 8)],   # the setup's bead z-stacks
               sign=-1, holdout=7, offset_adu=106.34, gain_e_per_adu=GAIN),
    psf=dict(init=['zero', 'f0.1', 'f0.2', 'f0.3', 'beads']),
    gpus=[1, 2, 0],
).freeze()                                 # read-only: acquisitions are derived with .updated(), the base cannot change by accident


def acquisition(name, movie, model):
    """one movie of the setup, localized with an existing model of simcode/models.json
    (train.model=None instead would train a PSF-matched model 'a<name>' in stage e)"""
    return base.updated(name=name, movie=f'{DATA}/{movie}', out=f'{OUT}/{name}',
                        train=dict(model=model), process=dict(model=model))


configs = [
    acquisition('1809_2',  '1809/cover1/2/simflux_5.5ms_500mW.tif',            'm1809'),
]

# values derived in code: the camera maps use at most the frames the spot finder looks at (its spot mask ends there)
SPOT_FRAMES = None                         # null: the whole movie (the default); e.g. 30000 for the first 30000 frames only
configs = [c.updated(spots=dict(maxframe=SPOT_FRAMES), camera=dict(map_frames=20000 if SPOT_FRAMES is None else min(20000, SPOT_FRAMES)))
           for c in configs]
