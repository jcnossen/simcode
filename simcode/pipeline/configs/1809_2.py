"""1809_2.yaml as a Python config (same resolved config; tests/test_config_dict.py compares the two).
DMD-SIMFLUX TIRF-SIM (NA 1.49 oil, Zyla 4.2 sCMOS, 642 nm, 108.33 nm px); the paper's low-density movie 1809/cover1/2.
Reference calibration (production): models/insitu/calib_1809_2_psf_v2.json, camera_maps_1809_2_v2.npz.

  python -m simcode.pipeline simcode/pipeline/configs/1809_2.py
"""
from simcode.util.config_dict import ConfigDict

DATA = '/mnt/data/dmdsf/bn_dnapaint'
GAIN = 0.56                                # SIMCODE paper value (camera_gain.py on 0.5 nM: 0.552); the measured value is reported

config = ConfigDict(
    name='1809_2',
    movie=f'{DATA}/1809/cover1/2/simflux_5.5ms_500mW.tif',
    out='/mnt/data/dmdsf/simcode2_out/auto/1809_2',
    beads=dict(                            # the setup's bead z-stacks (fitted by simcode.psf.beads)
        stacks=[f'{DATA}/PSF_calib/raw/zstack{i}-1.tif' for i in range(2, 8)],
        sign=-1, holdout=7, offset_adu=106.34, gain_e_per_adu=GAIN),
    camera=dict(
        gain_e_per_adu=GAIN, offset_adu=99.3,
        readnoise_from='insitu/camera_maps_0.5nM.npz',     # same sensor (as the reference calibration)
        map_frames=20000),
    psf=dict(init=['zero', 'f0.1', 'f0.2', 'f0.3', 'beads']),   # multi-start (focus-staggered; focus / z gauge), lowest base-fit NLL wins
    process=dict(model='m1809'),
    gpus=[1, 2, 0],
)
