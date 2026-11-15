"""The 4Pi-SIMFLUX acquisitions (setup fourpi, simcode/setups/fourpi.py): the MT (Cell03) and NPC (Cell07) movies of
/mnt/data/4pi-sf (SIMCODE_4PI_DATA / config key data4pi), 4 interference channels of 360 x 200 px at the raw columns 217,
652, 1420, 1855 (channels 2 and 3 mirrored left-right), gain 0.23 e/ADU, read noise 1.14 e, 128 nm px, 5.575 ms frames;
camera offset maps, the reference SIM pattern, the 4Pi PSF model and the networks in the 4Pi work directory
(<out>/4pi, config key fourpi). delta_deg = per-angle 4Pi phase offset (SMLM_REPORT 5.2).

  python -m simcode.pipeline simcode/pipeline/configs/fourpi.py --only fourpi_MT
  from simcode.setups.fourpi import acquisition; c = acquisition('MT')
"""
from simcode import config as K
from simcode.util.cfgdict import ConfigDict

P4 = K.root('fourpi')
DATA = K.root('data4pi') or '/mnt/data/4pi-sf'
X0, MIRROR = (217, 652, 1420, 1855), ('none', 'none', 'lr', 'lr')


def acquisition(cell, movies, zstack, bead_align, delta_deg):
    return ConfigDict(
        name=f'fourpi_{cell}', setup='fourpi',
        movie=f'{DATA}/{movies}', out=f'{P4}/pipeline/{cell}',
        camera=dict(gain_e_per_adu=0.23, readnoise_e=1.14, offset_map=f'{P4}/smlm/camera/camera_{cell}.npz'),
        optics=dict(pixel_um=0.128, NA=1.35, wavelength_um=0.60, n_imm=1.406, n_med=1.406),
        channels=[dict(name=f'c{i}', crop=[x, 0, 360, 200], mirror=m) for i, (x, m) in enumerate(zip(X0, MIRROR))],
        registration=dict(mode='insitu'),
        fourpi=dict(delta_deg=delta_deg, zstack=f'{DATA}/{zstack}', bead_align=f'{DATA}/{bead_align}',
                    psf=f'{P4}/psf4pi/NPC_f_dxy15.json', pattern_ref=f'{P4}/smlm/diag/pattern_{cell}.json',
                    detector=f'{P4}/models/det4pi/final.pt', emitter='v10', coh=1.56),
        gpus=[1, 2])


configs = [
    acquisition('MT', 'MT/Cell03_MT/Cell03_561_000_*.dcimg', 'MT/Cell02_beads_z_stack/Cell02_561_*_000.dcimg',
                'MT/Cell01_beads_align/Cell01_561_000_000.dcimg', -10.4),
    acquisition('NPC', 'NPC/movies/Cell07_561_000_*.dcimg', 'NPC/zstack/Cell06_561_*_000.dcimg',
                'NPC/beads_align/Cell05_561_000_000.dcimg', -13.9),
]
