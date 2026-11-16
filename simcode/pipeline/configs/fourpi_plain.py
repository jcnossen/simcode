"""Plain 4Pi (4Pi-SMS, setup fourpi_plain, simcode/setups/fourpi.py): the public 4Pi-SMS example data set of Li & Ries
(Zenodo 4022827, CC BY 4.0; Nup96-SNAP / AF647 in U2OS, TDE index-matching buffer), Cell11: 4 dcimg files x 3000
single frames (50 ms), 2048 x 168 px camera frames with 4 interference channels side by side (ch2 / ch3 mirrored
left-right), 192 x 168 px channel crops. Derived in out/fourpi_refactor/public_4pi/STATUS.md: layout, camera (gain 0.455
e/ADU from their own 4-channel sum, offset 103 ADU from the dark columns, read noise about 1 e), 128 nm px, the in-situ
registration of file 000 (used for every file: registration mode 'file'), the 4Pi PSF fitted to their bead stack
(PSF4Pi, NA 1.35 / n 1.406 / lambda 0.67 assumed; astigmatism scaled to the bead data: analysis/fourpi_plain/psf_astig.py).
Work directory (models, PSF, map, outputs): <out>/plain4pi (SIMCODE_OUT / config key out).

  python -m simcode.pipeline simcode/pipeline/configs/fourpi_plain.py --only fourpi_plain_Cell11
  python -m simcode.processing.fourpi simcode/pipeline/configs/fourpi_plain.py --files 0
Training (fourpi_train section): analysis/fourpi/train_det4pi.py --acq <this file>, analysis/fourpi/train_head4pi_w.py --acq <this file>
"""
from simcode import config as K
from simcode.util.config_dict import ConfigDict

W4 = K.out_dir('plain4pi', make=False)
DATA = '/mnt/data/dmdsf/public_4pi_sms/zenodo_4022827/extracted'
X0, MIRROR = (90, 460, 1399, 1768), ('none', 'none', 'lr', 'lr')
CW, CH = 192, 168


def acquisition(cell, movies):
    return ConfigDict(
        name=f'fourpi_plain_{cell}', setup='fourpi_plain',
        movie=f'{DATA}/{movies}', out=f'{W4}/pipeline/{cell}',
        camera=dict(gain_e_per_adu=0.455, readnoise_e=1.0, offset_adu=103.0),
        optics=dict(pixel_um=0.128, NA=1.35, wavelength_um=0.67, n_imm=1.406, n_med=1.406),
        channels=[dict(name=f'c{i}', crop=[x, 0, CW, CH], mirror=m) for i, (x, m) in enumerate(zip(X0, MIRROR))],
        registration=dict(mode='file', map=f'{W4}/reg/reg_insitu_{cell}_000.json'),
        fourpi=dict(frame_dt_s=0.05, xlim=[4, 188], file_field=-2, z_period_um=0.24,
                    psf=f'{W4}/psf/psf4pi_{cell}_astig.json',
                    detector=f'{W4}/models/det/final.pt', emitter=f'{W4}/models/emitter.pt',
                    tile=12.0, tblk=200, coh=1.0, min_photons=150.0, sb=1.0,
                    zstarts=[0.0], unwrap_r_px=8.0),           # the astigmatic prior's 4Pi period, then the structure's (STATUS.md)
        fourpi_train=dict(bg=[24.1, 19.6, 23.2, 20.0], emitter_size=128, emitter_scenes=16, emitter_steps=12000,
                          reg_shift_sd=0.02, reg_lin_sd=0.01),
        gpus=[0])


configs = [acquisition('Cell11', 'Cell11/Cell11_642_*_000.dcimg')]
