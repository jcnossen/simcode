"""Plain 4Pi Cell11 with the in-situ PSF (analysis/fourpi_plain/insitu4pi.py, p4spline_refine.py): the acquisition of
fourpi_plain.py with fourpi.psf = the FourPiSplinePSF refined on the Cell11 spots (refined from the in-situ FourPiPSF fit
v2; psf json + knots in <out>/plain4pi/p4psf/psf) and the emitter network fine-tuned on it (<out>/plain4pi/p4psf/models).
The detector and the registration are those of fourpi_plain.py.

  training: analysis/fourpi/train_head4pi_w.py --acq simcode/pipeline/configs/fourpi_plain_insitu.py --init <fourpi_plain emitter.pt>
  processing (from the crop cache): analysis/fourpi_plain/p4process.py --cfg simcode/pipeline/configs/fourpi_plain_insitu.py --out <run dir>
"""
from simcode import config as K
from simcode.pipeline.configs.fourpi_plain import acquisition as plain

W4 = K.out_dir('plain4pi', make=False)
P4 = f'{W4}/p4psf'


def acquisition(cell, movies, psf='psf4pi_Cell11_spline_r3.json', emitter='emitter_r3.pt'):
    c = plain(cell, movies)
    c['name'] = f'fourpi_plain_insitu_{cell}'
    c['out'] = f'{P4}/pipeline/{cell}'
    c['fourpi']['psf'] = f'{P4}/psf/{psf}'
    c['fourpi']['emitter'] = f'{P4}/models/{emitter}'
    c['fourpi_train']['emitter_steps'] = 4000           # fine-tuning from the fourpi_plain network
    return c


configs = [acquisition('Cell11', 'Cell11/Cell11_642_*_000.dcimg')]
