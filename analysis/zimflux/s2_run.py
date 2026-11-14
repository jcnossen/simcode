"""Run a simcode.insitu module (fit_insitu2, psf_heldout) with the ZIMFLUX optics: those modules build
VectorialPSF(R, NA, wavelength=0.70, n_med) with the package defaults for the pixel (108.33 nm), n_imm (1.518) and the
Zernike set (Noll 5-21). This thin wrapper substitutes a VectorialPSF subclass with pixel 65 nm, lambda 715 nm,
n_imm 1.52 and Noll 5-22 (the zimflux set includes Z(6,0) = Noll 22) before the module is imported. Nothing else changes.
usage: python s2_run.py <module> [module args...]   e.g. s2_run.py simcode.insitu.fit_insitu2 --tracks T.npz ..."""
import sys, runpy
sys.path.insert(0, '/home/jc/simcode2')
import simcode.psf.vectorial as V

_Base = V.VectorialPSF


class ZimfluxPSF(_Base):
    def __init__(self, roisize, pixelsize=None, NA=1.49, n_imm=None, n_med=1.33, wavelength=None, zernike_indices=None, **kw):
        super().__init__(roisize, pixelsize=0.065, NA=NA, n_imm=1.52, n_med=n_med, wavelength=0.715,
                         zernike_indices=tuple(range(5, 23)), **kw)


V.VectorialPSF = ZimfluxPSF
mod = sys.argv[1]; sys.argv = [mod] + sys.argv[2:]
runpy.run_module(mod, run_name='__main__')
