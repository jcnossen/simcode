"""4Pi PSF model until torch-psf ships it (torch_psf.FourPiPSF / FourPiSplinePSF on the torch-psf branch multichannel,
bit-identical; SIMCODE switches when it pins that version): psf4pi.py (coherent two-arm vectorial 4Pi PSF), lut4pi.py
(its spline table with the analytic 4Pi phase). The rest of the 4Pi code: channels/, setups/fourpi.py, processing/fourpi*.py,
solve/fourpi.py, emitter/fourpi.py, simulate/fourpi_patches.py, simulate/multichannel.py."""
