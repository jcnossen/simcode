| quantity | value | unit | note |
|---|---|---|---|
| PSF model | calib_1809_2_psf_v2.json |  | in-situ vectorial PSF fit (Zernike Noll 5-21) |
| rms aberration (Noll >= 5) | 68 | mλ |  |
| focal plane | 200 | nm | above the coverslip |
| emitter height, median | 117 | nm | pass-1 z of the plotted SIMCODE set |
| emitter height, q25-q75 | 85-150 | nm | q05-q95 44-209 nm |
| rows without z | 0.40 | % | no pass-1 partner |
| signal, median (PSF intensity parameter) | 784 | photons / 6-frame window | network photons of the plotted set |
| collected fraction at the median height | 0.72 |  | 41 x 41 px PSF image |
| signal, median (detected) | 565 | photons / 6-frame window | intensity parameter x collected fraction |
| background, median | 1.92 | photons / px / frame | pass-1 network bg, p >= 0.63, disjoint windows, crop |
| camera read noise | 1.33 | e- rms | mean variance 1.78 e^2 over the crop |
| camera gain | 0.4256 | e- / ADU |  |
| pattern pitch | 224.3 | nm | mean of the 2 orientations, estimated from this run (pattern_est.pt); per orientation 224.3 / 224.3 nm |
| modulation depth | 0.91 |  | mean of the 2 orientations, estimated from this run (pattern_est.pt); per orientation 0.90 / 0.93 |
| pattern | 2 x 3 | orientations x phase steps | relative step intensities of the run included |
| CRLB SMLM x / y / z | 13.7 / 13.6 / 63 | nm | 3D fit (x, y, z, I, bg) |
| CRLB SMLM x / y, z known | 13.6 / 13.4 | nm | 2D fit at the emitter height |
| CRLB SIMFLUX x / y / z | 5.2 / 5.4 / 54 | nm | 3D fit (x, y, z, I, bg) |
| CRLB SIMFLUX x / y, z known | 5.2 / 5.4 | nm | 2D fit at the emitter height |
| SIMCODE rows in the figure (n_frc) | 1 349 239 |  | fair-FRC set |
