| quantity | value | unit | note |
|---|---|---|---|
| PSF model | calib_1809_2_psf_v2.json |  | in-situ vectorial PSF fit (Zernike Noll 5-21) |
| rms aberration (Noll >= 5) | 68 | mλ |  |
| focal plane | 200 | nm | above the coverslip |
| emitter height, median | 112 | nm | pass-1 z of the plotted SIMCODE set |
| emitter height, q25-q75 | 79-145 | nm | q05-q95 39-195 nm |
| rows without z | 0.09 | % | no pass-1 partner |
| signal, median (PSF intensity parameter) | 943 | photons / 6-frame window | network photons of the plotted set |
| collected fraction at the median height | 0.73 |  | 41 x 41 px PSF image |
| signal, median (detected) | 686 | photons / 6-frame window | intensity parameter x collected fraction |
| background, median | 1.89 | photons / px / frame | pass-1 network bg, p >= 0.63, disjoint windows, crop |
| camera read noise | 1.33 | e- rms | mean variance 1.78 e^2 over the crop |
| camera gain | 0.4256 | e- / ADU |  |
| pattern pitch | 224.3 | nm | mean of the 2 orientations, estimated from this run (pattern_est.pt); per orientation 224.3 / 224.3 nm |
| modulation depth | 0.91 |  | mean of the 2 orientations, estimated from this run (pattern_est.pt); per orientation 0.90 / 0.92 |
| pattern | 2 x 3 | orientations x phase steps | relative step intensities of the run included |
| CRLB SMLM x / y / z | 11.7 / 11.7 / 53 | nm | 3D fit (x, y, z, I, bg) |
| CRLB SMLM x / y, z known | 11.7 / 11.5 | nm | 2D fit at the emitter height |
| CRLB SIMFLUX x / y / z | 4.5 / 4.6 / 46 | nm | 3D fit (x, y, z, I, bg) |
| CRLB SIMFLUX x / y, z known | 4.5 / 4.6 | nm | 2D fit at the emitter height |
| SIMCODE rows in the figure (n_frc) | 796 509 |  | fair-FRC set |
