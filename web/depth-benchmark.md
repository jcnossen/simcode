# Dense-bead modulation-depth benchmark

## Default pixel photometry at 240 nm

Both 400 × 400 TIFFs contain 40 exposures of 40 nm beads. The random slide has 3,000 beads. The annulus has 495. The table averages the quadratic depth maps at the simulated bead positions, so the empty centre of the annulus contributes no samples. Truth was used for scoring only.

| Movie | Axis | Truth | Independent Python fit | WebGPU upload |
|---|---|---:|---:|---:|
| Random slide | A | 0.83877 | 0.84298 | 0.84232 |
| Random slide | B | 0.88053 | 0.88527 | 0.88538 |
| Annulus | A | 0.83830 | 0.83437 | 0.82235 |
| Annulus | B | 0.87938 | 0.87478 | 0.86416 |

To isolate the photometry port, Python also started from the browser's fitted pupil, focus, initial XY, drift and intensities. The two implementations then differed by less than 0.000006 at every evaluated bead. The independent runs include differences introduced in the initial optical and bead fits.

TIFF SHA-256 digests are `94ccf7aa9e702e756313db1db22a7dcd06be8f2d2feff0b30c7d8bfb9551770e` (random slide) and `1844dffef7f370dc06b92dc9cb4689ad519f67ceba2f1d56af2b9b5980c43cf5` (annulus).

## Earlier 200 nm benchmark

The comparison uses one 400 × 400 field containing 12,000 stationary 40 nm beads at independent uniform XY positions, simulated with the web example (LCG seed 557), 108.33 nm camera pixels and a 200 nm illumination pitch. Each 100-frame movie uses two axes and three equally spaced phases per axis. The two axial excursions share the bead locations, brightnesses, aberrations and illumination parameters. Within each column, every recovery uses the same noisy TIFF.

## Where the attenuation enters

Values are fitted modulation depths at the field center, in axis 0 / axis 1 order. The first movie reproduces the earlier ±500 nm example and uses the previous pupil estimate. The second uses the revised pupil estimate with the new ±200 nm example. The comparisons within a column hold the fitted bead catalog, per-phase brightness initialization and the number of photon updates fixed. True drift substitutes all three coordinates, including absolute focus relative to the bead plane.

| Inputs to bead photometry | ±500 nm movie | ±200 nm movie |
|---|---:|---:|
| Fitted pupil, fitted drift | 0.224 / 0.355 | 0.451 / 0.523 |
| True pupil, fitted drift | 0.231 / 0.364 | 0.464 / 0.536 |
| Fitted pupil, true drift | 0.452 / 0.489 | 0.516 / 0.554 |
| True pupil, true drift | 0.465 / 0.502 | 0.527 / 0.569 |
| Analytic per-bead intensities, true positions and drift | 0.840 / 0.880 | 0.840 / 0.880 |
| Image-power depth | 0.848 / 0.880 | 0.846 / 0.876 |
| Simulated illumination depth | 0.840 / 0.880 | 0.840 / 0.880 |

The ideal-intensity control includes the finite-bead attenuation in the simulated photons and reverses it in the depth estimate. The fitted catalog had 24,373 candidates for the original movie and 19,790 for the revised movie. The count alone does not identify which candidates correspond to real beads.

**Photon-sharing check.** On two six-exposure cycles of the ±500 nm movie, we fixed the true bead positions, pupil, motion and 25-photon background, and initialized every bead to 4,500 photons. Increasing the multiplicative photon updates from 12 to 96 raised median per-cycle bead depth from 0.36 / 0.37 to 0.52 / 0.54. Shared pixels and incomplete intensity separation attenuate the traces even with correct optical parameters. This check uses a per-cycle median, distinct from the field-center surface fits in the table.

## Revised estimate and limits

SIMCODE measures the DC and first-harmonic spatial power in local tiles of the camera movie. A difference of Gaussians with standard deviations of 1.15 and 2.6 camera pixels selects bead texture. Each complete phase cycle contributes a shot-noise-corrected covariance matrix. The demodulation weights use the chosen or individually estimated phase steps. For independent bead positions, the ratio of harmonic to DC power estimates squared modulation depth after the sphere correction. The JSON records the method, contributing tile count, and bead-trace depth coefficients separately.

A held-out field with seed 123, independent aberrations, ±200 nm drift, and weaker true depths of 0.350 / 0.600 yielded 0.348 / 0.589 in the full application. Intensity traces alone gave 0.223 / 0.352. These tests cover randomly dispersed, nonblinking beads. Clustering, rapid motion within a phase cycle or changing backgrounds can bias the image-power estimate.

TorchPSF also fits the asymmetric third-order spatial correlation of the image, refines the focus between sampled planes, and includes registration blur when comparing bead profiles. On 11 sampled exposures of the ±200 nm seed-557 movie, normalized point-PSF image error fell from 39.6% to 23.2% (square root of total squared pixel error divided by total squared truth intensity). Signed focus remains ambiguous without axial calibration. The held-out field had 295 nm signed axial RMS error despite its accurate depth estimate.
