# Simulation table (experimental photons and background of 1809/2 (median 621 detected photons per 6-frame window, 1.95 photons/px/frame); 60 000 frames)

n = localizations in the FRC set (disjoint 6-frame windows, σ < 0.15 px); GT-FRC = FRC(reconstruction, ground truth), 1/7; frames to 30 nm GT-FRC ("-" = not reached in 60 000 frames, values <= 250 = reached in the first 250 frames); fair FRC = half-split FRC; Jaccard at 10 nm; false positive = a localization with no emitter ON in its window within 3 own σ. GT-FRC, frames to 30 nm and fair FRC: Resolution = 1 / the spatial frequency at which the FRC (5-sample moving average) falls below 1/7 and stays below it (for at least 0.01 nm⁻¹; sustained crossing): the labels sit on a shell about 25-30 nm from the microtubule axis, so the ground-truth spectrum has a null near 1/(60-65 nm) where every curve dips and recovers; that dip is not taken as the resolution.

| density | method | n | GT-FRC (nm) | frames to 30 nm | fair FRC (nm) | Jaccard 10 nm | false positive fraction |
|---|---|---|---|---|---|---|---|
| 0.5 | SMLM | 31613 | 42.5 | - | 85.0 | 0.732 | 0.278 |
| 0.5 | SIMFLUX (estimated pattern) | 16071 | 23.1 | 21782 | 39.2 | 0.867 | 0.126 |
| 0.5 | SIMFLUX (true pattern) | 16049 | 19.6 | 22605 | 38.4 | 0.870 | 0.103 |
| 0.5 | DECODE (retrained) | 45988 | 33.3 | - | 43.3 | 0.837 | 0.046 |
| 0.5 | SIMCODE | 59767 | 17.9 | 15572 | 32.6 | 0.946 | 0.020 |
| 1 | SMLM | 45291 | 40.1 | - | 88.1 | 0.725 | 0.414 |
| 1 | SIMFLUX (estimated pattern) | 18761 | 22.0 | 31907 | 34.3 | 0.876 | 0.209 |
| 1 | SIMFLUX (true pattern) | 18655 | 20.5 | 16759 | 41.0 | 0.882 | 0.165 |
| 1 | DECODE (retrained) | 74186 | 32.1 | - | 46.6 | 0.830 | 0.073 |
| 1 | SIMCODE | 109086 | 18.2 | 8229 | 29.7 | 0.950 | 0.025 |
| 2 | SMLM | 50799 | 72.2 | - | 80.6 | 0.698 | 0.571 |
| 2 | SIMFLUX (estimated pattern) | 14633 | 31.2 | - | 41.0 | 0.843 | 0.340 |
| 2 | SIMFLUX (true pattern) | 14366 | 21.8 | 46664 | 40.1 | 0.852 | 0.277 |
| 2 | DECODE (retrained) | 102546 | 31.9 | - | 43.7 | 0.813 | 0.120 |
| 2 | SIMCODE | 172958 | 19.0 | 6414 | 26.6 | 0.945 | 0.032 |
| 4 | SMLM | 38173 | 71.1 | - | 67.7 | 0.673 | 0.700 |
| 4 | SIMFLUX (estimated pattern) | 6946 | 56.3 | - | 73.6 | 0.720 | 0.545 |
| 4 | SIMFLUX (true pattern) | 6838 | 37.7 | - | 65.7 | 0.729 | 0.485 |
| 4 | DECODE (retrained) | 126700 | 35.0 | - | 41.2 | 0.787 | 0.209 |
| 4 | SIMCODE | 222591 | 19.1 | 5137 | 23.9 | 0.938 | 0.045 |
