# Simulation table (D_v2psf_wsm3d, SIMCODE = m1809v2; 60 000 frames)

n = localizations in the FRC set (disjoint 6-frame windows, σ < 0.15 px; SIMCODE p >= 0.63, calibrated σ); GT-FRC = FRC(reconstruction, ground truth), 1/7 sustained crossing (stays below 1/7 for 0.01 nm^-1); frames to 30 nm GT-FRC ("-" = not reached in 60 000 frames, values <= 250 = reached in the first 250 frames); fair FRC = half-split FRC; Jaccard at 10 nm; false positive = a localization with no emitter ON in its window within 3 own σ.

| density | method | n | GT-FRC (nm) | frames to 30 nm | fair FRC (nm) | Jaccard 10 nm | false positive fraction |
|---|---|---|---|---|---|---|---|
| 0.5 | SMLM | 21458 | 69.9 | - | 88.1 | 0.698 | 0.321 |
| 0.5 | SIMFLUX (estimated pattern) | 8335 | 30.3 | - | 81.1 | 0.804 | 0.135 |
| 0.5 | SIMFLUX (true pattern) | 8367 | 30.3 | - | 81.1 | 0.805 | 0.128 |
| 0.5 | DECODE (retrained) | 22785 | 49.3 | - | 71.0 | 0.797 | 0.073 |
| 0.5 | SIMCODE | 48467 | 26.6 | 36430 | 40.1 | 0.913 | 0.057 |
| 1 | SMLM | 33214 | 74.7 | - | 91.5 | 0.683 | 0.492 |
| 1 | SIMFLUX (estimated pattern) | 9212 | 33.1 | - | 79.3 | 0.812 | 0.208 |
| 1 | SIMFLUX (true pattern) | 9280 | 31.6 | - | 79.3 | 0.814 | 0.198 |
| 1 | DECODE (retrained) | 31768 | 54.9 | - | 68.7 | 0.788 | 0.109 |
| 1 | SIMCODE | 91167 | 25.8 | 19328 | 34.1 | 0.905 | 0.060 |
| 2 | SMLM | 45469 | 76.1 | - | 88.1 | 0.658 | 0.644 |
| 2 | SIMFLUX (estimated pattern) | 6596 | 64.7 | - | 77.6 | 0.755 | 0.383 |
| 2 | SIMFLUX (true pattern) | 6517 | 64.7 | - | 74.4 | 0.757 | 0.366 |
| 2 | DECODE (retrained) | 36969 | 66.7 | - | 48.1 | 0.777 | 0.168 |
| 2 | SIMCODE | 152142 | 27.1 | 36768 | 34.9 | 0.893 | 0.061 |
| 4 | SMLM | 49860 | 85.0 | - | 75.8 | 0.641 | 0.689 |
| 4 | SIMFLUX (estimated pattern) | 834 | 131.4 | - | 147.2 | 0.239 | 0.771 |
| 4 | SIMFLUX (true pattern) | 3533 | 76.1 | - | 86.1 | 0.616 | 0.641 |
| 4 | DECODE (retrained) | 41585 | 69.9 | - | 46.6 | 0.770 | 0.223 |
| 4 | SIMCODE | 199932 | 30.1 | - | 34.1 | 0.885 | 0.058 |
