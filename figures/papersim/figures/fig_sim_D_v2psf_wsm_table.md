# Simulation table (500 detected photons per 6-frame window, background 2.5 photons/px/frame; 60 000 frames)

n = localizations in the FRC set (disjoint 6-frame windows, σ < 0.15 px); GT-FRC = FRC(reconstruction, ground truth), 1/7; frames to 30 nm GT-FRC ("-" = not reached in 60 000 frames, values <= 250 = reached in the first 250 frames); fair FRC = half-split FRC; Jaccard at 10 nm; false positive = a localization with no emitter ON in its window within 3 own σ. GT-FRC, frames to 30 nm and fair FRC: Resolution = 1 / the spatial frequency at which the FRC (5-sample moving average) falls below 1/7 and stays below it (for at least 0.01 nm⁻¹; sustained crossing): the labels sit on a shell about 25-30 nm from the microtubule axis, so the ground-truth spectrum has a null near 1/(60-65 nm) where every curve dips and recovers; that dip is not taken as the resolution.

| density | method | n | GT-FRC (nm) | frames to 30 nm | fair FRC (nm) | Jaccard 10 nm | false positive fraction |
|---|---|---|---|---|---|---|---|
| 0.5 | SMLM | 14494 | 74.7 | - | 87.1 | 0.662 | 0.334 |
| 0.5 | SIMFLUX (estimated pattern) | 6416 | 33.6 | - | 83.6 | 0.761 | 0.149 |
| 0.5 | SIMFLUX (true pattern) | 6448 | 33.3 | - | 83.6 | 0.766 | 0.135 |
| 0.5 | DECODE (retrained) | 14358 | 45.6 | - | 74.7 | 0.773 | 0.073 |
| 0.5 | SIMCODE | 23014 | 30.3 | - | 42.0 | 0.888 | 0.025 |
| 1 | SMLM | 24071 | 78.8 | - | 99.2 | 0.658 | 0.493 |
| 1 | SIMFLUX (estimated pattern) | 7269 | 36.1 | - | 83.0 | 0.778 | 0.231 |
| 1 | SIMFLUX (true pattern) | 7280 | 35.8 | - | 83.0 | 0.780 | 0.211 |
| 1 | DECODE (retrained) | 19826 | 66.7 | - | 73.4 | 0.769 | 0.099 |
| 1 | SIMCODE | 33854 | 29.5 | 33235 | 35.8 | 0.914 | 0.028 |
| 2 | SMLM | 37462 | 80.3 | - | 86.1 | 0.641 | 0.628 |
| 2 | SIMFLUX (estimated pattern) | 5906 | 45.6 | - | 75.1 | 0.730 | 0.406 |
| 2 | SIMFLUX (true pattern) | 5849 | 42.1 | - | 75.1 | 0.735 | 0.373 |
| 2 | DECODE (retrained) | 22803 | 72.2 | - | 52.2 | 0.759 | 0.148 |
| 2 | SIMCODE | 38434 | 31.6 | - | 35.2 | 0.911 | 0.039 |
| 4 | SMLM | 48675 | 90.3 | - | 86.1 | 0.628 | 0.665 |
| 4 | SIMFLUX (estimated pattern) | 589 | 154.8 | - | 238.0 | 0.171 | 0.762 |
| 4 | SIMFLUX (true pattern) | 3613 | 76.1 | - | 80.0 | 0.627 | 0.628 |
| 4 | DECODE (retrained) | 24042 | 72.2 | - | 47.1 | 0.749 | 0.198 |
| 4 | SIMCODE | 30738 | 33.9 | - | 40.5 | 0.890 | 0.057 |
