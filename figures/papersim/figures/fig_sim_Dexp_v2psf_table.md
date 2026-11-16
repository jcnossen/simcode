# Simulation table (experimental photons and background of 1809/2 (median 621 detected photons per 6-frame window, 1.95 photons/px/frame); 60 000 frames)

n = localizations in the FRC set (disjoint 6-frame windows, σ < 0.15 px); GT-FRC = FRC(reconstruction, ground truth), 1/7; frames to 30 nm GT-FRC ("-" = not reached in 60 000 frames, values <= 250 = reached in the first 250 frames); fair FRC = half-split FRC; Jaccard at 10 nm; false positive = a localization with no emitter ON in its window within 3 own σ. GT-FRC, frames to 30 nm and fair FRC: Resolution = 1 / the spatial frequency at which the FRC (5-sample moving average) falls below 1/7 and stays below it (for at least 0.01 nm⁻¹; sustained crossing): the labels sit on a shell about 25-30 nm from the microtubule axis, so the ground-truth spectrum has a null near 1/(60-65 nm) where every curve dips and recovers; that dip is not taken as the resolution.

| density | method | n | GT-FRC (nm) | frames to 30 nm | fair FRC (nm) | Jaccard 10 nm | false positive fraction |
|---|---|---|---|---|---|---|---|
| 0.5 | SMLM | 39600 | 33.3 | - | 47.6 | 0.455 | 0.324 |
| 0.5 | SIMFLUX (estimated pattern) | 19050 | 18.0 | 3831 | 22.3 | 0.883 | 0.172 |
| 0.5 | SIMFLUX (true pattern) | 18881 | 15.0 | 2716 | 21.9 | 0.909 | 0.117 |
| 0.5 | DECODE (retrained) | 46832 | 24.4 | 11608 | 31.4 | 0.630 | 0.057 |
| 0.5 | SIMCODE | 61862 | 15.1 | 1349 | 20.6 | 0.878 | 0.023 |
| 1 | SMLM | 53753 | 35.5 | - | 45.8 | 0.416 | 0.468 |
| 1 | SIMFLUX (estimated pattern) | 20105 | 21.6 | 6460 | 21.5 | 0.826 | 0.308 |
| 1 | SIMFLUX (true pattern) | 19702 | 14.7 | 2363 | 21.1 | 0.890 | 0.187 |
| 1 | DECODE (retrained) | 73009 | 23.3 | 7808 | 30.1 | 0.590 | 0.084 |
| 1 | SIMCODE | 109365 | 15.0 | 758 | 18.5 | 0.855 | 0.028 |
| 2 | SMLM | 55099 | 40.9 | - | 47.0 | 0.376 | 0.614 |
| 2 | SIMFLUX (estimated pattern) | 14406 | 32.4 | - | 22.0 | 0.723 | 0.555 |
| 2 | SIMFLUX (true pattern) | 13619 | 16.2 | 3520 | 21.1 | 0.854 | 0.325 |
| 2 | DECODE (retrained) | 99394 | 24.1 | 6173 | 27.9 | 0.541 | 0.138 |
| 2 | SIMCODE | 166459 | 15.3 | 509 | 18.3 | 0.834 | 0.036 |
| 4 | SMLM | 34227 | 47.6 | - | 38.0 | 0.374 | 0.706 |
| 4 | SIMFLUX (estimated pattern) | 874 | 117.2 | - | 89.2 | 0.227 | 0.866 |
| 4 | SIMFLUX (true pattern) | 5769 | 20.3 | 13206 | 18.9 | 0.770 | 0.516 |
| 4 | DECODE (retrained) | 123550 | 24.5 | 11097 | 28.1 | 0.491 | 0.227 |
| 4 | SIMCODE | 206192 | 15.4 | 528 | 17.8 | 0.813 | 0.048 |
