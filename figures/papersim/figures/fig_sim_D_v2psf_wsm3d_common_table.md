# Simulation table (500 detected photons per 6-frame window, background 2.5 photons/px/frame; 60 000 frames)

n = localizations in the FRC set (disjoint 6-frame windows, σ < 0.15 px; SIMCODE p >= 0.63, posterior σ); GT-FRC = FRC(reconstruction, ground truth), 1/7; frames to 30 nm GT-FRC ("-" = not reached in 60 000 frames, values <= 250 = reached in the first 250 frames); fair FRC = half-split FRC; Jaccard at 10 nm; false positive = a localization with no emitter ON in its window within 3 own σ. GT-FRC, frames to 30 nm and fair FRC: Resolution = 1 / the spatial frequency at which the FRC (5-sample moving average) falls below 1/7 and stays below it (for at least 0.01 nm⁻¹; sustained crossing): the labels sit on a shell about 25-30 nm from the microtubule axis, so the ground-truth spectrum has a null near 1/(60-65 nm) where every curve dips and recovers; that dip is not taken as the resolution.

| density | method | n | GT-FRC (nm) | frames to 30 nm | fair FRC (nm) | Jaccard 10 nm | false positive fraction |
|---|---|---|---|---|---|---|---|
| 0.5 | SMLM | 21458 | 69.9 | - | 88.1 | 0.698 | 0.321 |
| 0.5 | SIMFLUX (estimated pattern) | 8335 | 30.3 | - | 81.1 | 0.804 | 0.135 |
| 0.5 | SIMFLUX (true pattern) | 8367 | 30.3 | - | 81.1 | 0.805 | 0.128 |
| 0.5 | DECODE (retrained) | 29463 | 43.4 | - | 73.4 | 0.806 | 0.072 |
| 0.5 | SIMCODE | 52994 | 26.3 | 21598 | 38.0 | 0.917 | 0.088 |
| 1 | SMLM | 33214 | 74.7 | - | 85.0 | 0.683 | 0.492 |
| 1 | SIMFLUX (estimated pattern) | 9212 | 33.1 | - | 79.3 | 0.812 | 0.208 |
| 1 | SIMFLUX (true pattern) | 9280 | 31.6 | - | 79.3 | 0.814 | 0.198 |
| 1 | DECODE (retrained) | 40773 | 43.8 | - | 72.2 | 0.791 | 0.105 |
| 1 | SIMCODE | 100689 | 25.7 | 16590 | 33.8 | 0.906 | 0.106 |
| 2 | SMLM | 45469 | 76.1 | - | 88.1 | 0.658 | 0.644 |
| 2 | SIMFLUX (estimated pattern) | 6596 | 64.7 | - | 77.6 | 0.755 | 0.383 |
| 2 | SIMFLUX (true pattern) | 6517 | 64.7 | - | 74.4 | 0.757 | 0.366 |
| 2 | DECODE (retrained) | 42548 | 60.2 | - | 71.0 | 0.775 | 0.155 |
| 2 | SIMCODE | 177166 | 26.9 | 17935 | 34.9 | 0.889 | 0.125 |
| 4 | SMLM | 49860 | 85.0 | - | 75.8 | 0.641 | 0.689 |
| 4 | SIMFLUX (estimated pattern) | 834 | 131.4 | - | 147.2 | 0.239 | 0.771 |
| 4 | SIMFLUX (true pattern) | 3533 | 76.1 | - | 86.1 | 0.616 | 0.641 |
| 4 | DECODE (retrained) | 32544 | 68.8 | - | 54.8 | 0.759 | 0.184 |
| 4 | SIMCODE | 263242 | 30.1 | - | 34.1 | 0.874 | 0.132 |
