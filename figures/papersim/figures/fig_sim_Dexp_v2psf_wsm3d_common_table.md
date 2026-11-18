# Simulation table (experimental photons and background of 1809/2 (median 621 detected photons per 6-frame window, 1.95 photons/px/frame); 60 000 frames)

n = localizations in the FRC set (disjoint 6-frame windows, σ < 0.15 px; SIMCODE p >= 0.63, posterior σ); GT-FRC = FRC(reconstruction, ground truth), 1/7; frames to 30 nm GT-FRC ("-" = not reached in 60 000 frames, values <= 250 = reached in the first 250 frames); fair FRC = half-split FRC; Jaccard at 10 nm; false positive = a localization with no emitter ON in its window within 3 own σ. GT-FRC, frames to 30 nm and fair FRC: Resolution = 1 / the spatial frequency at which the FRC (5-sample moving average) falls below 1/7 and stays below it (for at least 0.01 nm⁻¹; sustained crossing): the labels sit on a shell about 25-30 nm from the microtubule axis, so the ground-truth spectrum has a null near 1/(60-65 nm) where every curve dips and recovers; that dip is not taken as the resolution.

| density | method | n | GT-FRC (nm) | frames to 30 nm | fair FRC (nm) | Jaccard 10 nm | false positive fraction |
|---|---|---|---|---|---|---|---|
| 0.5 | SMLM | 35936 | 37.4 | - | 73.6 | 0.778 | 0.229 |
| 0.5 | SIMFLUX (estimated pattern) | 19906 | 17.9 | 15901 | 37.2 | 0.890 | 0.097 |
| 0.5 | SIMFLUX (true pattern) | 19877 | 17.3 | 15901 | 37.2 | 0.891 | 0.090 |
| 0.5 | DECODE (retrained) | 73700 | 28.1 | 28504 | 39.7 | 0.885 | 0.039 |
| 0.5 | SIMCODE | 71798 | 17.6 | 6961 | 28.9 | 0.953 | 0.048 |
| 1 | SMLM | 48439 | 39.1 | - | 73.6 | 0.767 | 0.365 |
| 1 | SIMFLUX (estimated pattern) | 22223 | 19.7 | 16784 | 33.1 | 0.898 | 0.163 |
| 1 | SIMFLUX (true pattern) | 22167 | 18.1 | 16242 | 33.4 | 0.900 | 0.144 |
| 1 | DECODE (retrained) | 124048 | 24.0 | 21754 | 38.3 | 0.873 | 0.063 |
| 1 | SIMCODE | 138608 | 16.0 | 4922 | 23.8 | 0.951 | 0.061 |
| 2 | SMLM | 47973 | 67.7 | - | 76.5 | 0.733 | 0.531 |
| 2 | SIMFLUX (estimated pattern) | 15782 | 24.4 | 46929 | 37.2 | 0.860 | 0.279 |
| 2 | SIMFLUX (true pattern) | 15606 | 20.7 | 32164 | 36.8 | 0.865 | 0.237 |
| 2 | DECODE (retrained) | 170889 | 25.2 | 46303 | 37.0 | 0.853 | 0.098 |
| 2 | SIMCODE | 248633 | 16.3 | 4674 | 24.1 | 0.941 | 0.083 |
| 4 | SMLM | 28893 | 78.8 | - | 70.2 | 0.699 | 0.678 |
| 4 | SIMFLUX (estimated pattern) | 6294 | 33.9 | - | 37.5 | 0.696 | 0.465 |
| 4 | SIMFLUX (true pattern) | 6177 | 33.1 | - | 43.6 | 0.698 | 0.431 |
| 4 | DECODE (retrained) | 174423 | 30.3 | - | 38.0 | 0.822 | 0.147 |
| 4 | SIMCODE | 401100 | 17.6 | 4171 | 26.7 | 0.924 | 0.113 |
