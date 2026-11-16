# Moving-microtubule simulation table (moving_exp_v2psf, SIMCODE v2 = m1809v2; 64000 frames, 64 blocks of 1000)

rows = all output rows; FRC set = disjoint 6-frame windows, σ < 0.15 px (SIMCODE v2: its σ ≤ 12 nm selection first); FRC per block = median over the blocks of the half-split FRC within each block; Jaccard = mean over blocks (GT = all trajectory points of the block); error = GT-matched (nearest ON emitter of the window within 100 nm); recall = GT emitter windows with >= 228 detected photons that have an FRC-set localization within 100 nm.

| method | rows | FRC-set rows per block | FRC per block, median (nm) | Jaccard 10 / 20 / 40 nm | RMS error (nm) | median error (nm) | unmatched or > 30 nm | recall |
|---|---|---|---|---|---|---|---|---|
| SMLM | 306436 | 740 | 101.3 | 0.179 / 0.340 / 0.547 | 33.2 | 36.8 | 0.630 | 0.073 |
| SIMFLUX (estimated pattern) | 23959 | 59 | - | 0.021 / 0.053 / 0.126 | 24.5 | 23.5 | 0.373 | 0.006 |
| SIMFLUX (true pattern) | 58380 | 148 | - | 0.051 / 0.122 / 0.249 | 12.0 | 10.6 | 0.081 | 0.016 |
| DECODE (retrained) | 638890 | 2400 | 56.5 | 0.421 / 0.698 / 0.912 | 20.3 | 19.8 | 0.280 | 0.260 |
| SIMCODE preprint (estimated pattern) | 232288 | 1121 | 56.5 | 0.245 / 0.490 / 0.762 | 11.3 | 11.3 | 0.056 | 0.111 |
| SIMCODE v2 | 495970 | 3838 | 43.7 | 0.431 / 0.716 / 0.916 | 10.4 | 10.2 | 0.041 | 0.349 |
