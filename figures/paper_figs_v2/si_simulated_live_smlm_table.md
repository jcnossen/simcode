# SI table: si_simulated_live_smlm (moving microtubules)

Moving-microtubule simulation (moving_exp_v2psf; 64000 frames, 64 blocks of 1000 frames; SIMCODE = model m1809fin, its pass-2 pattern solve with p >= 0.63 and posterior max(sigma) < 0.15 px; the other methods all output rows; SIMFLUX with its own pattern estimate). Jaccard = mean over the blocks (GT = all trajectory points of the block), as Fig. B. FRC = median over the blocks of the half-split FRC within each block, on the FRC set (disjoint 6-frame windows, sigma < 0.15 px); "-" = too few localizations per block. RMS error = GT-matched error of the FRC-set localizations (nearest emitter ON in the window, within 100 nm; per axis); "> 30 nm" = fraction unmatched or off by more than 30 nm.

| method | localizations | Jaccard 10 nm | Jaccard 20 nm | Jaccard 40 nm | FRC per block, median (nm) | RMS error (nm) | unmatched or > 30 nm |
|---|---|---|---|---|---|---|---|
| SMLM | 306436 | 0.179 | 0.340 | 0.547 | 101.3 | 33.2 | 0.63 |
| SIMFLUX | 23959 | 0.021 | 0.053 | 0.126 | - | 24.5 | 0.37 |
| DECODE | 795406 | 0.531 | 0.818 | 0.949 | 55.6 | 17.1 | 0.19 |
| SIMCODE | 812165 | 0.588 | 0.848 | 0.963 | 45.7 | 12.8 | 0.09 |

Source: out/papersim/moving_exp_v2psf/full/metrics_m1809fin.json, out/papersim/moving_exp_v2psf/full/moving_gterr_m1809fin.json (analysis/papersim/fig_moving_pp.py write_table)
