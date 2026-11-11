# SIMCODE v2 vs SIMCODE: current results

This is the honest headline as of critics 38–40 (2026-11-11). All numbers come from the development runs' output files,
given as `<out>/...` (the output root). The scripts behind each table
are versioned in `analysis/`, and text snapshots of their outputs are in `analysis/results/`; section 7 ("Reproduce") gives
the command and settings per table. How each metric is defined, and what it can and cannot measure, is in `EVALUATION.md`.

## Conventions

- **ours p1** = pass 1: det_n1 + sc5nb + pattern solve, with the fixed-c real calibration.
- **ours p2L** = pass 2: p2_det + p2_head_long + solve.
- **SIMCODE (b)** = the SIMCODE paper networks and pipeline with the phase-drift sign bug fixed, run by us. The paper code
  applies the estimated phase drift with the wrong sign; see `analysis/baselines/run_simcode.py`.
- **SIMCODE (published)** = the paper's own result files.
- **Real data**: DNA-PAINT microtubules, 0.5 nM (90k frames), 2.5 nM movie 2 (180k), 20 nM movie 2 (120k). Each method is
  corrected with its own DME v2 drift. Crop 5–115 px.
- **Settings behind the real-data numbers:** the full-movie runs used `--cand_pmin 0.5` and the fixed-c real calibration
  (`--autoc 0`, `calib_sc_hd_sc5nbb15_pm_bright_real`); pass 2 `calib_p2L_b10_real`. The shipped CLI defaults are
  `--cand_pmin 0.3 --autoc 1`. The two differ slightly (one-chunk check in section 5), so a default run does not reproduce
  these numbers bit for bit; use the flags in section 7.
- **Window grid:** G3 = window starts % 3 == 0 (all of ours' rows; ~2/3 of SIMCODE's rows, since its window starts cluster at % 3 == 0), G6 = % 6 == 0
  (used at 20 nM). Comparisons put both methods on the same grid.

## 1. Real-data resolution: coverage-matched FRC

FRC std (nm) = 1/7 threshold, alternating 300-frame halves, 5-nm render, on the drift-corrected set. Smaller is better.
All rows of the table use one protocol (the covpool protocol): common window grid (G3 at 0.5 / 2.5 nM, G6 at 20 nM), own
DME v2 drift, 5–115 px crop after the correction. Sources: `<out>/evalproto/covpool_real*.txt` and, for the published row, `analysis/results/evalproto/frc_published_real*.txt` (`analysis/evalproto/frc_published.py`,
recomputed for critic 40 from the published `sfhd-ndi.hdf5`, read-only).

| set | 0.5 nM | 2.5 nM | 20 nM |
|---|---|---|---|
| SIMCODE, published result files (`sfhd-ndi.hdf5`), G3 subset | 18.9 | 25.4 | – (not published) |
| SIMCODE (b), sign-fixed, G3 / G6 subset of its output | 17.7 | 24.8 | 20.9 |
| **ours p1, strict product** (per-10-px-tile σ quantile, coverage-matched) | **15.9** | **21.3** | **18.4** |
| **ours p2L, strict product** | **15.6** | **20.5** | **17.9** |

- On all rows (stride-1 window starts, the protocol of the earlier "published 19.3 / 25.0" numbers) the two SIMCODE sets
  give published 19.26 / 24.99 and (b) 18.25 / 24.94 nm (same files). The G3 subset has a third fewer rows; it is the
  right comparison for ours, whose windows start every 3 frames.
- **The sign fix and these numbers.** The SIMCODE paper code applies the estimated phase drift with the wrong sign
  (`analysis/baselines/run_simcode.py`). What the fix does and does not do:
  - After own-DME drift correction there is no gain on the same rows: FRC 25.5 vs 25.5 nm, consecutive-window NN median
    11.00 vs 10.97 nm (paper sign vs fixed). The bug's error is mostly a global time-dependent shift, which the drift
    correction absorbs.
  - The fix changes which rows pass SIMCODE's filters. With the paper sign SIMCODE keeps about 18 % fewer rows (1.134M vs
    1.376M), concentrated at the start and end of the movie where the phase error is largest; only 659k rows are shared
    (`<out>/critic34/signrows.txt`). The (b)-only rows are worse on their own (FRC 28.6 nm). The all-row FRC
    difference (25.4 → 24.9 nm) is this change of row set.
  - It matters for absolute positions (the raw published positions carry a time-varying error up to ±20 nm), for the
    drift trace (ours vs (b) agree to 2–4 nm, vs 11 nm with the paper sign), and for sims scored against GT without drift
    correction.
  - **Our advantage over the published result is not caused by the bug:** on the table's protocol published and (b) differ
    by 0.6–1.1 nm, while published is 3.0–4.9 nm and (b) 1.8–4.3 nm above ours' strict product.

- The ours rows are the per-tile strict product (`simcode2/strict.py`; covpool "tileq"). Within every 10-px tile it keeps
  the q best rows by calibrated σ, with q set so that the set covers exactly as many occupied 20-nm bins as SIMCODE's
  output (fixed-c runs). That gives q = 0.28 / 0.12 / 0.13 for p1 and 0.25 / 0.12 / 0.10 for p2L. The shipped default
  (no SIMCODE run available) sets q from the GT-free row density instead; on these movies it reproduces SIMCODE's
  coverage to 0.98–1.02 under both the fixed and the automatic c (section 5). The product is spatially neutral: the kept fraction is 0.95–1.00 of
  ours-all across the illumination envelope and the depth field.
- Same pools, NN lag-12–36 ratio ours / SIMCODE: p1 0.945 / 0.884 / 0.880, p2L 0.992 / 0.919 / 0.891.
- With per-tile coverage matched (tilecov: the same spatial weighting as SIMCODE), the NN ratio is p1 0.92 / 0.88 / 0.90
  and p2L 0.95 / 0.92 / 0.91. In sim, the pool NN ratio understates the GT gain (m05pe p2L tileq: NN 0.90 vs GT 0.81).
- **At 0.5 nM, the spatially neutral per-tile product ties SIMCODE by NN (p2L 0.99).** The sim GT gain there (−19 %) is
  not demonstrated GT-free on real data.
- These numbers are a **selection-level** comparison (which rows each method keeps), not a per-localization one. At random
  equal N, SIMCODE's self-filtered output is better: NN +25–31 %, FRC +23–37 %. The FRC of a full overlapping-window set is
  not resolution evidence (`EVALUATION.md`).

## 2. Per localization: identical rows, real data

NN = doubly-shared nearest-neighbour precision on identical windows (same window start, mutual nearest < 0.5 px), the mean
ratio ours / SIMCODE (b) over lags 12–36 frames. CI = event-block bootstrap. ,
`evalproto/nnlag_real*.txt`.

| | 0.5 nM | 2.5 nM | 20 nM |
|---|---|---|---|
| p1 | 0.930 (0.927–0.932) | 0.979 (0.976–0.981) | 0.927 (0.925–0.929) |
| p2L | 0.942 (0.940–0.944) | 0.966 (0.964–0.968) | 0.914 (0.912–0.917) |

- The NN proxy has a systematic error far larger than its CI. Validated on the four punctate / envelope sims (NN ratio vs
  the GT RMS ratio on the same identical rows, `<out>/evalproto/nnlag_sim{m05pd,m25pd,m05pe,m25pe}.txt`):
  - p1: GT − NN = +0.023…+0.040, i.e. the NN ratio flatters p1.
  - p2L: the NN ratio reads above GT by 0.003–0.040, i.e. NN understates p2L's gain.
  - So **quote the raw ratios ±0.05**.
- Corrected for that sim bias with the density-matched sims (m05pd / m05pe for 0.5 nM; m25pd / m25pe for 2.5 nM and, for
  lack of a realistic 20 nM-density sim, for 20 nM), ours / SIMCODE per window:
  - 0.5 nM: p1 0.953–0.970 (**~3–5 % better**), p2L 0.926–0.939 (~6–7 %)
  - 2.5 nM: p1 1.004–1.005 (a tie), p2L 0.926–0.932 (~7 %)
  - 20 nM: p1 0.952–0.953 (~5 %), p2L 0.874–0.880 (~12 %); in sim, m20qh gives paired RMS/CRLB −9 % (p1) / −13 % (p2L)
    at 200–2000 ph (older, 1.5–2× too bright, spatially uniform recipe; section 3b)
  - The correction transfers a sim bias to real data while the sims do not reproduce the real head loss (section 5), so
    these are indicative. The critic-38 rounding of the same numbers was 0.5 nM p2L ~3–7 % / p1 0–4 %, 2.5 nM a tie to
    ≤ 5 %, 20 nM 6–10 %.
- **Event level** (gap300 FRC on identical rows, 4 block phases):
  - 0.5 nM: not measurable (the sensitivity is below the phase scatter)
  - 2.5 nM: leans ours, Δ −1.0 / −1.4 ± 0.6 nm (n.s.)
  - 20 nM: ours better, −1.7 / −2.2 ± 0.2 nm (about a +3 nm-iid equivalent)
- SIMCODE's pattern-free net transfers to real data better than our pattern-free head: the prior / SC-net NN ratio is 1.11
  (0.5 nM) and 1.18 (2.5 nM). **All of our real advantage comes from the explicit pattern solve and pass 2.**

## 3. Simulation with ground truth

Realistic punctate benchmarks m05pd / m25pd and their envelope versions m05pe / m25pe (`analysis/simreal/recipes.sh`). They
are out of recipe relative to the training generator. 

- **Identical rows, GT RMS ratio ours / SIMCODE (b):** p1 0.92 / 0.93, p2L 0.88 / 0.87 (m05pe / m25pe); the same on the
  pd sims. In short, −8 % (p1) and −12…−14 % (p2L). With the real-data x-elongated PSF added to m05r (m05rb), the
  identical-row result is −12 %.
- **Coverage-matched pools** (same number of distinct GT events as SIMCODE, GT RMS per axis):

  | sim | SIMCODE | p1 cov | p2L cov |
  |---|---|---|---|
  | m05pd | 10.90 | 9.17 (0.84) | 8.47 (0.78) |
  | m25pd | 12.97 | 11.35 (0.88) | 9.86 (0.76) |
  | m05pe | 11.17 | 9.52 (0.85) | 8.77 (0.79) |
  | m25pe | 13.20 | 11.87 (0.90) | 10.44 (0.79) |

  - That is −10…−15 % for p1 and −19…−22 % for p2L. Tile stratification costs ≤ 2 % RMS.
  - Every error component is lower for ours (dynamic, event, site).
- **Gate scenes against the multi-emitter CRLB** (`analysis/gate/run_gate.sh`): iso p1 RMS / CRLB_SF_multi 1.17
  [1.14, 1.20]. Per photon bin, 200–400 / 400–700 / 700–1200 / 1200–2000 / > 2000: 1.38 / 1.26 / 1.17 / 1.13 / 1.28. Paired
  vs SIMCODE ni6_3 on the same targets, 200–2000: 1.09 vs 1.15 (iso), 1.19 vs 1.32 (pair2), 1.39 vs 1.46 (rs05qm).

## 3b. Standing requirements vs sign-fixed SIMCODE, per sim (critic 40)

Every number is read from an existing output file (paths relative to `<out>/`; text snapshots in
`analysis/results/`). Δ = ours − SIMCODE, negative = ours better; 95 % CIs in brackets. † = m20qh / m25qh, the older
recipe: 1.5–2× too bright and spatially uniform, so the margins there are larger than on the realistic sims.

**(a) Paired RMS/CRLB on the same targets** ("SIMCODE vs ours, Δ"; bins in GT label photons)

Gate scenes (`p2/long/gate_long_vs_simcode.txt`; the file gives SC − ours):

| scene / ours | 400–700 | 700–1200 | 1200–2000 | 2000–4000 | > 4000 | 200–2000 |
|---|---|---|---|---|---|---|
| rs05qm p1 | 1.19 vs 1.14, −0.05 [−0.16, +0.04] (n 54) | 1.45 vs 1.39, −0.07 | 1.51 vs 1.43, −0.08 | 1.83 vs 1.66, −0.18 | 2.17 vs 1.80, −0.37 (n 70) | 1.47 vs 1.39, −0.07 [−0.10, −0.05] |
| rs05qm p2L | 1.19 vs 1.14, −0.05 [−0.15, +0.04] | 1.46 vs 1.36, −0.11 | 1.49 vs 1.36, −0.14 | 1.83 vs 1.61, −0.22 | 2.17 vs 1.86, −0.31 | 1.46 vs 1.34, −0.12 [−0.14, −0.10] |
| rs25qm p1 | – (no SC targets) | 1.54 vs 1.35, −0.19 (n 84) | 1.65 vs 1.47, −0.18 | 1.74 vs 1.57, −0.17 | 2.23 vs 2.13, −0.09 [−0.43, +0.45] | 1.63 vs 1.44, −0.18 [−0.24, −0.13] |
| rs25qm p2L | – | 1.54 vs 1.26, −0.28 | 1.67 vs 1.45, −0.22 | 1.75 vs 1.46, −0.29 | 1.93 vs 1.78, −0.16 [−0.44, +0.27] | 1.64 vs 1.41, −0.23 [−0.30, −0.17] |

Movies (`simreal2/cci_<movie>_{est,p2L}.txt`, `n20/cci_m20qh_{est,p2L}.txt`, `p2/long/cci_m25qh_{est,p2L}.txt`; PAIRED block vs
SC_mrg_fix):

| movie / ours | 200–700 | 700–2000 | 2000–4000 | > 4000 | 200–2000 |
|---|---|---|---|---|---|
| m05r p1 | 1.65 vs 1.54, −0.108 [−0.141, −0.071] (n 2433) | 2.02 vs 1.88, −0.148 | 2.63 vs 2.26, −0.364 | 3.88 vs 3.09, −0.79 [−1.13, +0.17] (n 32) | 1.91 vs 1.78, −0.136 [−0.153, −0.121] |
| m05r p2L | 1.63 vs 1.47, −0.159 | 2.01 vs 1.80, −0.205 | 2.63 vs 2.30, −0.328 | 3.88 vs 2.87, −1.01 [−1.33, −0.09] | 1.90 vs 1.71, −0.192 [−0.211, −0.175] |
| m25rc p1 | 0.98 vs 0.97, **−0.014 [−0.050, +0.019]** (n 1551) | 1.73 vs 1.62, −0.107 | 2.23 vs 2.03, −0.200 | 2.82 vs 2.30, −0.52 [−0.78, −0.10] (n 76) | 1.46 vs 1.38, −0.077 [−0.095, −0.057] |
| m25rc p2L | 0.94 vs 0.88, −0.062 [−0.089, −0.039] | 1.71 vs 1.54, −0.166 | 2.24 vs 1.99, −0.249 | 2.82 vs 2.22, −0.60 [−0.96, −0.04] | 1.44 vs 1.31, −0.132 [−0.149, −0.115] |
| m20qh † p1 | 1.17 vs 1.04, −0.130 (n 95) | 2.24 vs 2.05, −0.192 | 2.88 vs 2.57, −0.313 | 3.39 vs 2.83, −0.560 | 2.22 vs 2.03, −0.190 [−0.212, −0.169] (−9 %) |
| m20qh † p2L | 1.14 vs 1.00, −0.146 (n 89) | 2.23 vs 1.93, −0.294 | 2.86 vs 2.41, −0.452 | 3.36 vs 2.70, −0.667 | 2.20 vs 1.91, −0.291 [−0.310, −0.272] (−13 %) |
| m25qh † p1 | 1.37 vs 1.27, −0.097 [−0.340, +0.178] (n 58) | 1.93 vs 1.72, −0.212 | 2.53 vs 2.12, −0.404 | 3.11 vs 2.37, −0.747 | 1.93 vs 1.72, −0.210 |
| m25qh † p2L | 1.42 vs 1.19, −0.229 (n 53) | 1.92 vs 1.65, −0.269 | 2.52 vs 2.05, −0.462 | 3.10 vs 2.36, −0.742 | 1.92 vs 1.65, −0.269 |

Punctate / envelope sims: no per-bin RMS/CRLB file exists. The substitute is the GT RMS ratio ours / SIMCODE (b) on
identical rows: p1 0.92 / 0.93 / 0.920 / 0.934, p2L 0.88 / 0.86 / 0.881 / 0.872 (m05pd / m25pd / m05pe / m25pe;
`evalproto/valid_m05pd.txt`, `valid_m25pd.txt`, `nnlag_simm05pe.txt`, `nnlag_simm25pe.txt`).

**(b) Recall at matched FP**

| sim | SIMCODE (filtered) at its FP | p1 at that FP | p2L at that FP | at FP fraction 0.01: SC net / p1 / p2L | source |
|---|---|---|---|---|---|
| rs05qm | 0.310 at 0 FP | – | **0.191** (needs thr 0.915) | SC unfiltered 0.653 at its FP → p2L 0.923 | `p2/long/gate_long_vs_simcode.txt` |
| rs25qm | 0.222 (FP 1) | – | 0.243 | SC unfiltered 0.528 → p2L 0.900 | same |
| m05r | 0.105 (FP 1) | **0.073** | 0.195 | 0.344 / 0.635 / 0.760 | `simreal2/cci_m05r_*.txt` |
| m25rc | 0.024 (FP 6) | 0.034 | 0.093 | 0.082 / 0.237 / 0.398 | `simreal2/cci_m25rc_*.txt` |
| m20qh † | 0.119 (FP 18) | 0.201 | 0.281 | 0.412 / 0.582 / 0.758 | `n20/cci_m20qh_*.txt` |
| m25qh † | 0.166 (FP 13) | 0.367 | 0.464 | 0.519 / 0.827 / 0.897 | `p2/long/cci_m25qh_*.txt` |
| m05pd / m25pd / m05pe / m25pe | not computed | | | | |

The 2.5 nM-like movies count the 300-ph dim binders as targets, so recall is low for every method.

**(c) Gross errors** (definitions differ between sources; compare methods within a row)

| sim | definition | SIMCODE | p1 | p2L | source |
|---|---|---|---|---|---|
| rs05qm | paired, % > 5 CRLB, 200–2000 / 2000–4000 | 1.21 / 4.23 | 0.95 / 1.81 | 0.82 / 2.32 | `p2/long/gate_long_vs_simcode.txt` |
| rs25qm | same | 1.33 / 4.46 | 0.89 / 2.08 | 1.03 / 2.08 | same |
| m05r | own set, \|e\| > 4 σ_cal, 200–400 / 400–700 / 700–1200 / 1200–2000 / 2000–4000 (%) | not in file | 4.72 / 2.42 / 1.86 / 2.07 / 1.98 | 3.02 / 1.65 / 1.16 / 1.14 / 0.77 | `simreal2/cci_m05r_*.txt` |
| m25rc | same | not in file | 5.06 / 2.68 / 1.46 / 1.14 / 0.94 | 3.05 / 1.79 / 1.16 / 0.90 / 0.64 | `simreal2/cci_m25rc_*.txt` |
| m05pd / m25pd | own pool, \|e\| ≥ 50 nm, best 70 % by σ; worst 30 % (%) | ≤ 0.48; 0.50 / ≤ 0.71; 1.33 | ≤ 1.16; 9.2 / ≤ 6.2; 20.5 | ≤ 0.61; 5.75 / ≤ 2.4; 10.6 | `evalproto/zcal_sim_m05pd_m25pd.txt` |
| m05pe / m25pe | same | ≤ 0.55; 0.51 / ≤ 0.66; 1.55 | ≤ 1.22; 10.3 / ≤ 7.1; 23.1 | ≤ 0.80; 6.7 / ≤ 3.3; 12.9 | `evalproto/zcal_sim_m05pe_m25pe.txt` |
| m20qh † | > 50 nm, 200–2000: own set; paired (%) | 0.74; 0.55 | 8.31; 0.33 | 4.45; 0.22 | `n20/merged_gross_m20qh.txt` |
| m25qh † | same | 0.21; 0.17 | 3.88; 0.14 | 2.33; 0.07 | `n20/merged_gross_m25qh.txt` |

SIMCODE's own-set gross is low because its filters keep only 5–12 % of the targets; on paired targets ours is lower or
equal. Ours' own-set gross sits in its worst-σ rows (the strict product drops them).

**(d) zcal (σ calibration; 1 = calibrated, > 1 = over-confident)**

| sim | estimator / bins | SIMCODE | p1 | p2L | source |
|---|---|---|---|---|---|
| rs05qm | robust sd of e/σ, paired, 200–2000 / 2000–4000 / > 4000 | 1.33 / 1.55 / 2.17 | 1.02 / 0.94 / 0.87 | 1.02 / 0.92 / 0.93 | `p2/long/gate_long_vs_simcode.txt` |
| rs25qm | same | 1.65 / 1.79 / 2.02 | 1.10 / 0.99 / 0.83 | 1.08 / 1.04 / 0.87 | same |
| m05r | 1.4826 median \|e/σ_cal\|, 200–400 / 400–700 / 700–1200 / 1200–2000 / 2000–4000 | not in file | 1.13 / 1.09 / 1.09 / 1.17 / 1.19 | 1.16 / 1.11 / 1.12 / 1.17 / 1.12 | `simreal2/cci_m05r_*.txt` |
| m25rc | same | not in file | 1.23 / 1.06 / 0.97 / 0.96 / 0.98 | 1.20 / 1.11 / 1.08 / 1.07 / 1.06 | `simreal2/cci_m25rc_*.txt` |
| m05pd / m25pd | rms(e/σ), \|e\| < 50 nm, full set [top 1 % by σ] | 2.07 [4.24] / 3.03 [5.51] | 1.26 [1.92] / 1.26 [1.83] | 1.19 [1.51] / 1.24 [1.59] | `evalproto/zcal_sim_m05pd_m25pd.txt` |
| m05pe / m25pe | same | 2.15 [4.82] / 2.93 [5.64] | 1.24 [1.92] / 1.20 [1.76] | 1.19 [1.52] / 1.23 [1.58] | `evalproto/zcal_sim_m05pe_m25pe.txt` |
| m20qh † | 1.4826 median \|e/σ\|, own set, 200–700 / 700–2000 / 2000–4000 / > 4000 | 3.87 / 2.40 / 2.64 / 3.09 | 1.15 / 1.02 / 0.93 / 0.79 | 1.20 / 1.12 / 1.05 / 0.98 | `hd_compare/score_m20qh.txt` |
| m25qh † | same | 3.40 / 2.26 / 2.63 / 3.27 | 1.12 / 1.03 / 0.94 / 0.80 | 1.11 / 1.09 / 1.02 / 0.95 | `hd_compare/score_m25qh.txt` |

Definitions and settings:

- CRLB reference: gate = each target's own CRLB_SF_multi (FOV bootstrap); movies = the gate scene's CRLB_SF_multi at the
  target's label photons (rs05pm for m05r, rs25pm for m25rc / m25qh / m20qh; event-block bootstrap). The m20qh ratios are
  inflated for every method because rs25pm lacks the 20 nM background.
- SIMCODE: the paper networks, ni6_3 (0.5 nM-like) and L64_5 checkpoint 3 (2.5 / 20 nM-like; the paper never ran SIMCODE
  at 20 nM). Gate SC uses the gate's pattern estimate (no sign issue); movie SC_mrg_fix is the sign-fixed paper pipeline
  with its filter chain; "SC net" is the network without filters.
- Ours: none of these rows uses the automatic c. Gate p1 b10 (`calib_sc_hd_sc5nbb15_pm_bright`), p2L `calib_p2L`; m05r /
  m25rc / punctate / envelope p1 b10, p2L `calib_p2L_movie_b10`; m25qh / m20qh b22 (c 0.22) for both passes (an autoc
  m25qh variant: `autoc/cmp/cci_m25qh.txt`, 200–2000 Δ −0.210 [−0.226, −0.190]).
- The zcal of the punctate / envelope rows is a non-robust rms with SIMCODE's σ = its CRLB, so it reads higher than the
  robust rows; do not compare across rows.
- Missing (would need a new analysis): SIMCODE zcal and gross on m05r / m25rc; per-bin RMS/CRLB and recall at matched FP on
  the punctate / envelope sims.

Reading: p2L is better than sign-fixed SIMCODE in paired RMS/CRLB in every populated bin of every sim with per-bin data
(the CI includes 0 only at rs05qm 400–700, n 54, and rs25qm > 4000). **p1 ties SIMCODE at 200–700 ph at 2.5 nM** (m25rc
−0.014 [−0.050, +0.019]; m25qh n.s., n 58) and is better elsewhere. Recall at matched FP is higher for ours except at
SIMCODE's zero/one-FP operating points on rs05qm (p2L 0.19 vs 0.31) and m05r (p1 0.073 vs 0.105). Ours' zcal is
0.8–1.26; SIMCODE's 1.3–3.9 (up to 5.6 in its best 1 %; ours 1.5–1.9 in its best 1 %).

## 4. Counts, drift, throughput

- Ours keeps far more usable rows than SIMCODE. At 20 nM, p1 has 5.3× and p2 9.4× SIMCODE's rows.
- More rows give better drift: DME v2 split-half error 1.13 (p1) / 0.87 (p2) vs 1.34 nm at 20 nM.
- Throughput, idle RTX 3090, 0.5 nM:
  - one-pass (pass 1 + pattern + solve): 324–339 frames/s
  - two-pass: ~190 frames/s
- Throughput at 20 nM, shared GPU: 84 frames/s (one-pass) and 39 frames/s (two-pass). The heads dominate the run time.

## 5. Evaluation-protocol caveats

- **Per-localization and selection comparisons are kept apart.** Random or photon-matched pools measure selection;
  identical rows measure per-window precision. Windows on a common grid only (G3 / G6): mixing SIMCODE's stride-1 starts
  with ours' stride-3 starts was an artefact (critic 35).
- **300-frame-half FRC rewards per-event static error.** Use gap-halves (gap300) on identical rows for the event level,
  with block-phase replicates. Sim FRC is not evidence (critic 36).
- **The NN proxy has a ±0.05 systematic** (sim validated) and cannot detect the winner's curse of the strict product.
- **The strict product's σ is overconfident in its best bins** (1.3–1.9×; real 2.5 nM ~15 % worse than sim). A
  per-density quantile rescale fixes it in sim only.
- **The global σ ≤ 6 nm / top-N_SC strict product is retired.** It had half the coverage and 2× the centre bias, and the
  earlier "15–29 % better at SIMCODE's N" headline was restated.
- **Strict-product q under the shipped defaults (critic 40 #1).** The earlier default (q matched to the global
  σ ≤ 7.0 / 7.4 nm set) was derived with the fixed c. Under the automatic c (the CLI default) σ is ~10–17 % larger and that
  rule kept only 0.82 / 0.71 / 0.68 × SIMCODE's occupied 20-nm bins (pass 1, 0.5 / 2.5 / 20 nM; pass 2 at 2.5 nM 0.89).
  The default is now a per-density q (row density ρ = good rows in the crop per window start per µm²; `simcode2/strict.py`).
  Check on the covpool protocol (`analysis/evalproto/strictq.py`, `analysis/results/evalproto/strictq_*.txt`), occupied
  bins / SIMCODE's:

  | | 0.5 nM | 2.5 nM | 20 nM (G6) |
  |---|---|---|---|
  | ρ (pass 1 / pass 2) | 0.49 / 0.64 | 0.87 / 1.69 | 1.25 / 2.22 |
  | pass 1, automatic c: q / bins ratio | 0.275 / 1.013 | 0.120 / 1.021 | 0.120 / 0.981 |
  | pass 1, fixed c: q / bins ratio | 0.275 / 0.987 | 0.120 / 0.990 | 0.120 / 0.976 |
  | pass 2 (p2L): q / bins ratio | 0.250 / 0.997 | 0.120 / 0.999 | 0.100 / 1.013 |
  | exactly matched q, pass 1 automatic / fixed c; pass 2 | 0.267 / 0.283; 0.252 | 0.113 / 0.124; 0.121 | 0.125 / 0.127; 0.099 |
  | old σ_ref rule, pass 1 automatic c: q / bins ratio | 0.171 / 0.817 | 0.045 / 0.711 | 0.059 / 0.682 |

  The anchors were set on these same movies, so this checks robustness to the calibration (automatic vs fixed c differ
  by ≤ 0.016 in the matched q), not transfer to new data. The FRC table in section 1 used the exactly matched q.
- **`--cand_pmin` 0.3 (default) vs 0.5 (the RESULTS runs), one chunk** (real 0.5 nM, frames 12000–18000, automatic c;
  `analysis/checks/cand_pmin_check.py`, `analysis/results/checks/cand_pmin_chunk12000_0.5nM.txt`):
  - The p ≥ 0.63 rows are identical (147,625).
  - Pattern: pitch within 1e-4 px, angles within 0.003°. The six phases shift by 0.011–0.026 rad (0.4–0.9 nm, nearly
    common to all steps), depth is +0.009…+0.016 higher with 0.3, relative intensities within 0.005. The pattern stage
    selected 75.7k vs 140.2k rows; the extra low-p candidates probably fail its isolation test as neighbours.
  - Automatic c: θ (s, c, c_a) = (1.010, 0.093, 0.078) with 0.3 vs (1.014, 0.131, 0.020) with 0.5. The c / c_a split
    moves a lot (√(c² + c_a²) 0.121 vs 0.132), so the split is weakly identified on one chunk. The full 0.5 nM movie
    (cut 0.5) gave (0.117, 0.055).
  - Finals of the identical rows: median shift 0.17 / 0.14 nm, robust sd 0.5 nm, rms 1.6 nm, 0.5 % move > 5 nm; median σ
    is 4 % smaller with 0.3.
  - So the two cuts are equivalent for positions at the < 1 nm level, but σ, θ and the σ_ref strict rule differ (q 0.222
    vs 0.170 on this chunk). The density-rule q is identical (0.260 on this chunk).
- **The sims do not reproduce the real head loss.** The retrain on realistic sims did not close the gap, and no sim knob
  except a PSF-shape change reproduces it.

## 6. Limitations

- **Absolute accuracy, per-site static error, the ±3–6 nm linear term: unknown for both methods.** The GT-free protocol
  cannot measure them; the sims show they are invisible to every GT-free metric. Measuring them needs a calibration sample
  (e.g. nanorulers / known geometry).
- **PSF ellipticity.**
  - The real PSF is x/y-elliptic relative to the model (x wider, ΔMxx +0.066 px²); its best one-parameter description is
    an x-blur of 0.36 px (39 nm) beyond the sims.
  - It is the same at 0.5 and 2.5 nM. About 20 % of it flips with pattern angle. There is no frame-position or DMD-switch
    effect and no diattenuation signature.
  - Its origin is not established. Detection-side optics (polarisation- or wavelength-dependent) is the leading
    candidate. Session vibration is disfavoured (the same anisotropy at 0.5 and 2.5 nM), and so is the camera pixel MTF
    (~2σ, bead stacks). The ±20 % angle-dependent part is compatible with photoselection (angle 1 is the more
    x-elongated), but the excitation polarisation is not known; frame position, DMD switching, rolling shutter and
    intra-frame drift are ruled out (critic 39).
  - Binned by our photons, the x-blur explains about all of the 0.5 nM x/y excess of (prior − final) up to 1500 ph and
    about a third above (24–49 % of the x rsd alone). PSF-diversity fine-tunes were not adopted.
- **The angle-0 brightness-dependent x term** is a ±3 nm accuracy systematic, common to both methods (critic 39).
- **Sub-pixel pairs (< ~1 px)** are only partly resolved. On the pair1 gate scene: recall 0.64 (p1) / 0.78 (p2L); gross
  errors (|e|/CRLB > 5, 200–2000 ph) 23 % for p1 → 4.5 % for p2s, the short-schedule pass-2 predecessor of p2L (recall: `<out>/p2/long/gate_long_vs_simcode.txt`).
- **2.5 nM event-static prior error.** It is event-specific, not caused by neighbours or static light. About 40 % of it
  leaks into the final position, and no cheap solve-side fix exists.
- **σ is overconfident in the strict product's best bins** (winner's curse: zcal 1.3–1.9× in the top 1 % by σ in sim;
  real 2.5 nM ~15 % worse than sim). Over the full set zcal is 1.19–1.26 (punctate / envelope sims) and 0.8–1.26 on the
  standing-requirement sims (section 3b). Quote NN (real) or GT RMS (sim) for the strict product, never its σ.
- **20 nM.** There is no realistic 20 nM-density sim. The 20 nM real-data claims are GT-free (NN proxy ±0.05, event-level
  gap300 FRC), and the only 20 nM sim with GT, m20qh, uses the older recipe (1.5–2× too bright, spatially uniform), on
  which the margins are larger than on the realistic m05r / m25rc. Its paired −9 % (p1) / −13 % (p2L) is therefore an
  upper estimate. SIMCODE never ran at 20 nM in the paper; its L64_5 network is used as a fallback.

## 7. Reproduce

Scripts are in `analysis/` (run with `PYTHONPATH=<repo>`). Text snapshots of the development outputs they reproduce are in
`analysis/results/` (its README gives each source). Real-data scripts take a `datasets.py` json spec
(`analysis/evalproto/example_real05.json`, `example_real20.json`, `example_sim.json`): per set the output dir, the tag
(`est` = pass 1, `est2` = pass 2), and the set's own drift.

**Real movies** (data root): `0.5nM/cover1/1/simflux_5.5ms_500mW.tif` (90k frames), `2.5nM/2/cover1_600mW_5.5ms_cf660.tif`
(180k), `20nM/2/cover1_500mW_5.5ms_cf660.tif` (120k).

**Our runs (all real-data tables).**

```bash
python -m simcode2.run --movie <tif> --calib insitu/calib_0.5nM.json | insitu/calib_2p5nM_m2_sc.json | insitu/calib_20nM_m2_sc.json \
    --out <dir> --pass2 --autoc 0 --cand_pmin 0.5
```

- Pass 1: `det_n1` + `hd_sc5nb`, solve calibration `calib/calib_sc_hd_sc5nbb15_pm_bright_real.json` (fixed c 0.10 + 0.10).
- Pass 2: `p2_det` + `p2_head_long`, `--p2_cal calib/calib_p2L_b10_real.json` (default) = the development tag `est2r`. The
  development run used `calib_p2L_movie_b10` plus a post-hoc real σ rescale; positions are the same.
- Automatic-c variants (`--autoc 1`): `calib/calib_autoc_r05 / r25 / r20.json` hold the full-run θ.
- 0.5 / 2.5 nM ran on the reference code path, 20 nM on the fast path; the fast path is equivalent (README "History").
- Strict product: `python -m simcode2.strict <dir> [--tag est2] --drift <own DME npz>`. The section 1 rows used the
  exactly coverage-matched q (`--strict_q` 0.283 / 0.124 / 0.127 for p1, 0.252 / 0.121 / 0.099 for p2L, the covpool tileq
  values). The default (density) q is checked with `analysis/evalproto/strictq.py` (section 5).

**SIMCODE (b), sign-fixed** (`analysis/baselines/run_simcode_real.py`; paper camera gain 0.56 / offset 106.34; checkpoint 3):

```bash
python analysis/baselines/run_simcode_real.py net <cache.npy> --movie <tif> --model sf_conv_g1.3_tio2_L64_ni6_3 --ckpt 3   # 0.5 nM
#                                                     --model sf_conv_g1.3_L64_5 --ckpt 3          # 2.5 and 20 nM
python analysis/baselines/run_simcode_real.py ndi <cache.npy> sc_b.npz --fix_sign 1
```

**Drift: DME v2** (the `dme` package, `github.com/jcnossen/drift-estimation`, through `simcode2/util/drift.py`; 'v3e' in file and method names is its development name). Each method gets its own drift, at time = window start + 3, with
crlb = its own σ and crlb < 1 px. Ours uses the disjoint rows (window start % 6 == 0, p ≥ 0.63) of pass 1 and pass 2
separately; SIMCODE uses all its rows.

- `analysis/baselines/dme_drift.py <locs_p1.npz | locs_p2.npz | sc_b.npz> <drift.npy>`, or
- `analysis/dme/dme_splithalf.py run <spec> <sets>`, which also does the split halves (v3e / grid_fpb20 / grid_fpb10; at
  20 nM crlb_min_pos).

**Grid and crop.** G3 at 0.5 / 2.5 nM, G6 at 20 nM. Crop 5–115 px after drift correction.

**Per section.**

- **§1, coverage-matched FRC:**
  - `analysis/evalproto/covpool.py <real spec>` (`covpool_real*.txt`)
  - published row: `analysis/evalproto/frc_published.py` (`frc_published_real*.txt`)
  - spatial neutrality: `envmap.py <spec with movie, camera_maps>`, then `spatial.py <spec with envmap (+ depth_field)>`
- **§2, identical-row NN:**
  - `analysis/evalproto/nnlag.py <spec>` (`nnlag_real*.txt`)
  - event level: `blockphase.py <spec>`
  - NN-proxy validation: `nnlag.py` / `valid.py` on the sim specs
  - GT-free protocol and filters: `realproto.py`, `filt.py`
- **§3 / §3b, sims.**
  - Movies: `analysis/simreal/recipes.sh m05pd | m25pd | m05pe | m25pe | m05r | m25rc` (m05rb = m05r + `SIM_PSF_BLUR=0.36,0`).
  - Methods: `analysis/simreal/eval_movie.sh <movie> <out> ni6_3 (m05*) | L64_5 (m25*)`.
    - Ours: the sim calibrations `calib/calib_sc_hd_sc5nbb15_pm_bright.json` (p1 b10) and `calib/calib_p2L_movie_b10.json`,
      with `insitu/calib_0.5nM.json`.
    - SIMCODE: `run_simcode.py --model ni6_3 | L64_5 --cam paper --fix_sign 1`.
  - Identical rows and coverage pools: `nnlag.py`, `covpool.py`, `valid.py <sim spec>`.
  - σ calibration: `zcal.py sim | real`, `zcal_fix.py`.
  - m05r / m25rc paired CIs, recall at FP, zcal / gross:
    `analysis/simreal/compare_ci_r.py <sim dir> rs05pm | rs25pm <out> ours=<run dir> SC_net=sc_net:<paper_est.npz> SC_mrg_fix=sc_mrg:<paper_fix.npz>`.
    For p2L, use a view dir with `pass2_est2_*` linked as `pass2_est_*`.
  - PSF shape and m05rb: `analysis/psfshape/evalgt.py`.
  - Gate: `GPU=1 bash analysis/gate/run_gate.sh iso` (expect iso 1.17 [1.14, 1.20]).
    - SIMCODE on the gate: `python -m simcode2.eff.simcode_gate --model ni6_3 --scen iso,pair2,rs05qm`.
    - Paired comparison: `analysis/gate/gate_paired.py _scode_ni6_3 _hd_sc5nbb15pmb10 iso,pair2,rs05qm --offset`.
    - Pass-2 gate vs SIMCODE (rs05qm / rs25qm in §3b): `python -m simcode2.eff.simcode_gate --model ni6_3 --scen pair1,pair1.5,rs05qm`
      and `--model L64_5 --scen rs25qm` (gate npz `_scode_<net>` / `_scodenf_<net>`), then `analysis/gate/merged_d1.py` and
      `analysis/gate/vs_simcode_p2.py` on the pass-2 gate dumps (`<out>/p2/fe`) → `<out>/p2/long/{gate_long,merged_d1}_vs_simcode.txt`
      (reproduces the snapshots in `analysis/results/gate/` exactly).
  - m05q / m25q / m05qh / m25qh vs SIMCODE: `analysis/baselines/compare.py`, CRLB_ref = gate rs05pm / rs25pm
    (`SIMCODE2_GATE_DIR`).
  - m25qh / m20qh (§3b †): `analysis/hd_compare/score.py <spec>`, with ours b22 = `analysis/hd_compare/calib_b22/*`.
    - Paired CIs: `compare_ci_r.py` with rs25pm.
    - Movies: `analysis/simreal/recipes.sh m25qh | m20qh` (`sim_movie.py` / `sim_movie20.py`; m05qh too). Classical inputs:
      `analysis/baselines/run_classic.py <sim> <work> classic_<m>.npz` (SMLM + SF, paper pipeline) and
      `analysis/baselines/run_smlm_tuned.py <sim> <work> smlmt_<m>_r6_m4_dt3.5.npz 6 4`; renders `analysis/hd_compare/render.py <spec>`;
      gross / merged: `analysis/simreal/merged_gross.py <sim> <out.txt> p1=ours_est:<dir> p2L=ours_p2L:<dir> SC_net=sc_net:<paper_est.npz> SC_mrg_fix=sc_mrg:<paper_fix.npz>`.
- **§4, counts / drift / throughput:**
  - Row counts: `realproto.py` header, or `dme_splithalf.py dry`.
  - Split-half: `dme_splithalf.py report <npz>` (reference grid_fpb10; snapshot `analysis/results/dme/splithalf_all.txt`).
  - Throughput: `analysis/tput/wait_and_bench.sh` (`bench_tput.py` on an idle GPU, chunk 12000–18000) +
    `gpu_sampler.sh` + `summarize.py`. Frames/s = 6000 / (pass-1 wall + pattern + solve).
  - 20 nM: the full-run stage times (`analysis/results/tput/stages_real20.log`).
- **§5 checks:**
  - `analysis/evalproto/strictq.py` (strict q)
  - `analysis/checks/cand_pmin_check.py` (cand_pmin)
- **Figures:** `analysis/figures/` (README there): CRLB (`crlb_all_v2.py`, references via `run_crlb_refs.sh`: LM `lm_gate.py` /
  `lm_icrlb.py`, joint SIMFLUX MLE `sf_gate.py`, `icrlb.py`), real renders (`recon_full*.py`, `hdzoom.py`, `render_matched20.py`,
  `render_strict20.py`, zooms `zoompick.py`), summaries (`fig_gap.py`, `scorecard.py --stack`, `frc_plot.py`,
  `field/plot_field_offsets.py` with its inputs `field/check1_offsets.py`, `field_maps.py`, `check3_pattern.py`).
- **In-situ PSF / camera calibration:** `simcode2/insitu/README.md`.
- **Not reproducible from this repo alone:**
  - the injection movies of evalgt
  - the real depth fields (`field/depth/real05|real25/field_own.pt`)

  The script headers name the development sources.
