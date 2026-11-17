# Evaluation protocol

This is the protocol behind every cross-method statement in `RESULTS.md`. It was built and validated against ground truth
on punctate repeated-site sims.

- Code: `simcode/evaluation/metrics.py` (library), `archive-2026-11-15:analysis/evalproto/{nnlag,blockphase,covpool}.py` (scripts, archived), and
  `archive-2026-11-15:analysis/evalproto/datasets.py` (dataset spec).
- Every script takes a dataset json that names, per set, a simcode output dir or a SIMCODE npz plus its drift; see the
  `datasets.py` docstring.
- On the real 0.5 nM data, `nnlag.py` and `blockphase.py` reproduce the development numbers exactly.

## Rows and grids

- **Ours**
  - Detector p ≥ 0.63 (`good`). Positions and σ: the protocol below was built on the pattern-solve output (the tables
    before 2026-11-17); since then a run's result is the pass-2 emitter network's own output (`locs_p2_net`: its position
    and the sd of its joint covariance, no factor), the solve tables (`locs_p{1,2}_solve`) are written next to it. Which
    table a statement uses must be said with it.
  - σ = mean of the two reported axes.
- **Uncertainty checks (2026-11-17, held-out simulated movies, `analysis/sigma_joint/net_vs_solve.py`, paper_figures branch).**
  Per photon bin: robust sd of the GT error, z = rms(error / reported σ), CRLBs (SMLM on the 6-frame sum; SIMFLUX with a
  background per frame at the true pattern). The pass-2 network's σ is calibrated (z 0.85-1.05); the solve was
  overconfident with the fixed c = 0.1 (z 1.3-1.4); the brightness noise c, c_a is now estimated per dataset with the
  pattern (METHOD.md stage 3) and the kinetic model averaged per row (stage 4).
  - Window start f; windows start every 3 frames.
- **SIMCODE (b)**
  - The paper networks and pipeline (MovieProcessor, eq.-2 NDI fit, filters), with the phase-drift sign fixed
    (`analysis/baselines/run_simcode*.py --fix_sign 1`).
  - Its window starts are stride 1.
- **Drift and crop.** Each method is drift-corrected with its own DME v2 drift, at time f + 3 (`analysis/baselines/dme_drift.py`,
  using the external `dme` package). Crop 5–115 px after correction.
- **Common window grid.** Comparisons use G3 (f % 3 == 0) at 0.5 / 2.5 nM and G6 (f % 6 == 0) at 20 nM. Mixing grids was
  an artefact (critic 35).

## Metrics

| metric | definition (`lib.py`) | what it measures |
|---|---|---|
| **identical rows** | `pair`: exact window start, mutual nearest neighbour < 0.5 px | the only basis for per-localization claims |
| **NN, lag-averaged** | `lagavg_shared` (implementation removed from the package 2026-11-15, no caller; git history of simcode/evaluation/metrics.py). For identical rows, the distance to the nearest row of the same method at f + lag (doubly shared: the partner is also an identical row). σ per axis from a Rayleigh + uniform mixture fit (`sig1d_fine`, 0.01-nm grid). Mean over lags 12, 18, 24, 30, 36. CI = event-block Poisson bootstrap (200 replicates; blocks = linked events). | window-to-window error. Exact on synthetic components; blind to all static error. In sim it flatters ours p1 by 0.03–0.05, so quote ratios ±0.05. |
| **FRC std** | `frc_std`: 1/7 threshold, alternating 300-frame halves, 5-nm render | ranks correctly but exaggerates gaps; rewards per-event and per-site static error. Secondary, never a magnitude. Not resolution evidence on overlapping windows. |
| **gap300 FRC** | `frc_gap300`: 300-frame blocks 300 frames apart, (0 vs 2) and (1 vs 3) of each 1200 frames | event level (window-to-window + per-event static) on **identical rows only**. Across pools it ranks by how many sites are sampled in both halves (3 of 7 wrong in sim). |
| **block phases** | `blockphase.py`: gap300 with the block boundaries shifted by 0 / 75 / 150 / 225 frames, plus the +3 nm iid injection per phase | write "not measurable" when the +3 nm sensitivity is below the phase scatter |
| **coverage pools** | `covpool.py`: random N_SC; global top-N_SC; global top-N at SIMCODE's coverage (cov); per-tile N (tileN); per-tile coverage (tilecov); per-tile σ quantile at the matched global coverage (tileq). Coverage = occupied 20-nm bins (real) or distinct GT events (sim). | selection-level statements; quote with NN (own set, lag-averaged) or FRC std |
| **spatial drivers** | kept fraction in the low / high tercile of the illumination envelope and of the depth field, relative to ours-all | 1 = spatially neutral; SIMCODE itself is 0.39–0.58 (centre weighted) |
| **revisit / event-revisit** | `revisit`, `event_revisit` | realism check of the sims (binding statistics); diagnostic only |
| **GT decomposition (sim)** | `truth`: per axis, dyn = var − cov(same event); event = cov(same event, ≥ 12 frames apart) − site; site = cov(different events, same site) | the sim ground truth all GT-free metrics are validated against |
| **zcal / M4** | rms(e / σ) per σ-quantile bin (sim); real: consecutive photon-disjoint windows of a site, z = Δr / √(σ_w² + σ_w+6²) | M4 reads 0.8–0.9 for a true zcal of 1 at > 700 ph (window-common errors cancel), so compare real M4 with the like-for-like sim M4, not with 1. Lag-averaged NN cannot detect the winner's curse of a σ-ranked selection. M4 is a diagnostic only: until 2026-11-17 the real-data σ multiplier (`sig_ph_real_mult` of the removed `_real` calibrations) was fitted from the real 0.5 nM M4 × the sim (true / M4) ratio; σ is no longer rescaled with anything measured on real data. |

## Rules

1. **Per-localization claims only on identical rows** of a common grid.
   - Window-to-window: NN lag 12–36, quoted as a ratio and in quadrature nm.
   - Event level: gap300 with block phases and the injection scale.
2. **Pools are selection statements.** That covers random, photon-matched and σ-filtered pools, and the strict product.
   - Compare them with NN or FRC std.
   - Never compare them with gap FRC or the revisit proxy across pools.
   - Report N, coverage (events / sites / occupied bins), median photons and the spatial drivers next to every pool.
3. **Coverage matching for filtered products.** A σ-filtered product is compared at SIMCODE's coverage. For SIMCODE's
   spatial weighting use tilecov; for a neutral product use the per-tile quantile (tileq = `simcode/table.py`). The
   global σ ≤ 6 nm / top-N_SC product is retired: it had half the coverage and 2× the centre bias.
4. **Unknown, not "tie".** Per-site static error, absolute accuracy and the linear / field terms are not GT-free
   measurable. They are stated as unknown for both methods. Measuring them needs a calibration sample.
5. **Sim FRC is not evidence; sim GT metrics are** (critic 36).
6. **The strict product's σ** is overconfident in its best bins. Quote GT RMS (sim) or NN (real) for it, never its σ
   threshold.
7. **Pre-register acceptance rules** before evaluating a new network. Decisions follow the rules, not post-hoc readings.

## Running it

```bash
# dataset spec: see archive-2026-11-15:analysis/evalproto/datasets.py (paths to the simcode output dirs, the SIMCODE npz, the DME drifts)
python archive-2026-11-15:analysis/evalproto/nnlag.py      my_dataset.json [out_dir]   # identical-row NN, lags 12-36
python archive-2026-11-15:analysis/evalproto/blockphase.py my_dataset.json [out_dir]   # event-level gap300, 4 block phases, +3 nm sensitivity
python archive-2026-11-15:analysis/evalproto/covpool.py    my_dataset.json [out_dir]   # coverage-matched / tile-stratified pools
```

Sim datasets (`"mode": "sim"`, `"gt": <sim_gt.npz>`) add the GT decomposition. The benchmark movies come from
`analysis/simreal/recipes.sh`, and the methods are run with `archive-2026-11-15:analysis/simreal/eval_movie.sh`.
