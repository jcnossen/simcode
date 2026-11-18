# SI note: DECODE training and its residual pixel bias

## Training protocol
- **Network and inputs.** DECODE (SigmaMUNet) on non-overlapping 6-frame sums, with the neighbouring sums as context (3 input channels).
- **Training simulations** (DECODE's own simulator):
  - In-situ PSF v2 model of the 1809/2 recording, emitter heights 0-400 nm.
  - Camera v2 (gain 0.4256 e/ADU).
  - Photons lognormal (median 600, log-sd 0.7 per 6-frame sum).
  - Background 6-30 photons / px / 6-frame sum.
  - 4-160 emitters per 40 x 40 px frame, mean on-time 2 sums.
  - These ranges cover the simulations and the 1809/2 recording.
- **Optimisation** follows the training settings published with DECODE:
  - AdamW, learning rate 6e-4, weight decay 0.1.
  - ReduceLROnPlateau on the test-set loss: factor 0.5, patience 20, cooldown 10.
  - Batch size 64, 32 768 simulated frames per epoch.
- **Departures from the published settings:**
  - A linear learning-rate warm-up over the first 512 steps. Without it, training from scratch at 6e-4 diverged in the first epoch, twice.
  - Gradient-norm clipping, as in the DECODE version used.
- **Stopping.** Trained from scratch to convergence: all monitored metrics flat from epoch ~200. The weights were frozen at epoch 300, after 9.8 M training frames, at learning rate 1.9e-5 after five halvings.
- **Earlier model, for reference.** It stopped at epoch 262 on a time limit (2.1 M frames), with learning rate 2e-4 halved to 1.2e-8.

## Residual pixel bias on the simulations (PSF v2, ground truth known)
Metrics:
- |c1| = first Fourier harmonic of the sub-pixel histogram (0 = uniform).
- Bias = pull toward the pixel centre (first-harmonic amplitude of the error vs the true sub-pixel position), x / y.

| data (all localizations) | earlier model | converged model |
|---|---|---|
| D1 density movie, \|c1\| | 0.30 / 0.25 | 0.12 / 0.14 |
| D1 density movie, bias (nm) | 6.7 / 4.3 | -0.2 / 1.8 |
| D4 density movie, \|c1\| | 0.35 / 0.34 | 0.16 / 0.17 |
| D4 density movie, bias (nm) | 9.4 / 6.0 | -0.6 / 2.2 |
| Dexp1 (Fig 2 photon statistics), \|c1\| | 0.20 / 0.19 | 0.043 / 0.058 |
| Dexp1, bias (nm) | 3.5 / 2.6 | 0.5 / 0.9 |
| isolated emitters, bias (nm) | 4.0 / 1.2 | 0.4 / 0.9 |
| isolated emitters, mean x error (nm) | 3.1 | 1.2 |

- **Precision on isolated emitters.** RMS / CRLB at 350 / 500 / 750 / 1200 photons per window was 1.06 / 1.09 / 1.16 / 1.24 and is now 1.07 / 1.05 / 1.09 / 1.12. Recall rose from 0.64 to 0.71.
- **Conclusion.** The pixel bias is reduced but does not go away: the histograms remain non-uniform (|c1| 0.04-0.17 against a ground-truth floor of 0.016), and a y bias of about 1-2 nm remains at higher density.
- **With the common selection rule** (max sigma < 0.15 px, si_pixel_bias.*):
  - D1: DECODE |c1| 0.040 / 0.100, bias 0.6 / 1.8 nm.
  - D0.5-D4: bias 0.6-0.8 nm (x) and 1.0-2.0 nm (y).
- **DECODE's published model in its own conditions** (its simulator and PSF; out/decode_checks/published_model/STATUS.md):
  - |c1| 0.05-0.06 in x.
  - Bias at most 0.6 nm for isolated emitters overall; about +2 nm in its low-photon range and at 50 emitters per frame.

## Real data: 1809/2
Rows: the fair-FRC set (disjoint windows, max sigma < 0.15 px, crop).

| model | \|c1\| x / y | fair FRC | RSP |
|---|---|---|---|
| converged PSF-v2 model | 0.067 / 0.077 | 28.9 nm | 0.599 |
| older real-data model (dec_1809_2) | 0.006 / 0.009 | 28.1 nm | 0.611 |

- The older model was trained on the averaged in-situ PSF with the real-data ranges and a long on-time (10.8 sums).
- On these data the PSF-v2-trained model pixel-locks more, in all photon terciles (dimmest 0.065 / 0.057, brightest 0.091 / 0.105).
- **All DECODE results of the paper use the converged PSF-v2 models** (user decision 2026-11-16: the fairest option; same PSF v2 as
  every other method, the published training settings, converged):
  - simulations and 1809/2 (Fig. 3): dec_m1809v3 (fair FRC 28.9 nm, RSP 0.599, 0.865 M fair-FRC rows);
  - 2.5 nM/2 (Fig. 4): dec_m25v3, trained with the same protocol on the 2.5 nM PSF v2 (from scratch, frozen at epoch 300 with metrics flat for ~100 epochs).
- **Earlier models, as a check:** 1809/2 with the older real-data model dec_1809_2: fair FRC 28.1 nm, RSP 0.611, 1.04 M rows; 2.5 nM/2
  with dec_2.5nM_2: fair FRC 42.1 vs 37.0 nm (dec_m25v3), RSP 0.62 vs 0.52, rows 1.53 M vs 1.47 M; |c1| x / y 0.037 / 0.041 vs
  0.032 / 0.031 in the fair-FRC set and 0.128 / 0.133 vs 0.065 / 0.083 over all rows (out/decode_checks/v3/real_compare_2.5nM_2.json,
  real_pixlock_2.5nM_2.md). Simulations with the earlier dec_sim1809v2: out/decode_checks/v3/compare_sims.md.
- The cause of the stronger pixel locking of the PSF-v2 model on 1809/2 was not isolated. A PSF-model mismatch at the pixel level is plausible: the PSF v2 and averaged-PSF images differ by 9-15 % (relative L1) at 50-200 nm height.

## SIMCODE's own residual on the same data, for comparison
- **Simulations**, with the same rule and metrics as DECODE:
  - Bias 0.8-1.9 nm (D0.5-D4: x 0.8-1.3, y 1.4-1.9 nm; 2026-11-17 final method, before: x 0.6-1.4, y 1.3-1.5 nm); isolated emitters (true pattern) 0.9 / 0.7 nm at 500 photons.
  - |c1| 0.09-0.14 (before: 0.09-0.12).
  - The SIMCODE pixel-bias study (out/pixel_bias_simcode/STATUS.md) traces most of this to the p >= 0.63 selection. Without the p cut its isolated-emitter bias is 0.1-0.2 nm.
- **Real 1809/2**, fair-FRC rows:

| method | \|c1\| x / y |
|---|---|
| SIMCODE | 0.032 / 0.048 |
| SMLM | 0.008 / 0.006 |
| SIMFLUX | 0.005 / 0.003 |

(SIMCODE values of the 2026-11-16 table. The final method's pass-2 solve gives the same sub-pixel statistics: on the raw, not drift-corrected positions of the fair-FRC rows |c1| x / y = 0.075 / 0.087 before and 0.075 / 0.085 after.)

## Sources
- Training: out/paper2/decode/lutpsf/dec_m1809v3/ (train_log.csv, eval_log.jsonl, train_v3.json; frozen weights model_e0300_converged.pt). Code: analysis/baselines/decode/train_v3.py (preset m1809), branch decodev3.
- Published settings: DECODE experimental_data_workflow param_run.yaml (/mnt/data/dmdsf/public_decode/).
- Simulation comparison: out/decode_checks/v3/compare_sims.md / .json.
- Isolated-emitter eval: out/decode_checks/v3/eval_ref_dec_sim1809v2.json and dec_m1809v3 eval_log.jsonl, epoch 300.
- SI figure numbers: out/paper_figs_v2/si_pixel_bias_values.json.
- Real data: out/decode_checks/v3/real_pixlock_1809_2.md and real_compare_1809_2.json.
- PSF comparison: out/decode_checks/v3/psf_v2_vs_avg.json.
