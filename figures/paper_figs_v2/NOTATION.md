# Notation of the SIMCODE paper figures (2026-11-15)

Rule (user, 2026-11-15): every figure and caption uses the symbols of the SIMCODE preprint; no letter of the preprint is
redefined. 4Pi-SIMCODE may add symbols where needed; they must not clash with a preprint symbol. Where a symbol is not
defined in the main text, the figure panel uses words.

Sources
- P = SIMCODE preprint, bioRxiv 10.1101/2025.03.09.642203 (13 March 2025):
  /mnt/data/dmdsf/simcode2_out/refs/simcode_biorxiv_2025.pdf (text: simcode.txt next to it). Page numbers = the preprint's
  page footer (x/13).
- PS = preprint SI: old_SI.tex (sections / equation order; the reviewed PDF SI_review.pdf is the marked-up
  diff of the same SI).
- R = revised main text main_after_1st_review.tex, RS = revised SI SI_after_1st_review.tex (read-only
  copies; the laptop session owns the text, so they may be stale).
- 4P = 4Pi-SIMFLUX (Nature Methods 2025, doi 10.1038/s41592-025-02908-8): names the four interference images P1, S1, P2, S2
  ("four-phase images", phase delays 0, pi/2, pi, 3 pi/2) and the six SIM sub-images X1-X3, Y1-Y3; it defines no formula
  symbols for the 4Pi phase.

## 1. Preprint symbols (unchanged; these are the letters the figures use)

| symbol | definition (exact wording where short) | where |
|---|---|---|
| i | modulation angle index, i in {1, 2} (also "exposure index" in PS 4.1, see section 5) | P p.6 (Data processing); P p.5 |
| j | phase step index, j in {1, 2, 3} | P p.6; P p.5 |
| x, y | 2D emitter position | P p.5, p.6 |
| N_{i,j} | emitter intensity (in number of photons) at the i-th angle and the j-th phase step | P p.5 (SIMCODE architecture) |
| b_{i,j} | background (in photons/pixel) at angle i, step j | P p.5 |
| theta | parameter vector: network: theta = (x, y, N_{1,1}, ..., N_{2,3}, b_{1,1}, ..., b_{2,3}); fit: theta = (x, y, N_1, N_2) | P p.5; P p.6 Eq. 1 |
| mu_theta, sigma_theta | GMM mean and covariance ("assumed to be diagonal") | P p.5 |
| mu-hat_x, mu-hat_y, mu-hat_{b_{i,j}}, mu-hat_{N_{i,j}} | network estimates of position, background, emitter photon counts | P p.2 (Results); P p.6 |
| sigma-hat_{N_{i,j}} | network-estimated standard deviation of the intensity ("intensities and variances (mu-hat_N, sigma-hat_N)") | P p.6 |
| phi | probability of having an emitter (per pixel) | P p.10 Fig. 1 caption; PS 1.1 (phi_i at pixel i) |
| mu, sigma | properties and uncertainties of the emitter (network output) | P p.10 Fig. 1 caption |
| e_{i,j}(theta) | expected intensity (modulation pattern), Eq. 1: e_ij = (2 N_i / A)(1 + m_i sin(q_i . (x, y) - phi_{i,j})) | P p.6 Eq. 1 |
| phi_{i,j} | modulation phase for angle i and phase j | P p.6 |
| q_i (vector) | modulation frequency for angle i | P p.6 |
| N_i | emitter intensity for angle i | P p.6 |
| m_i | modulation contrast of the illumination pattern for angle i (PS: "modulation depth") | P p.6; PS Eq. 4 |
| A | normalization factor (Eq. 1); also area (PS Eq. for p_on), set A (Jaccard), number of angles (PS 1.1) | P p.6, p.7; PS |
| theta-hat | Eq. 2: argmin sum_ij ((mu-hat_N_ij - e_ij(theta)) / sigma-hat_N_ij)^2 | P p.6 Eq. 2 |
| R_{i,j}(theta), r(theta) = vec(R) | residual matrix / vector of the LM fit | P p.6 Eq. 3, 4 |
| J(theta), J_{k,l}, k, l | Jacobian and its indices | P p.6 |
| D, lambda | LM damping matrix and damping constant (100) | P p.6-7 Eq. 4 |
| theta-hat_dl, sigma_dl | position estimate of the deep-learning model (no pattern information) and its uncertainty (from the GMM) | P p.7 |
| theta-hat_pf, sigma_pf | position estimate from the pattern fit and its uncertainty (CRLB) | P p.7 |
| theta-hat', a, b | MVU combination a theta-hat_dl + b theta-hat_pf | P p.7 Eq. 5-8 |
| J(A, B) | Jaccard index of sets A (localizations) and B (ground truth) | P p.7 Eq. 9 |
| FRC, RSP, CRLB | Fourier ring correlation, resolution-scaled Pearson coefficient, Cramer-Rao lower bound | P p.7, p.12 |
| sigma (PSF) | standard deviation of the Gaussian PSF (1.3 px) | PS 2 (Simulation) |
| N_f, eta_f, f | intensity of a spot at frame f, N_f = N (1 + eta_f), eta_f ~ N(0, 0.5); N mean spot intensity | PS 1.1 |
| L, L_gmm, L_count, w_count, p_gmm, W^2, K, E, mu_count, sigma_count, mu_{i,k}, sigma_{i,k} | loss terms, GMM density, pixels per frame, parameter count, number of emitters, count-loss moments, per-pixel GMM mean / variance | PS 1.1 Eqs. 1-4 |
| k_on, k_off, p_on, rho | blinking transition probabilities, on-probability, density (p_on = rho A / N, N = number of binding sites) | PS 2 Eq. 5 |
| theta_x, theta_y, theta_{N_i} | localization parameters used for the pattern estimation, i = exposure index | PS 3.1 |
| gamma | modulation error, gamma = max_ij abs(mu_N_ij / sum mu_N - e_ij(theta)); filter gamma > 0.1 | PS 3.2 Eq. 6 |
| e_{i,j}(theta) (normalized) | (1/6)(1 + m_i sin(q_i . (x, y) - phi_{i,j})) | PS 3.2 Eq. 7 |
| r_i, p, k_x, k_y, X, Y, theta, Delta, theta_1, theta_2, f_1..f_3, d, NA_ill, phi_i | DMD pattern (sine and binary), pitch wavenumbers, mirror coordinates, grating orientation and period, diffraction angles, focal lengths, order distance, illumination NA, DMD phase steps | PS 4 Eqs. 8-15 |
| bg ~ U(1, 6) | per-frame background of the CRLB test (photons/px) | PS Fig. (crlb_vs_precision) |
| mu_{N_1} / sum_i^6 mu_{N_i} | normalized intensity of the pattern plots | PS Fig. (pattern_plots) |
| "NN Output", "SIMCODE" | legend names of the CRLB figure | PS Fig. (crlb_vs_precision) |

## 2. Added by the revised manuscript (R / RS; the figures follow them)

| symbol | definition | where | note |
|---|---|---|---|
| p | detection probability; selection p >= 0.63 | R Methods (comparison protocol, Data processing) | the preprint calls it phi; p avoids the clash with the phase phi_ij |
| sigma_x, sigma_y | calibrated precision; selection max(sigma_x, sigma_y) <= 0.15 px (16.25 nm; common rule, 2026-11-15) | R | |
| e_{i,j}(theta) = N r_{i,j} (1 + m_i sin(q_i . (x, y) - phi_{i,j} - eps_i(t))) | revised Eq. 1 | R Data processing | the figures use exactly this form |
| N | emitter intensity (one per window); theta = (x, y, N) | R | replaces N_1, N_2 of P |
| r_{i,j} | relative illumination intensity of angle i and step j | R | clashes with the residual vector r(theta) / R_ij in the same section (see 6) |
| eps_i(t) | phase drift of angle i at the time t of the window | R; RS (eps(t)) | |
| Sigma, C (Sigma = C C^T) | intensity covariance from the network, its Cholesky factor | R | RS writes Sigma = L L^T (see 6) |
| "6 x 6", full covariance | "the covariance of the 6 intensities is a full 6 x 6 matrix" | R (SIMCODE architecture); RS 1.1 | see 6: the figures no longer show it |
| z | emitter height (PSF, in-situ estimation; 3D prior) | RS 2 | |
| beta, chi^2 | slope of the intensity-regression loss term; chi^2 per pixel of the PSF fit | RS | |

## 3. Index conventions used in the figures

- Frames of a window: angle i = 1, 2 and phase step j = 1, 2, 3 (1-based, as P). The code's frame t of a window (0-5) is
  t = 3 (i - 1) + (j - 1); the code's a = i - 1, s = t % 3 = j - 1. Panels write "angle 1 / 2", "step j = 1, 2, 3".
- t is TIME (of the window), only in eps_i(t) and the 4Pi phase field, as in R. No figure uses t as a frame index any more.
- Phase convention: the preprint's sin(q_i . (x, y) - phi_{i,j} - ...). The code uses cos(k_a . r - psi_s - ...), so
  phi_{i,j} = psi_s - pi/2 (s = 3 (i - 1) + j - 1). si_phase_decouple shows phi_ij (values shifted by -90 deg); the
  SIMCODE panel of si_pattern_plots is plotted in the sin convention (phase + pi/2), so both panels now share one convention.
- 4Pi channels: ch1-ch4 in panels (1-based; ch1 = the reference channel = code c0; ch3, ch4 = the mirrored code c2, c3). If the
  text needs an index: c in {1, 2, 3, 4} (new, no clash in P).

## 4. Figure symbols: old -> new

### fig_pipeline_v2 (main Fig. 1)
| old (panel) | new | meaning |
|---|---|---|
| frame axis `t` (C) | removed; "6-frame window = 2 angles (i) x 3 phase steps (j)" | defines i, j |
| logit + sub-pixel offset; p >= 0.3 (D) | "detection prob. p + sub-pixel offset" | p as in R |
| I_t = w_t . (y_t - bg_t) + correction (D) | words: "intensity = learned weighted sum of the pixels minus background, + correction" | |
| I_1 ... I_6, "full 6x6 covariance (Cholesky L)", covariance glyph (D) | "intensities mu-hat_{N_{i,j}} +- sigma-hat_{N_{i,j}}" (bars with error bars); glyph removed | the preprint's description; ablation: off-diagonals no measurable effect in pass 2 (out/ablation_cov/REPORT.md) |
| bg_1 ... bg_6 (D) | "backgrounds mu-hat_{b_{i,j}}" | |
| 3D prior (x, y, z) + 3x3 covariance; z given x, y: z + b . Delta r (D) | "position theta-hat_dl = (x, y, z) +- sigma_dl (pattern-free)" | theta-hat_dl of P, extended by z |
| I_t, r on the D -> E arrow | mu-hat_N, theta-hat_dl | |
| I_t = B rel_s [1 + m_a cos(k_a . r - psi_s - eps_a(t))] (E) | e_{i,j} = N r_{i,j} [1 + m_i sin(q_i . (x, y) - phi_{i,j} - eps_i(t))] | R Eq. 1 |
| axis k_a . r - psi_s - eps_a(t), I_t / B (E) | q_i . (x, y) - phi_{i,j} - eps_i(t), mu-hat_{N_{i,j}} / N | |
| k_a, psi_s, rel. intensities, m_a, eps_a(t) (E) | q_i, phi_{i,j}, r_{i,j}, m_i, eps_i(t) | |
| "NLL along k_a" (G) | "cost along q_i" | |
| LM fit of (x, y, B [, t_on, t_off]) to I_t with Sigma(mu) = s Sigma_net + c^2 diag(mu^2) + c_a^2 mu mu^T, fixed c, c_a (G) | "LM fit of theta = (x, y, N): e_{i,j}(theta) to mu-hat_{N_{i,j}}, weighted by sigma-hat_{N_{i,j}} (+ a term proportional to e_{i,j}^2 for brightness fluctuations); + Gaussian prior term theta-hat_dl, sigma_dl" | |
| on/off kinetics kept if Delta(-2 log L) > 9.21 (G) | "switching on / off within the window: kept if it lowers -2 log-likelihood by > 9.21" | L would clash with the loss L / Cholesky L |
| z = z_p + b . (r - r_p) (H) | words: "its z is that candidate's z (D), corrected for the solved x, y" | b would clash with b_{i,j} |
| selection p >= 0.63, max(sigma_x, sigma_y) <= 0.15 px (16.25 nm; common rule, 2026-11-15); table x, y, z, sigma_x, sigma_y, sigma_z | unchanged | R |

### si_network_structure_v3 (SI network structure)
| old | new |
|---|---|
| sign(y) log(1 + abs(y)/30); (y - b-bar)/100; b-bar (y = photons clashes with the coordinate y) | "log-scaled photons"; "linear (photons - border bg)"; formula in the caption facts with n = patch photons |
| logit (p = sigma(.)), d_{x,y} = 1.5 tanh, log sigma_off = -5 + 6 sigma(.) (sigma = sigmoid clashes with the uncertainties) | "detection probability p (sigmoid of a logit); sub-pixel offset (<= 1.5 px) and its uncertainty" |
| weight map -> m, per-frame maps -> D_t (clash with m_i, D) | "weight modulation", "per-frame weight changes" |
| b_{0,t} scalars | "border-median background per frame" |
| read-out box g, w_0, w, w_t, I_lin,t = w_t . (y_t - bg_t) | words (template-matched weights x modulation, unit response to the template, + per-frame changes; weighted sum of photons - background) |
| 41-output legend: dI, log diag L, off-diag L, dx dy, log sigma_xy, bg, z / log s_z|xy / b_x / b_y | intensity corrections, intensity uncertainties, intensity correlations, x, y offset, sigma_x, sigma_y, backgrounds, z, sigma_z, z slopes |
| outputs box: bg_t = b0 + q sqrt(.), I_t = I_lin + c dI, 6x6 Cholesky L, L_p = [[...]] (c, q, b, L clash) | "backgrounds mu-hat_{b_{i,j}}", "intensities mu-hat_{N_{i,j}}", "uncertainties sigma-hat_{N_{i,j}} (with their correlations, as a Cholesky factor)", "position theta-hat_dl, sigma_x, sigma_y, z with sigma_z and slopes" |
| ResBlock(c, d): x + ... | ResBlock(channels, dilation d): h + ... |

### si_4pi_pipeline (SI 4Pi-SIMCODE pipeline): words in the panels
| old | new (panel) | proposed symbol if the SI text needs one (no clash with P) |
|---|---|---|
| c0-c3 | ch1-ch4 | c in {1, ..., 4} (channel index) |
| (dx_c, dy_c) (red) | "channel offset(s)", "sub-pixel offset kept per channel (ch1: 0)" | (Delta x_c, Delta y_c) |
| Y_tck, "96 read-outs Y_tck", "4 x 4 cov. per (t, c)" | "96 read-outs: 4 per frame and channel", "+- uncertainties (analytic, x learned scale)" | none needed |
| W_ck, D, D_ck, y_tc, bg_c, p_0 | "read-out windows", "window changes", "pixels", "backgrounds: one per channel", "prior position" | none needed |
| Y_tck = sum_px (W_ck + D_ck)(y_tc - bg_c) | "(window + change) . (pixels - background), per frame, channel and window" | |
| SIM pattern k_a, psi_t, m | q_i, phi_{i,j}, m_i | preprint symbols |
| 4Pi phase field Phi_0(x, y, t), per-angle offset delta_a | "4Pi phase field (per file; tiles x time)", "+ an offset per angle" | Psi(x, y, t) (4Pi phase field), Psi_i (offset of angle i). Not Phi: too close to phi / phi_ij |
| E[Y_tck] = N_t sum W'_ck g_c(x + dx_c, ...; Phi_0 + delta_a); N_t = F [1 + m cos(k_a . r - psi_t)] | "expected read-out = e_{i,j}(theta) x the window applied to the channel 4Pi PSF at (x, y, z) + offset, with the 4Pi phase of the field"; e_{i,j}(theta) = N [1 + m_i sin(q_i . (x, y) - phi_{i,j})] | |
| Lambda = 257 nm; 3 starts prior z, +- Lambda/2 | "period 257 nm"; "3 starts: prior z, +- half a period" | Lambda (fringe period; P uses lambda only for the LM damping / wavelength) |
| -log L vs z | "cost vs z: the prior picks the fringe" | |
| LM fit of x, y, z, F [, t_on, t_off], bg_c (4) | "LM fit of theta = (x, y, z, N), a background per channel [, on / off switching]" | N (= code F) |
| SIMCODE note "I_1...I_6, 6 x 6 cov., bg_1...bg_6, prior with 3 x 3 cov." | "6 intensities +- uncertainties, 6 backgrounds, position (x, y, z)" | |
| table (f): I_t: 6 / Y_tck: 6x4x4; covariance 6x6 / 4x4 per (t, c); solve x, y, B / x, y, z, F, bg_c; Lambda/2; p, sigma_xy | read-outs "6 intensities" / "4 per frame and channel"; uncertainties "per intensity" / "per frame and channel"; fit x, y, N / x, y, z, N, bg per channel; "half a fringe apart"; p, sigma_x, sigma_y | |

### si_phase_decouple
psi_s -> phi_{i,j} (plotted as psi_s - 90 deg, sin convention); eps_a(t) -> eps_i(t); "angle 0 / 1" -> "angle i = 1 / 2";
"step 0, 1, 2" -> "step j = 1, 2, 3"; caption: orientation a -> angle i; block phases phi_s(b) -> blockphase_ij(b) (phi
is the fixed phase); residual r_s(b) -> res_ij(b) (r_ij is the relative intensity).

### si_pattern_plots
"pattern 0: orientation 0, phase step 0" -> "angle i = 1, phase step j = 1"; I_k / sum_6 I -> mu-hat_{N_{i,j}} /
sum_ij mu-hat_{N_{i,j}} (P / PS); SIMCODE panel moved to the preprint's sin convention (it was cos, pi/2 off the SIMFLUX
panel); m_eff defined in words ("effective modulation depth"). Not redrawn: si_pattern_plots_extended (internal variant; its
redraw needs the 10 M-row pass-1 table from /mnt/data; it still uses the code symbols).

### si_4pi_npc / si_4pi_mt / si_4pi_sim
sqrt(sx^2 + sy^2 + sz^2) -> sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2); "sigma_xy" / panel text "sigma_xy < 5" ->
max(sigma_x, sigma_y) (the code's definition, analysis/fourpi_sim/sweep4pi_eval.py); NPC legend "r ..., w ..." -> "radius
..., width ..."; "(25 < r < 85 nm)" -> "(radius 25-85 nm)". N > 300 (photo-electrons) kept: N = emitter intensity as in P.

### unchanged (already consistent)
si_crlb_precision (bg ~ U(1, 6), "NN Output", "SIMCODE bound", CRLB: as PS), si_dme_drift, si_simulated_live_smlm,
si_review-single-frame (correction note moved to si_review-single-frame_notes_internal.txt), Table S1.

## 5. Not touched here (other worker's scripts): changes for fig_sim_fig2.py / fig_exp_new.py / psfpanel.py

- Fig 2 counts legend "emitter ON within 3σ" (fig_sim_fig2.py l.120): sigma is defined only in the caption (the
  localization's own precision). Suggest "true positive" (legend) with the definition in the caption, or keep and leave the
  caption sentence.
- psfpanel.crlb_text (used by the PSF panels when the text block is drawn): "bg ... ph/px/frame" -> "background ...
  photons/px/frame", "ph / 6-frame window" -> "photons / 6-frame window"; "Gaussian PSF sigma 1.3 px" is the preprint's sigma
  (fine).
- Fig 3 / Fig 4 (fig_exp_new.py): panel text has no pattern / network symbols (FRC, Localizations (10^6), RSP, PSF xy / xz);
  caption "sigma < 0.15 px" -> "precision < 0.15 px (CRLB for SMLM / SIMFLUX, network uncertainty for DECODE, calibrated
  uncertainty for SIMCODE)" as in R; "max(sigma_x, sigma_y)" and "p >= 0.63" as in R. "depth" = modulation depth m_i.

## 6. For the manuscript text (paper session)

1. Network output: R says "the covariance of the 6 intensities is a full 6 x 6 matrix" and "The key difference ... their full
   covariance"; RS 1.1 has L_I with Sigma = L L^T. The ablation (out/ablation_cov/REPORT.md) found no measurable effect of
   the off-diagonal elements in pass 2 (the product). The figures now describe the output as P does (intensities with
   uncertainties sigma-hat_N_ij); the text can say the same, with at most one sentence that the network also predicts their
   correlations (used by the fit, no measurable effect).
2. Cholesky letter: R uses C (Sigma = C C^T), RS uses L (Sigma = L L^T; L is also the loss in RS 1.1). Pick C in both.
3. r_{i,j} (relative illumination intensity, R Eq. 1) vs r(theta) / R_{i,j}(theta) (residual vector / matrix, R Eq. 3-4, same
   section): different objects with the same letter. Options: keep r_ij and write the residual as rho(theta)... but that
   changes a preprint letter; or rename the new relative intensity (e.g. w_{i,j} or kappa_{i,j}) and keep the preprint's r.
   The figures currently use r_{i,j} as in R; if the text renames it, fig_pipeline_v2 panel E (two places) and its caption
   follow (one-line change each).
4. Detection probability: P writes phi (Fig. 1 caption, SI loss), R writes p (selection p >= 0.63). phi also is the phase
   phi_{i,j}. Keep p in the text and say once "detection probability p (phi in the loss of SI 1.1)".
5. Position: theta-hat_dl is 2D in P / R ("x, y the 2D position"); the network now also gives z. Fig. 1 writes
   theta-hat_dl = (x, y, z) +- sigma_dl; the text should say that the network position (and so theta-hat_dl) includes z, and
   that z of the final localizations comes from pass 1.
6. Index i: PS 3.1 uses i as "exposure index" (theta_{N_i}) and PS 1.1 as pixel index (phi_i, mu_{i,k}); in the main text i
   is the angle. The SI pattern-estimation paragraph could write theta_{N_{i,j}} (angle i, step j) to match.
7. Phase convention: R Eq. 1 uses sin(... - phi_ij - eps_i(t)); the code uses cos with psi_s = phi_ij + pi/2. Any phase value
   quoted in the text (e.g. the phase-drift range) is a difference and unaffected; absolute phases quoted must be in the sin
   convention (si_phase_decouple_caption.txt lists them: phi_ij deg).
8. 4Pi section: channels ch1-ch4 (or "the four interference channels P1, S1, P2, S2" as 4P names them, if the mapping is
   checked: code c0-c3 order vs P1/S1/P2/S2 is not verified here); new symbols only if needed: channel index c, channel offset
   (Delta x_c, Delta y_c), 4Pi phase field Psi(x, y, t) with per-angle offset Psi_i, fringe period Lambda (257 nm). The 4Pi
   fit's intensity is N (code F), its pattern q_i, phi_ij, m_i (no eps_i(t) within a file; the drift is in the per-file
   estimates).
9. The brightness-fluctuation term of the fit: R says "plus a term proportional to the squared model intensities, which
   accounts for errors in the pattern model". The ablation calls it brightness fluctuation of the emitter beyond photon
   noise (c = 0.10; the per-angle c_a has no measurable effect). Fig. 1 G says "+ a term proportional to e_ij^2 for brightness
   fluctuations"; align the wording of the text and the figure.

## 7. Symbols not mapped / open

- The 4Pi read-out uncertainty is a 4 x 4 covariance per frame and channel (analytic); the figure says "uncertainties"; the
  caption facts keep the matrix. No preprint counterpart.
- The 4Pi depth m: code m_t per frame (repeated per angle); written m_i.
- The channel order c0-c3 vs the 4Pi-SIMFLUX names P1, S1, P2, S2 is not verified, so the figures use ch1-ch4.
- si_pattern_plots_extended still uses the code symbols (not redrawn, internal).
