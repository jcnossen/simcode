# Per-pattern-angle PSF: photoselection and the in-situ 3D PSF

Note of 2026-11-11 (psf3d work). It covers the model, its derivation in the vectorial PSF, and what the in-situ fits on the paper
movies show. Numbers come from `out/psf3d/` (tables: `python analysis/psf3d/summary.py`). The log is in `out/WORKLOG.md`.

## 1. Setup

The SIMCODE setup (preprint, Methods) has, after the DMD:
- a quarter-wave plate
- then a pizza (azimuthal) polarizer in the Fourier plane, "to ensure the purity of the desired s-polarization state at the
  back-focal plane of the objective for both x- and y-oriented patterns"
- then a 3-mm dichroic, which "can subtly modify the polarization state"

The two beams of pattern angle a interfere into fringes with wavevector k_a. The beams' in-plane wavevectors are ±k_a/2. s-polarization
is perpendicular to the plane of incidence, which contains k_a and z. So the excitation field in the sample is in-plane and
perpendicular to k_a:

    ê_a = ẑ × k̂_a  (in-plane, ⟂ k_a).

The calibration k (x = column, y = row, rad/px) is k_0 = (3.04, 0.06) and k_1 = (−0.07, 3.03). So:
- angle 0 (frames 0–2 of the cycle, frame % 6 // 3 = 0) is excited with E ∥ y: ê_0 = (−0.021, 1.000)
- angle 1 is excited with E ∥ x: ê_1 = (−1.000, −0.023)

## 2. Model and derivation

**Absorption.** A dye with absorption dipole μ̂ is excited with a probability ∝ |μ̂·E_a|² = μ̂ᵀK_aμ̂. Here
K_a = Re(E_aE_a^†)/|E_a|² is the excitation polarization tensor (trace 1). For pure s-polarization, K_a = ê_aê_aᵀ.

**Emission, rigid dye.** Assume an ensemble of orientations that is isotropic before excitation, collinear absorption and emission
dipoles, and no rotation during the excited-state lifetime. Using ∫μ_iμ_jμ_kμ_l dΩ/4π = (δ_ijδ_kl + δ_ikδ_jl + δ_ilδ_jk)/15, the
emission-dipole second-moment tensor of the excited population is

    M = ∫(μ̂ᵀKμ̂) μ̂μ̂ᵀ dΩ / ∫(μ̂ᵀKμ̂) dΩ = (I + 2K)/5.

**Partial mobility.** Rotation within the lifetime scales the anisotropic part by the orientation correlation ⟨P₂⟩ over the
emission time:

    M_a = (1 − γ) I/3 + γ K_a,   γ = (2/5)⟨P₂⟩ = r (the steady-state fluorescence anisotropy, r₀ = 2/5).

- A dye that wobbles fast in a cone of half-angle α about a random axis has ⟨P₂⟩ = S², with S = cos α (1 + cos α)/2.
- A freely rotating dye has γ = 0.

**Image.** Let G_i(r) be the vectorial image field of a unit dipole along i. The model already computes the 6 components
[E_x, E_y] × [dipole x, y, z]. Emission from different molecules is incoherent, so

    I_a(r) = Σ_ij (M_a)_ij Re[G_i(r)·G_j(r)*].

The current model is M = I/3. The image is linear in M, so

    PSF_a = PSF_iso + γ (PSF_{K_a} − PSF_iso).

Code: `VectorialPSF.dipM` / `forward(dipM=...)`, `vectorial.photosel_tensor`.

**Imperfect s-polarization.** Suppose a fraction ε_a of the evanescent excitation intensity is p-polarized. At
n_eff = NA_ill = 1.47 and n_med = 1.33, the p field has |E_∥|² : |E_z|² = (n_eff² − n_med²) : n_eff² = 0.154 : 0.846, with E_∥ along k̂.
Then

    K_a = (1 − ε_a) ê_aê_aᵀ + ε_a (0.154 k̂_ak̂_aᵀ + 0.846 ẑẑᵀ).

The two beams of an angle are mirror images, so the p interference term is |E_z|² − |E_∥|² (vs +1 for s). The depth is then

    m_a = 1 − 2 · 0.154 · ε_a = 1 − 0.31 ε_a

(times the other contrast factors). The measured depths 0.925 / 0.955 would need ε_0 − ε_1 ≈ 0.1 if the p admixture were their only
cause. In situ, ε is not constrained: the free (γ, ε_0, ε_1) fit gives ε ≈ 0.4–0.5 and no held-out gain (§4). So p-polarization is
neither supported nor needed by the PSF data. The depth difference may have other causes (beam balance, DMD fidelity).

**What photoselection cannot do.** With pure s-polarization, the angle average is

    (M_0 + M_1)/2 = (1 − γ) I/3 + γ (x̂x̂ᵀ + ŷŷᵀ)/2.

That is isotropic in-plane, so it cannot produce the constant (angle-independent) x-elongation. That part needs a
detection-side term: here an image-plane line blur (`VectorialPSF.blur`).

The angle average also lowers the z-dipole weight, to (1 − γ)/3. The data reject that coupling: at a frozen line-blur PSF, the
held-out NLL rises monotonically with γ (+95 / +317 / +809 / +1773 nats at γ = 0.03 / 0.06 / 0.10 / 0.15, 0.5 nM). They prefer a slightly
larger z weight. The fitted form therefore frees it:

    M_a = (1 − γ) diag((1 − ζ)/2, (1 − ζ)/2, ζ) + γ ê_aê_aᵀ   (photosel mode 3; ζ = 1/3 is the isotropic case).

**Sign check (vectorial model, calibrated focus, z = 0.05 µm).**
- A pure x-dipole image is x-elongated: Mxx − Myy = +0.36 px² in r < 3 px.
- A pure y-dipole image is y-elongated: −0.26 px².
- So s-polarization predicts that angle 1 (E ∥ x) is the more x-elongated angle. That is the measured sign in all 4 movies.

## 3. Measured per-angle difference (data, model-free)

The statistic is the held-out residual moments of isolated-emitter tracks: aperture r < 3 px about the fitted centre, fitted
background removed. The angle contrast [Mxx − Myy](angle 1) − [Mxx − Myy](angle 0) of the DATA is:

| movie | contrast (px²) |
|---|---|
| 0.5 nM | +0.039 ± 0.007 |
| 1809/2 | +0.047 |
| 2.5 nM/2 | +0.024 |
| 20 nM/2 | +0.032 |

The same tool on sims without any angle term gives +0.002 to +0.004 (tool baseline). The sign and size agree across the movies.

**A within-emitter effect, not only a population effect.** Bin the samples by their brightness ratio between the angles: the contrast
is present in every bin (+0.02 to +0.07 px²). It is largest for emitters that are brighter in angle 0.

**Orientation memory.** The brightness ratio ln(A_1/A_0) of an event correlates:
- 0.70 between consecutive 12-frame samples of the same event
- 0.62 → 0.46 → 0.29 between events at the same site, for Δt = 13–60 / 60–300 / > 300 frames
- 0.25 between nearby sites
- 0.0 between simultaneous far-apart emitters

So part of the per-angle brightness ratio is per event or slowly varying. This is consistent with partially oriented dyes and with
pattern geometry. It is a population/orientation-diversity effect that a single per-angle PSF does not model.

## 4. In-situ fits: which per-angle model

Protocol (pre-registered in the WORKLOG):
- per movie: fit on checkerboard parity cb0, evaluate held-out on cb1 (6000 samples)
- each added term is tested on top of the FROZEN line-blur fit, so the test sees only the new term
- held-out NLL: paired, trimmed z (|d − median| < 10 nats)

| added per-angle term (2 params each) | 0.5 nM | 1809/2 | 2.5 nM/2 | 20 nM/2 |
|---|---|---|---|---|
| photoselection (γ, ζ): ΔNLL, trim z | −245, **+10.3** | −62, **+5.3** | −3473, **+5.9** | −1677, −0.2 |
| photoselection: angle contrast, model vs data (px²) | 0.025 / 0.039 | 0.034 / 0.048 | −0.004 / 0.023 | 0.000 / 0.034 |
| traceless per-angle blur Δ: ΔNLL, trim z | −54, **+4.7** | −72, **+5.3** | −468, **+4.1** | −58, **+5.4** |
| per-angle blur: angle contrast, model vs data | 0.040 / 0.039 | 0.053 / 0.049 | 0.008 / 0.025 | 0.043 / 0.033 |
| photoselection + per-angle blur vs photoselection alone (0.5 nM) | −6 nats (n.s.) | | | |

Photoselection vs per-angle blur, head to head (photoselection better = +):

| movie | trimmed z | untrimmed |
|---|---|---|
| 0.5 nM | +8.5 | |
| 1809/2 | +2.8 | equal |
| 2.5 nM | +5.8 | |
| 20 nM | −0.7 | +2.3 |

Fitted photoselection parameters (fits on all samples; Fisher SE from 1500 held-out samples, 0.5 nM; corr(γ, ζ) = 0.63):

| movie | γ | ζ |
|---|---|---|
| 0.5 nM | 0.080 (cb0 fit: 0.074 ± 0.008) | 0.365 |
| 1809/2 | 0.094 | 0.361 |
| 2.5 nM | 0.057 | 0.419 |
| 20 nM | 0.043 | 0.41 |

As a cone half-angle for fast wobble (γ = 0.4 S²), α ≈ 53–62°: a mobile dye, far from rigid (γ_rigid = 0.4). This
interpretation assumes collinear absorption/emission dipoles, an isotropic axis distribution, and wobble that is fast vs the
lifetime. The dense movies' γ is less reliable (z/focus gauge at z ≈ 0, contamination; section 6).

**Outcome.**
- Photoselection passes the held-out rule on 3 of 4 movies, with the largest likelihood gains. It is therefore the accepted per-axis
  model and the recommended one.
- It reproduces 65–70 % of the measured angle contrast on the low-density movies (0.5 nM, 1809/2). On the sims through the
  simulator, it reproduces the real per-angle moments within 0.01 px²; the contrast is within 2 SE (section 5).
- The phenomenological per-angle blur passes on all 4 movies and matches the contrast better. It is delivered as the
  alternative (`*_psf3d_ad*.json`).
- Adding the per-angle blur on top of photoselection gives no significant gain (0.5 nM).

## 5. Per-axis difference of the delivered 0.5 nM PSF and its consequences

Delivered PSF:
- constant line blur: covariance [0.211, 0.000, 0.0005] px² (σ = 0.46 px along x)
- photoselection: γ = 0.080, ζ = 0.365, ê_0 ∥ y, ê_1 ∥ x

| z | Mxx − Myy, angle 1 − angle 0 (r < 3 px) | FWHM x/y, angle 0 | FWHM x/y, angle 1 |
|---|---|---|---|
| 0.00 µm | 0.052 px² (610 nm²) | | |
| 0.05 µm | 0.050 px² (580 nm²) | 3.052 / 3.037 px (1.005) | 3.123 / 2.897 px (1.078) |
| 0.10 µm | 0.040 px² (470 nm²) | | |

The angle-averaged PSF has FWHM 3.088 / 2.966 px at z = 0.05 (the old model: 2.892 / 2.968). The per-angle blur alternative gives
0.031 / 0.034 / 0.036 px².

The difference image (PSF_0 − PSF_1) is quadrupolar: angle 0 has more light along ±y, angle 1 along ±x. It peaks at 2.4 % of the
peak (`out/psf3d/final/perangle_r05_blps3_diff.png`).

**Precision.** The pattern-free CRLB of a 6-frame window (1000 ph/frame, bg 10) with the true per-angle PSFs vs with their average
differs by ≤ 0.08 % in x / y.

**Bias from fitting with the angle-averaged PSF.**
- With equal light in both angles, the first-order terms cancel by symmetry: ≤ 0.001 nm.
- An emitter on only during the angle-0 frames is the worst case: ≤ 0.3 nm in x, 0.5 nm in y, 1.7 nm in z. The grid was
  sub-pixel (0, 0), (−0.4, −0.4), (0.25, −0.1), (0.5, 0.5), (0.45, 0), (0, 0.45) px and z 0.02 / 0.06 / 0.12 µm.

So the per-axis PSF matters for simulation realism (network training, PSF-shape statistics), not for the information content
or the position bias of a fit.

**Intensity side (the pattern information of SIMFLUX / SIMCODE).** This compares the per-axis PSFs with their average. Script:
`archive-2026-11-15:analysis/psf3d/intensity.py`; results in `out/psf3d/intensity/{m1809,m25}.json`. All numbers are on expected (noise-free) data with
Poisson MLE. The grid is the sub-pixel grid above, z 0.02 / 0.06 / 0.12 µm, 8×8 and 11×11 ROIs, 1000 and 300 ph/frame, bg 10.

1. **Per-frame intensity CRLB** (position known; I and bg free), per-axis truth vs average:

| movie | angle 0 | angle 1 |
|---|---|---|
| 1809/2 | +0.1 to +0.2 % | −0.1 to −0.2 % |
| 2.5 nM | +0.2 to +0.5 % | −0.2 to −0.5 % |

The worst case is the 8×8 ROI at 300 ph.

2. **Per-frame intensity bias of an averaged-PSF fit to per-axis data:**

| movie | common per-angle factor, angle 0 / angle 1 (8 px / 11 px) | residual sd over position and z (8 px / 11 px) |
|---|---|---|
| 1809/2 | 0.9979 / 1.0021 (1000 ph); 0.9970 / 1.0030 (300 ph) | 0.14–0.18 % / 0.09–0.14 % |
| 2.5 nM | 0.995 / 1.005 (1000 ph); 0.994 / 1.006 (300 ph) | 0.04–0.06 % / 0.02–0.03 % |

- Fitting with the average under-reads angle 0 and over-reads angle 1. The 11 px ROI halves the common factor.
- A per-angle relint absorbs the common factor; the pattern estimate fits relint per step from the data.
- The residual is a modulation error in the 3-frame demodulation of an angle (relint known): a phase error equivalent to
  ≤ 0.019 nm position (rms 0.006–0.009 nm), and a depth error ≤ 0.0014 (rms ≤ 0.0005).

3. **SIMFLUX joint 6-frame fit** (known pattern: calib k and phases, depth 0.925 / 0.955, relint, 4 pattern offsets):
- CRLB(x, y) with the per-axis vs the average PSF: within ±0.25 %. Median CRLB at 1000 ph is 1.39 / 1.40 nm (1809, 8×8) and
  1.64 / 1.59 nm (2.5 nM); at 300 ph, 3.5 / 3.4 and 4.2 / 4.0 nm.
- Position bias of the averaged-PSF fit to per-axis data:

| amplitude model | max bias | rms |
|---|---|---|
| one amplitude (relint fixed) | ≤ 0.021 nm | ≤ 0.010 nm |
| per-angle amplitude (relint absorbs the common factor; fitted A1/A0 = 1.002–1.008) | ≤ 0.020 nm | ≤ 0.010 nm |

4. **Meaning for SIMCODE v2's explicit pattern solve** (network intensities).
- At the model level, the per-axis PSF reaches the intensities as a common per-angle scale of ±0.2–0.6 %, which the per-step
  relint of the pattern estimate absorbs.
- What remains varies with position and z: 0.02–0.18 %, i.e. ≤ 0.02 nm of phase-equivalent position and ≤ 0.0014 in depth.
  That is two orders of magnitude below the per-emitter angle brightness factor in the data and in the training sims (AcquisitionSim
  ang_sd 0.14; measured per-event ln(A1/A0) correlation, section 3).
- So per-axis training cannot change precision or bias through the intensity channel, as far as an MLE-like read-out is
  concerned. If it matters, it is through how the networks generalise to the real PSF shape. That has to be measured on the
  networks themselves: heads trained on the per-axis vs the averaged PSF, scored on per-axis sims and on real data.

**Sim check (simulator LUT path, real camera maps; statistic = the old-model fit's residual moments).**

0.5 nM:

| | overall | angle 0 | angle 1 | contrast |
|---|---|---|---|---|
| real | +0.034 | +0.015 | +0.054 | +0.038 |
| sim with this PSF | +0.029 | +0.015 | +0.045 | +0.028 |
| sim with the old PSF | −0.006 | | | +0.002 |

1809/2:

| | overall | angle 0 | angle 1 | contrast |
|---|---|---|---|---|
| real | +0.018 | −0.004 | +0.043 | +0.047 |
| sim with this PSF | +0.017 | −0.003 | +0.039 | +0.043 |
| sim with the old PSF | −0.018 | | | +0.010 |
| per-angle-blur alternative | +0.017 | | | +0.063 (overshoots) |

## 6. Caveats

- **2.5 / 20 nM.** The fits sit at z ≈ 0 with focus ≈ 0.08 µm (0.14 µm at 0.5 nM / 1809). Their residual variance is 1.4, and they
  show aperture-dependent residuals (r < 4 px: +0.07 / +0.12 px²) and per-tile structure (χ²/dof 15–38 vs ≈ 10 for the tool
  itself on sims). Isolated tracks in dense movies are contaminated. These PSFs are less certain than the low-density ones.
- **Orientation diversity.** The per-event brightness-ratio correlation (section 3) shows orientation memory. A single per-angle
  PSF is the population average. Per-event dipole orientation (a PSF that varies per event, correlated with the angle brightness
  ratio) is not modelled.

## 7. Files

- `models/insitu/calib_<movie>_psf3d.json`: per axis (`psf_blur_px2` + `psf_dipole_M_angle`). Movies: 0.5nM, 1809_2, 2p5nM_m2,
  20nM_m2.
- `..._psf3d_avg.json`: the average of the two axis PSFs (`psf_dipole_M` = mean tensor, exact), for SMLM / SIMFLUX / DECODE.
- `out/psf3d/final/zstack_<movie>_psf3d.npz`: z-stacks (mean / angle0 / angle1).
- `*_psf3d_ad*.json`: the per-angle-blur alternative.
