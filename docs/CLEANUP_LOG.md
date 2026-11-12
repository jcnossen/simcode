# Cleanup log (branch `cleanup`, started 2026-11-12 from 60cc637)

Execution of `docs/CLEANUP_PLAN.md` with the user decisions of 2026-11-12 (keep only the two comparisons; rename the
package; source-only repository; one file handling for both pipelines; audit items). Done in an isolated worktree; the
main tree (`/home/jc/simcode2`) was not touched. One section per step: what changed, the commit, the check and its result.

## How results were checked

All checks ran in the worktree, with the networks / calibrations / 4Pi inputs of the main tree read through the config
file, and all outputs under `<worktree>/out/cleanup` (not tracked).

| check | what it is | reference |
|---|---|---|
| T | `tests/test_fast_paths.py` (GPU) | must print `ALL OK` |
| S | `tests/check_imports.py`: every tracked `.py` compiles, every package import resolves to an existing module and name, no shell script or doc names a missing file | must print `tree ok` |
| P | two chunks (frames 0-12000) of the 1809/2 movie through pass 1, the pass-1 solve and pass 2 with the m1809 set and the production flags (`--autoc 0 --cand_pmin 0.3 --pass2`), from a local copy of the first 12012 frames (one 4 s read of the HDD) and the reference run's pattern; `tests/compare_npz.py`: every array of `pass1_{0,6000}.npz`, `pass2_est_*`, `pass2_est2_*`, `pass2_est2b_*` must be `np.array_equal` | `out/paper2/1809_2/v2_matched` (the production run of 2026-11-11) |
| Tr | 20 steps of the four production trainers (det, head, pass-2 det, pass-2 head; m1809 recipe, batch 4, fixed seeds): the step-0 log line (first batch = generator random stream + forward pass) | the same run with the code of 60cc637 |
| G4 | 4Pi isolated-emitter gate of the learned-window head v10 (`analysis/fourpi/gatek4pi.py --head v10`: fixed linear read-out reference, learned head, pixel MLE; sets A and B): all 181 numbers of the json | the code of 60cc637 (which itself reproduces `out/4pi/sim/gatek_v10_trk.json` exactly) |
| N4 | 4Pi network pipeline on 2400 frames of NPC file 10 (`real_w4pi.py --head v10 --coh 1.56`): every array of the table | the code of 60cc637 on the same frames |
| C4 | classical 4Pi: validation on isolated emitters (`val_iso.py`), per-file pipeline on 2400 frames of NPC file 10 (`real_cl.py --src dcimg`), full driver read + fit stages (`full_cl.py`) on the same frames | the code of 60cc637 |
| PS | paper-simulation smoke: a 1200-frame density-1 movie through `run_all.py --steps v2,metrics --netset m1809` | the code of 60cc637 on the same movie |

Baseline facts established before any change (code of 60cc637, run in the worktree):
- P: 8 of 8 chunk files bit-identical to the production run (so the check is meaningful on this machine, on a GPU shared
  with other jobs).
- T: `ALL OK`.
- G4: identical to the stored gate result (181 numbers, max difference 0).
- Tr: training on the GPU is not bit-reproducible between two runs of the same code: weights after 20 steps differ by up
  to 3e-4, the step-0 losses agree to 2e-6 relative (identical on the same GPU for three of four trainers, 7th digit
  between GPUs). The trainer check is therefore: integer fields of the step-0 line equal, float fields within 1e-5
  relative. This detects a change of the random stream or of the model / loss, not a rounding-level change.
- No checkpoint in `models/` or `out/4pi/models/` (49 files) pickles a class of this package (`tests/check_checkpoints.py`:
  only `torch` and `collections` globals), so the rename needs no alias module.

`pearson_corr` (the preprint's RSP code in `~/smlmtorch/scripts`, imported by `analysis/paper/paper_common.py`) is on the
list of external modules of check S.

## Step 1: removals (user decision 1)

Order: the user's numbering (1 removals, 2 rename, 3 source-only) instead of the plan's R1-R3 then A1: fewer files to rename.

Kept on purpose, after checking the import graph (`grep` + a reference graph over `.py` / `.sh` / `.md`):
- `simcode2/fourpi/head4pi.py` (Head4Pi): the learned-window head v10 (`head4pi_w.py`) builds a frozen Head4Pi inside
  itself as its prior (weights are part of `head4pi_v10.pt`; trained as `v3L`), and the simulated showcase uses the
  stand-alone `v3L` checkpoint as the candidate front. `analysis/fourpi/train_head4pi.py` (its trainer) is kept for the
  same reason: without it v10 cannot be retrained.
- `simcode2/fourpi/linread4pi.py` (fixed linear read-out): the gate reference (`gatek4pi.py` runs the v10 head with
  `learned=False`) and `simlin4pi.py` without `--head`.
- `analysis/fourpi/gatek4pi.py` + `bench_classical.py`: the isolated-emitter gate of v10 and its pixel-MLE reference.
- `analysis/fourpi/fit_psf4pi.py`: listed as a one-off in the plan, but it produces the 4Pi PSF model every kept script loads.
- `analysis/fourpi_classical/val_iso.py`: the classical validation on isolated emitters (used as a check here).
- `analysis/psf3d_nets/sigcal.py`, `analysis/psfmatch/fit_cal_set.py`: part of the production recipe (sigma rescale json,
  calibration fit of a network set).
- the 4Pi calibration chain of commit 1106fbb (`process.py`, `register.py`, `camera.py`, ...): run once per data set, its
  outputs are read by every later script. Not moved to a sub-folder (they import each other through `sys.path`).
- `simcode2/eff/simcode_gate.py`: runs the preprint model on the gate scenes = a preprint-vs-v2 comparison.
- first-generation 4Pi files whose functions the current pipeline imports (`front4pi.py`, `solve_real4pi.py`,
  `simsolve4pi.py`): handled in step B4, where the functions move into the package.

Edits that go with the removals:
- `hd/sc.py`: the unused base head `SCHead` (+ `wls`, its trainer `train()`) and the sc7bg hooks removed; `load()` accepts
  only `arch == 'sc4n'` (all 11 head checkpoints in `models/` except `hd_sc7bg.pt` are `sc4n`); `hd/sc4.py train` refuses
  another `--arch`; `sc_front_fast.py`: sc7bg refusal removed.
- `fourpi/solve4pi.py`: the full-covariance branch of the removed `head4pi_p1` (read-outs with a `U` factor) removed;
  `fourpi/loadhead.py`: loads `arch 'w'` and the plain Head4Pi only; `WINS_K` moved to `fourpi/windows4pi.py`.
- Network sets: `simcode2/netsets.json` is now the single definition (sets m1809, m25, g13; `_default` m1809; the paper's
  production flags and datasets under `_production`); `config.netset(name)`. The constants `config.DET / HEAD / P2_DET /
  P2_HEAD / CAL_P1_REAL / CAL_P1_SIM / CAL_P2_REAL / CAL_P2_SIM` now name the files of the default set (m1809) instead of
  the old default set (det_n1, hd_sc5nb, p2_det, p2_head_long). `analysis/papersim/*` read the sets through
  `config.netset`; their `--netset` default changed from `default` to `m1809`.
- `tests/test_fast_paths.py` uses the m1809 set, the 1809/2 calibration (also for the simulated test chunk) and
  `tests/data/pattern_est_1809_2.pt` (= `out/paper2/1809_2/v2_matched/pattern_est.pt`). **The test reference data was
  regenerated with m1809**; tolerances unchanged, all tests pass.
- `analysis/papersim/reproduce.sh` holds the commands of the removed queue scripts.
- `analysis/figures/{crlb_all_v2.py, lm_robust.py, run_crlb_refs.sh}`: sc7bg rows removed.

Not removed although tied to the old default set (see "Open points"): the calibration json of `hd_sc5nb` / `p2_head_long`
(templates of the calibration fit of every new set, sigma table of the opt-in `--autoc 1`), `analysis/figures/`,
`analysis/gate/run_gate.sh` (still names `hd_sc5nb.pt` explicitly).

Checks: T `ALL OK` (m1809); S `tree ok`; P 8 of 8 files identical, run **without** network flags (the new defaults
resolve to the production files); Tr det / head / pass-2 det: same (max relative difference 2e-7 / 2e-7 / 0); pass-2 head: all step-0 fields identical except
`w_il` (mean |weight| of the zero-initialised illumination input after the first optimiser step), 5e-5 relative: the
reference evaluation before training now runs the default set's pass-1 networks instead of det_n1 + hd_sc5nb (key
`p1_ref` in `eval_ref.json`, was `sc5nb_detn1`); with the old two files put back the difference is 8e-6, i.e. run-to-run
level, so it is a rounding-level effect of what ran on the GPU before, not a change of the random stream or the loss; G4 / N4 / C4: run
after the path changes of step 2 (the 4Pi scripts of this tree import the package through a hard-coded path until then).

### Removed files

| path | lines | why |
|---|---|---|
| `analysis/checks/cand_pmin_check.py` | 46 | one-off check; no caller |
| `analysis/checks/lm_check_paper2.py` | 111 | one-off check; no caller |
| `analysis/fourpi/decomp4pi.py` | 160 | review / decomposition diagnostics of the 4Pi heads v2-v8 (Head4Pi + Solve4Pi read-outs); results are in out/4pi/REVIEW_NET.md |
| `analysis/fourpi/eval_head4pi.py` | 153 | review / decomposition diagnostics of the 4Pi heads v2-v8 (Head4Pi + Solve4Pi read-outs); results are in out/4pi/REVIEW_NET.md |
| `analysis/fourpi/field_bias4pi.py` | 90 | one-off 4Pi check; no caller, decision recorded in out/4pi/*.md |
| `analysis/fourpi/front4pi_full.py` | 116 | first-generation 4Pi network driver (v2 / v3 / p1 tables); the learned-window pipeline replaces it |
| `analysis/fourpi/gate4pi.py` | 210 | gate / tracker script of the 4Pi heads v2-v9 (v10 gate = gatek4pi.py, gate_movies_w.sh: kept) |
| `analysis/fourpi/gate_movies_det.sh` | 17 | gate / tracker script of the 4Pi heads v2-v9 (v10 gate = gatek4pi.py, gate_movies_w.sh: kept) |
| `analysis/fourpi/gate_v4.sh` | 21 | gate / tracker script of the 4Pi heads v2-v9 (v10 gate = gatek4pi.py, gate_movies_w.sh: kept) |
| `analysis/fourpi/gate_v5.sh` | 21 | gate / tracker script of the 4Pi heads v2-v9 (v10 gate = gatek4pi.py, gate_movies_w.sh: kept) |
| `analysis/fourpi/info4pi.py` | 159 | one-off 4Pi check; no caller, decision recorded in out/4pi/*.md |
| `analysis/fourpi/linbound4pi.py` | 114 | review / decomposition diagnostics of the 4Pi heads v2-v8 (Head4Pi + Solve4Pi read-outs); results are in out/4pi/REVIEW_NET.md |
| `analysis/fourpi/post4pi.py` | 126 | first-generation 4Pi network driver (v2 / v3 / p1 tables); the learned-window pipeline replaces it |
| `analysis/fourpi/prior_slope4pi.py` | 25 | review / decomposition diagnostics of the 4Pi heads v2-v8 (Head4Pi + Solve4Pi read-outs); results are in out/4pi/REVIEW_NET.md |
| `analysis/fourpi/ref_lin_movies.sh` | 16 | one-night queue script |
| `analysis/fourpi/review4pi.py` | 211 | review / decomposition diagnostics of the 4Pi heads v2-v8 (Head4Pi + Solve4Pi read-outs); results are in out/4pi/REVIEW_NET.md |
| `analysis/fourpi/review4pi_bg.py` | 113 | review / decomposition diagnostics of the 4Pi heads v2-v8 (Head4Pi + Solve4Pi read-outs); results are in out/4pi/REVIEW_NET.md |
| `analysis/fourpi/review4pi_cov.py` | 163 | review / decomposition diagnostics of the 4Pi heads v2-v8 (Head4Pi + Solve4Pi read-outs); results are in out/4pi/REVIEW_NET.md |
| `analysis/fourpi/run_full4pi.sh` | 32 | first-generation 4Pi network driver (v2 / v3 / p1 tables); the learned-window pipeline replaces it |
| `analysis/fourpi/run_full4pi_v3.sh` | 28 | first-generation 4Pi network driver (v2 / v3 / p1 tables); the learned-window pipeline replaces it |
| `analysis/fourpi/showcase4pi_queue.sh` | 11 | one-night queue script |
| `analysis/fourpi/simoffset4pi.py` | 68 | one-off 4Pi check; no caller, decision recorded in out/4pi/*.md |
| `analysis/fourpi/solve_follow.sh` | 25 | first-generation 4Pi network driver (v2 / v3 / p1 tables); the learned-window pipeline replaces it |
| `analysis/fourpi/track_gate.sh` | 22 | gate / tracker script of the 4Pi heads v2-v9 (v10 gate = gatek4pi.py, gate_movies_w.sh: kept) |
| `analysis/fourpi/track_gatek.sh` | 18 | gate / tracker script of the 4Pi heads v2-v9 (v10 gate = gatek4pi.py, gate_movies_w.sh: kept) |
| `analysis/fourpi/train_head4pi_k.py` | 83 | trainer of a 4Pi head v4-v9 |
| `analysis/fourpi/train_head4pi_p1.py` | 178 | trainer of a 4Pi head v4-v9 |
| `analysis/fourpi/train_head4pi_s.py` | 78 | trainer of a 4Pi head v4-v9 |
| `analysis/fourpi/train_head4pi_v4.py` | 70 | trainer of a 4Pi head v4-v9 |
| `analysis/fourpi/verify_v3.sh` | 18 | gate / tracker script of the 4Pi heads v2-v9 (v10 gate = gatek4pi.py, gate_movies_w.sh: kept) |
| `analysis/fourpi_classical/compare_det.py` | 51 | one-off check (classical detector vs aborted v3 candidates) |
| `analysis/papersim/netsets.json` | 39 | moved to simcode2/netsets.json (sets default and psf3dmix dropped, m25 added) |
| `analysis/papersim/rerun_metrics.sh` | 10 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_density_exp.sh` | 11 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_density_full.sh` | 14 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_g13.sh` | 11 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_gauss_base.sh` | 13 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_hollow.sh` | 12 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_hollow_analysis.sh` | 12 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_hollow_final.sh` | 15 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_hollow_final2.sh` | 15 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_hollow_x1rest.sh` | 8 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_queue2.sh` | 9 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/papersim/run_queue3.sh` | 8 | one-night queue script; its commands are in analysis/papersim/reproduce.sh |
| `analysis/psf3d/gamma_profile.py` | 18 | one-off PSF check; no caller |
| `analysis/psf3d/perangle.py` | 150 | one-off PSF check; no caller |
| `analysis/psf3d/simtracks.py` | 42 | one-off PSF check; no caller |
| `analysis/psf3d_nets/accept.py` | 84 | acceptance / calibration script of the mixture set psf3dmix (removed set) |
| `analysis/psf3d_nets/fit_cal_new.py` | 46 | acceptance / calibration script of the mixture set psf3dmix (removed set) |
| `analysis/psf3d_nets/m20r_stats.py` | 40 | acceptance / calibration script of the mixture set psf3dmix (removed set) |
| `analysis/psf3d_nets/spotshape.py` | 48 | acceptance / calibration script of the mixture set psf3dmix (removed set) |
| `analysis/psfmatch/accept3.py` | 30 | one-off acceptance check of a finished decision |
| `analysis/psfmatch/accept_g13.py` | 14 | one-off acceptance check of a finished decision |
| `analysis/psfmatch/gatecheck.py` | 27 | one-off acceptance check of a finished decision |
| `analysis/speed/autoc_iso_check.py` | 15 | one-off speed / equivalence check of the speed branch; no caller (spatial hash is tested in tests/test_fast_paths.py) |
| `analysis/speed/hash_verify.py` | 103 | one-off speed / equivalence check of the speed branch; no caller (spatial hash is tested in tests/test_fast_paths.py) |
| `analysis/speed/head_prof.py` | 26 | one-off speed / equivalence check of the speed branch; no caller (spatial hash is tested in tests/test_fast_paths.py) |
| `analysis/speed/head_variants.py` | 38 | one-off speed / equivalence check of the speed branch; no caller (spatial hash is tested in tests/test_fast_paths.py) |
| `analysis/speed/head_vs_fp32.py` | 62 | one-off speed / equivalence check of the speed branch; no caller (spatial hash is tested in tests/test_fast_paths.py) |
| `analysis/speed/pattern_hash_vs_file.py` | 31 | one-off speed / equivalence check of the speed branch; no caller (spatial hash is tested in tests/test_fast_paths.py) |
| `analysis/speed/pattern_kd.py` | 24 | one-off speed / equivalence check of the speed branch; no caller (spatial hash is tested in tests/test_fast_paths.py) |
| `analysis/speed/prof_pattern.py` | 25 | one-off speed / equivalence check of the speed branch; no caller (spatial hash is tested in tests/test_fast_paths.py) |
| `analysis/tput/wait_and_bench.sh` | 37 | one-night queue script |
| `models/calib/calib_p2L_psf3dmix.json` | 86 | calibration of the mixture set psf3dmix / its pair variant (removed sets) |
| `models/calib/calib_p2L_psf3dmix_pair.json` | 86 | calibration of the mixture set psf3dmix / its pair variant (removed sets) |
| `models/calib/calib_p2L_psf3dmix_pair_real.json` | 129 | calibration of the mixture set psf3dmix / its pair variant (removed sets) |
| `models/calib/calib_p2L_psf3dmix_real.json` | 129 | calibration of the mixture set psf3dmix / its pair variant (removed sets) |
| `models/calib/calib_sc_hd_psf3dmix.json` | 85 | calibration of the mixture set psf3dmix / its pair variant (removed sets) |
| `models/calib/calib_sc_hd_psf3dmix_real.json` | 130 | calibration of the mixture set psf3dmix / its pair variant (removed sets) |
| `models/calib/calib_sc_hd_sc7bgb15_pm.json` | 18 | calibration of the sc7bg head |
| `models/calib/sigu/sigu_p1_new.json` | 48 | sigma rescale of the mixture set psf3dmix |
| `models/calib/sigu/sigu_p2_new.json` | 48 | sigma rescale of the mixture set psf3dmix |
| `simcode2/det/train.py` | 114 | earlier detector trainer (det_stem lineage); no importer, produced no kept network |
| `simcode2/fourpi/head4pi_k.py` | 80 | 4Pi head v4-v9 architecture; v10 (head4pi_w) needs only WINS_K, moved to windows4pi.py |
| `simcode2/fourpi/head4pi_p1.py` | 160 | 4Pi head v4-v9 architecture; v10 (head4pi_w) needs only WINS_K, moved to windows4pi.py |
| `simcode2/fourpi/head4pi_s.py` | 121 | 4Pi head v4-v9 architecture; v10 (head4pi_w) needs only WINS_K, moved to windows4pi.py |
| `simcode2/hd/sc7bg.py` | 253 | sc7bg head variant (dense BgNet background); not production, refused by the fast path |
| `simcode2/insitu/coma_anchor_scan.py` | 185 | diagnostic pair, self-declared not part of the method; no importer |
| `simcode2/insitu/fit_insitu.py` | 136 | superseded by fit_insitu2.py; its fixed camera values are recorded in insitu/README.md |
| `simcode2/insitu/lm_refine.py` | 125 | diagnostic pair, self-declared not part of the method; no importer |
| `tests/data/pattern_est_0.5nM.pt` | - | test pattern made with the old default set; replaced by tests/data/pattern_est_1809_2.pt (m1809) |

Total: 5524 lines in the files above.

