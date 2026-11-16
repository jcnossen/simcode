# Cleanup log (branch `cleanup`, started 2026-11-12 from 60cc637)

Execution of `docs/CLEANUP_PLAN.md` (archived 2026-11-15: `git show archive-2026-11-15:docs/CLEANUP_PLAN.md`) with the user decisions of 2026-11-12 (keep only the two comparisons; rename the
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


Addendum to the step-1 trainer check: the pass-2 head's `w_il` field varies between runs of the same code by 2e-6 to 5e-5
relative (three later runs of the new code against the same baseline: 5e-5, 5e-5, 7e-6), so the 5e-5 above is run-to-run
noise of that field, not an effect of the changed reference evaluation.

## Step 2: package rename `simcode2` -> `simcode`, hard-coded paths (user decision 2)

- `git mv simcode2 simcode`, `simcode2.example.json` -> `simcode.example.json`; `simcode2` / `SIMCODE2` replaced in every
  tracked text file (868 replacements in 209 files: imports, `python -m simcode.…`, docs, pyproject name, console scripts
  `simcode-run` / `simcode-strict`, package data `netsets.json`, `fourpi/datasets.json`).
  Not replaced on purpose: the repository directory `/home/jc/simcode2`, `/mnt/data/dmdsf/simcode2_out`, the code-host name
  `jcnossen/simcode2`, `docs/CLEANUP_PLAN.md`, the output records under `analysis/results/`, and the calibration json
  files under `models/` (their `source` / `note` strings still say `simcode2/...`; they are data records).
- Old names still accepted (`simcode/config.py`; tested in `tests/test_fileio.py::test_config_names`): environment
  variables `SIMCODE2_*` (the new `SIMCODE_*` wins when both are set; `config.env(name)`), the config files
  `<repo>/simcode2.json` and `~/.config/simcode2.json` (after `simcode.json`), a `simcode2_out` directory as the default
  output root when no `simcode_out` exists. The main tree's `simcode2.json` therefore keeps working unchanged.
- Checkpoints: `tests/check_checkpoints.py` on `models/` and `out/4pi/models/` (49 files): none refers to a class of the
  package, so no alias module was added; T, P, G4, N4 below load the production checkpoints under the new name.
- Hard-coded `/home/jc/simcode2` removed from all code (was in 61 `.py` and 8 `.sh` files): scripts derive `REPO` from
  their own location; the 4Pi work directory is `simcode.fourpi.setup.P4` = config root `fourpi` (default `<out>/4pi`,
  env `SIMCODE_4PI`); other outputs use `config.root('out')`; 4Pi shell scripts `cd` to the repository of the script and
  ask the package for the 4Pi directory. The 4Pi data set (data root `/mnt/data/4pi-sf` unless config `data4pi` / env
  `SIMCODE_4PI_DATA`, channel crop layout, gain, pixel size, patch size, per-cell phase offset, file globs) is defined
  once in `simcode/fourpi/datasets.json`, read by `simcode/fourpi/setup.py`. (`analysis/fourpi/dc.py` still has its own
  copy until step B1.)
- Removed with this step (they hard-coded the speed worktree / one optics scan): `analysis/speed/clean_bench.sh` (27
  lines; one-off benchmark of the speed branch, calls scripts removed in step 1), `analysis/fourpi/run_psf_variants.sh`
  (17 lines; one-off optics-sensitivity scan).
- `tests/check_imports.py` now treats any remaining `simcode2` import, `python -m simcode2…` or `simcode2/<file>` mention
  as an error.

Checks (all with the renamed package, config through the worktree's `simcode.json`): S `tree ok`; T `ALL OK`; P 8 of 8
files identical; Tr det / head / pass-2 det / pass-2 head same (max relative difference 2e-7 / 3e-7 / 0 / 7e-6);
G4 identical (181 numbers; this also covers the step-1 edits of `solve4pi.py`, `loadhead.py`, `windows4pi.py`); N4 table identical
(1 of 1 files, all arrays); C4: `val_iso.npz` and `val_iso.txt` identical, per-file table identical, full-driver table,
patch meta and patch file identical; PS 8 of 8 files identical (chunk files and tables), metrics json identical.
The 4Pi pipelines and the gate are deterministic on this machine, so bit-identity is the criterion for them as well.

## Step 3: source-only repository (user decision 3)

- `git rm --cached` (the files stay on disk; history is **not** rewritten): the 24 tracked network / fit-state files under
  `models/` (`*.pt`: the sets m1809, m25, psf3dmix, psf3dmix_pair, the old default set det_n1 / hd_sc5nb / p2_det /
  p2_head_long, hd_sc7bg, the two ablation heads, the two DECODE models, the bead-fit state) and
  `models/insitu/beads/psf_fit_sign-1_na149_ball_drift_xz.png`. `.gitignore` already excluded `models/**/*.pt`; `*.png`
  under `models/` added.
- Kept tracked, flagged for a decision: the four camera-map files `models/insitu/camera_maps_*.npz` (85-130 kB each; every
  in-situ calibration json refers to one), `analysis/simreal/fields/field_own10_real{05,25}.pt` (25 kB each; pattern-field
  inputs of the simulated-movie recipes), `tests/data/pattern_est_1809_2.pt` (9 kB test fixture), and the text result
  snapshots under `analysis/results/` (140 files, 1.1 MB; outputs, but not binaries).
- Added: `models/README.md` (what is in git, the three production sets with file names, sizes and sha256 prefixes, which
  other weights the kept code can use, the recipe of a set, the note that weights are distributed separately and that the
  location is not decided). No fetch script: there is nowhere to fetch from yet.
- Added to git: the six calibration json files of the set g13 (`models/calib/calib_{sc_hd,p2L}_g13{,_real}.json`,
  `models/calib/sigu/sigu_p{1,2}_g13.json`), copied unchanged from the main tree's `models/`, where they are untracked.
  Without them the set g13 of `netsets.json` is incomplete in a clone.

Check: `git archive` of the commit into an empty directory (= a fresh checkout without weights), config pointing `models`
at the directory that holds the weights: S, `tests/test_fileio.py` and T pass (result below).
Result of the fresh-checkout check (commit d918493 exported with `git archive`, the four m1809 weight files linked into its
`models/`): S `tree ok`, T `ALL OK`.

## Step B1 + B2: shared file handling (`simcode/fileio/`; user decision 4)

The sub-package is called `fileio`, not `io` as in the plan: a directory `simcode/io` shadows the standard library's `io`
for any script started from inside `simcode/`.

- `fileio/movie.py`: one reader interface (`n`, `shape`, `parts`, `read(f0, f1, y0, x0, S)`), back ends `TiffMovie` (the
  former `pipeline/common.Movie`, logic unchanged), `DcimgMovie` (the former `analysis/fourpi/dc.Dcimg`; with a channel
  layout `read()` returns the mirrored channel crops), `ArrayMovie` (array or `.npy`), `open_movie` by extension, and
  `Prefetch` (the former `PrefetchMovie`, now back-end independent; it replaces the three 4Pi reader threads).
  `pipeline/common.py` keeps `Movie` and `PrefetchMovie` as aliases; `analysis/fourpi/dc.py` is a thin module over the
  package reader and `fourpi/setup.py` (same names as before, so the calibration scripts are unchanged).
- `fileio/work.py`: `Units` (output path pattern, done, claim marker with host / pid / process start time and takeover of a
  dead process's claim, failed marker, per-unit record json, the worker loop `todo` which with claiming on ends only when
  every unit is done), `atomic_savez / atomic_save / atomic_torch_save / atomic_json / atomic_text`, `disk_guard`.
- `fileio/launch.py`: `python -m simcode.fileio.launch --gpus 1,2 [--log prefix] -- <command>` starts the command once per
  GPU.
- `util/ckpt.py` (moved from `fourpi/ckpt.py`): the atomic, disk-guarded checkpoint save is now used by all trainers
  (`det/up2.py`, `hd/sc4.py`, `p2/det2.py`, `p2/head2.py` and the 4Pi trainers).

Checks:
- `tests/test_fileio.py` (CPU, new): TIFF across three parts with crops; a synthetic dcimg (header, raw, footer, crops
  with mirroring); array / `.npy`; Prefetch hits and a miss; atomic write leftovers are neither done nor matched by
  `*.npz`; resume; redo; failed marker; records; disk guard; 3 processes x 24 units: every unit exactly once, no claim
  left; a claim of a dead pid is taken over, a live one is respected and the loop waits; `kill -9` during a 1.9 GB write
  leaves no output; old / new environment and config-file names; network sets resolve. Result: `ALL OK`.
- Old readers (code of 60cc637) against the new ones on real data (`np.array_equal`): the read plan and the reads of
  chunks 0 and 6000 of the 1809/2 copy, the same through Prefetch (2 hits), and the channel crops, raw frames and footer
  of frames 0-600 of NPC file 10 and MT file 3: all identical.

## Step B3: the 2D pipeline on the shared driver (`pipeline/run_uncalibrated.py`, `run.py`)

- The chunk loops of pass 1, the pass-1 solve and pass 2 are `Units` loops; every chunk output is written atomically
  (audit defect 1); the pass-2 head-prior file `pass2_<tag>b_<f0>.npz` is written before the file that marks the chunk as
  done, so a run killed in between repeats both. `pattern_est.pt`, `pattern_est.json`, `calib_autoc.json`, the tables,
  `summary.json` and `runtime.json` are atomic as well.
- Each chunk leaves a record (`pass1_<f0>.json`, `pass2_est_<f0>.json`, `pass2_est2_<f0>.json`: timings, counts);
  `runtime.json` is merged from the records (audit defect 2: no file is rewritten from one process's partial view).
- The duplicated block that loads the pass-1 rows is one function, `pipeline/common.load_pass1_rows` (pattern stage and
  pattern-field stage; `autoc.py` and `strict.py` keep their own loaders: they read other columns / other files).
- New flags: `--claim 1` (several copies of the command share the chunks; pass 1 and pass 2 are claimable, the pattern
  stage, the automatic c and the tables run in one copy after all chunks of the previous stage exist; prefetch is off
  with claiming), `--min_free_gb` (default 2: a chunk is not started below it).
- Chunk boundaries, the read plan and the padding rule are unchanged.

Checks:
- T `ALL OK`; P 8 of 8 chunk files bit-identical to the production run; the two tables of those chunks identical to the
  ones the code of 60cc637 wrote (all previous columns).
- Claim test (`out/cleanup/claim_test.sh`: 12000 frames in 4 chunks of 3000, pattern estimated in the run, pass 2, tables):
  one process against (a) two processes on two GPUs with `--claim 1` through the launcher: 18 of 18 files identical
  (pass 1, solve, pass 2, tables), chunks split 2 + 2 (pass 1) and 1 + 3 (pass 2), the pattern stage ran in one process, no
  claim left; (b) `kill -9` in the middle of pass-1 chunk 3, restart: 2 complete chunk files after the kill, no truncated
  file, the restart processed chunks 6000 and 9000, 18 of 18 files identical; (c) the same with `--claim 1`: the claim of
  the dead process was taken over ("pass1 3000: took over the claim of the dead process …"), 18 of 18 identical.
- `runtime.json` has all four chunks per stage in the one-process and in the two-process run (12 record files each).

## Step B4 + B5: the 4Pi pipelines on the shared reader and driver

- `simcode/fourpi/run.py` (new): the learned-window network pipeline. `process_file(reader, ctx, args, file, nframes)` is
  the former body of `analysis/fourpi/real_w4pi.py` (operation for operation); the file loop is a `Units` loop (finished
  files skipped, atomic table, claim per file, disk guard **before** a file starts instead of dropping a finished result:
  audit defects 3 and 4); the per-file summary is a record json, the merged `locs_<cell>_<head><tag>_summary.json` is
  rebuilt from the records (defect 2). `analysis/fourpi/real_w4pi.py` is a 10-line wrapper (old command line unchanged);
  `analysis/fourpi/run_full_w4pi.sh` calls the launcher (`GPUS=1,2`), the shell claim logic is gone.
  Output names are unchanged (`<4Pi dir>/net/locs_<cell>_<file>_<head><tag>.npz`), so every reader of the tables works
  as before; the plan's new directory layout is not introduced (open question 5 of the plan).
- Functions that the pipeline imported from first-generation scripts now live in the package: `fourpi/front.py` (`Front`,
  `PolyMap`, `blur`, registered sum, nearest-candidate distance), `fourpi/insitu.py` (`repattern_xy`, `pattern_psi`,
  `phase_field`, `slotZ`, `refine_k`, `emitter_phase`). Removed: `analysis/fourpi/front4pi.py` (196 lines),
  `solve_real4pi.py` (174), `simsolve4pi.py` (128): their `main()`s were the first-generation drivers (v2 / v3 tables).
  `analysis/fourpi/register.py` and `pattern.py` import the moved functions, so the calibration chain is unchanged.
- Classical pipeline: `real_cl.py` (per-file loop = `Units`, `--claim`, atomic table and summary; dcimg source through
  `DcimgMovie`), `full_cl.py` (reader stage through `DcimgMovie` + `Prefetch`, temporary patch set written atomically with
  its meta file as the ready marker; fitter stage claims the table through `Units` instead of renaming `.ready` to
  `.claimed`; a failed file gets a `.failed` marker and no longer blocks the fitter's exit).
- 4Pi constants are defined once (`fourpi/setup.py` from `datasets.json`): 17 repeated definitions of `GAIN`, `P`, `XLIM`,
  `PX`, `DELTA` in `analysis/fourpi*` replaced by imports (values identical; `simmovie4pi.py` keeps its own simulation
  constants).
- Not done: the simulated-movie drivers (`simfront4pi.py`, `simlin4pi.py`, `sim_cl.py`; audit 2.9) still read `movie.npy`
  themselves and keep their own loops. They share the solve and the in-situ functions with the real pipeline, not the
  driver. Merging them changes the code path of the showcase figures and needs a showcase re-run to verify.

Checks (code of 60cc637 as reference, 2400 frames of NPC file 10): N4 network table identical (all arrays), merged
summary identical except the timings; G4 identical; C4: validation on isolated emitters identical (npz and text), per-file
classical table identical, full driver (read stage, fit stage): table, saved patch meta and patch file identical.
4Pi claim test (`out/cleanup/claim_test4pi.sh`, two files x 1200 frames): one process against two claiming workers
started by the launcher, where the worker that held file 11 was killed with `-9` after 20 s: the surviving worker took
the file over ("locs 11: took over the claim of the dead process …"), both tables identical to the one-process run, no
claim left, both files in the merged summary.

## Audit items (user decision 5)

**Selection inside the package (2.6) and the crop (defect 5).** `simcode/strict.py`:
- new column `main` = `isfinite(strict_u) & (strict_u <= u_max)`, `u_max` default 0.45 (`U_MAX`; `--u_max`), written into
  the table (`main`, `u_max`) and the info json (`u_max`, `n_main`); `select(T, rule)` for `main / strict / sel / good /
  all`; `write_subset` / `--subset main,strict` write `locs_p2_u045.npz` / `locs_p2_strictonly.npz` in the format of the
  former ad-hoc heredocs of `runs/psfmatch/{prod,u045}.sh`.
- the crop of the tiles and of the coverage is a parameter (`build_table(crop=)`, `--crop`), by default the output region
  of the run (read from the `crop` entry of its pass-1 files); `analysis/papersim/run_simcode.py` passes it instead of
  patching `strict.CROP` and two function defaults at run time.
- `--sig_u` is resolved through the models directory; `--dest` writes the table files elsewhere; table writes are atomic.
- Column name `main`: the plan's proposal; the words are the user's to decide (plan section 4).

**Network sets in one place (2.5).** `simcode/netsets.json` (step 1) + `run.py`: `--netset <name>` (default m1809),
`--cal_kind real|sim`, `--dataset <name>` = movie, calibration and set of a paper dataset plus the production recipe
(flags `--autoc 0 --cand_pmin 0.3`, pass 2, `--sig_u set`, `--subset main,strict`). `--autoc` now defaults to 0 (was 1):
the automatic-c sigma table exists only for the removed default head, so 1 with the default set would be an unvalidated
combination. `run.py` passes its crop and `--u_max` to the table step.

**One FRC implementation (2.1).** `simcode/evalproto/frc.py`: `frc(xa, ya, xb, yb, px, units='px'|'nm',
return_curve=False)`, `frc_halves`, and the two variants that give other numbers and are kept so that no figure changes:
`frc_hann` (4Pi showcase evaluation) and `frc_fov` (`hd_compare/score.py`). Call sites now import it:
`evalproto/lib.py` (`frc`, `frc_split`), `analysis/paper/frc_fair.py` (`frc_curve`), `analysis/figures/smoke_metrics.py`,
`analysis/fourpi_classical/fig_cl.py`, `analysis/fourpi/showcase4pi_eval.py`, `analysis/hd_compare/score.py`.
(`papersim/metrics.py` and `figures/lib20.py` already went through `frc_fair` / `smoke_metrics`.)

**Drift as library functions (2.3, 2.4).** `simcode/util/drift.py`: `drift_rows` (the row selection that was only in
`analysis/baselines/dme_drift.py`) and `drift_of_table(T, nframes)`; `dme_drift.py` is its command line;
`analysis/fourpi/post_w4pi.py` and `analysis/fourpi_classical/post_cl.py` call the function instead of a subprocess;
`analysis/paper/frc_fair.py` uses `drift_rows`. `analysis/fourpi/render.py` (calibration chain; it picks a GPU for the
drift) still starts the command line. `simcode/fourpi/zdrift.py`: `zdrift_struct`, replacing the definition in
`fig_cl.py` and its declared copy in `npc_site.py`; `post_w4pi.py` imports it from the package.

**Process titles.** `Units.todo` sets `simcode:<stage> <unit>` for every unit of both pipelines (`simcode:pass1 6000`,
`simcode:4pi-w NPC 10`, `simcode:4pi-cl NPC 10`); title length limit raised from 20 to 32 characters. The `auto('...')`
calls of the entry scripts stay.

**Tests added** (plan section 4, "missing tests"): 1, 2, 7 in `tests/test_fileio.py`; 3, 4 in `tests/test_tables.py`
(table columns and selections on synthetic rows, non-default crop, subset files, drift row selection, FRC sanity);
8 = `tests/check_checkpoints.py` (static) + the GPU test, which loads the default set. Not added: 5 and 6 (they belong to
the LM / CRLB merge, which was not done).

**Documentation.** `simcode/FILE_DESCRIPTION.md` rewritten for the final tree (every package file once); `README.md` (run,
network sets, production recipe, selections, several GPUs, paths, models, 4Pi, layout), `docs/METHOD.md` (network sets
and production recipe; the training commands are labelled as those of the ancestor networks), `models/README.md`,
`simcode/insitu/README.md`.

Checks:
- Tables: `python -m simcode.strict` on the production run `out/paper2/1809_2/v2_matched` (read only, written elsewhere
  with `--dest`) with the m1809 sigma rescale: `locs_p1.npz` and `locs_p2.npz` identical to the stored production tables
  in every previous column (9 214 144 and 6 328 751 rows); new columns `main`, `u_max`; `--subset main,strict` gives files
  identical row for row to the stored `locs_p2_u045.npz` (1 404 302 rows) and `locs_p2_strictonly.npz` (780 180 rows); info json: old values equal, new
  keys `crop`, `n_main`, `u_max`.
- `run.py` without network flags resolves to the production files (check P runs that way).
- PS (a run with crop `0,0,40`): the tables the paper simulations use (`locs_p1_sim.npz`, `locs_p2_sim.npz`) are identical in
  every previous column and the metrics json is identical. **One intended change**: the tables that `run.py` itself writes
  in a run with a non-default `--crop` (`locs_p1.npz`, `locs_p2.npz` of such a run) now use that crop for the tiles, so
  `strict_u`, `strict` and the rescaled `sx`, `sy` differ from before, where they were computed on the default crop
  (audit defect 5). They now equal the `_sim` tables. Runs with the default crop (all real movies) are unaffected (P, and
  the production tables above).
- FRC: every old implementation against the new module on the two-chunk 1809/2 table and on a 4Pi table
  (`out/cleanup/check_a4.py`): `evalproto.lib.frc` / `frc_split`, `smoke_metrics.frc`, `frc_fair.frc_curve` (resolution,
  frequencies, curve), `hd_compare.score.frc_res`, `fig_cl.frc`, `showcase4pi_eval.frc`: identical floats.
- z drift: `fig_cl.zdrift_struct` and the `npc_site.py` copy against `fourpi/zdrift.py` on a 4Pi table (two block sizes)
  and a synthetic set: identical.
- Lateral drift: the row selection of `drift_rows` equals the old script's lines exactly (158 536 rows). The drift itself:
  old script against `drift_of_table` 0.002 nm (2D table, drift range 24 / 33 nm) and 0.03 nm (a 6939-row 4Pi table);
  DME on the GPU is not bit-reproducible, the old script against itself gives 0.0015 nm and 0.19 nm on the same two
  tables, so the differences are within the estimator's own run-to-run variation.
- CPU tests `tests/test_tables.py`: `ALL OK`. `tests/test_solve4pi_oracle.py` (moved from `analysis/fourpi/`, now with
  assertions): the 4Pi solve returns the simulator truth (p99 0.0026 px, 5 nm in z).
- Trainers with the atomic checkpoint save: step-0 lines of det / head / pass-2 det equal within 2e-7; pass-2 head: all
  fields equal except `w_il` (6e-5; the run-to-run range of that field, see the step-1 addendum); the four `final.pt` were
  written through the atomic save.

## Not done, and why

| item | status |
|---|---|
| B6, throughput of 1 vs 2 workers on one TIFF movie | not measured: the data HDD is being read by the running 4Pi job, a timing now would be meaningless and would slow that job. The claim mechanism itself is verified (step B3). README says the speed-up is not measured. |
| 2.9, simulated 4Pi drivers on the shared driver | see step B4 + B5 |
| new 4Pi output layout `out/4pi/<cell>/<head>/…` with an old-name resolver (plan 1.2) | not introduced: every reader of the tables uses the flat names, a whole-movie run is writing them now, and the plan lists it as an open question. `Units` takes the name as a pattern, so the change is one line plus the resolver. |
| Phase C of the plan (trainer loop merge 2.2, 4Pi pattern / solve tidy 2.10 / 2.15, LM / CRLB merge 2.7 / 2.8, table loaders 2.12, renders 2.13, naming 2.16-2.19) | outside the user's list for this pass; each needs a re-run of trainers or figures to verify |
| 2D constants (`PX = 108.33` in 23 analysis files, `P_GOOD`) | not deduplicated (audit priority B; no behaviour involved) |
| 4Pi calibration chain into `analysis/fourpi/calib/` | not moved: the scripts import each other through `sys.path`; a move changes nothing functionally |
| `analysis/gate/run_gate.sh`, `analysis/figures/` | still tied to the earlier default head by name (see open points) |
| history rewrite | not done, as instructed |

## Commits of the branch `cleanup` (base 60cc637)

| commit | step |
|---|---|
| a1085bb | verification helpers (`tests/compare_npz.py`, `compare_ckpt.py`, `check_imports.py`, `check_checkpoints.py`), the audit plan |
| 557c399 | step 1: removals; network sets in `netsets.json`; test data regenerated with m1809 |
| ab9a8f3 | step 2: package rename, no hard-coded repository path, 4Pi set-up module |
| d918493 | step 3: source-only (weights untracked), `models/README.md`, g13 calibration json |
| 05bc2c9 | steps B1 + B2: `simcode/fileio` (readers, work driver, launcher), atomic checkpoints for all trainers, `tests/test_fileio.py` |
| 10f8f1c | step B3: the 2D pipeline on the work driver |
| 114e914 | selection `main` / subsets / crop in `strict.py`; `run.py --netset / --dataset / --claim` |
| 16792ba | steps B4 + B5: 4Pi network and classical pipelines on the shared reader and driver; three first-generation scripts removed |
| 111e7cb | one FRC implementation, drift and z drift as library functions, `tests/test_tables.py`, 4Pi solve oracle test |
| (the commit after 111e7cb) | documentation: this log, `FILE_DESCRIPTION.md`, `README.md`, `docs/METHOD.md` |

The checks of steps 1-3 ran on the tree of that step. The later steps were developed together and checked on two tree
states: the 2D checks (T, P, tables, claim / kill tests) with the 2D changes in place, the 4Pi checks with the 4Pi changes
in place, and all checks once more on the final tree:

| check | result on the final tree |
|---|---|
| S, `tests/test_fileio.py`, `tests/test_tables.py` | `tree ok`, `ALL OK`, `ALL OK` |
| T | `ALL OK` |
| P | 8 of 8 chunk files bit-identical to `out/paper2/1809_2/v2_matched`; the two tables identical to those of 60cc637 |
| Tr | det / head / pass-2 det within 2e-7; pass-2 head equal except `w_il` (6e-5, run-to-run range) |
| G4 | identical |
| N4 | identical |
| C4 | validation, per-file table, full-driver table, patch meta and patch file identical |
| PS | chunk files and `_sim` tables identical, metrics json identical (the run's own tables differ by the crop fix, see "Audit items") |


## Open points for the user

1. **Earlier default set.** det_n1 / hd_sc5nb / p2_det / p2_head_long are no longer a selectable set, but `hd_sc5nb.pt` is
   the initialisation of every PSF-matched head, `det_n1.pt` initialised the 4Pi detector, and the hd_sc5nb calibration
   json are the templates of the calibration fit of a new set. Kept: those json files; `models/README.md` lists the two
   weight files as needed for retraining. Decide whether that lineage stays, or the sets get a from-scratch recipe.
2. **`run --autoc` default 0** (was 1), see "Audit items". Say so if 1 should stay the default.
3. **Names of the selection columns**: `main` (new) next to `good`, `sel`, `strict`, `strict_u`; subset files
   `locs_p2_u045.npz`, `locs_p2_strictonly.npz` (names of the existing files kept).
4. **`analysis/figures/`, `analysis/gate/run_gate.sh`, `docs/RESULTS.md`, `analysis/results/`** describe the development
   phase with the earlier default head (result tags `_hd_sc5nb…`, `hd_sc5nb.pt` by name, expected gate value 1.17). Not on
   the removal list, kept unchanged. Remove, or re-point at a network set?
5. **4Pi head v3L** (`head4pi.py`, `train_head4pi.py`) is kept because v10 contains it and the simulated showcase uses the
   v3L checkpoint as candidate front.
6. **Tracked non-source files** kept for now: `models/insitu/camera_maps_*.npz` (4), `analysis/simreal/fields/*.pt` (2),
   `tests/data/pattern_est_1809_2.pt`, `analysis/results/` (text snapshots).
7. **20 nM / 2 movie**: no PSF-matched set exists (`netsets.json` has `netset: null` for it; `--dataset 20nM_2` asks for
   `--netset`).
8. **4Pi output layout** and the **simulated 4Pi drivers**: see "Not done".
9. **Weights location** (not decided): `models/README.md` has the file list with checksums; a fetch script can be added
   once there is a location.

## Notes for the merge into main

- The branch deletes 25 tracked files under `models/` from the index (`git rm --cached`). A merge removes files that the
  branch deleted from the working tree of main as well. **Before merging, copy `/home/jc/simcode2/models` aside (or run
  the same `git rm --cached` on main first)**, then put the weights back.
- The branch also deletes the calibration json of psf3dmix / pair / sc7bg and `sigu_p*_new.json`; they stay in the
  history (tag the pre-merge commit, e.g. `pre-cleanup` on 60cc637).
- Uncommitted work in the main tree at the time of branching touches files the branch renamed or removed:
  `analysis/papersim/netsets.json` (g13 entry: carried over into `simcode/netsets.json`; the file is removed here),
  `analysis/papersim/fig_overview.py` (modified), `analysis/papersim/{make_movie.py, si_crlb_figure.py, run_g13b.sh}`
  (new; they still import `simcode2`), `docs/CLEANUP_PLAN.md` (a copy is committed here), six untracked g13 calibration
  json under `models/calib` (committed here unchanged), `models/insitu/calib_gauss1.3.json` (modified in main; not
  touched here). Commit them on main before the merge. `python tests/check_imports.py` lists every `simcode2` reference
  left after the merge.
- `runs/` (gitignored, 74 files) still says `simcode2` and `python -m simcode2.…`; the production recipe it holds is now
  `python -m simcode.run --dataset <name>`.
- The main tree's `simcode2.json` keeps working (fallback). For the 4Pi scripts nothing has to be added: the 4Pi work
  directory defaults to `<out>/4pi`, the data root to `/mnt/data/4pi-sf` (`simcode/fourpi/datasets.json`).
- Claim markers of the old shell driver (`out/4pi/net/locs_*_v10.claim`) are not read by the new driver (its markers are
  `locs_*_v10.npz.claim`); delete them after the running job has ended.
- After the merge: `python tests/check_imports.py`, `python tests/test_fileio.py`, `python tests/test_tables.py`,
  `CUDA_VISIBLE_DEVICES=<g> python tests/test_fast_paths.py`.

## Addendum: the production recipe through `--dataset`

`python -m simcode.run --dataset 1809_2 --out <dir> --nframes 12000 --chunk_list 0,6000` (movie, calibration and network
set from `netsets.json`, the movie read from the data root, the reference run's pattern placed in the directory): 8 of 8
chunk files bit-identical to `out/paper2/1809_2/v2_matched`; tables with the set's sigma rescale (every column except the
rescaled `sx`, `sy` equal to the table of 60cc637 on the same chunks); `locs_p{1,2}_u045.npz` and `_strictonly.npz` written,
their rows = the `main` / `strict` rows. `simcode.example.json`: `data` now points at the directory the dataset paths are
relative to (`…/bn_dnapaint`).

## Addendum: no `main` column (user decision, 2026-11-12)

The boolean column `main` is not written any more: it duplicated `strict_u <= u_max`. The table keeps `strict_u` and the
stored `u_max` (npz entry and info json, default 0.45; `n_main` stays in the info json as a count). Readers get the row
mask from `simcode.strict.select(table, u_max=None)` (finite `strict_u` and `strict_u <= u_max`; default = the stored
value). `--subset main` still writes the `_u045` file. Where this log says "column `main`" above, read `select(table)`.
No kept script read the column. Checks: `tests/test_tables.py` `ALL OK`; `python -m simcode.strict` on
`out/paper2/1809_2/v2_matched`: `locs_p2.npz` identical to the stored table in every column (only new entry: `u_max`),
`locs_p2_u045.npz` (1 404 302 rows) and `locs_p2_strictonly.npz` identical row for row.

The other boolean columns of the 2D table, and which kept code still reads them (grep for `['<name>']`; 4Pi tables have
columns of the same names `good` / `sel` with their own meaning, those readers are not listed):

| column | read by |
|---|---|
| `good` (p >= 0.63) | `simcode/util/drift.py` (drift rows: good & disjoint), `simcode/strict.py` itself (the rank is defined on the good rows), `analysis/psf3d_nets/sigcal.py`, `analysis/evalproto/strictq.py` |
| `sel` (phase-calibrated) | `analysis/paper/frc_fair.py` (loader of v2 tables and subset files), `analysis/papersim/metrics.py`, `analysis/papersim/pvalid_real.py` |
| `strict` (density-rule product) | `analysis/papersim/metrics.py`, `analysis/papersim/selection.py`, `analysis/psf3d_nets/sigcal.py`, `analysis/evalproto/strictq.py`; `--subset strict` (the `_strictonly` file) |
| `disjoint` (frame % 6 == 0) | `simcode/util/drift.py` |

## 2026-11-14: options clean-up (user principle: an option is either better and always on, or worse and removed)

Branch `optclean` (worktree `.claude/worktrees/optclean`, base 42f36ba), done while the generic pass-2 chain (gnr80) ran on
the main tree; merged after it ended. Audit: the detband worker's list (out/detband), decisions of the user / main session
of 2026-11-14. Verification scripts and outputs: `out/optclean/` (`verify.sh`, `run_ref.sh`, `train_smoke.sh`,
`fitcheck.sh`, `cmp_trainlog.py`; baselines from a second, unmodified worktree of 42f36ba).

Checks (all on GPU 2):
- P (as above): two chunks of 1809_2, pass 1 + pattern solve + pass 2, m1809, now run **without any flag** (`--autoc 0
  --cand_pmin 0.3 --pass2` are the only behaviour): 8 of 8 chunk files bit-identical to `out/paper2/1809_2/v2_matched`
  (also after the trainer / PSF step).
- Tr: 20 steps of the five production trainers (det, hd, p2det, p2head, p2headL; m1809 recipe, batch 4): the new command
  lines (flags that became defaults dropped; p2head with `--pert_model v1`) vs the old explicit flags on 42f36ba: step-0 lines
  of det / hd / p2det / p2headL equal (max rel. diff 0 / 0 / 0 / 4e-7), p2head equal except `w_il` (2e-5, the known
  run-to-run range of that field, see above). The old command lines (`TRV=old`, the removed flags dropped by
  retired.py) give the same for det / hd / p2det / p2headL; the old p2head line (no `--pert_model`) now trains with `real`
  (step-0 loss 2e-2 different), hence `--pert_model v1` in the p2head stage of chain2.sh, auto/train.py and models/README.md.
- G: gate scenes iso + rs05qm, seed 301, 400 targets: new code without env (default PSF = the model's) vs 42f36ba with
  `SIMCODE_GATE_CALIB=insitu/calib_1809_2_psf3d_avg.json`: iso identical; rs05qm identical except CRLB entries (26 of 7956
  `gt_fr_m` differ by <= 6e-5), the same as two runs of the old code (30 of 7956, <= 3e-5).
- PSF fit: `fit_insitu2` stages bl / blps3 / blad (300 samples, 8 iterations), old vs new: bl, blad json identical; blps3
  identical except `resid_var` (1e-8 relative; nll, gamma, zeta identical).
- T `ALL OK`, F `ALL OK`, TT `ALL OK`; S: only the known `analysis/psfmatch/prod_m20_m25.sh` lines (`python -m simcode2.run /
  strict`, present before this step).

Older scripts may pass removed options: `simcode/util/retired.py` drops an option given with its fixed value (one stderr
line) and stops with an error for any other value (tables `RETIRED` in run.py, run_uncalibrated.py, det/up2.py, hd/sc4.py,
p2/det2.py, p2/head2.py). No run changes meaning silently.

| file | change | evidence | check |
|---|---|---|---|
| run.py | `--pass2` removed (pass 2 always), `--autoc` / `--autoc_fast` removed (autoc no longer a stage), `--cand_pmin`, `--fast`, `--compile`, `--compile_head`, `--pattern_field`, `--strict_sigma_ref` removed; `--sig_u` default `set`, `--subset` default `main,u,strict` (the `--dataset` recipe is now every run's); `--model` (alias `--netset`); summary key `model` | production = `--autoc 0 --cand_pmin 0.3 --pass2 --sig_u set --subset main,u,strict` (models.json `_production`, the paper runs) | P |
| pipeline/run_uncalibrated.py | `--fast` (reference path only in tests), `--compile` (always `auto`), `--cand_pmin`, `--compile_head`, `--solve_opts`, `--pattern_field` (est / est_depth: no refit exists), `--p2_pf_head`, `--p1_solve` (sc2p never runs the pass-1 solve; run.py runs it with `--frontend sc`) removed | run.py passed `--fast 1 --compile auto --compile_head off --p1_solve 0`; no kept script used the others with a non-default value | P |
| pipeline/common.py, sc_front.py, sc_front_fast.py, p2/front.py | one `CAND_PMIN = 0.3` (was 0.5 in three modules, overridden by run.py); QueuedFront without compile_head (blk / trunk) / channels_last; pass 2 without the `pf` head prior and the `_dir` calibration keys (`kap_bins_dir`, `sig_ph_dir`, `prior_off_dir`: only in the b10 calibrations, removed there; est2b of the b10 calibrations changes, no kept model uses them); `run_chunk_pattern_solve_fast` without `field` / `solve_opts` | grep: `p2_prior` was `own` in every calibration | P, T |
| hd/sc.py, hd/msolve_fast.py, hd/msolve.py | solve keys required: kap_bins, prior_off, cmult, cmult_ang, box, mix_T, sig_a, sig_g, sig_ph (all shipped calibrations have them); `mix` hard-wired to the posterior mean (`pm` in every calibration; key removed), `ms_xdet` (false in all 35) + its branch, the non-mix `solve_ms` path (unreachable), `kap_mult` / `kap_mult_src`, `cmult_logdet` (no calibration sets them), `sig_scale` (1.0 in the 5 autoc calibrations; key removed) removed; msolve_fast without the approximate options (tol, freeze_cov, prune, prune_k, cov_mode, stats; only reachable through `--solve_opts`) and the Student-t prior `nu` of msolve | calibration key census of models/calib | P, T |
| autoc.py | kept as an analysis estimator; its covariance scale s now scales L (sqrt(s) L) instead of the calibration key sig_scale; analysis/nanoruler/abl_gate.py the same | -- | S |
| strict.py | `sigma_ref` rule and `--strict_sigma_ref` removed (and `n_for_cov`); `build_table(..., strict_q, *, sig_u=...)` keyword-only after strict_q (callers updated) | opt-in earlier rule, valid only with the fixed c | TT |
| fastutil.py | env `SIMCODE_COMPILE` (the duplicate of `--compile`) removed | -- | T |
| models/calib/*.json (tracked) | keys `ms_xdet`, `mix`, `sig_scale`, `p2_prior` and the b10 `_dir` keys removed | the code no longer reads them | P |
| hd/gate.py, hd/sc.py, sim/crlb.py | `--det` replaces env `SC_DET`; the scene PSF defaults to the default model's PSF (models.json `psf`; was insitu/calib_0.5nM.json; env SIMCODE_GATE_CALIB still overrides); PSF stack chunk size automatic (halved on out-of-memory; env GATE_PSF_CHUNK removed), `GATE_PSF_INNER` / `GATE_CRLB_DIV` envs removed (constants 1024 / 3, the defaults); rs05r / rs25r(b) scenes removed (no kept figure / paper script uses them; only listed in analysis/gate/run_gate.sh's comment); `--pover` / `--sover` flags removed (no script passed them; the POVER / SOVER dicts stay: analysis/ablation_cov/gate_dump.py sets them for the paper's covariance ablation). hd/sc4.py prep() keeps the 0.5 nM PSF for its scenes | grep of SC_DET / GATE_* users: all at the default values | G |
| det/up2.py | `--up` (1), `--em_photons` (40,2500) fixed; `--band` (band_extra, band_off_loss; user: do not change the detector), `--psum`, `--gen r1`, `--pair_sub` removed; `--calib` required | every model's recipe | Tr det |
| det/model.py, p2/det2.py | psum (summed-p output) removed; det2 `--band`, `--psum`, `--gen`, `--pair_sub` removed, `--calib` required, `--init_n1` default = the default model's detector | -- | Tr p2det |
| hd/sc4.py, hd/data.py | `--arch` (sc4n), `--paper` (1), `--bright_tail` (0.1,2000,20000) fixed (stored in args.json / checkpoints as before), `--tau` default 15,80, `--paper_med` / `--paper_sd`, `--psfdiv` (+ data.Gen psfdiv / ev_shift), `--gen r1`, `--pair_sub` removed; `--calib` required | every model's recipe | Tr hd |
| p2/head2.py, p2/data.py | `--pert_model` default `real` (p2headL); the p2head stage passes `--pert_model v1` explicitly (chain2.sh, auto/train.py, models/README.md); `--gen`, `--pair_sub` removed; `--calib` required; `--hold_v1` kept (the v1 model is used by p2head) | -- | Tr p2head, p2headL |
| hd/realgen.py | removed (RGen; networks p1r / p2r not adopted), with the realsim `pf_*` per-FOV pattern-field keys (used only by RGen and the rs05r / rs25r scenes) | -- | Tr, G |
| hd/realsim.py | `pf_*`, `ev_shift`, the `psfdiv` preset key, `pair_sub`, the scene key `halo2` removed; `add_halo2`, `filament_static_map`, `psf_div_luts`, `smooth_field` kept (used by analysis/simreal / nanoruler simulators). The pattern draw (pattern(), _pgen) untouched | -- | Tr, G |
| insitu/fit_insitu2.py | kept: `--blur line / angle_delta`, `--photosel 3`, `--init-fit`, `--freeze-base`, `--split` / `--tile`, `--pol-calib`, numeric settings; `--intensities` default 0 (passed by all 20 calls); removed: `--psf`, `--fit-zernike`, `--fit-apod`, `--fit-na`, `--dipole`, `--fit-saf`, `--gamma-init`, `--photosel 1 / 2`, `--blur global / angle / angle_line` | flags of every fit_insitu2 call in out/, runs/, analysis/ (out/psf3d/run_final*.sh: tracks, out, intensities, init-fit, freeze-base, blur, photosel, split) | PSF fit |
| psf/vectorial.py, analysis/psf3d/make_calib.py | calibration keys `psf_dipole_w`, `saf` no longer read (no json in models/insitu or out/nanoruler has them); make_calib writes a single dipole weight as `psf_dipole_M` and refuses `saf` | key census of 28 PSF calibrations | P |
| config.py, netsets.json -> models.json, auto/* | "network set" -> "model": `simcode/models.json`, `config.model_set()` / `model_sets()` (`netset` / `netsets` aliases), env SIMCODE_MODEL (SIMCODE_NETSET accepted), dataset key `model`, `_production` keeps `cal_kind` only (the other flags are the only behaviour); auto config keys `train.model`, `process.model` (`netset` accepted; `process.pass2` removed), stage e writes `<out>/train/model.json` (netset.json still read), runwrap key `model` | user, 2026-11-14 | F, P |
| tests/test_fast_paths.py | compiled-head tests removed (option gone); the compiled solve's median tolerance 5e-5 px: with the stored-candidate cut 0.3 the test rows include dimmer candidates (median 2.3e-5 px; 8.6e-6 px on the p >= 0.5 rows, as before; eager unchanged < 1e-5); the pass-2 comparison builds Pass2P2Fast with `compile='off'` (its default became `auto`; the test compares the eager solve, as before: median 7.6e-6 px) | -- | T |
| docs | README, docs/METHOD.md, docs/AUTO_PIPELINE.md, simcode/FILE_DESCRIPTION.md, models/README.md: model wording, removed options, the training recipe without the flags that became defaults | -- | S |

Kept (real parameters, user decision): `--calib` (now required by the trainers), `--dens` / `--tau` (per-model recipe),
`--strict_q`, `--ref_pattern`, `--no_illum` (paper ablation), the rule `u` / `--u_max`, the RealSim realism keys (not folded
into presets: they are already the presets' keys), the generic pass-2 options (`--pgen`, illum model `realw`: another
worker). Not changed: simcode/fourpi, the pattern-draw code of realsim, p2/illum.py, the pgen parts of det2 / head2.
Callers updated: analysis/papersim/run_simcode.py (`--model`, no `--cal_kind autoc` / `--no_pass2`), the build_table callers,
analysis/ablation_cov/capture.py and analysis/nanoruler/abl_capture.py (no solve_opts), analysis/speed/prof_stages.py,
bench_head.py and analysis/tput/bench_tput.py (`compile=` instead of solve_opts), analysis/evalproto/strictq.py (no sigma_ref
runs), analysis/figures/lmref.py (constant-1 kap_bins / sig_ph), the SC_DET / GATE_PSF_CHUNK settings of the analysis shell
scripts (all at the default values; analysis/p2generic left as it is, another worker's running scripts). Scripts that
measured a removed option now stop with an error: analysis/detband/train_ab.sh (`--band`), the compile_head / channels_last
variants of analysis/speed/bench_head.py, bench_p2.py and analysis/tput/bench_tput.py; they are records of finished studies.

## 2026-11-14: renaming pass (aliases out; user-approved queue of out/HANDOFF_SESSION.md)

Scope: renames and the removal of every compatibility alias. Table formats and output locations are unchanged (the HDF5
dataset format, converter and `runs/` layout of `docs/DATASET_FORMAT.md` are the next pass). The 52 delete candidates of
`out/cleanpass/DELETE_CANDIDATES.md` are not deleted (not approved); only the mechanical `simcode2` -> `simcode` import
rename was applied to the 7 of them that imported `simcode2` (so that `tests/check_imports.py` is clean).

| what | change |
|---|---|
| `simcode2/` | the import alias package deleted; every `import simcode2` / `python -m simcode2.*` / `simcode2/<file>` mention in the tree is `simcode` now (docs/CLEANUP_LOG.md and CLEANUP_PLAN.md keep the historical names) |
| `simcode/config.py` | no `SIMCODE2_*` variables, no `simcode2.json` (the repository config is `simcode.json`; there was no `simcode2.json` on this machine), no `netset()` / `netsets()` / `NETSETS_FILE`, no `SIMCODE_NETSET`; `model_set(name)` takes a name of `simcode/models.json` or the path of a model folder / `model.json` (one model; paths relative to the folder) |
| `simcode/util/retired.py` | deleted; `run.py`, `run_uncalibrated.py` and the trainers (`det.up2`, `hd.sc4`, `p2.det2`, `p2.head2`) parse their options directly, so a removed option is an argparse error |
| `simcode/locs.py`, `Localizations` | deleted |
| `simcode.run` | `--model` required (or from `--dataset` / `--config`); `--netset`, `--det`, `--head`, `--cal`, `--p2_det`, `--p2_head`, `--p2_cal`, `--u_max`, `--strict_q` removed; `--subset` = thresholds in nm (default `--sigma_max`, i.e. `locs_p*_sig12.npz`); a re-run (`--config`) of a record made with single files registers them as model `<name>@record`; `runconfig.backfill` reads old command lines itself |
| `simcode/strict.py` | `select(T, max_precision_nm=12, min_score=0.63, in_region=True, crop=None, px_nm=None)` from x, y (x_dc, y_dc with `--drift`), sx, sy, p and the crop; the rule names (`main`, `u`, `strict`, `sel`, `good`, `all`), the u rule (`strict_u`, `U_MAX`, `u_max`), the strict product (`strict` column, `strict_product`, `Q_DENSITY`, `q_from_density`, `bin20`, coverage / q info) removed. The per-10-px-tile rank (`tile_rank`) stays internal: the model's selection-aware sigma rescale (`sig_u`) is a function of it. New tables keep every other column, including `good` and `sel` (until the format pass); the info json has `n_main` (rows of the selection), `n_good_crop`, `crop`, `sigma_max`, `median_photons` (of the selection) |
| `simcode/dataset.py` | snake_case only: `load_npz` / `save_npz` (v2 table; was `from_table` / `loadNumpy`), `load_npy` / `save_npy` (preprint layout; was `from_smlmtorch_npy` / `saveNumpy`), `load_hdf5` / `save_hdf5` (Picasso; was `from_picasso` / `save_picasso` / `saveHDF5` / `loadHDF5`), `load_csv_thunderstorm` / `save_csv_thunderstorm`, `from_estimates` (was `from_arrays` / `fromEstimates`), `add_fields`, `num_frames` (property; was `nframes` / `numFrames`), `indices_per_frame`, `crop_from_edge`, `apply_drift` / `undo_drift` / `estimate_drift` / `is_drift_corrected`; `merge`, `fields`, `background` aliases removed. `select(max_precision_nm=12, min_score=0.63, in_region=True)` / `mask(...)`; in-place variants `filter_`, `crop_`, `crop_from_edge_`, `select_`, `crlb_filter_`, `autocrop_` (return the object itself) |
| `simcode/auto` | no `netset` config key / `netset.json` / `process.pass2`; the training stage's gate scenes get `SIMCODE_GATE_CALIB` / `SIMCODE_GATE_DIR` (the `SIMCODE2_` names it used are no longer read); runwrap's crop override patches `tile_of` / `tile_rank` / `in_crop`; the report's FRC uses `Dataset.select()` |
| 4Pi | `simcode.fourpi.insitu.slotZ` -> `slot_z` (and its analysis callers) |
| `PSFLUT` | no leftovers (renamed to `ConvSplinePSF`, `simcode/psf/conv_spline.py`, in 2804e7a) |
| pyproject | package data `models.json` |

Checks:
- T: `pytest tests` all pass (44); `tests/test_tables.py`, `test_dataset.py`, `test_runconfig.py`, `test_fileio.py`
  updated for the new APIs (the old names must be absent; model folders; the selection from x, y, sx, sy, p).
- S: `tests/check_imports.py`: no `simcode2` left.
- P: `python -m simcode.run --movie <1809/2> --calib insitu/calib_1809_2_psf3d.json --model m1809 --nframes 12000` with the
  reference run's pattern: 8 of 8 chunk files (`pass1_{0,6000}`, `pass2_est_*`, `pass2_est2_*`, `pass2_est2b_*`)
  bit-identical to `out/paper2/1809_2/v2_matched` (out/cleanpass/bitid).
- Table: `strict.build_table` on the whole `out/paper2/1809_2/v2_matched` (pass 2, m1809 sigma rescale): every column of the
  new table bit-identical to the existing `locs_p2.npz`; `select(table)` = the stored-rank rule (2 141 721 rows);
  `Dataset.select()` rows = `locs_p2_sig12.npz` (also for 2.5nM_2).
- Fig. 3 / 4 (fig_exp_new.py / frc_fair.py, 12 nm rule): the SIMCODE entry recomputed from `Dataset.select()` of the full
  table with the cached drift: FRC 17.529 nm (1809_2) / 22.257 nm (2.5nM_2), n_frc, no-filter and 30-frame FRC identical
  to the stored `*_fig3_matched_sig12_values.json` / `*_fig4_matched_sig12_values.json`; the SMLM / SIMFLUX / DECODE
  loaders give the stored row counts.
- PS: `analysis/papersim/metrics.py` density metrics on `out/papersim/Dexp0.5` (60k frames, m1809) recomputed: every number
  of the remaining methods (smlm, sf, sf_gtp, decode, v1, v2sig) identical to the stored `metrics_m1809.json`; the methods
  `v2` (strict product) and `v2q45` (u <= 0.45) are gone, the key `netset` is `model` (out/cleanpass/papersim_check.json).
- R: `python tests/test_runconfig.py --gpu` (2 chunks, re-run from the record, bit-identical); the file count of a run is 12
  npz now (subsets: sig12 only).

Analysis scripts (kept ones; commits 603f9cb, dcfd268, 1cbdaa1):
- `simcode.run` callers use `--model` (a name, or a one-model json: `out/nanoruler/models/netset_nr80.json`, or one written
  by the new `analysis/nanoruler/mkmodel.py` from the single files a script used to pass); the nanoruler sim / evaluation
  runs pass `--sig_u ''` (unrescaled tables for the sigma-rescale fits, as when they were made); the removed trainer and
  `run_uncalibrated` options are dropped from the calls.
- The u rule / strict product rows of the paper-simulation tables and figures are removed (papersim/metrics.py keeps v2sig
  = the 12 nm rule on the simulation FOV); the nanoruler ablation's main selection u45 is now m12 (12 nm, `strict.select`;
  new outputs are named `*_m12*`; the existing u45 files stay as history). `--netset` -> `--model` everywhere; output file
  names unchanged (`metrics_<model>.json` ...), new json outputs write `model`.
- `psf3d_nets/sigcal.py`, `nanoruler/ev_sigcal.py` recompute the rank u with `strict.tile_rank` (= the stored strict_u to
  float32 rounding); the fit tables must be built without `--sig_u`.
- untracked `runs/psfmatch/*.sh` (not in git): prod.sh = `simcode.run --model $S`; eval*.sh write a model json; FRC inputs
  `locs_p2_sig12.npz` instead of strictonly.

Left as they are (their subject is the removed u rule / strict product; they no longer run on new tables; proposed for the
deletion list): `analysis/evalproto/strictq.py`, `analysis/ablation_cov/movie_abl.py`,
`analysis/selection_global/{sweep_real,sweep_sim,extract,fig_sweep,render_compare}.py` (+ `run_figs.sh` still works: it only
copies cached drifts and calls fig_exp_new with the sig12 files), untracked `runs/psfmatch/u045.sh`.
Delete candidates that would need changes (not touched): detband/movie.py, hdunbias/movie.py (`RS.netset`, `--netset`),
detband/train_ab.sh, detband/ft_head.sh, hdunbias/ft.sh (removed trainer options), speed/e2e_prefetch.sh, speed/prof_stages.py
(removed options / strict product), psfmatch/m20ghost_real.py, m20ghost_run.py, m20check_ghost_sim.py (strict_u, single files),
papersim/run_g13b.sh (`--netset`), nanoruler/nr_p2gen.sh, ev_flipreal.sh (single files, `--pass2 --autoc 0`).
Not renamed (on-disk names; layout pass): `out/nanoruler/models/netset_nr80.json`; `analysis/papersim/netsets.json`
(untracked leftover, no reader left). Kept until the format pass: the table columns `good`, `sel`; old tables' `strict_u`,
`strict`, `u_max` are read as plain columns and not used. smlmtorch's own camelCase methods (fromEstimates, numFrames) in
analysis/baselines are that package's API and stay.

## 2026-11-14: dead analysis scripts deleted (user approved)
Deleted (they remain in git history): analysis/detband/, analysis/hdunbias/, analysis/speed/ (whole directories); zimflux/s2_{crlbcheck,run,tracks}.py;
psfmatch/m20check_{dump,fig,gates,ghost_sim,psf}.py, m20ghost_{real,run}.py, psf_compare.py; papersim/run_g13b.sh, papersim/netsets.json;
p2generic/{t1_eval.py,t1_run.sh,cont_gen.sh}; fourpi/after_mt_w4pi.sh; nanoruler/{move_cache.sh,r24_memlog.sh,r24_refshift.py,ev_flipreal.sh,nr_p2gen.sh,
r2040_{cycle,meta,pattern,sample,tables}.py,meta.py}; studies of the removed u / strict rules: evalproto/strictq.py, ablation_cov/movie_abl.py,
selection_global/{sweep_real,sweep_sim,extract,fig_sweep,render_compare,write_subsets}.py; runs/psfmatch/u045.sh (untracked).
Reasons per file: out/cleanpass/DELETE_CANDIDATES.md and the rename-pass report. Checks after deletion: tests 44 passed, check_imports clean except the known tests.test_tables note.

## 2026-11-14: generic pattern draw = the only pass-2 training draw (user decision)
"The generic pass 2 is completely functional, so we don't need any of the dedicated things anymore. From now on all pass-2
models are generic."
| file | change |
|---|---|
| hd/realsim.py | the RealSim preset key `pgen` (opt-in dict) removed; the generic draw `RealSim._generic` (was `_pgen`) uses the fixed ranges `GENERIC` (= out/p2generic/presets/pgen_v1 without p_std; its unused alternatives `phase: steps`, `ksign: false` removed) and the attribute `RealSim.p_std` (None = the SIMCODE pattern: pass-1 training, gate scenes, sim movies) |
| p2/data.py | `Gen2(pgen=None)` -> `Gen2(p_std=P_STD)`, P_STD = 0.5, 0 <= p_std < 1 (no SIMCODE-only pass-2 training) |
| p2/det2.py, p2/head2.py | `--pgen <json>` removed; `--p_std` (default 0.5): share of FOVs with the SIMCODE pattern |
| auto/train.py, auto/config.py | stage e = the generic recipe: p2det (p_std 0.25) -> p2det_c (p_std 0.5, the model's pass-2 detector); p2head, p2headL (0.25) -> p2headL2 (0.5, the model's pass-2 head); steps keys p2det_c, p2headL2 |
| analysis/p2generic/{chain_gen,chain_gen2,cont_det,cont_gen2}.sh | `--pgen <preset>` -> `--p_std <share>` (pgen_v1 = 0.25, pgen_v1s50 = 0.5); p2head stage with `--pert_model v1` |
| models/README.md, docs/AUTO_PIPELINE.md, docs/METHOD.md | recipe / description |
Check: Gen2 batches (frames, patterns, estimates, steps, illumination maps; both p_std values, 'paper' / 'det' mixes, v1 / real
error models) bit-identical to the former `--pgen pgen_v1` / `pgen_v1s50` with the same seeds; RealSim.scene without
p_std bit-identical to before (out/z3d/checks/gen_bitid.py).

## 2026-11-15: package reorganisation (user approved)

Descriptive package layout, one network class per network for both passes, no aliases or old-name re-exports; the tree
and the full old -> new table are in `docs/LAYOUT.md` (section "Renamed"). In short: `auto/` -> `pipeline/`, the former
`pipeline/` -> `processing/`; `det/` + `p2/det2.py` -> `detector/` (one class `Detector`, one trainer); `hd/sc4.py` +
`p2/head2.py` + the network part of `hd/sc.py` -> `emitter/` (one class `EmitterNet`, one trainer, `heldout.py`); `core.py`,
`hd/msolve*.py`, `autoc.py`, the solve and z parts of `hd/sc.py` -> `solve/`; `p2/illum.py` -> `pattern/illumination.py`;
`hd/realsim.py` (RealSim) -> `simulate/acquisition.py` (AcquisitionSim), `hd/data.py` + `p2/data.py` + `det/up2.batch` ->
`simulate/batches.py` (TrainingBatches), `sim/` -> `simulate/`; `hd/gate*.py`, `eff/` -> `benchmark/`; `insitu/` ->
`calibration/`, `evalproto/` -> `evaluation/`, `nn/` -> `layers/`, `fileio/` -> `io/`, `fastutil.py` -> `util/compile.py`,
`strict.py` -> `table.py`; `p2/` removed. Trainer command lines: `python -m simcode.detector.train` (pass 2: `--illum 1
--init <pass-1 detector>`, `--init_n1` -> `--pass1_ref`), `python -m simcode.emitter.train` (pass 2 when `--detector` / `--init` is
a pass-2 checkpoint), `python -m simcode.emitter.heldout` (was `simcode.hd.sc4 prep` / `prep5`).
Checks (GPU 0): every checkpoint under models/ (33 detectors, 34 emitter networks: models.json models, model folders,
archive_2d, archive_p2dedicated, the other weights) and the 4Pi detector / heads v10, v3L give bit-identical outputs on a
fixed batch before / after (973 tensors, `tests/check_model_outputs.py`); `simcode.run --dataset 1809_2 --nframes 2000`
bit-identical to the z3d code (every chunk file and table); the merged trainers run (6 smoke trainings of the recipe
stages + the zero-init checks), with the same batches, targets and step-0 losses as the former trainers up to the GPU's
run-to-run noise; pytest, `tests/test_fast_paths.py`, `tests/check_imports.py`.

## 2026-11-15: the remaining old names (user approved)

Every remaining name from the internal codes is descriptive now (table in `docs/LAYOUT.md`, "Renamed"):
- model definition keys det / head / p2_det / p2_head / p2_cal_sim / p2_cal_real -> detector / emitter / detector_pass2 /
  emitter_pass2 / cal_sim_pass2 / cal_real_pass2 (simcode/models.json, models/*/model.json; the loader reads only the new
  keys and refuses the old ones with the name of the converter `analysis/maintenance/rename_model_keys.py`); config.DET /
  HEAD / P2_DET / P2_HEAD / CAL_P1_* / CAL_P2_* -> DETECTOR / EMITTER / DETECTOR_PASS2 / EMITTER_PASS2 / CAL_REAL / CAL_SIM /
  CAL_REAL_PASS2 / CAL_SIM_PASS2 / CAL_AUTOC*.
- run record format `simcode.run_config/2` with the same key names (settings, model files, fixed emitter_batch /
  compile_emitter); records of format 1 are converted when read (`util/runconfig.upgrade`), so `simcode.run --config <old
  record>` still re-runs them (checked on a record written by the code before).
- CLI: run_uncalibrated `--frontend sc|sc2p` -> `--pass2_mode solve|networks`, `--sc_det / --sc_head / --sc_cal / --p2_det /
  --p2_head / --p2_cal / --head_batch` -> `--detector / --emitter / --cal / --detector_pass2 / --emitter_pass2 / --cal_pass2 /
  --emitter_batch`; benchmark.gate `--det / --sc / --sccal` -> `--detector / --emitter / --cal`; emitter.train `--det` ->
  `--detector`; pipeline training stages det / hd / p2det / p2det_c / p2head / p2headL / p2headL2 -> detector / emitter /
  detector_pass2 / detector_pass2_cont / emitter_pass2 / emitter_pass2_long / emitter_pass2_final (acquisition config
  train.steps / init / extra).
- batch keys patt_p2 / illum_p2 -> patt_input / illum_input; front constructors and attributes head / det -> emitter / detector;
  head_z -> emitter_z.
- architecture tags: new emitter-network checkpoints store arch 'emitter_net' (the tag 'sc4n' of the existing checkpoints is
  read through emitter.model.ARCH_TAGS; no .pt file was rewritten), new detector checkpoints arch 'detector'.
- files: calibration/fit_insitu2.py -> fit_psf_tracks.py, benchmark/gate_report_xdet.py -> gate_report_detector_matched.py.
- `if __name__ == '__main__':` guards in the script-like modules (calibration/camera_gain, camera_maps, extract_tracks,
  fit_pattern, fit_psf_sums, fit_psf_tracks, pipeline/acceptance): importing them no longer runs them.
- output paths: `<out>/sc4/heldout{4,5}.pt` -> `<out>/emitter_heldout/`, `<out>/p2/fe` -> `<out>/gate_dumps`, `<out>/p2/long` ->
  `<out>/pass2_gate/long`. The data stay where they are (/mnt/data/dmdsf/simcode2_out/{sc4,p2}): the new names are symlinks
  next to the old ones in ~/simcode2/out; the old names `out/sc4` and `out/p2` are removed once the z3d workers that use them
  are done.
- Still to do at the merge (the z3d workers read them with the old code): the model.json files of the live models directory
  (~/simcode2/models/{m1809,m25,g13}/model.json and models/archive_*/*/model.json, not in git there) are rewritten with
  `python analysis/maintenance/rename_model_keys.py ~/simcode2/models` (`--check` lists the files that still have old keys).
Not renamed (data formats on disk): run-output file names and columns (pass1_<f0>.npz, pass2_est2_<f0>.npz, locs_p*.npz),
gate npz / dump columns and front labels (sc_*, s1 / s2 / p1 / p2L), checkpoint file names, output directories of finished
studies (out/p2generic, out/z3d, ...). simcode/fourpi/ is left for the 4Pi refactor.
Checks: every checkpoint bit-identical (973 tensors, tests/check_model_outputs.py), simcode.run --dataset 1809_2 --nframes
2000 bit-identical to the code before, pytest, tests/test_fast_paths.py, tests/check_imports.py.

## 2026-11-15: processing options and clean-up (user decisions)

Processing options (`simcode.run`, `simcode.processing.run_uncalibrated`, `simcode.table`):

- `--format` removed: the table output is the torch_psf Dataset HDF5 per pass, `locs_p1.hdf5` / `locs_p2.hdf5` (fields
  `docs/DATASET_FORMAT.md`; metadata pixel_nm, crop, tag, info). CSV / Picasso are exports of the loaded Dataset.
  `simcode.table.write_table(T, info, base)` writes it; `Dataset.load_table` / `Dataset.from_run` read it, `Dataset.columns()`
  gives the table columns back (x, y, sx, sy, p, I, ...). The `.npz` tables of earlier runs are still read (`from_run` tries
  `.hdf5`, then `.npz`); `python -m simcode.table <out> --tag est|est2` writes the `.hdf5` table of such a run from its chunks.
- `--subset` / `--sigma_max` removed: no selection files; the paper's selection is applied at load (`Dataset.select()`,
  `select(14)` for another threshold). `info['n_main']` still counts the 12-nm selection. `analysis/paper/fig_exp_new.py`
  and `frc_fair.py` take the run directory / table and `--v2_sigma_max`.
- `--prefetch` removed (always on, off only with `--claim 1`); `--sig_u` removed (always the model's own sigma rescale; the
  table CLI reads it from the run record's model).
- run_uncalibrated `--pass2_mode solve|networks` replaced by the stage names `--stages pass1,pattern,pass1_solve,pass2`
  (default all; `simcode.run` calls them in that order).
- `--crop` default: the whole frame (square sensors; on a non-square sensor the largest top-left square, a limitation of the
  square output region); the paper datasets carry `"crop": "5,5,110"` in `models.json` `_production`, the paper scripts pass
  their crop explicitly. run_uncalibrated without `--crop` takes the crop of existing pass-1 files.
- `--ref_pattern` removed from run_uncalibrated: the GT-pattern diagnostic of the simulations is in
  `analysis/papersim/run_simcode.py` (`gt_pattern_pass2`: pass 2 with the GT pattern in `<run>/gt_pattern/`).
- old run records rerun through `--config` (the removed keys are dropped by `util/runconfig.rerun_argv`; the backfill
  parser still reads the old flags of old command logs).

Clean-up (decisions in `docs/CLEANUP_CANDIDATES.md`): archive mechanism = git tag `archive-2026-11-15` (the state before
the removal) + removal from the tree, index `docs/ARCHIVE.md` (an archive/ directory would hold scripts whose imports no
longer resolve). 284 files archived, 30 deleted (zimflux and the delete category; also in the tag). Mentions in the
documents are written `archive-2026-11-15:<path>` (tests/check_imports.py skips them). Model-weight paths of the kept
p2generic / ablation scripts point to the model folders (`archive_2d/<set>/`, `archive_p2generic/`). out/ and models/:
`analysis/maintenance/cleanup_outputs.py` (dry run with sizes; `--apply` after the merge, then the model.json keys).

Checks: `simcode.run --dataset 1809_2 --nframes 2000`: chunk files bit-identical to the code before; `locs_p1/p2.hdf5`
equal to the old `locs_p*.npz` in every column, `select()` = the old `locs_p*_sig12.npz` rows (31424 / 35149);
`simcode.run --config <record of the code before>` (settings with subset / sigma_max / format / sig_u / prefetch):
identical chunk files and tables; `simcode.table` CLI = the run's table; 973 network tensors identical
(tests/check_model_outputs.py); pytest; tests/check_imports.py; pyflakes (no new undefined names).

## 2026-11-15: out/ and models/ clean-up applied (after the merge into main)

`python analysis/maintenance/cleanup_outputs.py --apply --log out/cleanup_2026-11-15.tsv` in ~/simcode2 at 06:44, with the one
change of the coordinator / user: the 13 superseded weight files without a copy elsewhere were MOVED to
`models/archive_superseded/` (deleting weights without a copy is not approved). Not touched (PROTECT): out/psf_v2,
out/z3d (incl. eval), out/fourpi_sim, out/shareloc_u373, out/registration, out/papersim, out/psf_compare_1809_25,
out/solvegap, out/crlbcmp, out/solvefix. The model.json files of the live models directory have the new key names
(models/{m1809,m25,g13} by the merge; archive_2d/*, archive_p2dedicated/* by rename_model_keys). Untracked files of the
main checkout that the merge replaced by their tracked versions are kept in `out/merge_backup_2026-11-15/` (fourpi_sim,
solvegap scripts; model.json with the old keys: equal to the tracked files after the key conversion). Full log with
sizes: `out/cleanup_2026-11-15.tsv` (184 actions); sizes = apparent size (du).

- out, delete: 136 entries, 3.30 GB in total
  `ablate_frr_hd_b.json` 730; `ablate.log` 4,4K; `ablate_polish_hd_b.json` 1,7K; `ablate_prior_hd_b.json` 1,3K; `ablate_r11.json` 4,3K; `ablate_r41.json` 4,3K; `bench` 34 (-> /mnt/data/dmdsf/simcode2_out/bench); `bench_hd_a_quick.log` 6,5K; `bright` 35 (-> /mnt/data/dmdsf/simcode2_out/bright); `calib_sc_hd_sc1.json` 284; `calib_sc_hd_sc1.json.p1` 189; `calib_sc_hd_sc2b15.json` 322; `calib_sc_hd_sc2b15.json.p1` 230; `calib_sc_hd_sc2.json` 290; `calib_sc_hd_sc2.json.p1` 191; `calib_sc_hd_sc2Lb15.json` 322; `calib_sc_hd_sc2Lb15.json.p1` 230; `calib_sc_hd_sc2Ln1b15.json` 321; `calib_sc_hd_sc2Ln1b15.json.p1` 229; `calib_sc_hd_sc3b15.json` 321; `calib_sc_hd_sc3b15.json.p1` 230; `calib_sc_hd_sc5db15.json` 322; `calib_sc_hd_sc5db15.json.p1` 230; `calib_sc_hd_sc5nb15.json` 322; `calib_sc_hd_sc5nb15.json.p1` 231; `calib_sc_hd_sc5nb15_pm.json` 299; `calib_sc_hd_sc5nbb15.json` 323; `calib_sc_hd_sc5nbb15.json.p1` 230; `calib_sc_hd_sc5nbb15_pm_autoc1.json` 2,4K; `calib_sc_hd_sc5nbb15_pm_autoc.json` 2,4K; `calib_sc_hd_sc5nbb15_pm_bright.json` 2,0K; `calib_sc_hd_sc5nbb15_pm_bright_real.json` 2,8K; `calib_sc_hd_sc5nbb15_pm.json` 548; `calib_sc_hd_sc6nb15.json` 322; `calib_sc_hd_sc6nb15.json.p1` 230; `calib_sc_hd_sc6nb15_pm.json` 444; `calib_sc_hd_sc7bgb15.json` 320; `calib_sc_hd_sc7bgb15.json.p1` 229; `calib_sc_hd_sc7bgb15_pm.json` 441; `critic34` 37 (-> /mnt/data/dmdsf/simcode2_out/critic34); `critic35` 37 (-> /mnt/data/dmdsf/simcode2_out/critic35); `decomp` 35 (-> /mnt/data/dmdsf/simcode2_out/decomp); `decomp_sc2L` 40 (-> /mnt/data/dmdsf/simcode2_out/decomp_sc2L); `decomp_sc2Ln1` 42 (-> /mnt/data/dmdsf/simcode2_out/decomp_sc2Ln1); `decomp_sc3` 39 (-> /mnt/data/dmdsf/simcode2_out/decomp_sc3); `decomp_sc5d` 40 (-> /mnt/data/dmdsf/simcode2_out/decomp_sc5d); `decomp_sc5n` 40 (-> /mnt/data/dmdsf/simcode2_out/decomp_sc5n); `decomp_sc5n_pm` 43 (-> /mnt/data/dmdsf/simcode2_out/decomp_sc5n_pm); `det_up2` 36 (-> /mnt/data/dmdsf/simcode2_out/det_up2); `drift` 34 (-> /mnt/data/dmdsf/simcode2_out/drift); `drift25` 36 (-> /mnt/data/dmdsf/simcode2_out/drift25); `ext` 32 (-> /mnt/data/dmdsf/simcode2_out/ext); `field` 34 (-> /mnt/data/dmdsf/simcode2_out/field); `full_0.5nM_p2L_real` 48 (-> /mnt/data/dmdsf/simcode2_out/full_0.5nM_p2L_real); `full_0.5nM_retr_real` 49 (-> /mnt/data/dmdsf/simcode2_out/full_0.5nM_retr_real); `full_0.5nM_sc5nb_autoc` 51 (-> /mnt/data/dmdsf/simcode2_out/full_0.5nM_sc5nb_autoc); `full_0.5nM_sc5nb_autoc_v1s` 55 (-> /mnt/data/dmdsf/simcode2_out/full_0.5nM_sc5nb_autoc_v1s); `full_0.5nM_sc5nb_bright_real` 57 (-> /mnt/data/dmdsf/simcode2_out/full_0.5nM_sc5nb_bright_real); `full_20nM_m2_fast` 46 (-> /mnt/data/dmdsf/simcode2_out/full_20nM_m2_fast); `full_20nM_m2_fast_autoc` 52 (-> /mnt/data/dmdsf/simcode2_out/full_20nM_m2_fast_autoc); `full_2.5nM_m2_p2L_real` 51 (-> /mnt/data/dmdsf/simcode2_out/full_2.5nM_m2_p2L_real); `full_2.5nM_m2_retr_real` 52 (-> /mnt/data/dmdsf/simcode2_out/full_2.5nM_m2_retr_real); `full_2.5nM_m2_sc5nb_autoc` 54 (-> /mnt/data/dmdsf/simcode2_out/full_2.5nM_m2_sc5nb_autoc); `full_2.5nM_m2_sc5nb_bright_real` 60 (-> /mnt/data/dmdsf/simcode2_out/full_2.5nM_m2_sc5nb_bright_real); `gate_gpu1.log` 754; `gate_gpu1.pid` 8; `gate_gpu2.log` 1,9K; `gate_gpu2.pid` 8; `gate_report_sc1.json` 58K; `gate_report_sc.json` 125K; `gate_report_v1.json` 136K; `gate_report_v2.json` 215K; `logs_move_paper2.sh` 1,3K; `miss_anatomy.py` 2,3K; `move_paper2_b.log` 2,5K; `move_paper2_b.sh` 1,4K; `move_paper2.log` 564; `n20` 32 (-> /mnt/data/dmdsf/simcode2_out/n20); `nuis` 33 (-> /mnt/data/dmdsf/simcode2_out/nuis); `p2prior` 36 (-> /mnt/data/dmdsf/simcode2_out/p2prior); `p2real` 35 (-> /mnt/data/dmdsf/simcode2_out/p2real); `plots` 34 (-> /mnt/data/dmdsf/simcode2_out/plots); `plots_check` 40 (-> /mnt/data/dmdsf/simcode2_out/plots_check); `plots_check2` 41 (-> /mnt/data/dmdsf/simcode2_out/plots_check2); `post_hd_b.log` 2,5K; `post_hd_b.pid` 8; `real20` 35 (-> /mnt/data/dmdsf/simcode2_out/real20); `realgap` 36 (-> /mnt/data/dmdsf/simcode2_out/realgap); `realism` 36 (-> /mnt/data/dmdsf/simcode2_out/realism); `refs` 33 (-> /mnt/data/dmdsf/simcode2_out/refs); `resolvability` 42 (-> /mnt/data/dmdsf/simcode2_out/resolvability); `retrain_accept.txt` 4,2K; `retrain_r` 38 (-> /mnt/data/dmdsf/simcode2_out/retrain_r); `run_baselines.log` 16K; `run_baselines.pid` 8; `run_baselines.sh` 832; `run_chain_b.sh` 388; `run_gate.sh` 486; `run_gpu2_a.pid` 8; `run_gpu2_a.sh` 603; `run_post.sh` 775; `run_sc1.log` 2,9K; `run_sc1.pid` 8; `run_sc2.log` 1,8K; `run_sc.sh` 855; `sc3` 32 (-> /mnt/data/dmdsf/simcode2_out/sc3); `sc6` 32 (-> /mnt/data/dmdsf/simcode2_out/sc6); `sc7` 32 (-> /mnt/data/dmdsf/simcode2_out/sc7); `scratch` 36 (-> /mnt/data/dmdsf/simcode2_out/scratch); `simcode_base` 41 (-> /mnt/data/dmdsf/simcode2_out/simcode_base); `simcode_base2` 42 (-> /mnt/data/dmdsf/simcode2_out/simcode_base2); `simreal2` 37 (-> /mnt/data/dmdsf/simcode2_out/simreal2); `smoke_0.5nM_sc5n` 45 (-> /mnt/data/dmdsf/simcode2_out/smoke_0.5nM_sc5n); `smoke_0.5nM_sc5nb` 46 (-> /mnt/data/dmdsf/simcode2_out/smoke_0.5nM_sc5nb); `smoke_0.5nM_sc5nb_bright` 53 (-> /mnt/data/dmdsf/simcode2_out/smoke_0.5nM_sc5nb_bright); `smoke_0.5nM_sc5nb_bright_real` 58 (-> /mnt/data/dmdsf/simcode2_out/smoke_0.5nM_sc5nb_bright_real); `speed` 34 (-> /mnt/data/dmdsf/simcode2_out/speed); `static25` 37 (-> /mnt/data/dmdsf/simcode2_out/static25); `summary_baselines.json` 40K; `time_sc.py` 2,0K; `validate_0.5nM_final.json` 3,4K; `validate_0.5nM_final.txt` 1,1K; `validate_0.5nM.json` 3,1K; `validate_0.5nM_v2.json` 3,4K; `validate_0.5nM_v3.json` 3,5K; `validate_0.5nM_v4.json` 3,4K; `validate_2.5nM_final.json` 3,3K; `validate_2.5nM_final.txt` 1,1K; `validate_2.5nM.json` 3,0K; `validate_2.5nM_v2.json` 3,3K; `validate_2.5nM_v3.json` 3,4K; `validate_2.5nM_v4.json` 3,4K; `zimflux` 1,3G; `sc4` symlink; `p2` symlink; `rename` 2,0G
- out, move to out/archive/: 11 entries, 20.01 GB in total
  `nanoruler` 7,9G; `tput` 656M; `hd_compare` 39; `evalproto` 38; `psfshape` 37; `autoc` 34; `calib_inputs` 41; `hdunbias` 7,9G; `detband` 677M; `cleanpass` 678M; `optclean` 2,2G
- models, delete (identical copy elsewhere): 13 entries, 0.08 GB in total
  `models/det_m1809.pt` 7,0M; `models/det_m25.pt` 7,0M; `models/det_g13.pt` 7,0M; `models/hd_m1809.pt` 5,3M; `models/hd_m25.pt` 5,3M; `models/hd_g13.pt` 5,3M; `models/p2det_m1809.pt` 7,0M; `models/p2det_m25.pt` 7,0M; `models/p2det_g13.pt` 7,0M; `models/p2headL_m1809.pt` 5,3M; `models/p2headL_m25.pt` 5,3M; `models/p2headL_g13.pt` 5,3M; `models/p2det_g2ss.pt` 7,0M
- models, move to archive_superseded/: 13 entries, 0.08 GB in total
  `models/p2det_gnr80.pt` 7,0M; `models/p2headL_gnr80.pt` 5,3M; `models/det_psf3dmix.pt` 7,0M; `models/hd_psf3dmix.pt` 5,3M; `models/p2det_psf3dmix.pt` 7,0M; `models/p2headL_psf3dmix.pt` 5,3M; `models/p2det_psf3dmix_pair.pt` 7,0M; `models/p2headL_psf3dmix_pair.pt` 5,3M; `models/hd_abl1809_avg.pt` 5,3M; `models/hd_abl1809_ax.pt` 5,3M; `models/hd_sc7bg.pt` 9,2M; `models/p2_det.pt` 7,0M; `models/p2_head_long.pt` 5,3M
- models, move to archive_p2generic/: 11 entries, 0.07 GB in total
  `models/p2det_g1809.pt` 7,0M; `models/p2det_g2c.pt` 7,0M; `models/p2det_g2s.pt` 7,0M; `models/p2det_g2w.pt` 7,0M; `models/p2det_g2cc.pt` 7,0M; `models/p2headL_g1809.pt` 5,3M; `models/p2headL_g2c.pt` 5,3M; `models/p2headL_g2cc.pt` 5,3M; `models/p2headL_g2s.pt` 5,3M; `models/p2headL_g2ss.pt` 5,3M; `models/p2headL_g2w.pt` 5,3M

Symlinked entries (size = the link, e.g. `sc3` 32 -> /mnt/data/dmdsf/simcode2_out/sc3): only the link was removed or
moved; the data under /mnt/data/dmdsf/simcode2_out stay (the data disk is read-only for this work; `--targets 1` would
delete them and was not used). So 3.30 GB in the out/ delete line is what was removed from ~/simcode2/out.

## 2026-11-16: names from the development iterations (branch `names`)

The model / calibration files and the modules that still carried development names (det_n1, hd_sc5nb, p2L, sc_hd, sigu,
hd_/det_/p2det_/p2headL_<set>, front, autoc, psf4pi, lut4pi, cfgdict, ckpt, gate) got descriptive names; the full old -> new
table is in docs/LAYOUT.md "Renamed (2026-11-16)". No aliases: the only old names the code still knows are the ones stored
in data it reads (the architecture tags of earlier checkpoints, the key names of format-1 run records), now in one module,
`simcode/util/legacy_names.py`, used only by the checkpoint / run-record readers.

Choices:
- **Checkpoint arch tags: load-time map, not a rewrite.** Every shipped pass-1 emitter network (and the archives, and every
  pass-1 checkpoint under out/) stores the development tag of the network; rewriting the checkpoints would change their
  sha256, which model.json `_files`, models/README.md and every run record carry (verify would report them as changed). So
  `legacy_names.EMITTER_ARCH_TAGS` maps it at load time; the 4Pi tags of the checkpoints before 2026-11-15 and the format-1
  run-record keys moved into the same module.
- **Run records: rewritten, no map.** The records under out/ that name a renamed file of the models directory (absolute
  paths in settings, model.files, model.definition) are rewritten once at merge time (`out/names/rewrite_records.py` of the
  branch worktree; dry run on the out/ of ~/simcode2 (master), 2026-11-16: 8825 json files scanned, 323 to rewrite: 46 run_config.json (+ their
  yaml), 86 runtime.json, 67 summary.json, 96 locs_*_info.json, 11 model.json, 8 papersim_run.json, 5 wsm_eval.json, 4 args.json).
  Re-running a record uses the model name (models.json), which already resolves to the new names; the rewrite keeps
  `runconfig verify` working (same sha256 at the new path). Not rewritten: the trainer arguments stored in checkpoints and
  their copies in records, calibration notes, `copied_from`, and paths into out/ (out/ files keep their names).
- **Weights: renamed by a merge-time script.** The tracked json files moved with `git mv`; the untracked weights and
  archive folders are renamed by `out/names/rename_models.sh [--apply]` (os.rename after a sha256 check of each file;
  `models_sha256.txt` = sha256 of all 171 renamed files at their new paths). The models/calib copies of the 2D models'
  calibrations (byte-identical to archive_2d/<set>/calib/) were removed; their users (ablation_cov, crlbcmp, solvegap,
  p2generic/eval_gen.sh) point at archive_2d now.
- Not renamed: class names (Pass1Front, LUT4Pi, PSF4Pi, ChannelFront), `channels/front.py` (the multi-channel front end),
  run-output and gate-scene file names (`pass1_<f0>.npz`, `<scene>__s<seed>_hd_<tag>.npz`), directories under out/ and the
  analysis/results snapshots (records with the names of the runs they come from), the DECODE folders `decode/hd_frame`,
  `decode/hd_sum6`, experiment ids used as folder names (g1809, g2ss, gnr80, psf3dmix, ...), the analysis scripts whose
  outputs on disk carry their name (`ablation_cov/autoc_est.py`, `run_autoc.sh`; `calib/dump_fe.py` / `blib.py` dump labels),
  the provenance notes inside calibration json files, and the one-off / old-checkout scripts listed in
  docs/CLEANUP_CANDIDATES.md "Scripts that work on the file names before 2026-11-16".

Checks: see the section "Verification" of the branch report (tests/check_model_outputs.py reference from master 674a0c4 (= the former main f8cbdf4) vs the
branch with the renamed files; one end-to-end simcode.run; pytest).
