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
