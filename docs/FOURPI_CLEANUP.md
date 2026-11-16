# 4Pi clean-up after the refactor (branch fourpi, 2026-11-15): keep / archive / delete list

**Nothing below is deleted yet; the user approves.** "archive" = remove from the tree after the archive tag
`archive-fourpi-2026-11-15` (= main b2ca5b7, the last commit with the old `simcode/fourpi/` modules and every script
below), as with `archive-2026-11-15` (`docs/ARCHIVE.md`). The refactor itself (`docs/LAYOUT.md`, section "4Pi refactor")
updated the imports of every script listed here, so all of them still import.

## Package modules (done in the refactor, not pending)

The planned archive of the "research variants" did not apply as planned: four of them are on the adopted path.

| module | status | reason |
|---|---|---|
| `fourpi/solvek4pi.py` SolveK | merged into `solve/fourpi.py` FourPiSolve | it IS the adopted solve (run.py used SolveK); Solve4Pi's own T / C read-out code removed (= SolveK with windows ('flat', 'gauss' 1.0); `tests/test_solve4pi_oracle.py` runs it that way, same numbers) |
| `fourpi/windows4pi.py` | merged into `emitter/fourpi.py` | the read-out windows of v10 and of the solve |
| `fourpi/solveloss4pi.py` | merged into `emitter/fourpi.py` (solve_loss) | the training loss of v10 (`train_head4pi_w.py`) and of the hd experiment |
| `fourpi/linread4pi.py` | merged into `emitter/fourpi.py` (linear_readout, border_bg) | border_bg is used by simlin4pi(_hd) on every run (also the running hd experiment); linear_readout = the fixed-window teacher of v10 |
| `fourpi/head4pi.py` plain head | `emitter/fourpi.py` FourPiPrior | v10's frozen prior (v3L) and the candidate front of the simulated movies |
| `fourpi/lm.py` | `analysis/fourpi_classical/pixel_lm.py` | classical baselines only; torch_psf.fit.lm_poisson is not numerically identical (damping start / floor, double step, data clamp) |
| `fourpi/coherent_psf.py`, `fourpi/spline_table.py` | kept until SIMCODE pins torch-psf `multichannel` | torch_psf.FourPiPSF / FourPiSplinePSF reproduce them bit-identically (all 33 stored states) |

## analysis/fourpi (42 files)

| verdict | file | reason |
|---|---|---|
| delete | run_full_w4pi.sh | now only calls `python -m simcode.pipeline simcode/pipeline/configs/fourpi.py --only fourpi_<cell>` (README has that command) |
| archive | register.py | bead + in-situ registration of the first pass; replaced by `channels/registration.py` (in situ only, per file) |
| archive | reg_npc_shift.py | wrote `smlm/reg/reg_insitu_NPC.json` (bead map + in-situ shift); the file stays (the existing tables used it), the script is replaced by the registration stage |
| archive | run_reg_insitu.py | wrote `smlm/reg/reg_insitu_MT.json`; same as above |
| keep | camera.py | 4Pi offset maps `smlm/camera/camera_<cell>.npz` (camera.offset_map of the configs); no 4Pi camera stage in the pipeline yet (candidate to port: dark strip + column FPN) |
| keep | dc.py | raw I/O of the calibration chain (now on the acquisition configs) |
| keep | process.py, quickloc.py, mle.py, refit.py, run_psf.py, psffit.py | calibration chain: summed-channel SMLM pass, refit, summed PSF (inputs of diag_pattern / zcal) |
| keep | diag_pattern.py, pattern.py, render.py | writes `smlm/diag/pattern_<cell>.json` = fourpi.pattern_ref of the configs |
| keep | diag_4pi.py, fourpi.py | channel / 4Pi phase calibration (delta_deg of the configs) |
| keep | zcal.py, fit_psf4pi.py | bead z-stack ROIs and the 4Pi PSF fit (`psf4pi/NPC_f_dxy15.json`, the PSF of v10) |
| keep | train_det4pi.py, train_head4pi.py, train_head4pi_w.py | trainers of det4pi, v3L, v10 |
| keep | gatek4pi.py, gate_movies_w.sh, bench_classical.py | v10 gates and the pixel-MLE reference |
| keep | simmovie4pi.py, simfront4pi.py, simlin4pi.py, simeval4pi.py | simulated 4Pi movies, their front end, network stage, GT evaluation (showcase, sweep, hd) |
| keep | showcase4pi_movie.py, showcase4pi_movies.sh, showcase4pi_run.sh, showcase4pi_eval.py, showcase4pi_fig.py, showcase4pi_table.py | simulated showcase (paper / SI) |
| keep | post_w4pi.py, cmp_real4pi.py, cmp_full_w4pi.py, dafl4pi.py, select_curve_mt.py, select_curve_npc.py, showcase_real_mt.py | real-data post-processing and comparisons (README entry points). Note: they read the old run outputs `<P4>/net/locs_<cell>_<file>_<head>.npz`; tables of the pipeline are `<out>/process/locs_<file>.npz` with the same arrays: point them at the new directory when the movies are re-run |

Later candidates (after the 4Pi PSF estimation and camera stages are in the pipeline): the calibration chain rows
(process / quickloc / mle / refit / run_psf / psffit / diag_pattern / pattern / render / diag_4pi / fourpi / zcal /
fit_psf4pi / camera / dc) -> archive.

## analysis/fourpi_classical (11 files)

| verdict | file | reason |
|---|---|---|
| keep | all (cl4pi, mov4pi_cl, real_cl, full_cl, post_cl, fig_cl, npc_site, sim_cl, val_iso, pixel_lm, __init__) | the classical 4Pi-SIMFLUX baseline of the paper. real_cl / full_cl now build their front end with `FourPiSetup.front(c, map_file=<P4>/smlm/reg/reg_insitu_<cell>.json)` = the whole-series map the existing classical tables used |

## analysis/fourpi_sim (16 files)

| verdict | file | reason |
|---|---|---|
| keep | mt_cells.py, render_mt_cells.py, sweep4pi_movie.py, sweep4pi_run.sh, sweep4pi_eval.py, sweep4pi_fig.py, sweep4pi_table.py, sweep4pi_realdens.py | density sweep (SI) |
| keep (until the hd experiment is decided) | hd_run.sh, hd_gate.py, hd_table.py, sim4pi_hd.py, simlin4pi_hd.py, sweep4pi_eval_hd.py, train_head4pi_hd.py | the running hd-head experiment; simlin4pi_hd / sweep4pi_eval_hd / train_head4pi_hd are COPIES of the analysis/fourpi scripts: merge (flags) or archive once decided |
| archive | compare_websmlm.py | one-off check of mt_cells.py against webSMLM (needs a local webSMLM checkout + Playwright); result recorded |

Totals: analysis/fourpi 42 (keep 38, archive 3, delete 1); fourpi_classical 11 (keep 11); fourpi_sim 16 (keep 15, archive 1).
