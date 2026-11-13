# Cleanup plan: one file handling for SIMCODE and 4Pi, redundancies, dead code

Read-only audit of the tree at b757ca8 (2026-11-12). Nothing was changed, nothing was run on a GPU. To be executed at the
quiet moment of the two QUEUED items in `out/HANDOFF_SESSION.md` (package rename `simcode2` -> `simcode`, source-only repository).

How the evidence was gathered, and its limits:
- Read in full: `pipeline/common.py`, `pipeline/run_uncalibrated.py`, `run.py`, `config.py`, `util/proctitle.py`, `util/drift.py`,
  `fourpi/loadhead.py`, `fourpi/ckpt.py`, `analysis/fourpi/{dc.py, real_w4pi.py, front4pi_full.py, run_full*.sh, showcase4pi_run.sh}`,
  `analysis/baselines/dme_drift.py`, the drivers of `analysis/fourpi_classical/{full_cl.py, real_cl.py}`.
- Everything else: docstrings, function lists, import graphs (`grep`) and line overlap (`diff`). "n of m lines identical"
  below is a `diff` count of the second file against the first; it measures copy-paste, not semantic equality.
- "Unused" means: no importer and no mention in any `.py`, `.sh`, `.md` of `simcode2/`, `analysis/`, `runs/`, `tests/`, `docs/`.
  Logs under `out/` were not searched; a script started by hand leaves no trace in this test. Treat every "delete" as "archive
  first" unless marked certain.
- Not checked: whether any checkpoint pickles a `simcode2.*` class (the trainers I read save `state_dict` + `args` dicts, so
  probably none; loading every file in `models/` and `out/4pi/models/` is step R2 below).

Size: package 13.8k lines / 105 files; `analysis/` about 25k lines / 380 files, of which `fourpi` + `fourpi_classical` 10.8k.

---

## 1. Movie, file and chunk handling

### 1.1 What exists now

| aspect | main pipeline | 4Pi learned-window (`real_w4pi.py` + `run_full_w4pi.sh`) | 4Pi classical (`full_cl.py`, `real_cl.py`) | older 4Pi (`front4pi*.py`, `process.py`) |
|---|---|---|---|---|
| reader | `pipeline/common.Movie` (multi-part TIFF, absolute frame, `read(f0,f1,y0,x0,S)`) | `dc.Dcimg.raw(f0,n)` + `dc.crops` | same | same |
| read-ahead | `PrefetchMovie` + `chunk_read_plan` (one thread, planned reads) | `front4pi.reader` thread + `queue.Queue(2)` | `full_cl.reader` (copy, `Queue(3)`); `real_cl.DcimgSource` (no thread) | `front4pi.reader`; `process._reader` (third copy) |
| unit of work | chunk of 6000 frames of one movie | one file (3600-frame dcimg), 600-frame read chunks, all patches in RAM | one file; reader stage writes patches to SSD, fitter stage claims them | one file |
| resume | `os.path.exists(pass1_<f0>.npz)` per stage and chunk, `--redo` | shell: `[ -f locs_...npz ] && continue` | `done()` = table exists; `--skip_done` | shell loop builds a TODO list |
| claiming | none (one process per movie) | shell `set -o noclobber; echo $$ > <out>.claim` | `os.rename(.ready -> .claimed)`, `.failed`; `real_cl.ready_files` (size + age poll) | none |
| write | `np.savez(fn)` directly | `np.savez(tmp)` + `os.replace` | (not checked line by line) | `np.savez(fn)` directly |
| disk guard | none | twice: shell `df /` < 12 GB, python `disk_usage(OUT)` < 10 GB | `--tmp_max_gb` on the patch directory | none; `fourpi/ckpt.py` guards checkpoints only |
| output names | `<out>/pass1_<f0>.npz`, `pass2_<tag>_<f0>.npz`, `locs_p1/p2.npz`, `runtime.json`, `summary.json` | flat `out/4pi/net/locs_<cell>_<fi:03d>_<head><tag>.npz`, one shared `locs_<cell>_<head><tag>_summary.json` | `out/4pi/classical/real/cl_<cell>_<fi:03d><tag>.npz` | `front_<cell>_<fi:03d>_<model>.npz` |
| paths | `config.resolve` (models / data / out roots) | `REPO = '/home/jc/simcode2'`, `dc.ROOT = '/mnt/data/4pi-sf'` | same | same |
| process title | `proctitle.auto('pipeline')`, `auto('run')` | `auto('4pi-real-w')` | `auto('4pi-classic')` | `auto('4pi-front1')` ... |
| GPU choice | `CUDA_VISIBLE_DEVICES` by the caller | `GPU=` env in the shell script | caller | hard-coded `CUDA_VISIBLE_DEVICES=2` in `run_full4pi.sh` |

Defects found while comparing (each is fixed by the shared design, none affects finished paper outputs):
1. `run_uncalibrated.py:134, 207, 238-239`: chunk outputs are written in place. A run killed during `np.savez` leaves a
   truncated `pass1_<f0>.npz` that the resume logic treats as done; the next stage then fails on `np.load` and the file has to be
   deleted by hand. `pass2_<tag>_<f0>.npz` and its `b` variant are two separate writes with one existence test (on the first).
2. `real_w4pi.py:162-164`: the per-movie summary json is read, updated and rewritten by every worker. With two workers
   (the documented use of `run_full_w4pi.sh`) one file's entry can be lost. `runtime.json` of the main pipeline has the same
   pattern and would have the same problem as soon as two processes share an output directory.
3. `run_full_w4pi.sh`: a crashed worker leaves `<out>.claim` and the file is skipped forever ("delete it", per its header).
   Two claims are in `out/4pi/net/` right now (NPC 005, 037); they may belong to live workers, I did not touch them.
4. `real_w4pi.py:154`: below 10 GB free the result of a finished file is dropped with a warning and the loop continues.
5. `strict.py:47, 124`: `CROP = (5, 5, 110)` is a module constant, while `run.py --crop` is a flag. A run with another crop
   gets a strict subset computed on the default crop. Paper runs use the default, so they are unaffected.
6. `run_uncalibrated.py:144-157` and `173-181`: the same 12 lines (load all pass-1 chunks, overlap rule, `prior_off`, `out`)
   appear twice; `autoc.run_dir` and `strict.load_rows` each have their own chunk loader as well.

### 1.2 Proposed design (new sub-package `simcode/io/`, about 350 lines)

`io/movie.py`: one reader interface, three back ends.
```
class MovieReader:            # protocol
    n: int                    # frames
    shape: (H, W)             # of one frame as returned by read()
    parts: list[str]          # source files
    def read(f0, f1, y0=0, x0=0, S=None) -> float32 [f1-f0, ...]   # ADU, absolute frame index
TiffMovie(path)               # = today's Movie, unchanged logic (movie.tif + movie_X1.tif ...)
DcimgMovie(path, layout=None) # = dc.Dcimg.raw; layout (4 crop windows, mirror flags) applies dc.crops -> [n, 4, 200, 360]
ArrayMovie(array | .npy path) # tests (tests/test_fast_paths.ArrayMovie) and simulated movies (showcase4pi_movie.py movie.npy)
open_movie(path, **kw)        # by extension
Prefetch(reader, plan)        # = today's PrefetchMovie, back-end independent; replaces the three reader threads
```
The dcimg channel layout (`X0`, `CW`, `CH`, `MIRROR`, `FRAME_DT_S`), the data root and the per-cell file globs of `dc.py`
become a small dataset json (`data/4pi_<cell>.json` or keys in the config file), not module constants.

`io/work.py`: one resumable driver for "units" (a chunk of a movie, or a file of a multi-file acquisition).
```
atomic_savez(path, **arrays)          # tmp + os.replace (generalises fourpi/ckpt.save and real_w4pi's tmp write)
disk_guard(path, min_free_gb)         # raises before the unit starts, never after the work is done
class Units(out_dir, stage, keys, pattern='{stage}_{key}.npz'):
    done(key)                         # output exists (complete by construction, because writes are atomic)
    claim(key) / release(key)         # O_CREAT|O_EXCL marker '<output>.claim' holding host, pid, start time;
                                      # a claim whose pid is dead on this host is taken over
    todo(order='fwd'|'rev', claim=False)   # iterator; claim=True for several workers on one output directory
    record(key, info)                 # one small json per unit ('<output>.json'); summary / runtime tables are built by
                                      # merging these, so no file is rewritten by two processes
```
Multi-GPU is then the same command started once per GPU with `--claim 1` (what `run_full_w4pi.sh` does), for both pipelines;
a 6-line launcher `python -m simcode.io.launch --gpus 1,2 -- <command>` replaces the per-pipeline shell loops. In the main
pipeline pass 1 and pass 2 chunks are claimable units; the pattern, autoc and table stages are barriers that start only when no
unit of the previous stage is missing or claimed. `Prefetch` already falls back to a synchronous read on a plan miss, so
claiming does not change the data, only the hit rate.

Output naming, one scheme: `<out root>/<dataset>/<method tag>/<stage>_<unit>.npz`, unit = `f<f0>` (chunk) or `<file:03d>`.
For 4Pi this means `out/4pi/<cell>/<head><tag>/locs_<fi>.npz` instead of the flat `out/4pi/net/` directory (which today mixes
front, locs, cmp, merged tables, drift and renders of 5 head versions). Existing outputs are not renamed; a resolver
`io/names.py: find(dataset, method, stage, unit)` knows the old patterns, so finished results stay readable. The main
pipeline's names (`pass1_<f0>.npz` ...) already fit and stay as they are.

Process titles: `Units.todo()` calls `proctitle.set_title(stage, key)`, so `nvidia-smi` shows the stage and the unit for
both pipelines; the 50 scattered `auto('...')` calls in entry scripts stay.

### 1.3 Moving each pipeline onto it

| step | change | verification |
|---|---|---|
| main, readers | `Movie` -> `io.movie.TiffMovie` (alias kept in `pipeline/common.py`), `PrefetchMovie` -> `io.movie.Prefetch` | `tests/test_fast_paths.py::test_prefetch_reader`; `np.array_equal` of `read()` on 3 chunks of 1809_2 old vs new |
| main, driver | the four chunk loops of `run_uncalibrated.py` use `Units` + `atomic_savez`; one `load_pass1_rows()` replaces the duplicated block and the loaders in `autoc.py` / `strict.py` | re-run 2 chunks of 1809_2 into a new directory, every array of `pass1_*.npz`, `pass2_est2_*.npz`, `locs_p2.npz` bit-identical to `out/paper2/1809_2/v2` (same GPU model; else the bf16 tolerances of the test) |
| 4Pi, reader | `dc.Dcimg` -> `io.movie.DcimgMovie`; `front4pi.reader`, `full_cl.reader`, `process._reader` -> `Prefetch` | `np.array_equal(dc.crops(Dcimg(f).raw(0, 600)), DcimgMovie(f, layout).read(0, 600))` on one NPC and one MT file |
| 4Pi, driver | body of `real_w4pi.main`'s file loop becomes `process_file(reader, ctx) -> dict`; the loop, claim, guard and write come from `Units`; `run_full_w4pi.sh` becomes the launcher call | one NPC file re-run, `locs_NPC_<fi>_v10.npz` bit-identical (the phase-field subsample uses `default_rng(0)`, so it is deterministic); kill a worker, restart, the stale claim is taken over |
| 4Pi classical | `full_cl.py` read/fit stages and `real_cl.follow` use `Units` (`.ready/.claimed/.failed` become claim + record) | one file, table bit-identical |

Risks. (a) All movies are on one HDD (`dc.py:38`: two interleaved streams drop to about 30 MB/s); several workers on one
TIFF movie may be slower than one. Measure before recommending multi-GPU for the main pipeline. (b) 4Pi per-file state
(pattern and phase field are estimated per file) must stay per unit; a 4Pi "chunk" is not an independent unit. (c) Do not
change chunk boundaries or the `l0/l1` padding rule of `chunk_read_plan`: outputs depend on them.

---

## 2. Duplicated functionality

Priority: A = worth doing at the rename, B = worth doing, C = note only.

| # | P | what is duplicated (evidence) | single home | risk | verify |
|---|---|---|---|---|---|
| 2.1 | A | **FRC, 8 implementations.** `evalproto/lib.frc` (13-27) is the reference; `analysis/paper/frc_fair.frc_curve` (76-92) is the same 12 lines returning the curve (its docstring says so); `analysis/figures/smoke_metrics.frc` + `frc_split` (78-110) identical to `evalproto/lib` up to one docstring line; `fourpi_classical/fig_cl.frc` (22-34) same in nm units; `fourpi/showcase4pi_eval.frc` (79-) different (Hann window, rectangular image); `hd_compare/score.frc_res`; `figures/lib20.frc`; `papersim/metrics.frc_set` (wraps). | `simcode/eval/frc.py`: `frc(xa, ya, xb, yb, px, units, window=None, return_curve=False)`; `frc_fair`'s protocol (halves, own drift) moves next to it | low; the Hann variant gives different numbers, keep it as an option and do not switch any figure silently | old vs new on the stored `locs_p2.npz` halves of 1809_2: identical float |
| 2.2 | A | **Training loops, 13 copies** of the same skeleton (argparse, AdamW + `OneCycleLR(total_steps=steps+1, pct_start)`, eval-hold, `ckpt_<step>.pt` / `final.pt`): `hd/sc.py:train`, `hd/sc4.py:train`, `hd/sc7bg.py:train`, `p2/head2.py:train`, `det/up2.py:train`, `p2/det2.py:train`, `det/train.py` (module-level script), `analysis/fourpi/train_head4pi{,_s,_v4,_p1,_k,_w}.py`, `train_det4pi.py`. `train_head4pi_w.py` shares 69 of 85 lines with `_k`; `_s` 44 of 78 with the base. Main trainers save with plain `torch.save`; only 4Pi uses `fourpi/ckpt.save` (atomic, disk guard), written after a checkpoint was destroyed on 2026-11-12. | `simcode/train/loop.py`: `fit(net, batch_fn, loss_fn, args, eval_fn)` + `add_common_args(ap)`; `ckpt.save` moves to `simcode/util/ckpt.py` and is used by all. Entry points become `simcode/train/{det,head,p2det,p2head}.py`, 4Pi ones `simcode/fourpi/train.py --arch w|k|s|p1` | medium: the RNG call order decides the trained weights. No retrain is needed for the paper; verify by step, not by result | fixed seed, 20 steps, loss sequence identical old vs new for det, head, p2 head, 4Pi w |
| 2.3 | A | **DME as a subprocess of a script.** `analysis/baselines/dme_drift.py` is the only place with the row selection (good & disjoint, sigma < 1 px, time = frame + 3); it is run through `subprocess` from `fourpi/post4pi.py:85`, `fourpi/post_w4pi.py:46`, `fourpi_classical/post_cl.py:41`, `fourpi/render.py:35`, and re-described in `paper/frc_fair.py:68`. | `util/drift.py: drift_of_table(T, nframes) -> (drift, info)`; `dme_drift.py` keeps its CLI as a 5-line wrapper | low | `np.array_equal` of the drift of `out/paper2/1809_2/v2/locs_p2.npz` (DME is deterministic on one GPU; if not, max difference < 0.01 nm) |
| 2.4 | A | **z drift**: `fourpi_classical/fig_cl.zdrift_struct` (62-80) and its declared copy `npc_site.zdrift_struct` (12-, "local copy with writable arrays"); `post_w4pi.py:54` imports it from the figure script. | `simcode/fourpi/zdrift.py` | low | same input, identical output |
| 2.5 | A | **Network sets defined in three places**: `config.py` constants (`DET = 'det_n1.pt'` ...), `analysis/papersim/netsets.json` (default / psf3dmix / m1809 / ...), `runs/psfmatch/env.sh` + `prod.sh` (`NETS="--det det_$S.pt ..."`, `declare -A PSF`, `MOV`, `CAL`). The paper production uses the PSF-matched sets, i.e. not the defaults of `config.py`. | `models/netsets.json` (moved), `config.netset(name)`, `run.py --netset m1809`; movie / calib table of the paper datasets in one `analysis/paper/datasets.json` | low | `--netset m1809` resolves to the same 9 paths as `prod.sh` prints in `command.txt` |
| 2.6 | A | **Selection rule outside the package.** The paper rule is "u <= 0.45" = `isfinite(strict_u) & (strict_u <= 0.45)`; it exists as an inline python heredoc in `runs/psfmatch/u045.sh:11-15`, as `kind == 'v2q45'` in `papersim/metrics.py:87`, in `papersim/pvalid_real.py:30` and `selection.py:196`; `strict.py`'s own `strict` column uses the density rule (`Q_DENSITY`), which the paper no longer uses as the main row. Result: three subset files per run (`locs_p2_strictonly.npz`, `locs_p2_u045.npz`, `locs_p2_q045.npz` in `out/paper2/1809_2/v2`) written by ad-hoc code. | `strict.py: select(T, rule='u<=0.45'|'strict'|'sel'|'good')` and `python -m simcode.strict <dir> --subset u045`; `phase_thr` wrapper in `figures/lib20.py:33` goes away | low; naming decision needed (section 4) | subset written by the new CLI equals `locs_p2_u045.npz` row for row |
| 2.7 | B | **LM / Poisson MLE, 6 implementations**: `fourpi/lm.py` (`lm_poisson`, `lm_poisson_j`, `fisher`, `crlb`; generic model callable); `analysis/fourpi/mle.py` (`lm_fit`, `loglik`, own damping; used by `refit.py`, `zcal.py`, `register.py`, `process.py`); `analysis/baselines/paper_pipeline.lm_mle` (81-); `analysis/figures/lmref.py` (`LMRef.fit`, `SFRef._lm`); `insitu/lm_refine.py` (only user: `coma_anchor_scan.py`); the intensity-domain LM in `hd/msolve.py` / `msolve_fast.py` (different objective, keep). `fourpi/lm.py:4` documents an `lm_gauss` that does not exist. | `simcode/fit/lm.py` = today's `fourpi/lm.py`; `mle.lm_fit` and `paper_pipeline.lm_mle` become calls of it. Leave `lmref.py` and `lm_refine.py` alone (analytic Jacobians, reference lines of figures) | medium for the baselines: damping and stopping rules differ, so SMLM / SIMFLUX baseline numbers would move at the 1e-3 px level. Do it only if the baselines are re-run anyway; otherwise C | baseline table of one 6000-frame chunk: median abs dx < 1e-3 px, same row count within 0.1 % |
| 2.8 | B | **CRLB, 6 places**: `sim/crlb.py:crlb_all` (multi-emitter, gate), `eff/crlb_nuis.py` (extends it), `fourpi/lm.fisher/crlb`, `analysis/papersim/s10.crlb` (80), `fourpi/info4pi.fisher/crlb` (42-50), `psf3d/perangle.crlb`, `psfmatch/psf_compare.crlb`. The four small ones are each `inv(J^T J / (mu + v))` in 3-5 lines. | `simcode/fit/lm.py: fisher_from_jac(J, var)`, `crlb(F)`; `sim/crlb.py` keeps the scene-specific part | low | `s10.py` CRLB column identical |
| 2.9 | B | **4Pi real vs simulated drivers are parallel copies**: `front4pi.py` / `front4pi_full.py` / `simfront4pi.py` (detector loop, patch cut, nearest-neighbour distance; the `cdist` block is in `front4pi_full.py:86-90` and `real_w4pi.py:80-84`), `solve_real4pi.py` / `simsolve4pi.py` (`repattern` vs `repattern_xy`), `real_w4pi.py` / `simlin4pi.py`. Line overlap is only 15-20 % because each was rewritten, not copied. | with section 1: `simcode/fourpi/pipeline.py: process_file(reader, ctx)`; real and simulated differ only in the reader (`DcimgMovie` vs `ArrayMovie`) and in where pattern / phase field come from (estimated vs oracle) | medium; the showcase figures depend on `simlin4pi.py` | one showcase tag (`sc_d03`) re-run, `eval.json` identical |
| 2.10 | B | **4Pi pattern and phase field**: `fourpi/pattern.refine_k` vs `fourpi/quickloc.refine_k`; `solve_real4pi.phase_field` (84-) vs `fourpi_classical/mov4pi_cl.phase_field_poly` (101-) and `field_fit`; `mov4pi_cl.estimate_pattern` vs `solve_real4pi.repattern`. The main pipeline's `pipeline/pattern_est.py` solves the same problem for the 2D case (k from an FFT peak, step phases, drift) but shares no code. I only compared signatures here, not the maths. | `simcode/fourpi/pattern.py` (`refine_k`, `repattern`, `phase_field(mode='tile'|'poly')`); a shared k-refinement with `pattern_est.estimate_k` is possible but not needed | medium | per-file pattern json (`psi_deg`, `m_median`, `R`) identical for one NPC file |
| 2.11 | B | **4Pi constants repeated**: `GAIN = 0.23` in 10 files, `P = 15` in 5, `XLIM = (55, 305)` in 2, `DELTA = {...}` in 3 (plus `DELTA_A` in `solve4pi.py` and two scalar `DELTA`), `PX = 128.0` in 7; `REPO = '/home/jc/simcode2'` in 51; camera offset map path spelled out in `front4pi.py:45`, `process.py:65`. Main side: `PX = 108.33` in 23 files, `PMIN / P_GOOD = 0.63` in 3. | `simcode/fourpi/setup.py` (or the dataset json of 1.2) and `pipeline/common.PX_NM` / `strict.P_GOOD` | low | grep: one definition each |
| 2.12 | B | **Table loaders**: 21 top-level `load()` functions; the ones that read a localization table with a method-specific column mapping are `evalproto/datasets.load_set`, `figures/lib20.load`, `paper/frc_fair.load`, `fourpi_classical/fig_cl.load`, `fourpi/showcase4pi_fig.load`, `hd_compare/score.load`, plus 25 scripts that glob `pass2_<tag>_*.npz` themselves. | `simcode/eval/tables.py: load_table(path, method)` -> common columns (`x, y, [z], sx, sy, frame, photons, mask`) | low, but touches many figure scripts; do it script by script when a figure is next regenerated | figure numbers unchanged |
| 2.13 | B | **Rendering**: 10 `render()` functions; `figures/recon_full.py`, `recon_full_dc.py`, `recon_full_dme_cross.py` each carry `render(x, y, lim, z, blur)`; `fig_cl.render/render_xz` and `simeval4pi.render_xy/render_xz` are the same idea for 3D; `fourpi/render.py`, `hd_compare/render.py`, `papersim/fig_density.render`, `papersim/common.Renderer` (that one renders camera frames, different). | `simcode/eval/render.py: hist_render(x, y, lim, px_nm, blur_nm, z=None)`; `zoompick.py` already centralises the zoom boxes | low | pixel-identical png for one zoom |
| 2.14 | C | **PSF and LUT**: this part is already layered correctly (`psf4pi.py` builds on `VectorialPSF.pupil_fields`, `lut4pi.py` on `PSFLUT`). Remaining overlap: "a `PSFLUT` built from a given knot stack" exists twice, `fourpi/lut4pi._lut_from_stack` (18-36) and `analysis/baselines/zpsf.ZStackLUT` (50-71); and `zpsf.py` (the PSF input of all baselines, also imported by `analysis/fourpi/refit.py`) lives in `analysis/baselines`. Further PSF wrappers: `fourpi/mle.GaussAstig` / `PSFModel`, `fourpi/refit.LUTPSF`, `fourpi/psffit.SumPSF`, `baselines/decode/dec_common.LutPSF`. | `PSFLUT.from_stack(...)` classmethod in `psf/lut.py`; move `zpsf.py` to `simcode/psf/zstack.py` | low | LUT values `allclose(atol=0)` on a grid |
| 2.15 | C | **Explicit solves**: `hd/msolve.py` (reference) vs `msolve_fast.py` (production) is a deliberate pair guarded by `tests/test_fast_paths.py::test_pattern_solve`; `run.py --autoc_fast 0` (default) still uses the reference solve for the autoc estimate, so both are live. `fourpi/solvek4pi.SolveK(Solve4Pi)` overrides `build_table`, `pattern`, `model`, `solve` (about half of its 227 lines parallel the parent's). The 2D and 4Pi solves share only `kinetics` ideas, not code. | keep both `msolve*`; in 4Pi make `Solve4Pi` the base with the read-out as the only overridden method once v2-v8 heads are archived (section 3) | medium | 4Pi: one file bit-identical |
| 2.16 | C | **Heads and detectors**: no copies. `SCHead4` -> `SCHead7`, `SCHeadP2`; `DetNet` -> `DetUp` -> `DetP2`; `Head4Pi` -> `Head4PiP1`; `Head4PiK`, `Head4PiS`, `Head4PiW` are separate modules (k / w share 15 of 92 lines). Small items: `norm_photons` defined 3 times (`det/model.py:28`, `hd/sc.py:39`, `fourpi/head4pi.py`); `focal` (`det/train.py:62`) vs `focal_w` (`det/up2.py:183`); `make_hold` / `eval_hold` in three trainers; `hd/sc.py`'s docstring calls `SCHead` the base class of `SCHead4`, which it is not. Loaders: `hd/sc.load`, `p2/head2.load`, `det/model.load`, `det/up2.load`, `p2/det2.load`, `fourpi/loadhead.load_head`, `p2/dump.load_head`, `analysis/fourpi/gate4pi.load_head`. | `norm_photons` -> `nn/blocks.py`; one `simcode/nn/load.py: load_net(path)` dispatching on `args['arch']` | low | `tests/test_fast_paths.py` |
| 2.17 | C | **Training-batch generators**: `hd/data.Gen` -> `hd/realgen.RGen`, `p2/data.Gen2` (inheritance, fine). `det/data.py` is used only by `det/train.py` and by `det/up2.targets` (`D.assign`). 4Pi: `sim4pi.Sim4Pi` (21 importers) and `realsim4pi.RealSim4Pi` (3 importers: `train_head4pi_p1.py`, `train_det4pi.py`, `gate4pi.py`) are independent of the 2D generators. | no merge proposed | - | - |
| 2.18 | C | **Camera calibration**: 2D reads `offset_adu` / `readnoise_e` maps through `pipeline/common.load_calib` (12 users) but 20 more files open the calib json or the maps npz directly; 4Pi has its own `analysis/fourpi/camera.py` (maps `off`) and a second key name (`offset_adu`) in `real_cl.PatchSource`. | `simcode/io/camera.py: load_camera(calib) -> offset, rn2, gain`, used by both | low | arrays identical |
| 2.19 | C | **Near-copies among scripts**: `simreal/sim_movie20.py` = `sim_movie.py` (124 of 134 lines); `figures/sf_gate.py` vs `lm_gate.py` (61 of 96); `figures/lm_icrlb.py` vs `icrlb.py` (49 of 92); `baselines/run_smlm_tuned.py` vs `run_classic.py` (35 of 54, both define `class _Det(_orig)`); `fourpi/gate_v5.sh` vs `gate_v4.sh` (19 of 21); `fourpi/run_full4pi_v3.sh` vs `run_full4pi.sh`; `hd/gate_report_xdet.py` vs `gate_report.py` (9 importers use `_xdet`, `gate.py` and README mention both). | parameterise (`sim_movie.py --recipe`, `lm_gate.py --model sf`), fold `gate_report_xdet` into `gate_report --match det` | low | gate report json identical |

---

## 3. Dead or superseded code and artefacts

"Used" was tested as described at the top. Recommendation: archive = `git mv` to `attic/` in one commit (history kept,
not importable, excluded from the package), delete = remove after that commit is tagged `pre-cleanup`.

| item | evidence | proposal |
|---|---|---|
| `hd/sc7bg.py`, `config.BGNET_HEAD`, `models/hd_sc7bg.pt`, `calib_sc_hd_sc7bgb15_pm.json` | used only by `figures/crlb_all_v2.py` ("[+ sc7bg if present]"), `figures/lm_robust.py`, `figures/run_crlb_refs.sh`; the fast path refuses it (`sc_front_fast.py:136`: `NotImplementedError('sc7bg head: use the original sc_front path')`); hooks in `hd/sc.py:149-152, 346-348` | archive, unless the CRLB figure keeps its sc7bg line (ask) |
| `det/train.py` | no importer; module-level script; docstring: "earlier pattern-free detector ... kept for ablations"; `up2.py` only cites it in a comment | archive; `det/psfpert.PertBank` then has `realsim.py` as its only user (keep) |
| `hd/sc.py: SCHead` class and `train()` | docstring: "kept for reference; the production head is trained with sc4.py"; `load()` falls back to it for checkpoints without `arch`. No file in `models/` was checked for that | archive `train()`; keep the class until step R2 shows no checkpoint needs it |
| `insitu/fit_insitu.py` | superseded by `fit_insitu2.py` (FILE_DESCRIPTION lists both as stage A); it is mentioned only in comments of `psf/fit_beads.py:2` and `fit_insitu2.py:63` (no import); those comments say it produced the shipped bead-fit init and the fixed camera values | archive, keep a note of that provenance in `insitu/README.md` |
| `insitu/lm_refine.py`, `insitu/coma_anchor_scan.py` | diagnostic pair, self-declared "not part of the method"; `util/neighbours.py` is used only by these and `fit_psf_sums.py` | move to `analysis/insitu_diag/` (not package code) |
| `pipeline/pattern_field.py`, `--pattern_field` | "Not used by default"; README of `run.py`: "not recommended by default"; adds the tags `estfld` / `estdep` and about 25 lines to `run_uncalibrated.py` | keep (opt-in feature with users in `simreal/sim_movie_r.py`, `evalproto/spatial.py`), but move its branch out of the driver |
| `pipeline/compare_pattern.py` | only user `analysis/papersim/common.py` | keep (evaluation), move to `simcode/eval/` |
| `pipeline/sc_front.py`, `hd/msolve.py` (reference paths) | used by tests, by `--fast 0`, by the default autoc estimate | keep; they are the reference of `tests/test_fast_paths.py` |
| network sets `psf3dmix`, `psf3dmix_pair`, `hd_abl1809_avg/ax` (10 `.pt`, 6 calib json, `sigu_*_new/cur`) | HANDOFF lists them as superseded ("production: m1809, m25, g13 when done; superseded: psf3dmix, pair, abl, sc7bg, older"); referenced from `netsets.json` ("psf3dmix": "2.5nM_2 / 20nM_2 production nets" - this label and the HANDOFF disagree, check which set produced the 2.5 nM / 20 nM paper tables), `psf3d_nets/accept.py`, `psfmatch/accept3.py`, `runs/psf3d/*` | weights leave git anyway; keep the calib json of any set that produced a table quoted in the paper, drop the rest from `netsets.json` |
| `models/det_n1.pt`, `hd_sc5nb.pt`, `p2_det.pt`, `p2_head_long.pt` (the `config.py` defaults) | still the defaults of `run.py` and of `tests/test_fast_paths.py`, but not the paper's production sets | keep; decide whether the default becomes `m1809` (changes the test's reference data) |
| 4Pi heads: `fourpi/head4pi_s.py` (v8), `head4pi_p1.py` (v4-v6), `head4pi_k.py` (v9) and `analysis/fourpi/train_head4pi_{s,v4,p1,k}.py`, `gate4pi.py`, `gatek4pi.py`, `gate_v4.sh`, `gate_v5.sh`, `track_gate.sh`, `track_gatek.sh`, `verify_v3.sh`, `review4pi*.py`, `decomp4pi.py`, `prior_slope4pi.py`, `front4pi_full.py`, `train_det4pi.py`* | the current path is v10 = `Head4PiW` (`real_w4pi.py`, `simlin4pi.py`, `showcase4pi_run.sh`, `train_head4pi_w.py`). `Head4PiS`: only `loadhead.py` + its trainer. `Head4PiP1`: `front4pi_full.py` (no caller), `gate4pi.py`, two trainers, one import in `solve4pi.py`. `Head4PiK`: `head4pi_w.py:23` imports only the constant `WINS_K` from it (move that to `windows4pi.py`); `solve4pi.py:177` imports `full_cov` from `head4pi_p1.py` for v4-v6 front files only. `out/4pi/models/` has `head4pi_v5_discarded.pt`, `det4pi_aborted_gpu0`. *`train_det4pi.py` made the detector that v10 uses: keep it | archive the v4-v9 heads, trainers and gate scripts together with a note of which commit trained which tag; keep `head4pi.py` (v2/v3): `showcase4pi_run.sh` still runs `simfront4pi.py --model v3L` as the candidate / patch front |
| 4Pi first-generation drivers: `front4pi.py main`, `solve_real4pi.py main`, `post4pi.py`, `run_full4pi.sh`, `run_full4pi_v3.sh`, `solve_follow.sh` | produced `front_*_v2/v3.npz`, `locs_*_v2.npz` (138 + 138 files in `out/4pi/net`). Their functions are still imported (`Front`, `reader`, `pattern_psi`, `phase_field`, `DELTA`) by `real_w4pi.py` and the classical code | move the imported functions into `simcode/fourpi/`, archive the `main()`s and the two shell drivers |
| 4Pi classical SMLM first pass: `process.py`, `quickloc.py`, `mle.py`, `refit.py`, `register.py`, `reg_xcorr.py`, `reg_npc_shift.py`, `run_reg_insitu.py`, `zcal.py`, `camera.py`, `frameorder.py`, `timeblocks.py`, `fo_analyze.py`, `tb_analyze.py`, `syncchk.py`, `inventory.py`, `chpsf.py`, `psffit.py`, `run_psf.py`, `run_psf_variants.sh`, `summary_fits.py`, `diag_4pi.py`, `diag_pattern.py` | commit 1106fbb, the calibration chain (camera offsets, registration `PolyMap`, in-situ PSF) whose outputs under `out/4pi/smlm/` every later script reads. One-off in the sense of "run once per data set", not dead | keep as `analysis/fourpi/calib/`; `register.PolyMap` and `camera` loading are needed by the pipeline and belong in the package |
| one-off checks | `analysis/checks/{cand_pmin_check, lm_check_paper2}.py`, `speed/{autoc_iso_check, hash_verify, pattern_hash_vs_file, pattern_kd, head_variants, head_vs_fp32, head_prof, prof_pattern}.py`, `fourpi/{test_solve_oracle, simoffset4pi, field_bias4pi, linbound4pi, info4pi, fit_psf4pi, bench_classical, eval_head4pi, dafl4pi}.py`, `fourpi_classical/{compare_det, val_iso}.py`, `psf3d/{gamma_profile, simtracks, perangle}.py`, `psf3d_nets/*`, `psfmatch/{accept3, accept_g13, gatecheck}.py`, `evalproto/zcal.py` (superseded by `zcal_fix.py`?) | no caller found; each documents a decision already taken (their results are in `out/*.md` and `docs/RESULTS.md`) | archive, except `test_solve_oracle.py` and `hash_verify.py`, which are tests in disguise: move to `tests/` |
| queue / orchestration scripts in `analysis/` | `papersim/run_{queue2,queue3,hollow,hollow_final,hollow_final2,hollow_x1rest,hollow_analysis,density_exp,density_full,g13,gauss_base}.sh`, `rerun_metrics.sh`; `fourpi/{showcase4pi_queue, showcase4pi_movies, gate_movies_det, gate_movies_w, ref_lin_movies}.sh`; `tput/wait_and_bench.sh` | they encode "wait for job X, then pick a GPU" for one night. The reproducible part (which command with which arguments) should be one `analysis/papersim/reproduce.sh` and `analysis/fourpi/reproduce.sh` | fold the commands into the two reproduce scripts, archive the rest |
| `analysis/results/` (140 tracked files, 1.1 MB) | outputs (json / txt / png summaries) inside the source tree | move out with the weights at the source-only step, or keep only what `docs/RESULTS.md` links |
| `simcode2/eff/` | `crlb_nuis.py` (used by `sim/crlb.py`), `simcode_gate.py` (runs the preprint model on the gate scenes; 7 users, all analysis) | `crlb_nuis` -> `sim/`; `simcode_gate.py` -> `analysis/gate/` (it depends on the old code base, not on the method) |

From `runs/` (gitignored), these are real functionality and should become code: `pickgpu` + `stage` with a GPU lock
(`runs/psfmatch/env.sh`) -> the launcher of 1.2; the production recipe (`prod.sh`: movie, calib, frozen flags
`--autoc 0 --cand_pmin 0.3` from `runs/psf3d/frozen_config.txt`, net set, sigma rescale, subset files) -> `analysis/paper/run_production.py`
reading `datasets.json` + `netsets.json`; the training chain (`chain.sh`, `chain2.sh`: det -> head -> calib -> p2 det -> p2 head
-> calib) -> `python -m simcode.train.chain --set <name> --psf <calib>`. The frozen flags differ from the defaults of
`run.py` (`--autoc 1`); that the paper tables come from non-default flags should be visible in the repository, not only in `runs/`.

---

## 4. Structure

**Package vs scripts.** Into the package (needed to run or evaluate the method on new data):
`analysis/fourpi/dc.py` (-> `io/movie.py`), `register.PolyMap`, `front4pi.Front`, `solve_real4pi.{pattern_psi, phase_field, repattern}`,
the body of `real_w4pi.py`, `fig_cl.zdrift_struct`, `baselines/zpsf.py`, `baselines/dme_drift.py`'s row selection,
`paper/frc_fair.py`'s protocol, `papersim/metrics.py` (GT metrics: Jaccard, FRC set, RMS; used by 7 scripts through
`sys.path`), `papersim/pvalid.py`'s feature and model code if p_valid becomes a table column (it writes a `.joblib` model;
today nothing in the package can apply it), `psf3d_nets/sigcal.py` (it produces the `sigu` json that `strict.build_table` consumes).
Out of the package: `eff/simcode_gate.py`, `insitu/coma_anchor_scan.py` + `lm_refine.py`, `hd/gate_report*.py` and
`p2/dump.py` (benchmark reporting; could be `simcode/bench/`), `evalproto/lib.py` joins `simcode/eval/`.
The classical 4Pi pipeline (`fourpi_classical/`, 2.2k lines) is a baseline: keep it in `analysis/`, but make it import the
shared reader / driver / pattern code instead of `sys.path` tricks.

**Training entry points inside model files.** `hd/sc.py`, `hd/sc4.py` (540 lines: model 170, hold-set preparation and
evaluation 140, trainer 150), `hd/sc7bg.py`, `p2/head2.py`, `p2/det2.py`, `det/up2.py` each contain model + targets + trainer +
`__main__`. Split into `model` and `train` (see 2.2). Importing a model then no longer pulls argparse and the generators.

**`sys.path.insert` and absolute paths.** 153 analysis scripts insert paths; 159 files contain `/home/jc/simcode2`
(78 in `analysis/fourpi`, 9 in `fourpi_classical`, 74 under `runs/`, and 2 inside the package: `fourpi/loadhead.py:7`,
`fourpi/realsim4pi.py:30`, against `config.py`'s "nothing in the package hard-codes an absolute path"). With `pip install -e .`
and `analysis/__init__.py` files the inserts are unnecessary; 4Pi paths go through `config.out_dir('4pi', ...)` and a new
`data4pi` root.

**Naming.**
- Selection columns: `good` (p >= 0.63), `sel` (phase-calibrated), `strict` (density-rule q), `strict_u` (the rank), and the
  paper's "u <= 0.45" which has no column. Labels in `papersim` call the same things `v2`, `v2sel` ("SIMCODE v2 all"),
  `v2q45`; subset files are `_strictonly`, `_u045`, `_q045`. Proposal: column `u` (= `strict_u`, old name kept as alias on
  read), boolean columns `good`, `sel`, `main` (= the paper rule, threshold stored in the table's info), `strict`; one subset
  file pattern `locs_p2.<rule>.npz`. The user decides the words (HANDOFF step 4 of the rename).
- Pass tags: files `pass2_est_*` = solve of the pass-1 rows, `pass2_est2_*` = pass 2, `pass2_est2b_*` = head-prior variant,
  `ref` / `ref2`, `estfld` / `estdep`, `est2r` (in `figures/lib20.py`); tables `locs_p1` / `locs_p2`; flags `--sc_*` vs `--p2_*`
  in `run_uncalibrated.py` but `--det/--head/--cal` in `run.py`. Proposal: stage names `p1`, `p1solve`, `p2` in new
  file names (`p1solve_est_<f0>.npz`), old names still read.
- 4Pi head tags `v2 ... v10b` mean different architectures (`arch` w / k / s / p1 / none); record `arch` in the file name of
  new models (`head4pi_w_v10.pt`).
- Position columns: 2D tables `x, y, sx, sy` in px; 4Pi tables px + um (`z`), `showcase` scripts nm. State units in the
  table info and convert in `load_table` (2.12).

**Documentation drift.** `FILE_DESCRIPTION.md` lists 6 files for `fourpi/`; there are 18 (missing: `head4pi_k/_s/_p1/_w`,
`solvek4pi`, `solveloss4pi`, `linread4pi`, `windows4pi`, `realsim4pi`, `loadhead`, `ckpt`) and `util/proctitle.py`. README's package tree has the same gaps.

**Missing tests** (all CPU or a few seconds of GPU; today there is one test file, GPU only):
1. `io.movie`: TIFF multi-part boundaries (frame index across `_X1`), dcimg header + `crops` mirror, `Prefetch` hit / miss.
2. `io.work`: claim exclusivity with two processes, stale-claim takeover, atomic write under `kill -9`, resume skips done units.
3. `strict`: table columns and the selection rules on a 1000-row fixture; non-default crop.
4. `eval.frc`: against a stored curve; `util.drift.drift_of_table` row selection (no DME needed).
5. `fit.lm`: Poisson fit of a Gaussian reaches the CRLB within 5 % (also guards 2.7).
6. 4Pi: `test_solve_oracle.py` as a test; `SolveK` on simulated read-outs returns the truth within 3 sigma.
7. `config`: `netset()` resolution; old env / config names still accepted after the rename.
8. Loading every shipped checkpoint (`load_net`) and one forward pass on zeros.

**What the rename touches** (counts of files containing the string `simcode2`, `__pycache__` excluded):
package 68 files (409 occurrences), `analysis/` 233 (638), `tests/` 1 (17), `docs/` 3 (29), README (14), `pyproject.toml` (4:
name, two console scripts, `packages.find`), `simcode2.example.json`, `.gitignore` (2), `runs/` 74 (226, gitignored, includes
running job scripts). 239 files import the package or call `python -m simcode2.`. Environment and config names:
`SIMCODE2_MODELS`, `SIMCODE2_DATA`, `SIMCODE2_OUT`, `SIMCODE2_CONFIG` (set in `config.py`; used in 13 files, e.g.
`analysis/gate/run_gate.sh`, `analysis/baselines/compare.py`), `simcode2.json`, `~/.config/simcode2.json`, default output
directory `simcode2_out`. The conda env `simcode-py311` (52 mentions) and the process-title prefix `simcode:` already have the
new name. The repository directory `/home/jc/simcode2` appears in 159 files; if the directory keeps its name these stay valid,
but they should go through `config` regardless (above).

---

## 5. Order of work

Rules: one commit per step; every step leaves `tests/test_fast_paths.py` green; "bit-identical" means `np.array_equal` on
every array of the named files, produced on the same GPU model with the same flags; where bf16 batching can differ, the
tolerances of the test file apply (5e-3 px, 3 ph). Reference outputs: `out/paper2/1809_2/v2` (2 chunks: `--chunk_list
12000,18000`), gate `iso` (expected RMS / CRLB 1.17 [1.14, 1.20], `analysis/gate/run_gate.sh`), one NPC file of
`out/4pi/net/locs_NPC_<fi>_v10.npz`. Effort in hours of focused work, without waiting for jobs.

Phase R: the queued items (preconditions in HANDOFF: no running `simcode:*` process, workers at a hand-over point).

| step | change | verify | h |
|---|---|---|---|
| R0 | commit everything, tag `pre-cleanup`, back up `.git` | `git status` clean | 0.5 |
| R1 | rename package, console scripts, config / env names with fallbacks (HANDOFF steps 2-3) | test file; `python -m simcode.run` on 2 chunks bit-identical to the reference; old `SIMCODE2_*` still honoured (new test 7) | 2 |
| R2 | load every checkpoint in `models/` and `out/4pi/models/` under the new name; add an alias module only if one fails | script exits 0 (becomes test 8) | 1 |
| R3 | source-only: untrack weights, `models/README.md`, fetch script; move `analysis/results/` | fresh clone + fetch + test file | 2 |

Phase A: no behaviour change, pure moves and helpers.

| step | change | verify | h |
|---|---|---|---|
| A1 | `attic/`: section 3 items marked archive (`det/train.py`, sc7bg after the user's answer, 4Pi heads v4-v9 + trainers + gate scripts, one-off checks, queue scripts) | test file; `python -c "import simcode.fourpi.loadhead"`; showcase `sc_d03` eval unchanged | 2 |
| A2 | constants and paths: 4Pi `setup.py` (GAIN, P, XLIM, DELTA, PX), `config` roots for 4Pi, remove `REPO = '/home/jc/...'` and `sys.path.insert` from live scripts | grep counts; one NPC file bit-identical | 3 |
| A3 | `util/ckpt.py` (from `fourpi/ckpt.py`) used by all trainers; `io.work.atomic_savez` used by `run_uncalibrated.py`, `strict.write_table` | 2 chunks bit-identical; `kill -9` during a chunk, restart completes | 1.5 |
| A4 | `simcode/eval/frc.py`, `util.drift.drift_of_table`, `fourpi/zdrift.py`; callers switched (2.1, 2.3, 2.4) | FRC of 1809_2 halves identical; drift array identical; z-drift identical | 3 |
| A5 | `models/netsets.json` + `config.netset` + `run.py --netset`; `strict.select` + `--subset` (2.5, 2.6); crop passed to `strict` (defect 5) | resolved paths equal `command.txt`; subset equals `locs_p2_u045.npz` | 3 |
| A6 | `FILE_DESCRIPTION.md`, README tree regenerated from the docstrings | every package file listed once | 1 |

Phase B: the shared file handling (section 1).

| step | change | verify | h |
|---|---|---|---|
| B1 | `io/movie.py` (`TiffMovie`, `DcimgMovie`, `ArrayMovie`, `Prefetch`) + tests 1; old names are aliases | reader equality checks of 1.3 | 3 |
| B2 | `io/work.py` (`Units`, claim, guard, per-unit records) + tests 2 | tests (CPU) | 4 |
| B3 | main pipeline on `Units`: chunk loops, single pass-1 row loader, per-unit timing merged into `runtime.json` | 2 chunks bit-identical; then two processes with `--claim 1` on 4 chunks: same files as one process | 4 |
| B4 | 4Pi learned-window on the reader + `Units`: `process_file`, launcher instead of `run_full_w4pi.sh`; new output layout with the old-name resolver | one NPC file bit-identical; two workers, kill one, restart | 5 |
| B5 | 4Pi classical and the simulated drivers on the same reader / driver (2.9) | one file table and `sc_d03` eval identical | 5 |
| B6 | throughput check: 1 vs 2 workers on one TIFF movie from the HDD | frames/s table; decide whether to document multi-GPU for the main pipeline | 1 |

Phase C: deeper merges, only where a re-run is planned anyway.

| step | change | verify | h |
|---|---|---|---|
| C1 | trainer loop + model / train split (2.2) | 20-step loss sequences identical per trainer | 8 |
| C2 | 4Pi pattern / phase-field module, `Solve4Pi` base tidy (2.10, 2.15) | pattern json and one file bit-identical | 5 |
| C3 | `fit/lm.py` + CRLB helpers (2.7, 2.8) | CRLB columns identical; baselines within 1e-3 px, or skipped | 4 |
| C4 | `eval/tables.py`, `eval/render.py` (2.12, 2.13), script by script | each regenerated figure's numbers unchanged | 6 |
| C5 | naming (section 4) after the user's decision; old names accepted on read | `strict` fixture test; figures unchanged | 3 |

Total about 65 h; phases R + A + B (the user's request proper) about 41 h. Steps within a phase are independent except
B3-B5 (need B1, B2) and A1 before C1 / C2.

Questions for the user before starting: (1) keep the sc7bg line in the CRLB figure? (2) which net set produced the paper's
2.5 nM / 20 nM tables (`netsets.json` and HANDOFF disagree about psf3dmix)? (3) should `run.py`'s defaults become the paper
configuration (`--netset`, `--autoc 0 --cand_pmin 0.3`)? (4) names for the selection columns. (5) rename existing
`out/4pi/net` files or only use the new layout for new runs (proposed: the latter)?
