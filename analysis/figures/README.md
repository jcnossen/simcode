# analysis/figures

Figure scripts for the paper and the summary plots. `<out>` is the out root of `simcode.config` (`SIMCODE_OUT`, or `out` in
`simcode.json`), and `<data>` is the data root. Inputs are read from `<out>` under the same relative paths as in the
development tree. Every script writes its PNG to `<out>/plots/` by default; `--out` or the first argument redirects it.
Run the scripts from the repo with `PYTHONPATH=<repo>`. Use `--help` on any script for its usage.

Check: each script below was run on the existing development outputs, and its PNG / text output was compared with the stored
one. Every checked output was pixel-identical or byte-identical, except as noted.

## CRLB

| script | makes | inputs |
|---|---|---|
| `crlb_all_v2.py [sc5nb,sc6n] [11z]` | `crlb_all_v2.png`, `<out>/sc7/crlb_all_v2.txt` | `<out>/gate/*.npz` (networks, `_scode_ni6_3`), `<out>/sc7/gate_lm/*_lm11*.npz`, `*_sf11.npz`, `<out>/sc7/icrlb_*.npz` |
| `icrlb_report.py` | `crlb_sc6_all.png`, `<out>/sc7/icrlb_report.txt` | `<out>/sc7/icrlb_*.npz` |
| `lm_robust.py` | `<out>/sc7/lm_robust.txt` | as `crlb_all_v2.py` |
| `plot_crlb_sc3.py` | `crlb_sc3.png` (generic gate CRLB plot; `curves()` is used by the scripts above) | `<out>/gate` |
| `crlb_vs_simcode.py` | `crlb_vs_simcode.png` | `<out>/gate` |
| `crlb_gate.py` | `crlb_gate.png`, the early figure of the development hybrid front-ends pf / det / hd | the untagged `<out>/gate/<scene>__s30?.npz` |

The evaluation-only references are generated on GPU with `run_crlb_refs.sh [icrlb|lm|sf|scode|all]`, which calls these scripts:

- `icrlb.py`: per-frame I_t and bg of a head vs their CRLBs.
- `lm_icrlb.py` and `lm_gate.py`: the basic single-emitter LM.
- `sf_gate.py`: the joint SIMFLUX MLE.
- `simcode.eff.simcode_gate`: SIMCODE, which needs smlmtorch.

`lmref.py` holds the LM and SIMFLUX-MLE references (`LMRef`, `SFRef`). They are not part of the method. It uses
`simcode.psf.lut.make_lut`, which is identical to the development `hd/solve.make_lut`. The first 175 detector candidates of
`iso` seed 301 FOV-batch 1 reproduce the stored `gate_lm` x_lm, x_sc and SFRef x bit for bit.

The `sc6n` network is not shipped in `models/`. Its `icrlb_sc6n_*` and `_hd_sc6nb15pm` records exist only as development
outputs.

## Real-data renders

| script | makes | inputs |
|---|---|---|
| `recon_full.py` | `recon_full_0.5nM.png` | `<out>/full_0.5nM_sc5nb_bright_real` (`--run`), SIMCODE `sfhd-ndi.hdf5` under `<data>` (`--simcode`) |
| `recon_full_dc.py` | `recon_full_0.5nM_dc.png`, `drift_full_0.5nM.png` | same; RCC drift `rcc_drift.py` |
| `recon_full_dme_cross.py` | `recon_full_0.5nM_dme_cross.png` | same + `<out>/drift/drift_{ours,simcode}_fpb50.npy` (`dme_full05.py`, DME via the `dme` package) |
| `hdzoom.py` | `recon_full_20nM_dense.png`, `<out>/real20/zooms_dense.json` | `lib20.py` sets: `<out>/full_20nM_m2_fast`, `<out>/real20/{simcode_real,dmev2}`, published hdf5 and the 20 nM movie under `<data>` |
| `render_matched20.py` | `recon_full_20nM_matched.png` | as `hdzoom.py` + `<out>/real20/zooms.json` |
| `render_strict20.py` | `recon_full_20nM_strict.png` | same |
| `smoke_metrics.py` | `<run>/smoke_metrics.txt / .json` (real-chunk metrics vs SIMCODE; also the loaders of the renders) | run dir, SIMCODE hdf5 |

`zoompick.py` picks the zoom boxes at filament crossings. Its skeleton comes from `skel.py`, a numpy port of
`skimage.morphology.skeletonize` (Zhang–Suen, with scikit-image's lookup table). The port gave the same output as skimage
0.26 on 800 random images, and it gives the same `recon_full_0.5nM_dme_cross.png` boxes.

## Summary figures

| script | makes | inputs |
|---|---|---|
| `fig_gap.py` | `realgap_summary.png` | none (the numbers are copied into the script from the development realgap analyses) |
| `scorecard.py --stack 'real05,m05qh,m05r@<title>' 'real25,m25qh,m25r@<title>' <png>` | `realism_scorecard.png` | `<out>/simreal2/scorecard.json` (built by `scorecard.py name=<run>:<f0>:<movie>:...`) |
| `frc_plot.py` | `frc_vs_N.png` | `<out>/critic34/frc_commonN.json` (`frc_commonN.py`), `frc_common.py` sets (`<out>/full_*`, `<out>/drift*/{simcode_real,dmev2}`) |
| `field/plot_field_offsets.py` | `field_dep_offsets.png` | `<out>/field/{maps_*,check3_*}.npz`, `check1_summary.json`, `maps_summary.txt` |

`realism_scorecard.png`: both panels are pixel-identical. The 25-px title lines were re-created, because the development
composite command was not saved. Their font rendering differs slightly.

`field/`: run `check1_offsets.py`, `check1_summary.py`, `field_maps.py` and `check3_pattern.py`, then `plot_field_offsets.py`.
On a fresh out root (`SIMCODE_OUT`) with links to the run dirs, these reproduced every stored `<out>/field` npz / txt / json
file and the PNG exactly.
