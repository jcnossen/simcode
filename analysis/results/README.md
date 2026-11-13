# Result snapshots

Text snapshots of the development-tree outputs behind `docs/RESULTS.md`, copied as of 2026-11-11. They are reference copies, not
regenerated here. Development paths are relative to `<out>/` (not shipped). The scripts that reproduce
them are in `analysis/` (see `docs/RESULTS.md`, "Reproduce").

| here | development source | script in simcode |
|---|---|---|
| `evalproto/{valid,zcal*,spatial,envmap,realproto,filt,covpool,nnlag,blockphase}_*` | `evalproto/` (same names) | `analysis/evalproto/<name>.py` |
| `evalproto/frc_published_*`, `evalproto/strictq_*`, `checks/` | written in simcode (not dev snapshots) | `analysis/evalproto/{frc_published,strictq}.py` |
| `psfshape/evalgt_{bc,pd}.txt` | `psfshape/` | `analysis/psfshape/evalgt.py` |
| `hd_compare/score_*.txt`, `topn_vs_sc_*.txt`, `summary.txt` (hand-written summary) | `hd_compare/` | `analysis/hd_compare/score.py` |
| `simcode_base2/cmp_*.txt`, `summary*.txt`, `gate_paired_*.txt`, `simcode_config.txt` | `simcode_base2/` | `analysis/baselines/compare.py`, `analysis/gate/gate_paired.py` |
| `simreal/cci_*.txt` | `simreal2/` | `analysis/simreal/compare_ci_r.py` |
| `simreal/cci_m20qh_*.txt`, `merged_gross_m{20,25}qh.txt`; `simreal/cci_m25qh_*.txt` | `n20/`; `p2/long/` | `analysis/simreal/compare_ci_r.py`, `analysis/simreal/merged_gross.py` |
| `realism/realstats.json`, `validate_{0.5,2.5}nM_final.{json,txt}` | `realstats.json`, `validate_*_final.*` (input paths rewritten to `dev:`) | development realstats.py / validate_sim.py (legacy hybrid front-end; not ported, see `simcode/hd/realsim.py`) |
| `plots/*.png` (crlb_all_v2, crlb_sc6_all, crlb_vs_simcode, crlb_gate, recon_full_0.5nM[_dc\|_dme_cross], recon_full_20nM_{dense,matched,strict}, realgap_summary, realism_scorecard, frc_vs_N, field_dep_offsets; not copied here) | sc7/, plots/, realgap/, simreal2/, critic34/, field/, scratch/hdzoom/, real20/ | `analysis/figures/` |
| `gate/gate_long_vs_simcode.txt`, `merged_d1_vs_simcode.txt` | `p2/long/` | `analysis/gate/vs_simcode_p2.py` (+ `merged_d1.py` for ideal_C) |
| `tput/bench_*.json`, `clean_*.json` | `tput/` | `analysis/tput/bench_tput.py`, `summarize.py` |
| `tput/stages_real20.log` | `real20/stages.log` | (wall time per stage of the 20 nM run) |
| `tput/summarize.txt` | `summarize.py` run on a copy of the dev `tput/` (2026-11-11) | `analysis/tput/summarize.py` |
| `dme/report_real20.txt` | `real20/report.txt` | `analysis/dme/dme_splithalf.py report` (split-half column) |
| `dme/compare_dmev2_{05,25}.log` | `drift/compare_dmev2.log`, `drift25/compare_dmev2_25.log` | `analysis/dme/dme_splithalf.py` |
| `dme/report_p2_{05,25}.log` | `p2real/report_{05,25}.log` | `analysis/dme/dme_splithalf.py` |
| `dme/splithalf_all.txt` | `dme_splithalf.py report` on the dev `*/dmev2*/*.npz` (2026-11-11) | `analysis/dme/dme_splithalf.py` |

Dev names: `real0.5 / 2.5 / 20` = the real movies, `m05pd / m25pd / m05pe / m25pe` = the punctate sims. The ported scripts name their
outputs after the dataset json (e.g. `zcal_real0.5_simm05pe.txt` for the dev `zcal_real0.5_m05pe.txt`).
