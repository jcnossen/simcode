# figures/

Text sources of the paper figures, as written by the figure scripts in `analysis/` (copies of the outputs under `out/`, 2026-11-16):

- `paper_figs_v2/`: main and SI figures (from `out/paper_figs_v2/`): per figure `<name>_caption.txt` (caption material),
  `<name>_values.json` (every number shown), `<name>_table.{md,csv}` / `<name>_psf_table.{md,csv}` (SI tables);
  `INDEX.md` (which script makes which figure), `NOTATION.md` (symbols), `FIG2_FACTS.md`, `PATTERN_EST_FACTS.md`,
  `SI_DECODE_TRAINING_NOTE.md`, `table_S1_conditions.md`.
- `papersim/figures/`: the per-density tables and captions of the simulation figures (from `out/papersim/figures/`).
- `paper_v2psf/`: fact sheets of the experimental (`FACTS.md`) and 4Pi (`FACTS_4PI.md`) sections (from `out/paper_v2psf/`).

Not included: rendered images (pdf / svg / png), LaTeX sources, caches, internal working notes and internal figure variants,
and the material of the schematic figures (Fig 1 pipeline, SI network structure, SI 4Pi pipeline). Paths in these files are
relative to the repository root (`out/...` are outputs that are not in the repository).
