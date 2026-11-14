# SIMCODE model g13

Trained networks and solve calibrations of this acquisition.

- `model.json`: the model (format of `simcode/models.json`, paths relative to this folder).
  - `_recipe`: how it was trained: the acquisition config, every command in order, steps per trainer, gate scenes, acceptance, git commit of the code.
  - `_files`: size and sha256 of every file; for the networks also the trainer arguments stored in the checkpoint.
- How a set of localization tables was calculated with this model: the `run_config.json` (readable copy `run_config.yaml`) in the folder of the tables: fully resolved settings, sha256 of every network / calibration / movie file, code commit, environment. Re-run: `python -m simcode.run --config run_config.json --out <new dir>`.
