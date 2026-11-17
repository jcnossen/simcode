# Baseline runs of the sigma study (2026-11-17)

The reference runs of `analysis/sigma_joint/evalsim.py` and `k1_compare.py`, made with the package of branch `siglayer`
(commit 1c7f26e8; that package stored the solve's raw quantities `s_pm_*`, `spread_pm_*`, ... in the chunk files):

- `id`: the production networks and solve until 2026-11-17 (kap_bins scales in the fit) with an identity sigma (the production
  sigma is recomputed from the stored raw quantities: `evalsim.prod_sigma`); x, y, z, photons bit-identical to the production
  code (checked on m05r3v2 and on the real 1809/2 movie)
- `k1`: the same networks, the solve without the kap_bins scales (kap_p = kap_I2 = 1; pass 1 and the pattern hard-linked from `id`)

`mk_models_id.py` (model folders `<m>_id`, `<m>_k1`), `runs_id.sh`, `runs_k1.sh`: as run (they import that branch's
`simcode.solve.sigma_layer`). Outputs: `out/sigma_layer/runs/<m>_{id,k1}/<movie>`.
