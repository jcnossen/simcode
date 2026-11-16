"""Process the Cell11 files from the crop cache (analysis/fourpi_plain/p4cache.py; the raw dcimg files are not read again)
with the real driver (simcode.processing.fourpi.process_file) and settings overrides, e.g. an in-situ PSF table:

  python analysis/fourpi_plain/p4process.py --out <run dir> [--cfg <config .py>] [--files 0 1 2 3] [--set lut_psf="<psf json>" emitter="<ckpt>"]
  -> <run dir>/process/locs_<file>.npz (+ .json records, locs_summary.json, config_used.json)
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from analysis.fourpi_plain.p4cache import acq, cache_path, CachedReader, CFG


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--files', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--set', nargs='*', default=[])
    p.add_argument('--cfg', default=CFG, help='acquisition config file (default fourpi_plain.py)')
    a = p.parse_args()
    from simcode.processing import fourpi as PF
    c, S = acq(a.cfg)
    for kv in a.set:
        k, v = kv.split('=', 1); c['fourpi'][k] = json.loads(v)
    od = os.path.join(a.out, 'process'); os.makedirs(od, exist_ok=True)
    json.dump(c, open(os.path.join(a.out, 'config_used.json'), 'w'), indent=1, default=str)
    st = PF.settings(c); ctx = PF.context(c)
    summ = {}
    for fi in a.files:
        fo = os.path.join(od, f'locs_{fi:03d}.npz')
        if os.path.exists(fo): print(fo, 'exists'); continue
        rd = CachedReader(cache_path(fi))
        fr = S.front(c, fi, PF.dev)
        out, fld, sm = PF.process_file(rd, fr, ctx, st, fi, rd.n)
        np.savez(fo + '.tmp.npz', **out, **fld); os.replace(fo + '.tmp.npz', fo)
        json.dump(sm, open(os.path.join(od, f'locs_{fi:03d}.json'), 'w'), indent=1, default=float)
        summ[fi] = sm
        print(f'file {fi}: chi2/dof median {sm["chi2_dof_median"]:.2f}, {sm["n"]} rows, {sm["t_total"]:.0f}s -> {fo}', flush=True)
    json.dump(summ, open(os.path.join(od, 'locs_summary.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
