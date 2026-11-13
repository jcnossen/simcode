"""Run a simcode module (simcode.run, a trainer, gate, dump) with the instrument overrides of the automatic pipeline applied
IN MEMORY (docs/AUTO_PIPELINE.md section 7; no package file is edited). The overrides come from a json file (env AUTO_WRAP):
  pitch_px [lo, hi]   default of pipeline.pattern_est.estimate_k(pitch_px=...)                       (assumption 3)
  crop [x0, y0, S]    strict.CROP and the bound defaults of strict.tile_of / strict.bin20            (assumption 4)
  det_batch n         argparse default of run_uncalibrated --det_batch                              (assumption 5)
  netset {name: {...}} added to simcode.config.netsets() (a set of this dataset, absolute paths)       (P8)
  presets {preset: {key: value}} merged into simcode.hd.realsim.PRESETS (pattern k, spacing, camera maps, bg, brightness) (7)
  px_nm v             simcode.hd.realsim.PX_NM                                                       (7)
  title s             process title (<= 20 characters)
usage: AUTO_WRAP=wrap.json python -m simcode.auto.runwrap <module> <module arguments...>
Without AUTO_WRAP (or with an empty json) the module runs unchanged: bit-identical to python -m <module>."""
import json, os, runpy, sys


def apply(W):
    if W.get('title'):
        from simcode.util.proctitle import set_title
        set_title(W['title'])
    if W.get('pitch_px'):
        from simcode.pipeline import pattern_est as PE
        d = list(PE.estimate_k.__defaults__); assert d[0] == (1.6, 2.8), 'estimate_k signature changed'
        d[0] = tuple(float(v) for v in W['pitch_px']); PE.estimate_k.__defaults__ = tuple(d)
    if W.get('crop'):
        from simcode import strict as SP
        crop = tuple(float(v) for v in W['crop'])
        assert SP.tile_of.__defaults__[0] == SP.CROP and SP.bin20.__defaults__[-1] == SP.CROP, 'strict.py signature changed'
        SP.CROP = crop
        SP.tile_of.__defaults__ = (crop,) + SP.tile_of.__defaults__[1:]
        SP.bin20.__defaults__ = SP.bin20.__defaults__[:-1] + (crop,)
    if W.get('det_batch'):
        import argparse as _ap
        _orig = _ap.ArgumentParser.add_argument
        def _add(self, *a, **k):
            if a and a[0] == '--det_batch': k['default'] = int(W['det_batch'])
            return _orig(self, *a, **k)
        _ap.ArgumentParser.add_argument = _add
    if W.get('netset'):
        from simcode import config
        config.netsets().update(W['netset'])
    if W.get('presets') or W.get('px_nm'):
        import simcode.hd.realsim as RS
        for name, p in RS.PRESETS.items():
            upd = {**W.get('presets', {}).get('common', {}), **W.get('presets', {}).get(name, {})}
            p.update({k: (tuple(v) if isinstance(v, list) and k != 'k' else v) for k, v in upd.items()})
        if W.get('px_nm'): RS.PX_NM = float(W['px_nm'])
    if W.get('px_nm'):          # analysis/calib/plib.py (sigma-table fits, gate statistics): PX 108.33
        import os as _os
        sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'analysis', 'calib'))
        import plib
        plib.PX = float(W['px_nm'])


def main():
    f = os.environ.get('AUTO_WRAP')
    W = json.load(open(f)) if f else {}
    apply(W)
    mod = sys.argv[1]; sys.argv = [mod] + sys.argv[2:]
    if W: print('auto.runwrap:', json.dumps({k: (v if k not in ('netset', 'presets') else '...') for k, v in W.items()}), flush=True)
    if mod.endswith('.py'): runpy.run_path(mod, run_name='__main__')
    else: runpy.run_module(mod, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
