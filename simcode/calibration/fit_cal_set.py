"""PSF-matched retrain (2026-11-11): solve calibration of a front-end (since 2026-11-17 only prior_off from the iso scene) from ONE PSF's averaged-PSF
gate scenes (as the earlier fit of the mixture set, which pooled the 3 per-axis PSFs; same plib.fit_cal procedure, prior_off from iso
of the same PSF; until 2026-11-16 analysis/psfmatch/fit_cal_set.py). This is the model's only solve calibration of that pass: it
is used for real and simulated movies alike (the real-data sigma multiplier of the earlier _real variant was removed 2026-11-17).
usage: python -m simcode.calibration.fit_cal_set <front> <dump directory, e.g. out/psfmatch/fe_m1809a> <fit scenes, comma> <base calib> <out calib>
The label in fit_scenes / note is the dump directory's basename without a leading 'fe_' (fe_m1809a -> m1809a)."""
import json, os, sys
from simcode.calibration import gate_stats as P
from simcode import config

if __name__ == '__main__':
    if len(sys.argv) != 6: raise SystemExit(__doc__)
    front, d, scen, base, out_sim = sys.argv[1:6]
    label = os.path.basename(os.path.normpath(d))
    if label.startswith('fe_'): label = label[3:]
    B = json.load(open(config.resolve(base)))
    Rs = [P.load(sc, front, seeds=(202,), d=d) for sc in scen.split(',')]
    iso = P.load('iso', front, seeds=(202,), d=d)
    cal = P.fit_cal(Rs, B, iso=iso)
    cal['fit_scenes'] = [f'{label}:{sc}' for sc in scen.split(',')]
    cal['note'] = (f'2026-11-11 PSF-matched retrain: sim-sigma calibration of front {front} (simcode.calibration.fit_cal_set = gate_stats.fit_cal), '
                   f'tuning seed 202 of the averaged-PSF gate scenes {cal["fit_scenes"]}, prior_off from iso ({label}); base {base}')
    os.makedirs(os.path.dirname(os.path.abspath(out_sim)), exist_ok=True)
    json.dump(cal, open(out_sim, 'w'), indent=1)
    print('wrote', out_sim)
