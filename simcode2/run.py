"""
SIMCODE v2 on a DMD-SIMFLUX DNA-PAINT movie, production settings.

  python -m simcode2.run --movie <movie.tif> --calib <in-situ calib.json> --out <dir> [--pass2]

Stages (each skipped if its outputs exist, unless --redo):
  1  pass 1      det_n1 detector + sc5nb head on every 6-frame window (start every 3 frames): network per-frame intensities
                 with covariance, bg, pattern-free position prior; candidates with detector p >= --cand_pmin are stored
  2  pattern     illumination pattern (k, per-step phase / depth / relative intensity, per-angle phase drift) estimated from
                 the pass-1 network intensities
  3  autoc       automatic per-dataset intensity nuisance (c, c_a) from the solve's own residuals (--autoc 1, default);
                 --autoc 0 uses the fixed c 0.10 + 0.10 real-data calibration instead
  4  solve       explicit intensity-domain pattern solve of the pass-1 rows (fringe posterior, kinetics, calibrated sigma)
  5  pass 2      (--pass2) pattern-aware pass-2 detector + head (p2L) fed the estimated pattern, then the same solve
  6  table       localization table(s) locs_p1.npz (and locs_p2.npz) with the phase-calibrated selection and the recommended
                 strict product (per-10-px-tile sigma quantile, q from the row density, matched in-sample to SIMCODE's coverage
                 on the real movies (anchors fitted there) under both the automatic and the fixed c; simcode2/strict.py); summary.json
Fast path (queued batching, exact fast solve, torch.compile of the solve's LM step when available) is the default.
Paths of networks / calibrations are relative to the models directory (simcode2/config.py); set CUDA_VISIBLE_DEVICES to
choose the GPU. A 6000-frame 0.5 nM chunk takes ~0.5 min (pass 1 + solve) on an idle RTX 3090.
"""
import argparse, json, os, sys, time
from simcode2 import config

CAND_PMIN = 0.3          # stored-candidate detection cut (the phase-calibrated selection needs candidates below p 0.63)


def _uncal(args):
    from simcode2.pipeline import run_uncalibrated
    run_uncalibrated.main(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1].strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--movie', required=True, help='movie .tif (further parts movie_X1.tif ... are found automatically)')
    ap.add_argument('--calib', required=True, help='in-situ PSF / camera calibration json (e.g. insitu/calib_0.5nM.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pass2', action='store_true', help='also run the pattern-aware pass 2 (p2L) -> locs_p2')
    ap.add_argument('--autoc', type=int, default=1, help='1: automatic (c, c_a) per dataset (default); 0: fixed c calibration')
    ap.add_argument('--autoc_fast', type=int, default=0, help='1: the autoc estimator uses the fast solve (3x faster; theta differs '
                    'by ~1e-3 through Nelder-Mead rounding); 0 (default): the reference solve, identical to the evaluated pipeline')
    ap.add_argument('--cand_pmin', type=float, default=CAND_PMIN)
    ap.add_argument('--fast', type=int, default=1)
    ap.add_argument('--compile', default='auto', help='torch.compile of the solve LM step: auto | on | off')
    ap.add_argument('--crop', default='5,5,110', help='output region x0,y0,size (px)')
    ap.add_argument('--chunk', type=int, default=6000)
    ap.add_argument('--chunk_list', default='', help="process only these chunk starts, e.g. '12000'")
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--pattern_field', default='', help="opt-in: 'est_depth' = depth-only field (not recommended by default)")
    ap.add_argument('--det', default=config.DET); ap.add_argument('--head', default=config.HEAD)
    ap.add_argument('--cal', default='', help='pass-1 solve calibration (default: autoc table + estimated c, or the fixed-c real calib)')
    ap.add_argument('--p2_det', default=config.P2_DET); ap.add_argument('--p2_head', default=config.P2_HEAD)
    ap.add_argument('--p2_cal', default=config.CAL_P2_REAL)
    ap.add_argument('--drift', default='', help='optional drift [F,2] px for the strict-product coverage (e.g. DME output)')
    ap.add_argument('--strict_q', type=float, default=None, help='fixed per-tile quantile (default: from the row density)')
    ap.add_argument('--strict_sigma_ref', type=float, default=None,
                    help='nm; opt-in earlier rule (7.0 pass 1 / 7.4 pass 2), valid with the fixed-c calibration (--autoc 0) only')
    ap.add_argument('--format', default='npz', help='table formats: npz,csv,hdf5')
    ap.add_argument('--sig_u', default='', help='opt-in selection-aware sigma rescale json per pass, "<pass1>,<pass2>" (strict.build_table)')
    ap.add_argument('--redo', type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t00 = time.time(); ST = {}
    common = ['--movie', a.movie, '--calib', a.calib, '--out', a.out, '--crop', a.crop, '--chunk', str(a.chunk),
              '--fast', str(a.fast), '--compile', a.compile, '--cand_pmin', str(a.cand_pmin), '--redo', str(a.redo),
              '--sc_det', a.det, '--sc_head', a.head]
    if a.chunk_list: common += ['--chunk_list', a.chunk_list]
    if a.nframes: common += ['--nframes', str(a.nframes)]
    # (2026-11-11) with an explicit --cal (e.g. a retrained head's calibration) its prior_off also feeds the pattern stage; default unchanged
    p1_fixed = config.resolve(a.cal) if a.cal else config.CAL_P1_REAL
    # 1-2: pass 1 + pattern (the pattern stage only uses the calibration's prior_off, identical in all pass-1 calibrations)
    t0 = time.time()
    need_pat = a.redo or not os.path.exists(f'{a.out}/pattern_est.pt')
    _uncal(common + ['--frontend', 'sc', '--sc_cal', p1_fixed, '--stages', 'pass1' + (',pattern' if need_pat else '')])
    ST['pass1_pattern_s'] = time.time() - t0
    # 3: automatic c
    cal = a.cal
    if not cal:
        if a.autoc:
            cal = os.path.join(a.out, 'calib_autoc.json')
            if a.redo or not os.path.exists(cal):
                t0 = time.time()
                from simcode2 import autoc
                print('autoc: estimating (s, c, c_a) on the pass-1 rows', flush=True)
                E = autoc.run_dir(a.out, fast=bool(a.autoc_fast))
                json.dump(dict(E, cal0=config.CAL_P1_SIM), open(os.path.join(a.out, 'autoc_est.json'), 'w'), indent=1, default=float)
                json.dump(autoc.make_calib(E['theta'], src=os.path.join(a.out, 'autoc_est.json')), open(cal, 'w'), indent=1)
                ST['autoc_s'] = time.time() - t0
                print('autoc theta s %.3f c %.3f c_a %.3f (s is a nuisance, not applied)' % tuple(E['theta']), flush=True)
        else:
            cal = p1_fixed
    # 4: pattern solve of the pass-1 rows
    t0 = time.time()
    _uncal(common + ['--frontend', 'sc', '--sc_cal', cal, '--stages', 'pass2'] + (['--pattern_field', a.pattern_field] if a.pattern_field else []))
    ST['solve_s'] = time.time() - t0
    # 5: pass 2
    if a.pass2:
        t0 = time.time()
        _uncal(common + ['--frontend', 'sc2p', '--p1_solve', '0', '--sc_cal', cal, '--p2_det', a.p2_det, '--p2_head', a.p2_head,
                         '--p2_cal', a.p2_cal, '--stages', 'pass2'])
        ST['pass2_s'] = time.time() - t0
    # 6: tables + strict product
    from simcode2 import strict as SP
    px = json.load(open(config.resolve(a.calib)))['pixelsize_um'] * 1e3
    summ = dict(movie=a.movie, calib=a.calib, cal_pass1=cal, pass2=a.pass2, cand_pmin=a.cand_pmin, fast=a.fast, autoc=a.autoc)
    if a.autoc and not a.cal and os.path.exists(os.path.join(a.out, 'autoc_est.json')):
        summ['autoc_theta'] = json.load(open(os.path.join(a.out, 'autoc_est.json')))['theta']
    tags = [('est', 'locs_p1')] + ([('est2', 'locs_p2')] if a.pass2 else [])
    su = (a.sig_u.split(',') + [''])[:2] if a.sig_u else ['', '']
    for (tag, name), su_ in zip(tags, su):
        T, info = SP.build_table(a.out, tag, px, a.drift or None, a.strict_q, a.strict_sigma_ref, config.resolve(su_) if su_ else None)
        files = SP.write_table(T, info, os.path.join(a.out, name), a.format.split(','))
        summ[name] = {k: v for k, v in info.items() if k != 'columns'}; summ[name]['files'] = files
        summ['columns'] = info['columns']
        print(f'{name}: {info["n_rows"]} rows, {info["n_sel"]} selected, strict {info["n_strict"]} (q {info["q"]:.3f}, '
              f'{info["coverage_bins"]} / {info["coverage_bins_good"]} occupied 20-nm bins) -> {files}', flush=True)
    ST['total_s'] = time.time() - t00
    summ['timing'] = ST
    json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, default=str)
    print('done', json.dumps(ST), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
