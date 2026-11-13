"""
SIMCODE v2 on a DMD-SIMFLUX DNA-PAINT movie, production settings.

  python -m simcode.run --movie <movie.tif> --calib <in-situ calib.json> --out <dir> [--netset m1809] [--pass2]
  python -m simcode.run --dataset 1809_2 --out <dir>          (a paper dataset: movie, calibration, network set, flags)

Stages (each skipped if its outputs exist, unless --redo):
  1  pass 1      detector + head on every 6-frame window (start every 3 frames): network per-frame intensities
                 with covariance, bg, pattern-free position prior; candidates with detector p >= --cand_pmin are stored
  2  pattern     illumination pattern (k, per-step phase / depth / relative intensity, per-angle phase drift) estimated from
                 the pass-1 network intensities
  3  autoc       (--autoc 1, opt-in) automatic per-dataset intensity nuisance (c, c_a) from the solve's own residuals. The
                 default (--autoc 0, the paper's setting) uses the fixed-c real-data calibration of the network set. The
                 sigma table of --autoc 1 exists only for the earlier default head (hd_sc5nb), not for the PSF-matched sets.
  4  solve       explicit intensity-domain pattern solve of the pass-1 rows (fringe posterior, kinetics, calibrated sigma)
  5  pass 2      (--pass2) pattern-aware pass-2 detector + head (p2L) fed the estimated pattern, then the same solve
  6  table       localization table(s) locs_p1.npz (and locs_p2.npz) with the selections (simcode/strict.py): main = the
                 paper's global precision rule max(sx, sy) <= sigma_max nm (--sigma_max), u = the earlier per-tile
                 "u <= 0.45" (--u_max), sel (phase-calibrated), strict (per-10-px-tile sigma quantile, q from the
                 row density); --subset writes selections as small files; summary.json
Networks and calibrations come from a network set (--netset, simcode/netsets.json; default m1809); --det / --head / --cal /
--p2_det / --p2_head / --p2_cal override single files. Paths are relative to the models directory (simcode/config.py).
--dataset <name> takes movie, calibration and network set of a paper dataset from the '_production' entry of netsets.json and
applies the production recipe: its flags, pass 2, the set's sigma rescale (--sig_u set) and the subset files of the main
selection and of the strict product (what the paper tables are made with).
Fast path (queued batching, exact fast solve, torch.compile of the solve's LM step when available) is the default.
Set CUDA_VISIBLE_DEVICES to choose the GPU; several GPUs: the same command once per GPU with --claim 1
(python -m simcode.fileio.launch --gpus 1,2 -- python -m simcode.run ... --claim 1): the chunks of pass 1 and pass 2 are
shared between the copies, the stages in between run in one of them.
A 6000-frame 0.5 nM chunk takes ~0.5 min (pass 1 + solve) on an idle RTX 3090.
"""
import argparse, json, os, sys, time
from simcode import config

PROD = config.production()['flags']
CAND_PMIN = PROD['cand_pmin']      # stored-candidate detection cut (the phase-calibrated selection needs candidates below p 0.63)


def _uncal(args):
    from simcode.pipeline import run_uncalibrated
    run_uncalibrated.main(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1].strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='', help="a paper dataset (netsets.json '_production': " + ', '.join(config.production()['datasets']) + '): '
                    'sets --movie, --calib, --netset and the production recipe')
    ap.add_argument('--movie', default='', help='movie .tif (further parts movie_X1.tif ... are found automatically)')
    ap.add_argument('--calib', default='', help='in-situ PSF / camera calibration json (e.g. insitu/calib_1809_2_psf3d.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--netset', default='', help='network set of simcode/netsets.json (default: its _default)')
    ap.add_argument('--cal_kind', default=PROD['cal_kind'], choices=['real', 'sim'], help="solve calibrations of the set: 'real' (real movies; default) or 'sim' (simulated movies)")
    ap.add_argument('--pass2', action='store_true', help='also run the pattern-aware pass 2 (p2L) -> locs_p2')
    ap.add_argument('--autoc', type=int, default=PROD['autoc'], help='0 (default, the paper): fixed-c calibration of the set; 1: automatic (c, c_a) per dataset '
                    '(sigma table only valid for the earlier default head)')
    ap.add_argument('--autoc_fast', type=int, default=0, help='1: the autoc estimator uses the fast solve (3x faster; theta differs '
                    'by ~1e-3 through Nelder-Mead rounding); 0 (default): the reference solve, identical to the evaluated pipeline')
    ap.add_argument('--cand_pmin', type=float, default=CAND_PMIN)
    ap.add_argument('--fast', type=int, default=1)
    ap.add_argument('--compile', default='auto', help='torch.compile of the solve LM step: auto | on | off')
    ap.add_argument('--compile_head', default='off', help="opt-in: 'trunk' = torch.compile of the heads' bf16 CNN trunk when available "
                    "(~1.5x faster heads; not bit-equivalent to the eager bf16 head: |dI| p99 ~0.7 ph, but at least as close to the fp32 head); 'off' (default)")
    ap.add_argument('--prefetch', type=int, default=1, help='read the next chunk in a background thread (identical data); 0 = off')
    ap.add_argument('--claim', type=int, default=0, help='1: several copies of this command share the chunks (one copy per GPU)')
    ap.add_argument('--min_free_gb', type=float, default=2.0, help='a chunk is not started with less free space on the output file system')
    ap.add_argument('--crop', default='5,5,110', help='output region x0,y0,size (px)')
    ap.add_argument('--chunk', type=int, default=6000)
    ap.add_argument('--chunk_list', default='', help="process only these chunk starts, e.g. '12000'")
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--pattern_field', default='', help="opt-in: 'est_depth' = depth-only field (not recommended by default)")
    ap.add_argument('--det', default='', help='pass-1 detector (default: the network set)'); ap.add_argument('--head', default='', help='pass-1 head')
    ap.add_argument('--cal', default='', help='pass-1 solve calibration (default: the set\'s calibration; with --autoc 1 the autoc table + estimated c)')
    ap.add_argument('--p2_det', default=''); ap.add_argument('--p2_head', default='')
    ap.add_argument('--p2_cal', default='', help='pass-2 solve calibration (default: the set\'s)')
    ap.add_argument('--drift', default='', help='optional drift [F,2] px for the strict-product coverage (e.g. DME output)')
    ap.add_argument('--strict_q', type=float, default=None, help='fixed per-tile quantile (default: from the row density)')
    ap.add_argument('--strict_sigma_ref', type=float, default=None,
                    help='nm; opt-in earlier rule (7.0 pass 1 / 7.4 pass 2), valid with the fixed-c calibration (--autoc 0) only')
    ap.add_argument('--sigma_max', type=float, default=None, help='nm; threshold of the main selection (max(sx, sy) <= sigma_max on the rows with a rank; stored in the table; default strict.SIGMA_MAX_NM)')
    ap.add_argument('--u_max', type=float, default=None, help="threshold of the per-tile rule 'u' (strict_u <= u_max, stored in the table; default 0.45)")
    ap.add_argument('--subset', default='', help="also write these selections as small files, e.g. 'main,strict' (strict.write_subset)")
    ap.add_argument('--format', default='npz', help='table formats: npz,csv,hdf5')
    ap.add_argument('--sig_u', default='', help='opt-in selection-aware sigma rescale json per pass, "<pass1>,<pass2>" (strict.build_table); '
                    "'set' = the rescale of the network set")
    ap.add_argument('--redo', type=int, default=0)
    a = ap.parse_args(argv)
    from simcode.util.proctitle import auto; auto('run')     # recognizable name in nvidia-smi / ps
    if a.dataset:
        D = config.production()['datasets'].get(a.dataset)
        if D is None: ap.error(f'unknown dataset {a.dataset!r} ({", ".join(config.production()["datasets"])})')
        if not (a.netset or D['netset']): ap.error(f'dataset {a.dataset}: no network set is defined for it; give --netset')
        a.movie = a.movie or D['movie']; a.calib = a.calib or D['calib']; a.netset = a.netset or D['netset']
        a.pass2 = a.pass2 or PROD['pass2']; a.sig_u = a.sig_u or 'set'; a.subset = a.subset or 'main,u,strict'
    if not a.movie or not a.calib: ap.error('--movie and --calib are required (or --dataset)')
    N = config.netset(a.netset or None)
    a.det = a.det or N['det']; a.head = a.head or N['head']; a.p2_det = a.p2_det or N['p2_det']; a.p2_head = a.p2_head or N['p2_head']
    a.p2_cal = a.p2_cal or N[f'p2_cal_{a.cal_kind}']
    if a.sig_u == 'set': a.sig_u = N['sig_u']
    from simcode.fileio.work import Units, atomic_json
    os.makedirs(a.out, exist_ok=True)
    t00 = time.time(); ST = {}
    common = ['--movie', a.movie, '--calib', a.calib, '--out', a.out, '--crop', a.crop, '--chunk', str(a.chunk),
              '--fast', str(a.fast), '--compile', a.compile, '--compile_head', a.compile_head, '--prefetch', str(a.prefetch), '--cand_pmin', str(a.cand_pmin), '--redo', str(a.redo),
              '--claim', str(a.claim), '--min_free_gb', str(a.min_free_gb), '--sc_det', a.det, '--sc_head', a.head]
    if a.chunk_list: common += ['--chunk_list', a.chunk_list]
    if a.nframes: common += ['--nframes', str(a.nframes)]
    # (2026-11-11) with an explicit --cal (e.g. a retrained head's calibration) its prior_off also feeds the pattern stage; default unchanged
    p1_fixed = config.resolve(a.cal) if a.cal else N[f'cal_{a.cal_kind}']
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
            for _ in Units(a.out, 'autoc', pattern='calib_autoc.json', claim=a.claim, redo=a.redo).todo(['']):
                t0 = time.time()
                from simcode import autoc
                print('autoc: estimating (s, c, c_a) on the pass-1 rows', flush=True)
                E = autoc.run_dir(a.out, fast=bool(a.autoc_fast))
                atomic_json(os.path.join(a.out, 'autoc_est.json'), dict(E, cal0=config.CAL_P1_AUTOC_START), indent=1, default=float)
                atomic_json(cal, autoc.make_calib(E['theta'], src=os.path.join(a.out, 'autoc_est.json')), indent=1)
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
    # 6: tables + selections (with --claim 1 the tables are written by one of the copies; the others skip them)
    from simcode import strict as SP
    px = json.load(open(config.resolve(a.calib)))['pixelsize_um'] * 1e3
    crop = tuple(float(v) for v in a.crop.split(','))
    u_max = SP.U_MAX if a.u_max is None else a.u_max
    sigma_max = SP.SIGMA_MAX_NM if a.sigma_max is None else a.sigma_max
    summ = dict(movie=a.movie, calib=a.calib, netset=N['name'], cal_pass1=cal, pass2=a.pass2, cand_pmin=a.cand_pmin, fast=a.fast, autoc=a.autoc)
    if a.autoc and not a.cal and os.path.exists(os.path.join(a.out, 'autoc_est.json')):
        summ['autoc_theta'] = json.load(open(os.path.join(a.out, 'autoc_est.json')))['theta']
    tags = [('est', 'locs_p1')] + ([('est2', 'locs_p2')] if a.pass2 else [])
    su = (a.sig_u.split(',') + [''])[:2] if a.sig_u else ['', '']
    wrote = False
    for (tag, name), su_ in zip(tags, su):
        for _ in Units(a.out, 'table', pattern=name + '.npz', claim=a.claim, redo=not a.claim, title=False).todo(['']):
            T, info = SP.build_table(a.out, tag, px, a.drift or None, a.strict_q, a.strict_sigma_ref, config.resolve(su_) if su_ else None,
                                     crop=crop, u_max=u_max, sigma_max=sigma_max)
            files = SP.write_table(T, info, os.path.join(a.out, name), a.format.split(','))
            files += [SP.write_subset(T, os.path.join(a.out, name), r, u_max, sigma_max, px) for r in a.subset.split(',') if r]
            atomic_json(os.path.join(a.out, name + '_info.json'), info, indent=1, default=str)
            summ[name] = {k: v for k, v in info.items() if k != 'columns'}; summ[name]['files'] = files
            summ['columns'] = info['columns']; wrote = True
            print(f'{name}: {info["n_rows"]} rows, main (max sigma <= {sigma_max:g} nm) {info["n_main"]}, u <= {u_max:g} {info["n_u"]}, {info["n_sel"]} selected, strict {info["n_strict"]} (q {info["q"]:.3f}, '
                  f'{info["coverage_bins"]} / {info["coverage_bins_good"]} occupied 20-nm bins) -> {files}', flush=True)
    ST['total_s'] = time.time() - t00
    summ['timing'] = ST
    if wrote or not os.path.exists(os.path.join(a.out, 'summary.json')):
        atomic_json(os.path.join(a.out, 'summary.json'), summ, indent=1, default=str)
    print('done', json.dumps(ST), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
