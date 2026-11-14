"""SIMCODE v2 on a papersim movie: `python -m simcode.run` unchanged (production stages: pass 1, pattern estimate, solve,
pass 2), with the movie's calib_sim.json (averaged 1809/2 PSF + simulated camera), the whole 40 x 40 (or 50 x 50) FOV as the
output region, and a model (--model: a name of simcode/models.json or a model folder; default m1809 = the model of the
realistic-PSF simulations). Solve calibrations: --cal_kind sim (default; the simulation sigma calibrations of the model, fixed c)
or real (the real-movie calibrations the production runs used).

After the run, the localization tables are rebuilt by simcode/strict.py on the simulation's FOV (build_table crop = (0, 0, W):
the selection region and the sigma-rescale tiles). Output: <out>/locs_p2_sim.npz (+ locs_p1_sim.npz) with columns as strict.py
(p, sel, ...; the selection = simcode.strict.select(T, crop=(0, 0, W))), and <out>/locs_sim_info.json (with the crop).
usage: CUDA_VISIBLE_DEVICES=0 python analysis/papersim/run_simcode.py <movie dir> [--model m1809] [--cal_kind sim|real]
          [--nframes N] [--out <movie dir>/v2_<model>] [--redo 0]"""
import os, sys, json, time, argparse, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)


def model(name):
    from simcode import config
    return config.model_set(name)


def strict_tables(out, W, px_nm, sig_u=None, pass2=True):
    import simcode.strict as SP
    crop = (0.0, 0.0, float(W))
    su = (sig_u.split(',') + [''])[:2] if sig_u else ['', '']
    from simcode import config
    info_all = {}
    for (tag, name), s in zip([('est', 'locs_p1_sim'), ('est2', 'locs_p2_sim')][:2 if pass2 else 1], su):
        T, info = SP.build_table(out, tag, px_nm, sig_u=config.resolve(s) if s else None, crop=crop)
        SP.write_table(T, info, os.path.join(out, name), ('npz',))
        info_all[name] = {k: v for k, v in info.items() if k != 'columns'}
        print(name, json.dumps(info_all[name], default=str), flush=True)
    json.dump(dict(crop=crop, **info_all), open(os.path.join(out, 'locs_sim_info.json'), 'w'), indent=1, default=str)
    return info_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('movie_dir')
    ap.add_argument('--model', default='m1809')
    ap.add_argument('--cal_kind', default='sim', choices=['sim', 'real'])
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--redo', type=int, default=0)
    ap.add_argument('--tables_only', action='store_true')
    ap.add_argument('--gt_diag', type=int, default=1, help='1: also pass 2 on the ground-truth pattern (diagnostic, locs_p2gt_sim.npz)')
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('psim-v2')
    md = os.path.abspath(a.movie_dir)
    g = json.load(open(os.path.join(md, 'gen.json')))
    W = int(g['W'])
    N = model(a.model)
    out = os.path.abspath(a.out or os.path.join(md, f'v2_{a.model}' + ('' if a.cal_kind == 'sim' else f'_{a.cal_kind}')))
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, '-m', 'simcode.run', '--movie', os.path.join(md, 'movie.tif'), '--calib', os.path.join(md, 'calib_sim.json'),
           '--out', out, '--crop', f'0,0,{W}', '--model', a.model, '--cal_kind', a.cal_kind, '--redo', str(a.redo)]
    if a.nframes: cmd += ['--nframes', str(a.nframes)]
    t0 = time.time()
    if not a.tables_only:
        with open(os.path.join(out, 'command.txt'), 'a') as f: f.write(time.ctime() + ': ' + ' '.join(cmd) + '\n')
        subprocess.run(cmd, check=True, cwd=REPO)
    T = time.time() - t0
    info = strict_tables(out, W, g['px_nm'], N['sig_u'])
    if a.gt_diag:
        # diagnostic (not the method): pass 2 with the GROUND-TRUTH pattern fed to the pass-2 networks and the solve
        # (run_uncalibrated --ref_pattern -> pass2_ref2_*.npz); table with the same crop and sigma rescale
        t1 = time.time()
        ucmd = [sys.executable, '-m', 'simcode.pipeline.run_uncalibrated', '--movie', os.path.join(md, 'movie.tif'),
                '--calib', os.path.join(md, 'calib_sim.json'), '--out', out, '--crop', f'0,0,{W}',
                '--sc_det', N['det'], '--sc_head', N['head'], '--frontend', 'sc2p', '--sc_cal', N[f'cal_{a.cal_kind}'],
                '--p2_det', N['p2_det'], '--p2_head', N['p2_head'], '--p2_cal', N[f'p2_cal_{a.cal_kind}'], '--stages', 'pass2',
                '--ref_pattern', os.path.join(md, 'gt_pattern.pt')]
        if a.nframes: ucmd += ['--nframes', str(a.nframes)]
        subprocess.run(ucmd, check=True, cwd=REPO)
        import simcode.strict as SP
        from simcode import config
        su = (N['sig_u'].split(',') + [''])[:2][1]
        Tg, ig = SP.build_table(out, 'ref2', g['px_nm'], None, sig_u=config.resolve(su) if su else None, crop=(0.0, 0.0, float(W)))
        SP.write_table(Tg, ig, os.path.join(out, 'locs_p2gt_sim'), ('npz',))
        print('locs_p2gt_sim (GT pattern, diagnostic)', json.dumps({k: v for k, v in ig.items() if k != 'columns'}, default=str), flush=True)
        print('gt_diag s', round(time.time() - t1, 1), flush=True)
    json.dump(dict(model=a.model, nets=N, cal_kind=a.cal_kind, nframes=a.nframes, run_s=T, cmd=cmd), open(os.path.join(out, 'papersim_run.json'), 'w'), indent=1)
    print('done', round(T, 1), 's', flush=True)


if __name__ == '__main__':
    main()
