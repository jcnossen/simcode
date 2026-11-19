"""SIMCODE v2 on a papersim movie: `python -m simcode.run` unchanged (production stages: pass 1, pattern estimate, solve,
pass 2), with the movie's calib_sim.json (averaged 1809/2 PSF + simulated camera), the whole 40 x 40 (or 50 x 50) FOV as the
output region, and a model (--model: a name of simcode/models.json or a model folder; default m1809 = the model of the
realistic-PSF simulations). Solve calibrations: the model's (cal_sim / cal_sim_pass2) + the brightness noise c estimated by
the run with its pattern (2026-11-17; the per-orientation c_a was removed 2026-11-19), as for real movies.

The run's own tables are the output: <out>/locs_p2_solve.hdf5 (the result since 2026-11-19: the pass-2 solve; + locs_p2_net,
locs_p1_net, locs_p1_solve), torch_psf Dataset format with the crop (0, 0, W) (the selection region); the selection =
Dataset.load_table(...).select().
--gt_diag 1 (diagnostic, not the method): pass 2 with the GROUND-TRUTH pattern fed to the pass-2 networks and the solve, in
<out>/gt_pattern/ (the run's pass-1 files, gt_pattern.pt as the pattern with the run's estimated brightness noise c; no
intensity map: the simulation has none) -> <out>/gt_pattern/locs_p2_solve.hdf5 (+ locs_p2_net).
usage: CUDA_VISIBLE_DEVICES=0 python analysis/papersim/run_simcode.py <movie dir> [--model m1809]
          [--nframes N] [--out <movie dir>/v2_<model>] [--redo 0]"""
import os, sys, json, time, argparse, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)


def model(name):
    from simcode import config
    return config.model_set(name)


def gt_pattern_pass2(out, md, W, N, nframes=0):
    """pass 2 (and its table) with the ground-truth pattern: a directory <out>/gt_pattern with links to the run's pass-1 chunk
    files, gt_pattern.pt as its pattern_est.pt and the run's record, then run_uncalibrated --stages pass2 and simcode.table"""
    import glob, shutil
    d = os.path.join(out, 'gt_pattern'); os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(out, 'pass1_*')):
        t = os.path.join(d, os.path.basename(f))
        if not os.path.exists(t): os.symlink(f, t)
    import torch
    G = torch.load(os.path.join(md, 'gt_pattern.pt'), weights_only=False); E = torch.load(os.path.join(out, 'pattern_est.pt'), weights_only=False)
    G['cmult'] = float(E['cmult'])                                               # the run's estimated brightness noise
    torch.save(G, os.path.join(d, 'pattern_est.pt'))
    shutil.copyfile(os.path.join(out, 'run_config.json'), os.path.join(d, 'run_config.json'))     # the run record (model)
    ucmd = [sys.executable, '-m', 'simcode.processing.run_uncalibrated', '--movie', os.path.join(md, 'movie.tif'),
            '--calib', os.path.join(md, 'calib_sim.json'), '--out', d, '--crop', f'0,0,{W}',
            '--detector', N['detector'], '--emitter', N['emitter'], '--cal', N['cal_sim'],
            '--detector_pass2', N['detector_pass2'], '--emitter_pass2', N['emitter_pass2'], '--cal_pass2', N['cal_sim_pass2'],
            '--stages', 'pass2']
    if nframes: ucmd += ['--nframes', str(nframes)]
    subprocess.run(ucmd, check=True, cwd=REPO)
    for t in ('locs_p2_net', 'locs_p2_solve'):
        subprocess.run([sys.executable, '-m', 'simcode.table', d, '--table', t, '--crop', f'0,0,{W}'], check=True, cwd=REPO)
    return os.path.join(d, 'locs_p2_solve.hdf5')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('movie_dir')
    ap.add_argument('--model', default='m1809')
    ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--redo', type=int, default=0)
    ap.add_argument('--gt_diag', type=int, default=1, help='1: also pass 2 on the ground-truth pattern (diagnostic, gt_pattern/locs_p2_solve.hdf5)')
    a = ap.parse_args()
    from simcode.util.proctitle import set_title; set_title('psim-v2')
    md = os.path.abspath(a.movie_dir)
    g = json.load(open(os.path.join(md, 'gen.json')))
    W = int(g['W'])
    N = model(a.model)
    out = os.path.abspath(a.out or os.path.join(md, f'v2_{a.model}'))
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, '-m', 'simcode.run', '--movie', os.path.join(md, 'movie.tif'), '--calib', os.path.join(md, 'calib_sim.json'),
           '--out', out, '--crop', f'0,0,{W}', '--model', a.model, '--redo', str(a.redo)]
    if a.nframes: cmd += ['--nframes', str(a.nframes)]
    t0 = time.time()
    with open(os.path.join(out, 'command.txt'), 'a') as f: f.write(time.ctime() + ': ' + ' '.join(cmd) + '\n')
    subprocess.run(cmd, check=True, cwd=REPO)
    T = time.time() - t0
    if a.gt_diag:
        t1 = time.time()
        print('GT-pattern diagnostic ->', gt_pattern_pass2(out, md, W, N, a.nframes), 'in', round(time.time() - t1, 1), 's', flush=True)
    json.dump(dict(model=a.model, nets=N, nframes=a.nframes, run_s=T, cmd=cmd), open(os.path.join(out, 'papersim_run.json'), 'w'), indent=1)
    print('done', round(T, 1), 's', flush=True)


if __name__ == '__main__':
    main()
