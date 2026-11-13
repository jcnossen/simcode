"""Classical baselines (2D SMLM on 6-frame sums, SIMFLUX) on the 40 nm / 20 nm nanoruler movies: bl_run.py with the dataset
entry adapted. bl_run.py is NOT edited and not copied: its source is read, patched IN MEMORY and executed, so every setting
and override documented in bl_run.py applies unchanged. Exact changes (each textual replacement is asserted to occur the
stated number of times):

  text patches of the bl_run.py source
    1. "set_title('nr-base')" -> "set_title('nr-r24b')"                                     (1 x; process title)
    2. "'nr80'" -> "'<movie>'" (nr40 / nr20)                                                (3 x: the pp.DATASETS key, the
       dataset argument of paper_pipeline and one mention in the docstring; only a label, it ends up in smlm/info.json
       and run.json)
    3. "n_tf == n_rd == 30000" -> "n_tf == n_rd == NFRAMES"                                 (1 x; 39996 / 99996)
  module globals set after executing the patched source (before main())
    4. MOVIE = /mnt/data/dmdsf/simcode2_out/nanoruler/raw_<movie>/movie_reord.tif
    5. CAM = out/nanoruler/camera_maps_<movie>.npz (keys 'gain' = 0.45 and 'offset_adu' present, checked here)
    6. CALIB = out/nanoruler/calib_<movie>.json (default of --psf = this movie's in-situ PSF)
    7. OUTROOT = out/nanoruler/<movie>/ref; the run directory OUTROOT/<tag> is created beforehand as a symlink to
       /mnt/data/dmdsf/simcode2_out/nanoruler/<movie>/<tag> (ROI cache on the data disk)
    8. NFRAMES (see 3)
  optional, only with --cap N (NOT in the default run):
    9. smlmtorch's SpotDetector is created by the localizer with a hard-coded spots_per_frame = 100: per 6-frame window only
       the 100 candidates with the highest detection score are kept (before the edge rejection). On nr80 / nr40 no window
       reaches this (nr40 maximum 51 after the edge rejection); on nr20 at threshold 3.5 nearly every window does (80-99 ROIs
       per window after the edge rejection). --cap N wraps SpotDetector.__init__ in memory and passes spots_per_frame = N.
  added around main(): a sampler thread that records this process's GPU memory (nvidia-smi, every 2 s) and torch's peak
  allocated / reserved memory; written to <run dir>/gpu_mem.json. No effect on the results.
Everything else (ROI 10, threshold 3.5, fix_depth 0.9, block mode, bg_min 1.0, batch 8000 = the bl_run defaults) is given
on the command line or left at the bl_run default; --det_sigma must be given (quick-fit PSF sigma of the movie: nr40 1.82,
nr20 1.83; the bl_run default 1.84 is the nr80 value).

usage (GPU, through the lock wrapper):
  bash analysis/nanoruler/r24_gpu.sh 6 python analysis/nanoruler/r24b_run.py nr40 --det_sigma 1.82 [--tag bl_run] [--cap N] [bl_run options]"""
import os, sys, json, threading, subprocess, time
REPO = '/home/jc/simcode2'
SRC = f'{REPO}/analysis/nanoruler/bl_run.py'
NF = dict(nr40=39996, nr20=99996)


def main():
    movie = sys.argv[1]; assert movie in NF, 'first argument: nr40 or nr20'
    argv = [SRC] + sys.argv[2:]
    assert any(x.startswith('--det_sigma') for x in argv), 'give --det_sigma (nr40 1.82, nr20 1.83)'
    cap = None
    if '--cap' in argv:
        k = argv.index('--cap'); cap = int(argv[k + 1]); del argv[k:k + 2]
    tag = argv[argv.index('--tag') + 1] if '--tag' in argv else 'bl_run'
    if '--tag' not in argv: argv += ['--tag', tag]
    src = open(SRC).read()
    for old, new, n in (("set_title('nr-base')", "set_title('nr-r24b')", 1), ("'nr80'", f"'{movie}'", 3),
                        ('n_tf == n_rd == 30000', 'n_tf == n_rd == NFRAMES', 1)):
        assert src.count(old) == n, (old, src.count(old)); src = src.replace(old, new)
    ns = dict(__name__='bl_run_patched', __file__=SRC)
    exec(compile(src, SRC, 'exec'), ns)
    outroot = f'{REPO}/out/nanoruler/{movie}/ref'; big = f'/mnt/data/dmdsf/simcode2_out/nanoruler/{movie}/{tag}'
    os.makedirs(outroot, exist_ok=True); os.makedirs(big, exist_ok=True)
    link = os.path.join(outroot, tag)
    if not os.path.lexists(link): os.symlink(big, link)
    assert os.path.realpath(link) == os.path.realpath(big), link
    ns.update(MOVIE=f'/mnt/data/dmdsf/simcode2_out/nanoruler/raw_{movie}/movie_reord.tif', CAM=f'{REPO}/out/nanoruler/camera_maps_{movie}.npz',
              CALIB=f'{REPO}/out/nanoruler/calib_{movie}.json', OUTROOT=outroot, NFRAMES=NF[movie])
    import numpy as np
    cam = np.load(ns['CAM']); assert 'gain' in cam.files and 'offset_adu' in cam.files and cam['offset_adu'].shape == (400, 400)
    print(f'r24b_run: {movie}, movie {ns["MOVIE"]}, cam {ns["CAM"]} (gain {float(cam["gain"])}, mean offset {cam["offset_adu"].mean():.2f}), psf {ns["CALIB"]}, run dir {link} -> {big}', flush=True)

    peak = dict(nvidia_smi_mib=0, samples=0); stop = threading.Event(); pid = str(os.getpid())
    def sample():
        while not stop.wait(2.0):
            try:
                t = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
                for ln in t.splitlines():
                    a, b = [x.strip() for x in ln.split(',')[:2]]
                    if a == pid: peak['nvidia_smi_mib'] = max(peak['nvidia_smi_mib'], int(b)); peak['samples'] += 1
            except Exception: pass
    th = threading.Thread(target=sample, daemon=True); th.start()
    if cap is not None:                                                                                     # 9 (see docstring)
        from smlmtorch.smlm import detection
        _init = detection.SpotDetector.__init__
        def init(self, *args, **kw):
            assert kw.get('spots_per_frame') == 100, kw.keys()
            kw['spots_per_frame'] = cap; _init(self, *args, **kw)
        detection.SpotDetector.__init__ = init
        print(f'r24b_run: detector candidates per window capped at {cap} instead of 100', flush=True)
    sys.argv = argv; t0 = time.time()
    try: ns['main']()
    finally:
        stop.set()
        import torch
        peak.update(torch_max_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20, torch_max_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
                    wall_s=time.time() - t0, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
        json.dump(peak, open(os.path.join(link, 'gpu_mem.json'), 'w'), indent=1); print('r24b_run gpu memory:', peak, flush=True)


if __name__ == '__main__':
    main()
